"""
SmartVisionX Arrival Fusion
============================

Runs FOUR independent "arrived?" signals in parallel on every frame
and combines them with a weighted vote.

Signals (rebalanced for indoor reliability):
  1. Scene classifier    (label match)                     weight 0.25  ★ high
  2. Object landmark     (YOLO detections vs. expected)    weight 0.25  ★ high
  3. Path metrics (M4)   (10-metric profile similarity)    weight 0.25  ★ high
  4. Visual fingerprint  (pHash + colour histogram)        weight 0.15  (supporting)

(Removed: path signature — too fragile with only 3 numbers.)

Arrival is declared when ALL of:
  1. at least MIN_AGREE_COUNT signals (2 of 4) are individually above
     their per-signal "agrees" threshold
  2. the weighted final score is at or above ARRIVAL_THRESHOLD
  3. the condition has been true for ARRIVAL_STREAK consecutive frames
  4. the path is not currently in a STOP / UNRECOGNIZED state
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Pipelines.path_metrics import (
    PathProfile, compute_metrics, profile_distance, navigation_gradient,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Final weighted score must be at or above this for arrival.
# 0.40 is reachable when 2 of 4 high-weight signals are at their "agrees"
# threshold:
#   0.25*0.50 + 0.25*0.45 + 0.25*0.65 = 0.400  (>= 0.40 ✓)
# Or with 3 strong signals:
#   0.25*0.50 + 0.25*0.45 + 0.25*0.65 + 0.15*0.65 = 0.498  (>= 0.40 ✓)
ARRIVAL_THRESHOLD    = 0.40

MIN_AGREE_COUNT      = 2     # of 4 — lowered from 3-of-5 to match the new design

# Two consecutive frames is enough — any longer makes the user feel the
# system is ignoring them at the moment of arrival.
ARRIVAL_STREAK       = 2
WINDOW_FRAMES        = 10
MIN_FRAMES_EVALUATED = 2

# Reduced from 180s — most indoor journeys are under 90s.
HARD_TIMEOUT_SEC     = 90.0

# Weights (must sum to 1.0) — rebalanced to favour the reliable signals.
WEIGHTS = {
    "scene":       0.25,    # ★ high — room classifier
    "object":      0.25,    # ★ high — landmark objects
    "path_metric": 0.25,    # ★ high — 10-metric M4 profile
    "visual":      0.15,    # supporting — pHash is lighting-sensitive
    # "path" removed: 3-number signature was too fragile
}

# Per-signal "agrees" thresholds (0..1).
# Visual threshold lowered (from 0.65 → 0.55) so pHash can still contribute.
THRESHOLDS = {
    "scene":       0.45,
    "object":      0.40,
    "path_metric": 0.55,
    "visual":      0.55,
}

BLOCKING_COMMANDS = ("STOP", "UNRECOGNIZED PATH")

_SCENE_ALIASES = {
    "indoor_passage":  "corridor",
    "street_road":     "street",
    "lecture_room":    "learning_space",
    "market":          "market_outdoor",
    "restaurant":      "eating_place",
}

# Active signals (used by helpers / status reports)
ACTIVE_SIGNALS = ("scene", "object", "path_metric", "visual")


# ----------------------------------------------------------------------
# Helper — synthetic safe mask from 3-region scores (M4)
# ----------------------------------------------------------------------

def _safe_mask_from_scores(safe_scores: Optional[Dict[str, float]],
                           frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Build a synthetic 128x256 safe mask from the 3-region scores so
    compute_metrics() can run. If a real mask is available via the
    navigation pipeline it is preferred; this proxy is used when only
    the L/C/R scores are passed in.
    """
    if not safe_scores:
        return None
    h, w = 128, 256
    mask = np.zeros((h, w), dtype=np.float32)
    third = w // 3
    center_val = float(safe_scores.get("center", 0.0))
    left_val = float(safe_scores.get("left", 0.0))
    right_val = float(safe_scores.get("right", 0.0))
    roi_h = int(h * 0.4)
    mask[roi_h:, :third] = left_val
    mask[roi_h:, third:2 * third] = center_val
    mask[roi_h:, 2 * third:] = right_val
    for row in range(roi_h, h):
        decay = 1.0 - 0.4 * (row - roi_h) / max(1, h - roi_h)
        mask[row, :] *= decay
    return mask


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class SignalReading:
    name: str
    score: float
    raw: Dict[str, Any]
    agrees: bool
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score":  round(float(self.score), 3),
            "agrees": bool(self.agrees),
            "weight": float(self.weight),
            "raw":    self.raw,
        }


@dataclass
class ArrivalResult:
    arrived: bool
    final_score: float
    agree_count: int
    streak: int
    blocked_by_safety: bool
    elapsed_sec: float
    signals: Dict[str, SignalReading]
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arrived":           self.arrived,
            "final_score":       round(self.final_score, 3),
            "agree_count":       self.agree_count,
            "streak":            self.streak,
            "blocked_by_safety": self.blocked_by_safety,
            "elapsed_sec":       round(self.elapsed_sec, 1),
            "reason":            self.reason,
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
        }


# ----------------------------------------------------------------------
# Individual signal evaluators (pure functions, no I/O between them)
# ----------------------------------------------------------------------

def _signal_visual(frame: np.ndarray,
                   target_loc: Dict[str, Any],
                   location_manager) -> SignalReading:
    """Visual fingerprint: pHash + colour histogram vs. all saved refs."""
    if frame is None or frame.size == 0 or not target_loc.get("references"):
        return SignalReading("visual", 0.0,
                             {"reason": "no frame / no refs"},
                             False, WEIGHTS["visual"])
    match = location_manager.match_location(frame, only=target_loc["name"])
    if match is None:
        return SignalReading("visual", 0.0,
                             {"reason": "no match"},
                             False, WEIGHTS["visual"])
    score = float(match["match_score"])
    return SignalReading(
        name="visual",
        score=score,
        raw={
            "match_score": score,
            "hash_sim":    match.get("hash_sim"),
            "hist_sim":    match.get("hist_sim"),
            "ref_count":   match.get("reference_count"),
        },
        agrees=score >= THRESHOLDS["visual"],
        weight=WEIGHTS["visual"],
    )


def _signal_objects(detections: Sequence[Dict[str, Any]],
                    target_loc: Dict[str, Any]) -> SignalReading:
    """Object landmark: fraction of expected_objects seen by YOLO now."""
    expected = [o.lower() for o in (target_loc.get("expected_objects") or [])]
    if not expected:
        return SignalReading("object", 0.0,
                             {"reason": "no expected objects stored"},
                             False, WEIGHTS["object"])
    seen = {str(d.get("name", "")).lower() for d in (detections or [])}
    if not seen:
        return SignalReading("object", 0.0,
                             {"expected": expected, "seen": []},
                             False, WEIGHTS["object"])

    matched = [o for o in expected if any(o in s or s in o for s in seen)]
    ratio = len(matched) / max(1, len(expected))
    return SignalReading(
        name="object",
        score=float(ratio),
        raw={"expected": expected, "seen": sorted(seen),
             "matched": matched, "ratio": round(ratio, 3)},
        agrees=ratio >= THRESHOLDS["object"],
        weight=WEIGHTS["object"],
    )


def _signal_scene(current_scene: str,
                  current_scene_conf: float,
                  target_loc: Dict[str, Any]) -> SignalReading:
    """
    Scene classifier: current label matches the saved expected label?
    ★ High-priority signal — boosted to count partial matches.
    """
    expected = (target_loc.get("expected_scene") or "").strip().lower()
    if not expected:
        return SignalReading("scene", 0.0,
                             {"reason": "no expected scene stored"},
                             False, WEIGHTS["scene"])

    cur = (current_scene or "").strip().lower()
    if not cur or current_scene_conf < 0.25:
        return SignalReading("scene", 0.0,
                             {"expected": expected, "current": cur,
                              "current_conf": current_scene_conf},
                             False, WEIGHTS["scene"])

    label_match = (cur == expected)
    if not label_match:
        cur_a = _SCENE_ALIASES.get(cur, cur)
        exp_a = _SCENE_ALIASES.get(expected, expected)
        label_match = (cur_a == exp_a)

    # If the label matches, the score is the classifier's confidence.
    # If the label doesn't match, the score is 0 — but we also accept
    # "near-miss" scenes at reduced confidence for robustness.
    if label_match:
        score = float(current_scene_conf)
    else:
        # Near-miss: same family of rooms. Use conf * 0.4 so it can
        # contribute without overpowering a true match.
        near_miss_families = {
            "bathroom":  {"bedroom", "kitchen"},
            "bedroom":   {"bathroom", "kitchen"},
            "kitchen":   {"restaurant", "eating_place", "supermarket"},
            "corridor":  {"indoor_passage", "staircase", "waiting_room"},
            "library":   {"learning_space", "lecture_room", "work_space"},
            "classroom": {"learning_space", "lecture_room"},
        }
        family = near_miss_families.get(expected, set())
        if cur in family:
            score = float(current_scene_conf) * 0.4
        else:
            score = 0.0

    return SignalReading(
        name="scene",
        score=float(score),
        raw={"expected": expected, "current": cur,
             "current_conf": round(current_scene_conf, 3),
             "label_match": bool(label_match)},
        agrees=score >= THRESHOLDS["scene"],
        weight=WEIGHTS["scene"],
    )


def _signal_path_metric(frame: np.ndarray,
                        nav_safe_scores: Optional[Dict[str, float]],
                        target_loc: Dict[str, Any]) -> SignalReading:
    """
    M4: 10-metric path profile similarity vs. saved profile.
    ★ High-priority signal — most reliable for indoor navigation.
    """
    refs = target_loc.get("references", [])
    profile_refs = [r for r in refs if r.get("path_profile")]
    if not profile_refs:
        return SignalReading("path_metric", 0.5,
                             {"reason": "no path_profile stored",
                              "neutral": True},
                             False, WEIGHTS["path_metric"])

    target_profile = PathProfile.from_dict(profile_refs[0].get("path_profile"))
    if target_profile.is_empty():
        return SignalReading("path_metric", 0.5,
                             {"reason": "empty path_profile",
                              "neutral": True},
                             False, WEIGHTS["path_metric"])

    cur_metrics = compute_metrics(
        safe_mask=_safe_mask_from_scores(nav_safe_scores, frame),
        safe_scores=nav_safe_scores,
    )
    sim = profile_distance(target_profile, cur_metrics)
    grad = navigation_gradient(cur_metrics, target_profile, nav_safe_scores)

    return SignalReading(
        name="path_metric",
        score=float(sim),
        raw={
            "similarity":         round(sim, 3),
            "gradient_command":   grad.get("command"),
            "gradient_confidence": grad.get("confidence"),
            "gradient_reason":    grad.get("reasoning"),
            "deltas":             grad.get("deltas"),
        },
        agrees=sim >= THRESHOLDS["path_metric"],
        weight=WEIGHTS["path_metric"],
    )


# ----------------------------------------------------------------------
# Path signature capture (called by the save tool / save endpoint)
# ----------------------------------------------------------------------

def compute_path_signature(safe_scores: Optional[Dict[str, float]]) -> List[float]:
    """Serialise the safe-path vector for storage alongside a reference."""
    if not safe_scores:
        return []
    return [
        float(safe_scores.get("left",   0.0)),
        float(safe_scores.get("center", 0.0)),
        float(safe_scores.get("right",  0.0)),
    ]


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

class ArrivalFusion:
    """Combines the four reliable signals into arrived / not-arrived."""

    def __init__(self) -> None:
        self._streak: int = 0
        self._best_score: float = 0.0
        self._score_history: deque = deque(maxlen=WINDOW_FRAMES)
        self._last_arrival: Optional[ArrivalResult] = None
        self._frames_evaluated: int = 0
        self._start_time: float = 0.0
        self._active: bool = False

    def begin(self) -> None:
        self._streak = 0
        self._best_score = 0.0
        self._score_history.clear()
        self._last_arrival = None
        self._frames_evaluated = 0
        self._start_time = time.time()
        self._active = True

    def end(self) -> None:
        self._active = False
        self._streak = 0
        self._score_history.clear()

    def is_active(self) -> bool:
        return self._active

    def update(self,
               *,
               frame: np.ndarray,
               detections: Sequence[Dict[str, Any]],
               current_scene: str,
               current_scene_conf: float,
               nav_safe_scores: Optional[Dict[str, float]],
               nav_command: str,
               target_loc: Dict[str, Any],
               location_manager) -> ArrivalResult:
        if not self._active:
            self.begin()

        # Run the four reliable signals
        sig_scene       = _signal_scene(current_scene, current_scene_conf, target_loc)
        sig_object      = _signal_objects(detections, target_loc)
        sig_path_metric = _signal_path_metric(frame, nav_safe_scores, target_loc)
        sig_visual      = _signal_visual(frame, target_loc, location_manager)

        signals = {
            "scene":       sig_scene,
            "object":      sig_object,
            "path_metric": sig_path_metric,
            "visual":      sig_visual,
        }

        # Weighted vote
        weighted_sum = sum(s.score * s.weight for s in signals.values())
        total_w = sum(s.weight for s in signals.values()) or 1.0
        final_score = weighted_sum / total_w

        # Count agreeing signals
        agree_count = sum(1 for s in signals.values() if s.agrees)

        # Safety override — never declare arrival while the nav is blocked
        blocked_by_safety = any(b in (nav_command or "")
                                for b in BLOCKING_COMMANDS)
        if blocked_by_safety:
            self._streak = 0
            self._score_history.append(0.0)
            self._frames_evaluated += 1
            return ArrivalResult(
                arrived=False, final_score=0.0, agree_count=agree_count,
                streak=0, blocked_by_safety=True,
                elapsed_sec=time.time() - self._start_time,
                signals=signals,
                reason=f"Path unsafe ({nav_command}); arrival blocked.",
            )

        # Streak tracking — both agree-count AND final score above threshold
        self._score_history.append(final_score)
        self._best_score = max(self._best_score, final_score)
        self._frames_evaluated += 1
        reached = (agree_count >= MIN_AGREE_COUNT) and (final_score >= ARRIVAL_THRESHOLD)
        self._streak = self._streak + 1 if reached else 0

        arrived = (
            self._frames_evaluated >= MIN_FRAMES_EVALUATED
            and self._streak >= ARRIVAL_STREAK
        )

        # Hard timeout
        elapsed = time.time() - self._start_time
        if elapsed > HARD_TIMEOUT_SEC and not arrived:
            self.end()
            return ArrivalResult(
                arrived=False, final_score=final_score, agree_count=agree_count,
                streak=self._streak, blocked_by_safety=False,
                elapsed_sec=elapsed, signals=signals,
                reason=f"timeout: failed to confirm arrival within "
                       f"{HARD_TIMEOUT_SEC:.0f}s",
            )

        if arrived:
            # Build a readable list of which signals agreed
            agreed_names = [name for name, s in signals.items() if s.agrees]
            reason = (f"Final score {final_score:.2f} >= {ARRIVAL_THRESHOLD}, "
                      f"{agree_count}/4 signals agree "
                      f"({', '.join(agreed_names)}), "
                      f"streak {self._streak}/{ARRIVAL_STREAK}.")
            self._last_arrival = ArrivalResult(
                arrived=True, final_score=final_score, agree_count=agree_count,
                streak=self._streak, blocked_by_safety=False,
                elapsed_sec=elapsed, signals=signals, reason=reason,
            )
            self.end()
            return self._last_arrival

        if final_score < 0.30:
            short = "no clear match"
        elif final_score < 0.50:
            short = f"approaching ({agree_count}/4 signals agree)"
        else:
            short = (f"high score but only {agree_count}/4 agree; "
                     f"streak {self._streak}/{ARRIVAL_STREAK}")
        return ArrivalResult(
            arrived=False, final_score=final_score, agree_count=agree_count,
            streak=self._streak, blocked_by_safety=False,
            elapsed_sec=elapsed, signals=signals, reason=short,
        )

    @property
    def last_arrival(self) -> Optional[ArrivalResult]:
        return self._last_arrival

    def smoothed_score(self) -> float:
        if not self._score_history:
            return 0.0
        return sum(self._score_history) / len(self._score_history)

    def progress_text(self, width: int = 16) -> str:
        score = self.smoothed_score()
        filled = int(round(min(1.0, max(0.0, score)) * width))
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {int(score * 100):3d}%   {self._streak}/{ARRIVAL_STREAK}"
