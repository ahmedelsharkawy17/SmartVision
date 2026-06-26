"""
SmartVisionX Path Metrics — composite path descriptor (M4)
==========================================================

Computes 10 scalar metrics from a single frame's navigation output,
and a "path profile" (mean/std/min/max/histogram) from a set of frames.

At runtime:
  - compute_metrics()        -> current frame -> dict of 10 scalars
  - profile_distance()       -> how far is the current frame from the saved profile?
  - navigation_gradient()    -> which direction to walk to reduce that distance?

Metrics are all in [0, 1] so they are directly comparable and fusion-friendly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Metric keys — keep this list as the single source of truth
# ---------------------------------------------------------------------------

METRIC_KEYS: List[str] = [
    "free_space_ratio",       # 1 - obstacle density in the lower 60% of the frame
    "corridor_score",         # how corridor-like the safe mask is
    "avg_safe_width",         # average horizontal safe width
    "obstacle_density",       # 1 - free_space_ratio
    "approach_clarity",       # how clear the centre column is
    "forward_safety_avg",     # mean of the centre safe score
    "lateral_safety_balance", # how balanced left vs right is
    "depth_openness",         # how open the depth looks
    "path_centrality",        # is the safe path in the middle?
    "vertical_openness",      # is the upper part of the frame open (sky/ceiling)?
]


# ---------------------------------------------------------------------------
# Frame-level metric computation
# ---------------------------------------------------------------------------

def _normalize_mask(mask: Optional[np.ndarray], out_h: int = 128, out_w: int = 256) -> np.ndarray:
    """Resize a safe mask to the canonical (OUT_H, OUT_W) used by the nav pipeline."""
    if mask is None:
        return np.zeros((out_h, out_w), dtype=np.float32)
    if mask.shape != (out_h, out_w):
        import cv2
        mask = cv2.resize(mask.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    return mask.astype(np.float32)


def _corridor_score(mask: np.ndarray) -> float:
    """A corridor looks like: a wide horizontal band in the lower part of the frame."""
    h, w = mask.shape
    if h == 0 or w == 0:
        return 0.0
    band = mask[int(h * 0.4):int(h * 0.85), :]
    if band.size == 0:
        return 0.0
    col_safe = band.mean(axis=0)
    wide = (col_safe > 0.4).mean()
    fill = band.mean()
    return float(np.clip(0.55 * wide + 0.45 * fill, 0.0, 1.0))


def _avg_safe_width(mask: np.ndarray) -> float:
    """Average horizontal width of safe region (in normalized units, max = 1.0)."""
    h, w = mask.shape
    if h == 0 or w == 0:
        return 0.0
    band = mask[int(h * 0.5):, :]
    widths = []
    for col in band.mean(axis=0):
        if col <= 0:
            continue
        widths.append(float(col))
    return float(np.mean(widths)) if widths else 0.0


def _lateral_balance(safe_scores: Dict[str, float]) -> float:
    """How balanced left/right are. 0.5 = perfectly balanced, 0 or 1 = all on one side."""
    l = float(safe_scores.get("left", 0.0))
    r = float(safe_scores.get("right", 0.0))
    total = l + r
    if total < 1e-6:
        return 0.5
    return float(l / total)


def _depth_openness(safe_mask: np.ndarray) -> float:
    """How open the far depth is — measured as the safety in the upper half."""
    h, w = safe_mask.shape
    if h == 0:
        return 0.0
    upper = safe_mask[:int(h * 0.5), :]
    return float(upper.mean())


def _path_centrality(safe_mask: np.ndarray) -> float:
    """How central the safe path is in the horizontal direction."""
    h, w = safe_mask.shape
    if h == 0 or w == 0:
        return 0.0
    band = safe_mask[int(h * 0.6):, :]
    if band.size == 0:
        return 0.0
    col_means = band.mean(axis=0)
    weighted_x = float(np.sum(np.arange(w) * col_means))
    total = float(np.sum(col_means))
    if total < 1e-6:
        return 0.0
    cx = weighted_x / total
    centre = (w - 1) / 2.0
    deviation = abs(cx - centre) / centre
    return float(np.clip(1.0 - deviation, 0.0, 1.0))


def _vertical_openness(safe_mask: np.ndarray) -> float:
    """How open the upper part of the safe mask is (proxy for sky / ceiling)."""
    h, w = safe_mask.shape
    if h == 0:
        return 0.0
    top = safe_mask[:int(h * 0.3), :]
    return float(top.mean())


def compute_metrics(safe_mask: Optional[np.ndarray],
                    safe_scores: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute the 10 path metrics from a single frame.

    Parameters
    ----------
    safe_mask   : 2D numpy array (HxW) in [0, 1] — the safe-path probability mask.
    safe_scores : dict with keys 'left', 'center', 'right' in [0, 1].

    Returns
    -------
    dict with the 10 metric keys, all in [0, 1].
    """
    safe_mask = _normalize_mask(safe_mask)
    safe_scores = safe_scores or {"left": 0.0, "center": 0.0, "right": 0.0}

    free_space = float(safe_mask[int(safe_mask.shape[0] * 0.4):, :].mean())
    obstacle = float(np.clip(1.0 - free_space, 0.0, 1.0))
    forward = float(safe_scores.get("center", 0.0))

    return {
        "free_space_ratio":       free_space,
        "corridor_score":         _corridor_score(safe_mask),
        "avg_safe_width":         _avg_safe_width(safe_mask),
        "obstacle_density":       obstacle,
        "approach_clarity":       forward,
        "forward_safety_avg":     forward,
        "lateral_safety_balance": _lateral_balance(safe_scores),
        "depth_openness":         _depth_openness(safe_mask),
        "path_centrality":        _path_centrality(safe_mask),
        "vertical_openness":      _vertical_openness(safe_mask),
    }


# ---------------------------------------------------------------------------
# Path profile (multi-frame statistics) — built during save
# ---------------------------------------------------------------------------

class PathProfile:
    """
    Stores mean / std / min / max / histogram for each of the 10 metrics,
    computed across a sequence of frames captured at save time.

    Compatible with JSON serialisation.
    """

    HIST_BINS = 8

    def __init__(self, frames_metrics: Optional[Sequence[Dict[str, float]]] = None):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        if frames_metrics:
            self.update(frames_metrics)

    def update(self, frames_metrics: Sequence[Dict[str, float]]) -> None:
        if not frames_metrics:
            return
        for key in METRIC_KEYS:
            values = np.array([float(m.get(key, 0.0)) for m in frames_metrics], dtype=np.float32)
            if values.size == 0:
                continue
            hist, _ = np.histogram(values, bins=self.HIST_BINS, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            s = hist.sum()
            if s > 0:
                hist = hist / s
            self.metrics[key] = {
                "mean":      float(values.mean()),
                "std":       float(values.std()),
                "min":       float(values.min()),
                "max":       float(values.max()),
                "histogram": hist.tolist(),
            }

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "keys": list(METRIC_KEYS), "metrics": self.metrics}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PathProfile":
        p = cls()
        if isinstance(data, dict) and "metrics" in data:
            p.metrics = data["metrics"]
        return p

    def is_empty(self) -> bool:
        return len(self.metrics) == 0


# ---------------------------------------------------------------------------
# Profile distance — how far is the current frame from the saved profile?
# ---------------------------------------------------------------------------

def _histogram_intersection(h1: Sequence[float], h2: Sequence[float]) -> float:
    """Histogram intersection similarity in [0, 1]."""
    a = np.asarray(h1, dtype=np.float32)
    b = np.asarray(h2, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.minimum(a, b).sum())


def _unit_hist(value: float, bins: int = 8) -> List[float]:
    """Build a soft one-hot histogram with a small spread, in [0, 1]."""
    h = np.full(bins, 0.05, dtype=np.float32)
    pos = float(np.clip(value, 0.0, 1.0)) * (bins - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    h[lo] += 0.7
    if hi != lo:
        h[hi] += 0.3
    s = h.sum()
    return (h / s).tolist()


def profile_distance(profile: PathProfile,
                     current_metrics: Dict[str, float]) -> float:
    """
    Returns a similarity score in [0, 1]:
        1.0 = current frame is highly consistent with the saved profile
        0.0 = current frame is nothing like the saved profile
    """
    if profile.is_empty():
        return 0.5  # neutral

    per_key_sim = []
    for key in METRIC_KEYS:
        m = profile.metrics.get(key)
        if not m:
            continue
        cur = float(np.clip(current_metrics.get(key, 0.0), 0.0, 1.0))
        hist_sim = _histogram_intersection(m.get("histogram", []), _unit_hist(cur))
        mean = float(m.get("mean", 0.5))
        std = max(float(m.get("std", 0.1)), 0.05)
        z = (cur - mean) / std
        gauss = float(np.exp(-0.5 * z * z))
        per_key_sim.append(0.5 * hist_sim + 0.5 * gauss)

    if not per_key_sim:
        return 0.5
    return float(np.clip(np.mean(per_key_sim), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Navigation gradient — which direction to walk?
# ---------------------------------------------------------------------------

def navigation_gradient(current_metrics: Dict[str, float],
                        target_profile: PathProfile,
                        safe_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Compare current metrics to the target profile and produce a navigation hint.

    Returns
    -------
    dict with:
        command    : "MOVE FORWARD" | "TURN LEFT" | "TURN RIGHT" | "STOP / SCAN" | "ALIGN CENTRE"
        confidence : 0..1
        reasoning  : short human-readable string
        deltas     : per-metric delta (target_mean - current)
    """
    safe_scores = safe_scores or {"left": 0.0, "center": 0.0, "right": 0.0}
    if target_profile.is_empty():
        return {
            "command":    "MOVE FORWARD",
            "confidence": 0.0,
            "reasoning":  "No target profile available.",
            "deltas":     {},
        }

    deltas: Dict[str, float] = {}
    for key in METRIC_KEYS:
        m = target_profile.metrics.get(key, {})
        mean = float(m.get("mean", 0.5))
        cur = float(np.clip(current_metrics.get(key, 0.0), 0.0, 1.0))
        deltas[key] = mean - cur

    fwd_delta = deltas.get("forward_safety_avg", 0.0)
    lat_delta = deltas.get("lateral_safety_balance", 0.0)
    cor_delta = deltas.get("corridor_score", 0.0)
    cen_delta = deltas.get("path_centrality", 0.0)

    left_safe = float(safe_scores.get("left", 0.0))
    right_safe = float(safe_scores.get("right", 0.0))

    command = "MOVE FORWARD"
    reasoning = ""
    confidence = 0.5

    if fwd_delta > 0.15 and cor_delta > 0.10:
        command = "MOVE FORWARD"
        confidence = float(np.clip(0.5 + 0.5 * min(fwd_delta, 0.5), 0.5, 1.0))
        reasoning = "Target is more open ahead — move forward."

    elif abs(lat_delta) > 0.18:
        if lat_delta > 0:
            if left_safe >= right_safe:
                command = "TURN LEFT"
                reasoning = "Target's open space is on the left — turn left."
            else:
                command = "MOVE FORWARD"
                reasoning = "Left side is more open — move forward and drift left."
        else:
            if right_safe >= left_safe:
                command = "TURN RIGHT"
                reasoning = "Target's open space is on the right — turn right."
            else:
                command = "MOVE FORWARD"
                reasoning = "Right side is more open — move forward and drift right."
        confidence = float(np.clip(0.5 + 0.5 * min(abs(lat_delta), 0.5), 0.5, 1.0))

    elif cen_delta > 0.20 and abs(lat_delta) <= 0.18:
        command = "ALIGN CENTRE"
        reasoning = "Target's path is centred — align yourself to the middle."
        confidence = 0.6

    elif fwd_delta < -0.20:
        command = "STOP / SCAN"
        reasoning = "Current area is wider than target — slow down and scan."
        confidence = 0.55

    else:
        command = "MOVE FORWARD"
        confidence = 0.6
        reasoning = "Direction is well aligned — continue forward."

    return {
        "command":    command,
        "confidence": round(float(confidence), 3),
        "reasoning":  reasoning,
        "deltas":     {k: round(v, 3) for k, v in deltas.items()},
    }
