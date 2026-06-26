"""
SmartVisionX Location Navigation — active guidance + multi-signal arrival
==========================================================================

State machine:

    idle → searching → approaching → arrived / failed

M4 additions:
  - On every update, the current frame's 10 path metrics are compared
    to the saved profile of the target location.
  - navigation_gradient() produces a directional cue that is appended
    to the spoken message ("Target is more open ahead — move forward.")

M5 additions (this update):
  - _compute_danger_alert() checks the nav pipeline + YOLO detections
    for imminent danger and returns (level, message). The danger info
    is passed to the decision engine so safety alerts ALWAYS take
    priority over location guidance (except arrived / failed).
  - _compute_location_direction() combines the safe-path nav command
    with the M4 path gradient to give TARGET-AWARE directions instead
    of the generic "move forward / turn left" from the nav pipeline.
  - The returned dict now contains:
        location_danger_level   : "none" | "caution" | "warning" | "stop"
        location_danger_message : spoken text for the danger alert
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Pipelines.arrival_fusion import (
    ArrivalFusion, ArrivalResult, compute_path_signature,
)
from Pipelines.path_metrics import (
    PathProfile, compute_metrics, navigation_gradient,
)


class LocationNavigator:
    TIMEOUT_SEC         = 120.0
    GUIDANCE_COOLDOWN   = 4.0
    STARTUP_GRACE_SEC   = 0.5

    def __init__(self, location_manager) -> None:
        self.location_manager = location_manager
        self.target: Optional[str] = None
        self.state: str = "idle"
        self.start_time: float = 0.0
        self._last_message: str = ""
        self._last_message_time: float = 0.0
        self._was_approaching: bool = False
        self._last_direction: str = ""
        self._last_direction_time: float = 0.0
        # M4: gradient cue tracking
        self._last_gradient_command: str = ""
        self._last_gradient_time: float = 0.0
        # M5: danger tracking
        self._last_danger_level: str = "none"
        self._last_danger_message: str = ""
        self.fusion = ArrivalFusion()

    def is_active(self) -> bool:
        return (self.target is not None
                and self.state not in {"arrived", "failed", "idle"})

    def set_target(self, target: str,
                   path_signature: Optional[List[float]] = None) -> Dict[str, Any]:
        if not target:
            return self.status()
        loc = self.location_manager.get_location(target)
        if loc is None:
            return {
                "location_state":   "failed",
                "location_target":  target,
                "location_message": f"I do not know a place called {target.replace('_', ' ')}. "
                                    f"Please save it first.",
                "location_score":   0.0,
                "location_danger_level":   "none",
                "location_danger_message": "",
            }
        self.target = target
        self.state = "searching"
        self.start_time = time.time()
        self._last_message_time = time.time()
        self._last_direction = ""
        self._last_direction_time = 0.0
        self._was_approaching = False
        self._last_gradient_command = ""
        self._last_gradient_time = 0.0
        self._last_danger_level = "none"
        self._last_danger_message = ""
        self.fusion.begin()

        display = loc.get("display_name") or target.replace("_", " ")
        msg = (f"Navigating to {display}. "
               f"I will guide you there with voice directions.")
        self._last_message = msg
        return {
            "location_state":   "searching",
            "location_target":  target,
            "location_message": msg,
            "location_score":   0.0,
            "location_danger_level":   "none",
            "location_danger_message": "",
        }

    def stop(self, message: Optional[str] = None) -> Dict[str, Any]:
        self.target = None
        self.state = "idle"
        self._was_approaching = False
        self._last_direction = ""
        self._last_gradient_command = ""
        self._last_danger_level = "none"
        self._last_danger_message = ""
        self.fusion.end()
        return {
            "location_state":   "idle",
            "location_target":  None,
            "location_message": message or "Navigation cancelled.",
            "location_score":   0.0,
            "location_danger_level":   "none",
            "location_danger_message": "",
        }

    def status(self) -> Dict[str, Any]:
        return {
            "location_state":   self.state,
            "location_target":  self.target,
            "location_message": "",
            "location_score":   round(self.fusion.smoothed_score(), 3),
            "location_danger_level":   self._last_danger_level,
            "location_danger_message": self._last_danger_message,
        }

    def update(self,
               frame: np.ndarray,
               nav_result: Dict[str, Any],
               detections: Optional[List[Dict[str, Any]]] = None,
               scene: Optional[str] = None,
               scene_conf: float = 0.0) -> Dict[str, Any]:
        if not self.target or self.state == "idle":
            return {
                "location_state":   "idle",
                "location_target":  None,
                "location_message": "",
                "location_score":   0.0,
                "location_danger_level":   "none",
                "location_danger_message": "",
            }

        target_loc = self.location_manager.get_location(self.target)
        if target_loc is None:
            self.state = "failed"
            self.fusion.end()
            return {
                "location_state":   "failed",
                "location_target":  self.target,
                "location_message": f"Unknown location: {self.target}.",
                "location_score":   0.0,
                "location_danger_level":   "none",
                "location_danger_message": "",
            }

        nav_result = nav_result or {}
        nav_safe_scores = nav_result.get("safe_scores", {}) or {}
        nav_command = (nav_result.get("command", "") or "").strip()
        nav_text = nav_result.get("text", "") or ""
        frame_width = frame.shape[1] if frame is not None and frame.size else 640

        # M5: compute danger alert on EVERY frame (always fresh)
        danger_level, danger_message = self._compute_danger_alert(
            nav_result, detections, scene, frame_width
        )
        self._last_danger_level = danger_level
        self._last_danger_message = danger_message

        # Run the multi-signal arrival fusion
        arrival: ArrivalResult = self.fusion.update(
            frame=frame,
            detections=detections or [],
            current_scene=scene or "",
            current_scene_conf=scene_conf,
            nav_safe_scores=nav_safe_scores,
            nav_command=nav_command,
            target_loc=target_loc,
            location_manager=self.location_manager,
        )

        display = target_loc.get("display_name") or self.target.replace("_", " ")
        message = ""
        landmark_hint = self._landmark_hint(target_loc, detections)

        # M4: compute gradient cue
        grad_message = self._gradient_hint(nav_safe_scores, target_loc)

        # ------------------------------------------------------------------
        # 1) DANGER — override everything except arrived / failed
        # ------------------------------------------------------------------
        if danger_level in ("stop", "warning"):
            self.state = "searching"
            message = danger_message

        # ------------------------------------------------------------------
        # 2) ARRIVED — exact wording required by the spec.
        # ------------------------------------------------------------------
        elif arrival.arrived:
            self.state = "arrived"
            self.target = None
            self._last_direction = ""
            message = f"You arrive to {display}."

        # ------------------------------------------------------------------
        # 3) SAFETY — nav pipeline is shouting STOP / UNRECOGNIZED.
        # ------------------------------------------------------------------
        elif arrival.blocked_by_safety:
            self.state = "searching"
            message = ""

        # ------------------------------------------------------------------
        # 4) TIMEOUT — give up and tell the user we could not confirm.
        # ------------------------------------------------------------------
        elif arrival.reason.startswith("timeout"):
            self.state = "failed"
            self._last_direction = ""
            message = (f"I could not confirm arrival at {display} after "
                       f"{int(arrival.elapsed_sec)} seconds. "
                       f"Please try again or save more reference photos of that place.")
            self.fusion.end()

        # ------------------------------------------------------------------
        # 5) WALKING — give active direction (gradient + nav).
        # ------------------------------------------------------------------
        else:
            smoothed = self.fusion.smoothed_score()
            self.state = "approaching" if smoothed >= 0.40 else "searching"
            elapsed = time.time() - self.start_time
            now = time.time()

            approaching_just_now = (self.state == "approaching"
                                    and not self._was_approaching)

            should_speak = (
                elapsed > self.STARTUP_GRACE_SEC and (
                    (now - self._last_message_time) >= self.GUIDANCE_COOLDOWN
                    or approaching_just_now
                )
            )

            if should_speak:
                # M5: use target-aware direction (gradient + nav)
                direction = self._compute_location_direction(
                    nav_command, nav_text, nav_safe_scores, target_loc
                )

                if (direction == self._last_direction
                        and (now - self._last_direction_time) < 8.0
                        and not approaching_just_now):
                    pass
                else:
                    if self.state == "approaching":
                        if direction:
                            message = (f"Getting closer to {display}. "
                                       f"{arrival.agree_count} of 4 signals agree. "
                                       f"{direction}")
                        else:
                            message = (f"Getting closer to {display}. "
                                       f"{arrival.agree_count} of 4 signals agree.")
                    else:
                        if direction:
                            message = f"Navigating to {display}. {direction}"
                        else:
                            message = f"Navigating to {display}."

                    if landmark_hint:
                        message += f" {landmark_hint}"

                    self._last_direction = direction
                    self._last_direction_time = now

                self._was_approaching = (self.state == "approaching")

        if message:
            self._last_message = message
            self._last_message_time = time.time()

        return {
            "location_state":          self.state,
            "location_target":         self.target,
            "location_target_display": display,
            "location_message":        message,
            "location_score":          round(arrival.final_score, 3),
            "location_raw_score":      arrival.signals["visual"].score,
            "location_arrived":        self.state == "arrived",
            "location_failed":         self.state == "failed",
            "location_progress":       self.fusion.progress_text(),
            "location_arrival":        arrival.to_dict(),
            "location_direction":      self._last_direction,
            "location_gradient":       grad_message,
            "location_danger_level":   danger_level,
            "location_danger_message": danger_message,
        }

    # ------------------------------------------------------------------
    # M5: Danger estimation for location navigation
    # ------------------------------------------------------------------

    def _compute_danger_alert(self, nav_result: Dict[str, Any],
                              detections: Optional[List[Dict[str, Any]]],
                              scene: Optional[str],
                              frame_width: int = 640) -> Tuple[str, str]:
        """
        Evaluate the current frame for danger and return (level, message).

        Levels (lowest to highest):
            "none"    → no danger detected
            "caution" → medium-danger object close ahead
            "warning" → high-danger object ahead OR unrecognized path
            "stop"    → STOP from nav pipeline OR very close high-danger object
        """
        nav_command = str(nav_result.get("command", ""))
        nav_text = nav_result.get("text", "")

        # 1) STOP from the navigation pipeline
        if "STOP" in nav_command:
            return ("stop", nav_text or "Stop. No safe path detected.")

        # 2) UNRECOGNIZED path → warning
        if "UNRECOGNIZED" in nav_command:
            return ("warning", nav_text or "Path is unclear. Move back or around.")

        # 3) YOLO detections
        for det in detections or []:
            danger = str(det.get("danger", "low"))
            area = float(det.get("area", 0))
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            third = frame_width / 3
            is_front = third < cx < 2 * third
            name = str(det.get("name", "obstacle"))

            if is_front and danger == "high" and area > 80_000:
                return ("stop", f"Danger! {name} is very close ahead. Stop.")
            if is_front and danger == "high":
                return ("warning", f"Warning. {name} ahead.")
            if is_front and danger == "medium" and area > 30_000:
                return ("caution", f"Caution. {name} is close ahead.")

        return ("none", "")

    # ------------------------------------------------------------------
    # M5: Target-aware direction (gradient + nav)
    # ------------------------------------------------------------------

    def _compute_location_direction(self, nav_command: str, nav_text: str,
                                    nav_safe_scores: Dict[str, float],
                                    target_loc: Dict[str, Any]) -> str:
        """
        Compute the best direction to guide the user toward the target.

        Priority:
          1. Safety overrides (STOP / UNRECOGNIZED / CAUTION)
          2. M4 path gradient (target-aware) — used for turns & alignment
          3. Nav pipeline command (generic safe path) — used as fallback
        """
        c = (nav_command or "").upper()

        # Safety overrides
        if "STOP" in c:
            return "Stop. Obstacle ahead."
        if "UNRECOGNIZED" in c:
            return "Path unclear. Move back or around."
        if "CAUTION" in c:
            return "Be careful. Obstacle ahead."

        # M4 path gradient (target-aware)
        refs = target_loc.get("references", [])
        profile_refs = [r for r in refs if r.get("path_profile")]
        if profile_refs:
            target_profile = PathProfile.from_dict(profile_refs[0].get("path_profile"))
            if not target_profile.is_empty():
                cur_metrics = compute_metrics(safe_mask=None, safe_scores=nav_safe_scores)
                grad = navigation_gradient(cur_metrics, target_profile, nav_safe_scores)

                # Use gradient for turns and alignment (target-aware)
                if grad["command"] in ("TURN LEFT", "TURN RIGHT") and grad["confidence"] >= 0.5:
                    return grad["reasoning"]
                if grad["command"] == "ALIGN CENTRE" and grad["confidence"] >= 0.5:
                    return grad["reasoning"]
                if grad["command"] == "STOP / SCAN":
                    return grad["reasoning"]
                # For MOVE FORWARD, only use gradient if very confident
                if grad["command"] == "MOVE FORWARD" and grad["confidence"] >= 0.75:
                    return grad["reasoning"]

        # Fallback to nav pipeline
        if c == "MOVE LEFT":
            return "Turn left."
        if c == "MOVE RIGHT":
            return "Turn right."
        if c == "MOVE FORWARD":
            return "Move forward."
        return (nav_text or "").strip()

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _direction_from_nav(nav_command: str, nav_text: str) -> str:
        """Translate the nav pipeline command into a short spoken direction."""
        c = (nav_command or "").upper()
        if "STOP" in c:
            return "Stop. Obstacle ahead."
        if "UNRECOGNIZED" in c:
            return "Path unclear. Move back or around."
        if "CAUTION" in c:
            return "Be careful. Obstacle ahead."
        if c == "MOVE LEFT":
            return "Turn left."
        if c == "MOVE RIGHT":
            return "Turn right."
        if c == "MOVE FORWARD":
            return "Move forward."
        return (nav_text or "").strip()

    @staticmethod
    def _landmark_hint(loc, detections):
        if not detections:
            return ""
        expected = {o.lower() for o in (loc.get("expected_objects") or [])}
        if not expected:
            return ""
        seen = {str(d.get("name", "")).lower() for d in detections}
        common = expected & seen
        if not common:
            return ""
        names = sorted(common)
        if len(names) == 1:
            return f"I see a {names[0]}."
        return f"I see {' and '.join(names)}."

    def _gradient_hint(self, nav_safe_scores: Dict[str, float],
                       target_loc: Dict[str, Any]) -> str:
        """M4: build a spoken cue from navigation_gradient()."""
        refs = target_loc.get("references", [])
        profile_refs = [r for r in refs if r.get("path_profile")]
        if not profile_refs:
            return ""
        target_profile = PathProfile.from_dict(profile_refs[0].get("path_profile"))
        if target_profile.is_empty():
            return ""
        cur_metrics = compute_metrics(safe_mask=None, safe_scores=nav_safe_scores)
        grad = navigation_gradient(cur_metrics, target_profile, nav_safe_scores)
        if grad["command"] in {"MOVE FORWARD"} and grad["confidence"] < 0.65:
            return ""
        return grad["reasoning"]