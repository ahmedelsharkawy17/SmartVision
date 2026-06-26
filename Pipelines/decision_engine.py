"""
SmartVisionX Decision Engine — Blind-Assist Edition
----------------------------------------------------
The decision engine picks the single message that the system will speak
each frame.

Priority rules:
  • Safety alerts (priority 0) — STOP, object very close — ALWAYS win.
  • Location messages:
        arrived   → priority 0  (always wins, even over a safety alert;
                                the user must hear "You arrive to X.")
        failed    → priority 0
        approaching → priority 2 (wins over nav guidance and scene)
        searching → priority 3   (wins over MOVE FORWARD guidance)
    A location message is inserted at position 0 only when its priority
    is <= the top message's priority (lower number = more important).
  • Navigation guidance (MOVE LEFT / RIGHT / CAUTION) → priority 3.
  • MOVE FORWARD                              → priority 5.
  • Scene / OCR auto                           → priority 5.
  • Silent                                     → priority 9.

M5 additions:
  • Location danger alerts (from LocationNavigator) are inserted FIRST
    so they always take priority over location guidance.
  • Danger sources are protected — location guidance can NEVER push a
    danger alert out of position 0 (only arrived / failed can).
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

_LOG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH   = os.path.join(_LOG_DIR, f"session_{_SESSION_ID}.csv")

_LOG_HEADERS = [
    "timestamp", "frame_id",
    "fps", "det_ms", "scene_ms", "ocr_ms", "nav_ms", "total_latency_ms",
    "runtime_mode", "scene", "scene_conf", "scene_risk",
    "nav_command", "nav_mode", "safe_left", "safe_center", "safe_right", "navigation_risk",
    "top_object", "object_danger", "object_position", "object_distance", "object_risk",
    "ocr_text", "ocr_source", "ocr_risk",
    "location_state", "location_target", "location_score",
    "location_danger_level", "location_danger_message",   # M5
    "risk_score", "decision_source", "message_spoken", "priority",
]


def _init_log() -> None:
    with open(_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_LOG_HEADERS).writeheader()
    print(f"[Log] Session log → {_LOG_PATH}")


def _write_log(row: dict) -> None:
    with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_LOG_HEADERS).writerow(
            {h: row.get(h, "") for h in _LOG_HEADERS}
        )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


_SCENE_ALIASES = {
    "street_road":     "street",
    "indoor_passage":  "corridor",
    "lecture_room":    "learning_space",
    "market":          "market_outdoor",
    "restaurant":      "eating_place",
}


def _normalize_scene(scene: str) -> str:
    scene = (scene or "unknown").strip()
    return _SCENE_ALIASES.get(scene, scene)


def _get_position(bbox, frame_width: int = 640) -> str:
    if not bbox or len(bbox) < 4:
        return "unknown"
    cx    = (float(bbox[0]) + float(bbox[2])) / 2.0
    third = frame_width / 3.0
    if cx < third:     return "to your left"
    if cx > 2 * third: return "to your right"
    return "in front of you"


def _is_front(bbox, frame_width: int = 640) -> bool:
    return _get_position(bbox, frame_width) == "in front of you"


def _get_distance(area) -> str:
    area = float(area or 0)
    if area > 80_000: return "very close"
    if area > 30_000: return "close"
    return "ahead"


_SCENE_RISK_VALUE = {
    "street":        0.75, "highway":       0.85, "crosswalk":     0.85,
    "parking_lot":   0.75, "transport_hub": 0.55, "market_outdoor":0.50,
    "supermarket":   0.45, "shopping_mall": 0.45, "staircase":     0.70,
    "corridor":      0.45, "elevator":      0.25, "waiting_room":  0.20,
    "work_space":    0.20, "learning_space":0.20, "library":       0.15,
    "hospital":      0.25, "mosque":        0.15, "bathroom":      0.25,
    "bedroom":       0.15, "beach":         0.25, "green_outdoor": 0.25,
    "eating_place":  0.25,
}

_DANGER_VALUE = {"high": 1.0, "medium": 0.60, "low": 0.25}


@dataclass
class RiskResult:
    runtime_mode:    str
    risk_score:      float
    object_risk:     float
    navigation_risk: float
    scene_risk:      float
    ocr_risk:        float
    top_detection:   Optional[dict]
    source:          str
    priority:        int
    message:         str


class DangerEstimator:
    def _scene_risk(self, scene: str, scene_conf: float) -> float:
        base = _SCENE_RISK_VALUE.get(_normalize_scene(scene), 0.35)
        if scene_conf < 0.45:
            return 0.20
        return _clip01(base * max(0.50, float(scene_conf)))

    def _navigation_risk(self, nav_result: dict) -> float:
        command = str(nav_result.get("command", ""))
        safe    = nav_result.get("safe_scores", {}) or {}
        center  = float(safe.get("center", 0.0) or 0.0)
        left    = float(safe.get("left",   0.0) or 0.0)
        right   = float(safe.get("right",  0.0) or 0.0)
        best    = max(left, center, right)
        if "STOP"    in command: return 0.95
        if command == "UNRECOGNIZED PATH": return 0.80
        if command in {"MOVE LEFT", "MOVE RIGHT"}: return 0.65
        if "CAUTION" in command: return 0.70
        return _clip01(0.35 * (1.0 - center) + 0.15 * (1.0 - best))

    def _object_risk(self, detections, frame_width: int):
        best_score = 0.0
        best_det = None
        for det in detections or []:
            danger      = str(det.get("danger", "low"))
            danger_val  = _DANGER_VALUE.get(danger, 0.25)
            area        = float(det.get("area", 0.0) or 0.0)
            conf        = float(det.get("confidence", 0.0) or 0.0)
            close_val   = 1.0 if area > 80_000 else (0.70 if area > 30_000 else 0.35)
            front_bonus = 0.25 if _is_front(det.get("bbox", []), frame_width) else 0.0
            score       = _clip01(0.55*danger_val + 0.25*close_val + 0.20*conf + front_bonus)
            if score > best_score:
                best_score, best_det = score, det
        return best_score, best_det

    def _ocr_risk(self, ocr_text: str, ocr_triggered: bool) -> float:
        if not ocr_triggered:
            return 0.0
        text = (ocr_text or "").strip().lower()
        if not text:
            return 0.10
        danger_words = ["stop", "danger", "warning", "exit", "fire", "closed", "private"]
        if any(w in text for w in danger_words):
            return 0.70
        return 0.30

    @staticmethod
    def _object_message(top_det, frame_width: int) -> str:
        obj    = str(top_det.get("name", "object"))
        pos    = _get_position(top_det.get("bbox", []), frame_width)
        dist   = _get_distance(top_det.get("area", 0))
        danger = str(top_det.get("danger", "low"))
        if danger == "high" and dist == "very close":
            return f"Danger! {obj} is very close {pos}. Stop immediately."
        if danger == "high":
            return f"Warning. {obj} is {dist} {pos}."
        if danger == "medium":
            return f"Caution. {obj} is {dist} {pos}."
        return f"{obj} detected {pos}."

    def estimate(self, *, scene, scene_conf, detections,
                 ocr_text, ocr_triggered, nav_result,
                 frame_width, ocr_source="auto") -> RiskResult:
        nav_result      = nav_result or {}
        object_risk, top_det = self._object_risk(detections, frame_width)
        navigation_risk = self._navigation_risk(nav_result)
        scene_risk      = self._scene_risk(scene, scene_conf)
        ocr_risk        = self._ocr_risk(ocr_text, ocr_triggered)
        nav_command     = str(nav_result.get("command", ""))
        nav_text        = str(nav_result.get("text", ""))
        manual_ocr      = bool(ocr_triggered and ocr_source == "manual")

        risk_score = _clip01(
            0.40 * object_risk +
            0.35 * navigation_risk +
            0.15 * scene_risk +
            0.10 * ocr_risk
        )

        def mk(mode, source, priority, message):
            return RiskResult(
                runtime_mode=mode, risk_score=round(risk_score, 3),
                object_risk=round(object_risk, 3),
                navigation_risk=round(navigation_risk, 3),
                scene_risk=round(scene_risk, 3),
                ocr_risk=round(ocr_risk, 3),
                top_detection=top_det, source=source,
                priority=priority, message=message,
            )

        if "STOP" in nav_command:
            return mk("safety", "navigation_stop", 0,
                      nav_text or "Stop. No safe path detected.")
        if nav_command == "UNRECOGNIZED PATH":
            return mk("safety", "navigation_stop", 0,
                      nav_text or "Safe path is not detected. Please move back or move around.")
        if top_det and object_risk >= 0.85 and _is_front(top_det.get("bbox", []), frame_width):
            return mk("safety", "object", 0,
                      self._object_message(top_det, frame_width))
        if manual_ocr:
            text = (ocr_text or "").strip()
            msg  = (f"Text reads: {text}." if text
                    else "No clear text found. Move closer and keep text centred.")
            return mk("ocr", "ocr", 1, msg)
        if top_det and object_risk >= 0.65:
            return mk("safety", "object", 2,
                      self._object_message(top_det, frame_width))
        if nav_command in {"MOVE LEFT", "MOVE RIGHT"} or "CAUTION" in nav_command:
            return mk("navigation", "navigation", 3,
                      nav_text or nav_command.replace("_", " ").title())
        if nav_command == "MOVE FORWARD":
            return mk("navigation", "navigation_forward", 5,
                      nav_text or "Path ahead is clear. Move forward.")
        if ocr_triggered and (ocr_text or "").strip():
            return mk("ocr_auto", "ocr", 4,
                      f"Text detected: {(ocr_text or '').strip()}.")
        if scene_conf >= 0.65:
            label = _normalize_scene(scene).replace("_", " ")
            return mk("scene", "scene", 5, f"You are in a {label}.")
        return mk("silent", "none", 9, "")


class AlertManager:
    COOLDOWN: dict = {
        "navigation_stop":       2.0,
        "object":                3.0,
        "ocr":                   0.8,
        "navigation":            4.0,
        "navigation_forward":    8.0,
        "ocr_auto":              6.0,
        "scene":                 6.0,
        "location_arrived":      0.0,
        "location_approaching":  4.0,
        "location_searching":    4.0,
        "location_failed":       0.0,
        "location_danger_stop":      1.5,   # M5 — safety, speak fast
        "location_danger_warning":   2.5,   # M5
        "location_danger_caution":   4.0,   # M5
        "none":                  3.0,
    }

    def __init__(self) -> None:
        self._last_spoken: dict = {}
        self._last_mode: str = ""

    def allow(self, source, message, priority, runtime_mode) -> bool:
        if not message:
            return False
        now      = time.time()
        key      = f"{source}:{priority}:{message[:80]}"
        cooldown = self.COOLDOWN.get(source, 5.0)
        if priority == 0:
            cooldown = min(cooldown, 2.0)
        if runtime_mode != self._last_mode:
            self._last_mode        = runtime_mode
            self._last_spoken[key] = now
            return True
        if now - self._last_spoken.get(key, 0.0) >= cooldown:
            self._last_spoken[key] = now
            return True
        return False


class DecisionEngine:
    def __init__(self) -> None:
        self.estimator     = DangerEstimator()
        self.alert_manager = AlertManager()
        self.last_message  = ""
        self._frame_id     = 0
        _init_log()

    def decide(self,
               scene:            str,
               scene_conf:       float,
               detections:       list,
               ocr_text:         str,
               ocr_triggered:    bool,
               nav_result:       Optional[dict] = None,
               frame_width:      int   = 640,
               ocr_source:       str   = "auto",
               fps:              float = 0.0,
               det_ms:           float = 0.0,
               scene_ms:         float = 0.0,
               ocr_ms:           float = 0.0,
               nav_ms:           float = 0.0,
               total_latency_ms: float = 0.0,
               location_status:  Optional[dict] = None) -> list:

        self._frame_id += 1
        nav_result  = nav_result or {}
        safe_scores = nav_result.get("safe_scores", {}) or {}

        risk = self.estimator.estimate(
            scene=scene, scene_conf=scene_conf, detections=detections or [],
            ocr_text=ocr_text or "", ocr_triggered=bool(ocr_triggered),
            nav_result=nav_result, frame_width=frame_width,
            ocr_source=ocr_source,
        )

        messages: list = []
        if self.alert_manager.allow(risk.source, risk.message, risk.priority, risk.runtime_mode):
            messages.append({
                "text":       risk.message,
                "priority":   risk.priority,
                "source":     risk.source,
                "mode":       risk.runtime_mode,
                "risk_score": risk.risk_score,
            })
            self.last_message = risk.message

        # ── Location awareness ────────────────────────────────────────
        # M5: danger alert from LocationNavigator is inserted FIRST
        #     so it always takes priority over location guidance.
        #     Arrived / failed still win (the user MUST hear them).
        if location_status:
            loc_state   = location_status.get("location_state", "idle")
            loc_message = location_status.get("location_message", "")
            loc_danger_level   = location_status.get("location_danger_level", "none")
            loc_danger_message = location_status.get("location_danger_message", "")

            # ---- (a) Location danger alert (highest priority except arrived/failed) ----
            if loc_danger_level != "none" and loc_danger_message:
                danger_pri = {"stop": 0, "warning": 1, "caution": 2}.get(loc_danger_level, 2)
                danger_src = f"location_danger_{loc_danger_level}"
                if self.alert_manager.allow(danger_src, loc_danger_message,
                                            danger_pri, danger_src):
                    messages.insert(0, {
                        "text":       loc_danger_message,
                        "priority":   danger_pri,
                        "source":     danger_src,
                        "mode":       danger_src,
                        "risk_score": 0.95 if danger_pri == 0 else 0.75,
                    })
                    self.last_message = loc_danger_message

            # ---- (b) Location guidance message ----
            if loc_state not in (None, "idle", "") and loc_message:
                if loc_state == "arrived":
                    loc_source, loc_priority = "location_arrived",     0
                elif loc_state == "approaching":
                    loc_source, loc_priority = "location_approaching", 2
                elif loc_state == "searching":
                    loc_source, loc_priority = "location_searching",   3
                else:  # failed
                    loc_source, loc_priority = "location_failed",      0

                top_priority = messages[0]["priority"] if messages else 9
                top_source   = messages[0].get("source", "") if messages else ""
                must_speak   = loc_state in ("arrived", "failed")

                # M5: never let location guidance override an active danger alert
                danger_sources = (
                    "navigation_stop", "object",
                    "location_danger_stop", "location_danger_warning",
                    "location_danger_caution",
                )
                if top_source in danger_sources and not must_speak:
                    should_insert = False
                else:
                    should_insert = must_speak or (loc_priority <= top_priority)

                if should_insert:
                    if self.alert_manager.allow(loc_source, loc_message, loc_priority,
                                                f"location_{loc_state}"):
                        messages.insert(0, {
                            "text":       loc_message,
                            "priority":   loc_priority,
                            "source":     loc_source,
                            "mode":       f"location_{loc_state}",
                            "risk_score": 0.0,
                        })
                        self.last_message = loc_message

        top_det = risk.top_detection
        _write_log({
            "timestamp":        datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "frame_id":         self._frame_id,
            "fps":              f"{float(fps):.2f}",
            "det_ms":           f"{float(det_ms):.2f}",
            "scene_ms":         f"{float(scene_ms):.2f}",
            "ocr_ms":           f"{float(ocr_ms):.2f}",
            "nav_ms":           f"{float(nav_ms):.2f}",
            "total_latency_ms": f"{float(total_latency_ms):.2f}",
            "runtime_mode":     risk.runtime_mode,
            "scene":            scene,
            "scene_conf":       f"{float(scene_conf):.3f}",
            "scene_risk":       f"{risk.scene_risk:.3f}",
            "nav_command":      str(nav_result.get("command", "")),
            "nav_mode":         str(nav_result.get("mode", "")),
            "safe_left":        f"{float(safe_scores.get('left',   0.0)):.3f}" if safe_scores else "",
            "safe_center":      f"{float(safe_scores.get('center', 0.0)):.3f}" if safe_scores else "",
            "safe_right":       f"{float(safe_scores.get('right',  0.0)):.3f}" if safe_scores else "",
            "navigation_risk":  f"{risk.navigation_risk:.3f}",
            "top_object":       top_det.get("name",   "") if top_det else "",
            "object_danger":    top_det.get("danger", "") if top_det else "",
            "object_position":  _get_position(top_det.get("bbox", []), frame_width) if top_det else "",
            "object_distance":  _get_distance(top_det.get("area", 0))              if top_det else "",
            "object_risk":      f"{risk.object_risk:.3f}",
            "ocr_text":         ocr_text,
            "ocr_source":       ocr_source if ocr_triggered else "",
            "ocr_risk":         f"{risk.ocr_risk:.3f}",
            "location_state":   (location_status or {}).get("location_state", ""),
            "location_target":  (location_status or {}).get("location_target", ""),
            "location_score":   (location_status or {}).get("location_score", ""),
            "location_danger_level":   (location_status or {}).get("location_danger_level", ""),     # M5
            "location_danger_message": (location_status or {}).get("location_danger_message", ""),   # M5
            "risk_score":       f"{risk.risk_score:.3f}",
            "decision_source":  risk.source,
            "message_spoken":   messages[0]["text"]     if messages else "",
            "priority":         messages[0]["priority"] if messages else "",
        })
        return messages