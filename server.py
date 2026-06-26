"""
SmartVisionX Mobile Server — Blind-Assist Edition
ADDED (M4 multi-signal arrival fusion):
  • POST /locations accepts an optional `path_signature` (the live
    left/center/right safe-path vector).
  • POST /locations accepts an optional `path_profile` (a 10-metric
    statistical descriptor computed client-side at save time).
  • /navigation_status returns the full five-signal breakdown
    (visual / object / scene / path / path_metric) + smoothed score
    + progress text + gradient cue.
  • analyze_frame response still contains `location_status` (unchanged
    shape) but now with a `location_arrival` sub-dict that includes the
    M4 gradient fields.
  • /navigate_to performs fuzzy target matching so a blind user who
    mispronounces a place name still gets the closest saved match.

ADDED (M5 danger alerts + camera flip):
  • FRAME_FLIP config — horizontal flip applied to the raw frame so
    every downstream pipeline (YOLO bboxes, position math, decisions)
    sees the same orientation the user actually sees.  Fixes the
    "table on the right but actually on the left" mirror bug.
  • analyze_frame now returns `location_danger_level` and
    `location_danger_message` inside `location_status` so the mobile
    app can render the danger banner.
  • POST /toggle_flip lets the phone flip the camera at runtime.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import threading
import time
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Pipelines.scene_pipeline        import ScenePipeline
from Pipelines.ocr_pipeline          import OCRPipeline
from Pipelines.object_pipeline       import ObjectPipeline
from Pipelines.decision_engine       import DecisionEngine
from Pipelines.navigation_pipeline   import NavigationPipeline
from Pipelines.location_manager      import LocationManager
from Pipelines.location_navigation   import LocationNavigator
from Pipelines.arrival_fusion        import compute_path_signature


# =========================================================
# PATHS
# =========================================================

BASE_DIR          = Path(__file__).resolve().parent
WEB_DIR           = BASE_DIR / "web"
LOG_DIR           = BASE_DIR / "Logs"
EVIDENCE_ROOT     = LOG_DIR / "evidence_frames"

MODEL_SCENE       = BASE_DIR / "Models" / "Scene Recognition"  / "Scene Recognition.pth"
MODEL_OCR         = BASE_DIR / "Models" / "Text Recognition"   / "best_smartvisionx_crnn_lowercase.pt"
MODEL_YOLO        = BASE_DIR / "Models" / "Object Detection"   / "yolov8n.pt"
MODEL_NAV_OUTDOOR = BASE_DIR / "Models" / "Navigation"         / "best_outdoor_navigation.pt"
MODEL_NAV_INDOOR  = BASE_DIR / "Models" / "Navigation"         / "best_indoor_rgbd_navigation.pt"


# =========================================================
# CONFIG
# =========================================================

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
SCENE_INTERVAL_SEC = 1.0
OCR_INTERVAL_SEC   = 3.0
AUTO_OCR_ENABLED   = False
OCR_MIN_EDGE_DENSITY = 0.08
OCR_CROP_W_FRAC = 0.56
OCR_CROP_H_FRAC = 0.25
MAX_DETECTIONS_RETURNED = 8
SAVE_EVIDENCE_FRAMES = True
EVIDENCE_COOLDOWN_SEC = 2.0
EVIDENCE_JPEG_QUALITY = 85
LOCATIONS_RELOAD_SEC = 5.0

# M5: horizontal frame flip.  Set True for selfie / mirrored cameras.
# The phone app can override this at runtime via POST /toggle_flip.
FRAME_FLIP = True


# =========================================================
# APP STATE
# =========================================================

app = FastAPI(title="SmartVisionX Blind Assist", version="2.4-M6")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/evidence_frames", StaticFiles(directory=str(EVIDENCE_ROOT)),
          name="evidence_frames")

_REFS_DIR = (BASE_DIR / "Locations" / "references")
_REFS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/location_references", StaticFiles(directory=str(_REFS_DIR)),
          name="location_references")


class FramePayload(BaseModel):
    image: str
    force_ocr: bool = False


class SaveLocationPayload(BaseModel):
    name:            str
    image:           str
    description:     str = ""
    path_signature:  Optional[List[float]] = None
    path_profile:    Optional[Dict[str, Any]] = None


class NavigatePayload(BaseModel):
    target: str


# =========================================================
# AUTO-EVALUATION LOGGER
# =========================================================

class AutoEvaluationLogger:
    HEADERS = [
        "timestamp", "frame_id",
        "fps", "latency_ms", "det_ms", "scene_ms", "ocr_ms", "nav_ms",
        "scene", "scene_conf",
        "nav_command", "nav_text", "nav_mode",
        "safe_left", "safe_center", "safe_right",
        "top_object", "top_object_conf", "object_danger", "object_position", "object_distance",
        "all_detections", "ocr_text", "ocr_triggered",
        "location_state", "location_target", "location_score",
        "location_danger_level", "location_danger_message",   # M5
        "arrival_final_score", "arrival_agree_count", "arrival_streak",
        "sig_visual", "sig_object", "sig_scene", "sig_path", "sig_path_metric",
        "gradient_command", "gradient_reason",
        "alert", "alert_mode", "risk_score", "priority", "evidence_frame",
        "frame_flip",   # M5
    ]

    def __init__(self, log_dir, evidence_root):
        log_dir.mkdir(parents=True, exist_ok=True)
        evidence_root.mkdir(parents=True, exist_ok=True)
        self.session_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path      = log_dir / f"mobile_session_{self.session_id}.csv"
        self.evidence_dir  = evidence_root / self.session_id
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.frame_id      = 0
        self._last_ev_time = 0.0
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.HEADERS).writeheader()
        print(f"[AutoEval] CSV  → {self.csv_path}")
        print(f"[AutoEval] Frames → {self.evidence_dir}")

    @staticmethod
    def _position(bbox, fw: int = FRAME_WIDTH):
        if not bbox: return ""
        cx = (bbox[0] + bbox[2]) / 2
        third = fw / 3
        if cx < third:   return "left"
        if cx > 2*third: return "right"
        return "front"

    @staticmethod
    def _distance(area):
        area = int(area or 0)
        if area > 80_000: return "very close"
        if area > 30_000: return "close"
        return "ahead"

    def _should_save(self, alert, nav_command, priority):
        if not SAVE_EVIDENCE_FRAMES: return False
        important = bool(alert) or "STOP" in str(nav_command)
        try:    important = important or int(priority) <= 1
        except Exception: pass
        if not important: return False
        now = time.time()
        if now - self._last_ev_time < EVIDENCE_COOLDOWN_SEC: return False
        self._last_ev_time = now
        return True

    def write(self, *, frame, result, messages):
        self.frame_id += 1
        dets     = result.get("detections", []) or []
        top      = dets[0] if dets else {}
        nav      = result.get("nav_result", {}) or {}
        safe     = nav.get("safe_scores", {}) or {}
        mod_ms   = result.get("module_ms", {}) or {}
        alert    = result.get("alert", "")
        priority = messages[0].get("priority", "") if messages else ""
        nav_cmd  = result.get("nav_command", "")
        loc      = result.get("location_status", {}) or {}
        arrival  = loc.get("location_arrival") or {}
        sigs     = arrival.get("signals") or {}

        ev_rel = ""
        if self._should_save(alert, nav_cmd, priority):
            fname = f"frame_{self.frame_id:06d}.jpg"
            fpath = self.evidence_dir / fname
            cv2.imwrite(str(fpath), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), EVIDENCE_JPEG_QUALITY])
            ev_rel = str(fpath.relative_to(BASE_DIR)).replace("\\", "/")

        all_dets = "; ".join(
            f"{d.get('name','')}:{d.get('danger','')}:{d.get('confidence','')}"
            for d in dets[:MAX_DETECTIONS_RETURNED]
        )
        grad_sig = sigs.get("path_metric", {}).get("raw", {}) or {}
        row = {
            "timestamp":     datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "frame_id":      self.frame_id,
            "fps":           result.get("fps_estimate", ""),
            "latency_ms":    result.get("latency_ms", ""),
            "det_ms":        mod_ms.get("det_ms", ""),
            "scene_ms":      mod_ms.get("scene_ms", ""),
            "ocr_ms":        mod_ms.get("ocr_ms", ""),
            "nav_ms":        mod_ms.get("nav_ms", ""),
            "scene":         result.get("scene", ""),
            "scene_conf":    result.get("scene_conf", ""),
            "nav_command":   nav_cmd,
            "nav_text":      result.get("nav_text", ""),
            "nav_mode":      nav.get("mode", ""),
            "safe_left":     safe.get("left", ""),
            "safe_center":   safe.get("center", ""),
            "safe_right":    safe.get("right", ""),
            "top_object":    top.get("name", ""),
            "top_object_conf": top.get("confidence", ""),
            "object_danger": top.get("danger", ""),
            "object_position": self._position(top.get("bbox", [])),
            "object_distance": self._distance(int(top.get("area", 0))) if top else "",
            "all_detections": all_dets,
            "ocr_text":      result.get("ocr_text", ""),
            "ocr_triggered": result.get("ocr_triggered", False),
            "location_state":  loc.get("location_state", ""),
            "location_target": loc.get("location_target", ""),
            "location_score":  loc.get("location_score", ""),
            "location_danger_level":   loc.get("location_danger_level", ""),     # M5
            "location_danger_message": loc.get("location_danger_message", ""),   # M5
            "arrival_final_score": arrival.get("final_score", ""),
            "arrival_agree_count": arrival.get("agree_count", ""),
            "arrival_streak":     arrival.get("streak", ""),
            "sig_visual":      sigs.get("visual", {}).get("score", ""),
            "sig_object":      sigs.get("object", {}).get("score", ""),
            "sig_scene":       sigs.get("scene",  {}).get("score", ""),
            "sig_path":        sigs.get("path",   {}).get("score", ""),
            "sig_path_metric": sigs.get("path_metric", {}).get("score", ""),
            "gradient_command": grad_sig.get("gradient_command", ""),
            "gradient_reason":  grad_sig.get("gradient_reason", ""),
            "alert":         alert,
            "alert_mode":    result.get("alert_mode", ""),
            "risk_score":    result.get("risk_score", ""),
            "priority":      priority,
            "evidence_frame": ev_rel,
            "frame_flip":    FRAME_FLIP,    # M5
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.HEADERS).writerow(row)
        return ev_rel or None


# =========================================================
# RUNTIME STATE
# =========================================================

class RuntimeState:
    def __init__(self):
        self.device             = "cpu"
        self.loaded             = False
        self._load_lock         = threading.Lock()
        self.scene_pipeline     = None
        self.ocr_pipeline       = None
        self.object_pipeline    = None
        self.nav_pipeline       = None
        self.decision           = None
        self.location_manager   = None
        self.location_navigator = None
        self.current_scene      = ("unknown", 0.0)
        self.last_scene_time    = 0.0
        self.last_ocr_time      = 0.0
        self.last_alert         = ""
        self.logger             = None
        self.last_locations_reload = 0.0


STATE = RuntimeState()


# =========================================================
# HELPERS
# =========================================================

def _decode_image(data_url: str) -> np.ndarray:
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw   = base64.b64decode(data_url)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")


def _resize(frame): return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def _centre_crop(frame):
    h, w = frame.shape[:2]
    cw = int(w * OCR_CROP_W_FRAC)
    ch = int(h * OCR_CROP_H_FRAC)
    x1 = max(0, w // 2 - cw // 2)
    y1 = max(0, h // 2 - ch // 2)
    return frame[y1:y1+ch, x1:x1+cw]


def _crop_has_text(frame):
    crop = _centre_crop(frame)
    if crop.size == 0: return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = float(edges.mean()) / 255.0
    return density > OCR_MIN_EDGE_DENSITY


def load_pipelines(device: str = "cpu"):
    with STATE._load_lock:
        if STATE.loaded and STATE.device == device:
            return
        print("=" * 70)
        print(f"Loading SmartVisionX pipelines (M5) | device={device}")
        print("=" * 70)
        STATE.device          = device
        STATE.scene_pipeline  = ScenePipeline(MODEL_SCENE, device=device)
        STATE.ocr_pipeline    = OCRPipeline(MODEL_OCR, device=device)
        STATE.object_pipeline = ObjectPipeline(MODEL_YOLO, device=device)
        STATE.nav_pipeline    = NavigationPipeline(MODEL_NAV_OUTDOOR, MODEL_NAV_INDOOR, device=device)
        STATE.decision        = DecisionEngine()
        STATE.location_manager   = LocationManager()
        STATE.location_navigator = LocationNavigator(STATE.location_manager)
        STATE.logger          = AutoEvaluationLogger(LOG_DIR, EVIDENCE_ROOT)
        STATE.loaded          = True
        print("All pipelines loaded.")


# =========================================================
# CORE ANALYSIS
# =========================================================

def analyze_frame(frame, force_ocr=False):
    if not STATE.loaded:
        load_pipelines(STATE.device)

    assert STATE.scene_pipeline  is not None
    assert STATE.ocr_pipeline    is not None
    assert STATE.object_pipeline is not None
    assert STATE.nav_pipeline    is not None
    assert STATE.decision        is not None
    assert STATE.logger          is not None
    assert STATE.location_manager   is not None
    assert STATE.location_navigator is not None

    # M5: horizontal flip — apply to the raw frame so detection
    # bboxes, position math, decisions, and the saved reference
    # fingerprints all share the same orientation as the user.
    if FRAME_FLIP:
        frame = cv2.flip(frame, 1)

    frame = _resize(frame)
    now = time.time()
    t0  = time.perf_counter()

    if (now - STATE.last_locations_reload) >= LOCATIONS_RELOAD_SEC:
        STATE.location_manager.reload()
        STATE.last_locations_reload = now

    # Scene
    scene_ms = 0.0
    if now - STATE.last_scene_time >= SCENE_INTERVAL_SEC:
        ts = time.perf_counter()
        lbl, conf, _ = STATE.scene_pipeline.predict(frame)
        scene_ms = (time.perf_counter() - ts) * 1000.0
        STATE.current_scene   = (lbl, conf)
        STATE.last_scene_time = now

    # Detection
    detections, det_ms = STATE.object_pipeline.detect(frame)

    # Navigation
    nav_result = STATE.nav_pipeline.predict(
        frame,
        scene      = STATE.current_scene[0],
        scene_conf = STATE.current_scene[1],
        detections = detections,
    )
    nav_ms = float(nav_result.get("ms", 0.0))

    # Location navigator (multi-signal + M5 danger)
    if STATE.location_navigator.is_active() or STATE.location_navigator.target:
        location_status = STATE.location_navigator.update(
            frame, nav_result, detections,
            scene=STATE.current_scene[0], scene_conf=STATE.current_scene[1],
        )
    else:
        location_status = {"location_state": "idle", "location_target": None,
                           "location_message": "", "location_score": 0.0,
                           "location_progress": "",
                           "location_danger_level": "none",     # M5
                           "location_danger_message": ""}        # M5

    # OCR
    ocr_text = ""
    ocr_ms   = 0.0
    ocr_triggered = False
    ocr_due = AUTO_OCR_ENABLED and ((now - STATE.last_ocr_time) >= OCR_INTERVAL_SEC)
    if force_ocr or (ocr_due and _crop_has_text(frame)):
        ts = time.perf_counter()
        try:
            ocr_text, _ = STATE.ocr_pipeline.read(frame)
            ocr_triggered = True
            STATE.last_ocr_time = now
        except Exception as exc:
            print(f"[OCR] error: {exc}")
        ocr_ms = (time.perf_counter() - ts) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    fps      = 1000.0 / max(1.0, total_ms)

    # Decision
    messages = STATE.decision.decide(
        scene=STATE.current_scene[0], scene_conf=STATE.current_scene[1],
        detections=detections, ocr_text=ocr_text, ocr_triggered=ocr_triggered,
        nav_result=nav_result, frame_width=FRAME_WIDTH,
        ocr_source="manual" if force_ocr else "auto",
        fps=fps, det_ms=float(det_ms), scene_ms=float(scene_ms),
        ocr_ms=float(ocr_ms), nav_ms=float(nav_ms), total_latency_ms=total_ms,
        location_status=location_status,
    )

    alert      = messages[0]["text"]                if messages else ""
    alert_mode = messages[0].get("mode", "silent")  if messages else "silent"
    risk_score = messages[0].get("risk_score", 0.0) if messages else 0.0
    if alert:
        STATE.last_alert = alert

    raw_safe = nav_result.get("safe_scores", {}) or {}
    safe_json = {
        "left":   float(raw_safe.get("left",   0.0)),
        "center": float(raw_safe.get("center", 0.0)),
        "right":  float(raw_safe.get("right",  0.0)),
    }
    dets_json = [
        {
            "name":       str(d.get("name", "")),
            "confidence": float(d.get("confidence", 0.0)),
            "bbox":       [int(x) for x in d.get("bbox", [])],
            "area":       int(d.get("area", 0)),
            "danger":     str(d.get("danger", "")),
        }
        for d in detections[:MAX_DETECTIONS_RETURNED]
    ]
    module_ms = {
        "det_ms": float(det_ms), "scene_ms": float(scene_ms),
        "ocr_ms": float(ocr_ms), "nav_ms": float(nav_ms),
    }

    log_result = {
        "alert": str(alert), "alert_mode": str(alert_mode),
        "last_alert": str(STATE.last_alert),
        "scene": str(STATE.current_scene[0]),
        "scene_conf": float(STATE.current_scene[1]),
        "ocr_text": str(ocr_text), "ocr_triggered": bool(ocr_triggered),
        "nav_command": str(nav_result.get("command", "")),
        "nav_text": str(nav_result.get("text", "")),
        "nav_result": nav_result,
        "safe_scores": safe_json, "detections": dets_json,
        "latency_ms": float(round(total_ms, 1)),
        "fps_estimate": float(round(fps, 1)),
        "module_ms": module_ms,
        "risk_score": float(round(risk_score, 3)),
        "location_status": location_status,
    }
    STATE.logger.write(frame=frame, result=log_result, messages=messages)

    return {
        "alert":         str(alert),
        "alert_mode":    str(alert_mode),
        "last_alert":    str(STATE.last_alert),
        "scene":         str(STATE.current_scene[0]),
        "scene_conf":    float(STATE.current_scene[1]),
        "ocr_text":      str(ocr_text),
        "ocr_triggered": bool(ocr_triggered),
        "nav_command":   str(nav_result.get("command", "")),
        "nav_text":      str(nav_result.get("text", "")),
        "nav_mode":      str(nav_result.get("mode", "")),
        "safe_scores":   safe_json,
        "detections":    dets_json,
        "latency_ms":    float(round(total_ms, 1)),
        "fps_estimate":  float(round(fps, 1)),
        "risk_score":    float(round(risk_score, 3)),
        "module_ms":     module_ms,
        "location_status": location_status,
        "frame_flip":    FRAME_FLIP,    # M5
    }


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def index():
    for candidate in (BASE_DIR / "index.html", WEB_DIR / "index.html"):
        if candidate.exists():
            return FileResponse(str(candidate))
    return JSONResponse({"error": "index.html not found."}, status_code=404)


@app.get("/health")
def health():
    model_paths = {
        "scene":       MODEL_SCENE, "ocr":         MODEL_OCR,
        "yolo":        MODEL_YOLO,  "nav_outdoor": MODEL_NAV_OUTDOOR,
        "nav_indoor":  MODEL_NAV_INDOOR,
    }
    missing = [k for k, v in model_paths.items() if not v.exists()]
    status_message = "All systems ready." if not missing else f"Missing models: {', '.join(missing)}"
    return {
        "ok":             True,
        "loaded":         STATE.loaded,
        "device":         STATE.device,
        "status_message": status_message,
        "missing_models": missing,
        "session_csv":    str(STATE.logger.csv_path)  if STATE.logger else None,
        "evidence_dir":   str(STATE.logger.evidence_dir) if STATE.logger else None,
        "models":         {k: str(v) for k, v in model_paths.items()},
        "models_exist":   {k: v.exists() for k, v in model_paths.items()},
        "locations_count": len(STATE.location_manager.location_names()) if STATE.location_manager else 0,
        "version":        "2.4-M5",
        "frame_flip":     FRAME_FLIP,    # M5
    }


@app.post("/analyze_frame")
def analyze(payload: FramePayload):
    try:
        frame = _decode_image(payload.image)
        return analyze_frame(frame, force_ocr=payload.force_ocr)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")


@app.get("/locations")
def get_locations():
    if not STATE.loaded:
        load_pipelines(STATE.device)
    return {"locations": STATE.location_manager.list_locations()}


@app.post("/locations")
def save_location(payload: SaveLocationPayload):
    if not STATE.loaded:
        load_pipelines(STATE.device)
    try:
        frame = _decode_image(payload.image)
    except HTTPException as e:
        raise e
    # M5: respect the same flip setting for saved references
    if FRAME_FLIP:
        frame = cv2.flip(frame, 1)
    objects: list = []
    scene_label = None
    scene_conf = 0.0
    try:
        dets, _ = STATE.object_pipeline.detect(frame)
        objects = [d["name"] for d in dets[:8]]
    except Exception:
        pass
    try:
        lbl, conf, _ = STATE.scene_pipeline.predict(frame)
        scene_label = lbl
        scene_conf = conf
    except Exception:
        pass

    path_sig = payload.path_signature
    if not path_sig:
        try:
            res = STATE.nav_pipeline.predict(
                frame, scene=scene_label, scene_conf=scene_conf, detections=[],
            )
            path_sig = compute_path_signature(res.get("safe_scores", {}) or {})
        except Exception:
            path_sig = None

    try:
        loc = STATE.location_manager.add_location(
            payload.name, frame,
            objects=objects, scene=scene_label, scene_conf=scene_conf,
            description=payload.description,
            path_signature=path_sig,
            path_profile=payload.path_profile,
        )
        return {"ok": True, "location": {
            "name": loc["name"], "display_name": loc["display_name"],
            "reference_count": len(loc["references"]),
            "expected_scene": loc.get("expected_scene"),
            "expected_objects": loc.get("expected_objects", []),
            "has_path_signature": any(
                r.get("path_signature") for r in loc.get("references", [])),
            "has_path_profile": any(
                r.get("path_profile") for r in loc.get("references", [])),
        }}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/locations/{name}")
def delete_location(name: str):
    if not STATE.loaded:
        load_pipelines(STATE.device)
    ok = STATE.location_manager.remove_location(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Location '{name}' not found.")
    return {"ok": True, "name": name}


# M5: runtime flip toggle for the phone client
@app.post("/toggle_flip")
def toggle_flip():
    global FRAME_FLIP
    FRAME_FLIP = not FRAME_FLIP
    return {"ok": True, "frame_flip": FRAME_FLIP}


def _resolve_target(raw: str) -> str:
    """Best-effort fuzzy match of a spoken/typed target against saved names."""
    if not raw:
        return raw
    norm = raw.strip().lower().replace(" ", "_")
    names = STATE.location_manager.location_names()
    if not names:
        return norm
    if norm in names:
        return norm
    for n in names:
        if norm in n or n in norm:
            return n
    close = get_close_matches(norm, names, n=1, cutoff=0.7)
    return close[0] if close else norm


@app.post("/navigate_to")
def navigate_to(payload: NavigatePayload):
    if not STATE.loaded:
        load_pipelines(STATE.device)
    if not payload.target or not payload.target.strip():
        raise HTTPException(status_code=400, detail="Target location is required.")
    target = _resolve_target(payload.target.strip())
    path_sig = None
    status = STATE.location_navigator.set_target(target, path_signature=path_sig)
    return {"ok": True, "status": status, "resolved": target}


@app.post("/stop_navigation")
def stop_navigation():
    if not STATE.loaded:
        load_pipelines(STATE.device)
    status = STATE.location_navigator.stop("Navigation cancelled.")
    return {"ok": True, "status": status}


@app.post("/get_my_location")
def get_my_location(payload: FramePayload):
    """
    Match the current frame against all saved locations and return
    a spoken response: "You are at <place>" or "I don't recognise this location."
    Accepts the same image payload as /analyze_frame.
    """
    if not STATE.loaded:
        load_pipelines(STATE.device)
    try:
        frame = _decode_image(payload.image)
    except HTTPException:
        raise

    if FRAME_FLIP:
        frame = cv2.flip(frame, 1)
    frame = _resize(frame)

    match = STATE.location_manager.match_location(frame)

    if match and match.get("match_score", 0.0) >= 0.45:
        display_name = (match.get("display_name") or match["name"]).replace("_", " ")
        spoken = f"You are at {display_name}."
        print(f"[WhereAmI] Matched '{match['name']}' score={match['match_score']}")
    else:
        spoken = "I don't recognise this location."
        print("[WhereAmI] No confident match found.")

    return {
        "ok":           True,
        "spoken":       spoken,
        "matched":      match is not None and match.get("match_score", 0.0) >= 0.45,
        "match":        {
            "name":        match["name"]         if match else None,
            "display_name": match["display_name"] if match else None,
            "match_score": match["match_score"]   if match else 0.0,
        } if match else None,
    }


@app.get("/navigation_status")
def navigation_status():
    if not STATE.loaded:
        load_pipelines(STATE.device)
    fusion = STATE.location_navigator.fusion
    last = fusion.last_arrival
    return {
        "status": STATE.location_navigator.status(),
        "smoothed_score": round(fusion.smoothed_score(), 3),
        "progress_text":  fusion.progress_text(),
        "last_arrival":   last.to_dict() if last is not None else None,
    }


# =========================================================
# ENTRY POINT
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="SmartVisionX Blind-Assist Server (M5)")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--device",     choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-flip",    action="store_true",
                        help="Disable horizontal camera flip")
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    global FRAME_FLIP
    if args.no_flip:
        FRAME_FLIP = False
    STATE.device = args.device
    if not args.no_preload:
        load_pipelines(args.device)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()