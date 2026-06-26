"""
SmartVisionX — Integrated Pipeline Runner
Pipelines: OCR + Scene Recognition + Object Detection + Navigation + Locations
+ Multi-signal Arrival Fusion

Keys:
    [R]    Force OCR read now
    [L]    List saved locations in the terminal
    [K]    Capture the current frame and save it as a new location
    [Z]    Cancel current location navigation
    [V]    Push-to-talk voice command (press once to start, again to stop)
    [F]    Toggle camera horizontal flip (for mirrored / selfie cameras)
    [1-9]  Start navigation to the 1st-9th saved location
    [Q]    Quit

Usage:
    python main.py
    python main.py --camera 0 --device cpu
    python main.py --test-pipelines
    python main.py --no-voice
    python main.py --no-flip        # disable horizontal frame flip
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from Pipelines.scene_pipeline        import ScenePipeline
from Pipelines.ocr_pipeline          import OCRPipeline
from Pipelines.object_pipeline       import ObjectPipeline
from Pipelines.decision_engine       import DecisionEngine
from Pipelines.navigation_pipeline   import NavigationPipeline
from Pipelines.location_manager      import LocationManager
from Pipelines.location_navigation   import LocationNavigator
from Pipelines.arrival_fusion        import compute_path_signature

try:
    from voice_listener import VoiceListener, ACTION_NAVIGATE, ACTION_STOP_NAV, \
        ACTION_LIST_LOCATIONS, ACTION_REPEAT, ACTION_READ_TEXT, ACTION_WHERE_AM_I, \
        CTRL_SILENCE, CTRL_UNSILENCE
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    # Dummy constants so the rest of the file doesn't NameError
    ACTION_NAVIGATE       = "NAVIGATE_TO"
    ACTION_STOP_NAV       = "STOP_NAVIGATION"
    ACTION_LIST_LOCATIONS = "LIST_LOCATIONS"
    ACTION_REPEAT         = "REPEAT"
    ACTION_READ_TEXT      = "READ_TEXT"
    ACTION_WHERE_AM_I     = "WHERE_AM_I"
    CTRL_SILENCE          = "__SILENCE__"
    CTRL_UNSILENCE        = "__UNSILENCE__"


# =========================================================
# PATHS
# =========================================================

BASE_DIR            = Path(__file__).resolve().parent
MODEL_SCENE         = BASE_DIR / "Models" / "Scene Recognition" / "Scene Recognition.pth"
MODEL_OCR           = BASE_DIR / "Models" / "Text Recognition"  / "best_smartvisionx_crnn_lowercase.pt"
MODEL_YOLO          = BASE_DIR / "Models" / "Object Detection"  / "yolov8n.pt"
MODEL_NAV_OUTDOOR   = BASE_DIR / "Models" / "Navigation"         / "best_outdoor_navigation.pt"
MODEL_NAV_INDOOR    = BASE_DIR / "Models" / "Navigation"         / "best_indoor_rgbd_navigation.pt"


# =========================================================
# CONFIG
# =========================================================

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

SCENE_INTERVAL_SEC    = 1.0
OCR_CROP_W            = 360
OCR_CROP_H            = 120
AUTO_OCR_ENABLED      = False
OCR_AUTO_EDGE_THRESH  = 0.08
OCR_AUTO_COOLDOWN_SEC = 5.0
AUTO_REPEAT_COOLDOWN  = 10.0
LOCATIONS_RELOAD_SEC  = 5.0

# M5: horizontal flip toggle — set True if your camera delivers a mirrored
# image (typical for front / selfie cameras).  Press [F] at runtime to
# toggle.  The flip is applied to the raw frame so every downstream
# pipeline (YOLO bboxes, position math, decisions) sees the same
# orientation the user does.
FRAME_FLIP_DEFAULT = True


# =========================================================
# TTS
# =========================================================

_speech_queue: queue.Queue = queue.Queue()
_speaking = threading.Event()


def _say_subprocess(text: str) -> None:
    safe = text.replace("'", " ").replace('"', " ")
    try:
        if sys.platform == "win32":
            cmd = [
                "powershell", "-NoProfile", "-WindowStyle", "Hidden",
                "-Command",
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = 1; $s.Speak('{safe}');",
            ]
            subprocess.run(cmd, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        elif sys.platform == "darwin":
            subprocess.run(["say", "-r", "175", safe], timeout=30)
        else:
            for exe in ("espeak-ng", "espeak"):
                try:
                    subprocess.run([exe, "-s", "160", safe], timeout=30)
                    break
                except FileNotFoundError:
                    continue
    except subprocess.TimeoutExpired:
        print("[TTS] timeout — skipping")
    except Exception as exc:
        print(f"[TTS] error: {exc}")


def _speech_worker() -> None:
    while True:
        text = _speech_queue.get()
        if text is None:
            break
        _speaking.set()
        try:
            _say_subprocess(text)
        finally:
            _speaking.clear()


threading.Thread(target=_speech_worker, name="tts-worker", daemon=True).start()


def speak(text: str) -> None:
    """Queue a message; never drop it. Use force_speak() to interrupt."""
    if not text:
        return
    try:
        _speech_queue.put_nowait(text.strip())
    except queue.Full:
        try:
            _speech_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _speech_queue.put_nowait(text.strip())
        except queue.Full:
            pass


def force_speak(text: str) -> None:
    """Interrupt anything currently being spoken and announce this NOW."""
    while not _speech_queue.empty():
        try:
            _speech_queue.get_nowait()
        except queue.Empty:
            break
    try:
        _speech_queue.put_nowait(text.strip())
    except queue.Full:
        pass


# =========================================================
# OCR HELPERS
# =========================================================

def get_ocr_crop(frame):
    h, w = frame.shape[:2]
    x1 = max(0, w // 2 - OCR_CROP_W // 2)
    y1 = max(0, h // 2 - OCR_CROP_H // 2)
    x2 = min(w, x1 + OCR_CROP_W)
    y2 = min(h, y1 + OCR_CROP_H)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _crop_has_text(crop) -> bool:
    if crop is None or crop.size == 0:
        return False
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, 50, 150)
    density = float(edges.mean()) / 255.0
    return density > OCR_AUTO_EDGE_THRESH


# =========================================================
# ASYNC PIPELINE WORKERS
# =========================================================

class _AsyncWorker:
    def __init__(self, fn, default_result: dict):
        self._fn      = fn
        self.result   = default_result
        self._lock    = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, item):
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        if isinstance(item, dict):
            payload = dict(item)
            if "frame" in payload and hasattr(payload["frame"], "copy"):
                payload["frame"] = payload["frame"].copy()
        else:
            payload = item.copy() if hasattr(item, "copy") else item
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def _run(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            try:
                result = self._fn(frame)
                with self._lock:
                    self.result = result
            except Exception as exc:
                print(f"[AsyncWorker] error: {exc}")

    def get(self) -> dict:
        with self._lock:
            return self.result

    def stop(self):
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


# =========================================================
# GUI HELPERS
# =========================================================

C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_AMBER  = (0,   200, 255)
C_CYAN   = (255, 220, 0)
C_GREEN  = (60,  180, 60)
C_GREY   = (160, 160, 160)
C_DARK   = (30,  30,  30)
C_PURPLE = (200, 120, 255)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# M5: danger palette for the location banner
C_DANGER_STOP    = (40,  40,  220)   # red
C_DANGER_WARNING = (40,  110, 220)   # orange
C_DANGER_CAUTION = (40,  180, 200)   # yellow


def _fill_rect(img, pt1, pt2, color, alpha=0.55):
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi     = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    img[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


def _label(img, text, pos, scale=0.55, color=C_WHITE, thickness=1):
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                FONT, scale, C_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_ocr_box(display, box, active=False):
    x1, y1, x2, y2 = box
    color = C_GREEN if active else C_CYAN
    t = 14
    for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        sx = 1 if px == x1 else -1
        sy = 1 if py == y1 else -1
        cv2.line(display, (px, py), (px + sx * t, py), color, 2)
        cv2.line(display, (px, py), (px, py + sy * t), color, 2)
    _label(display, "OCR READING..." if active else "[R] Read Text",
           (x1, y1 - 6), 0.48, color)


def draw_detection(display, det):
    x1, y1, x2, y2 = det["bbox"]
    color = {"high": (40, 40, 220), "medium": (40, 140, 220),
             "low": (40, 180, 80)}[det["danger"]]
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    label = f"{det['name']}  {det['confidence']:.0%}"
    tw, th = cv2.getTextSize(label, FONT, 0.45, 1)[0]
    cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(display, label, (x1 + 3, y1 - 5),
                FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)


def draw_hud(display, scene, scene_conf, fps,
             det_ms, scene_ms, ocr_ms, nav_ms,
             last_ocr_text, last_alert_text,
             location_status=None, frame_flip=True):   # M5: frame_flip param
    h, w = display.shape[:2]
    _fill_rect(display, (0, 0), (320, 60), C_DARK, 0.60)
    _label(display,
           f"Scene: {scene.replace('_', ' ').title()}  ({scene_conf:.0%})",
           (8, 22), 0.58, C_AMBER)
    _label(display, f"FPS  {fps:.1f}", (8, 46), 0.48, C_GREY)
    if last_ocr_text:
        _fill_rect(display, (0, 64), (420, 88), C_DARK, 0.55)
        _label(display, f"OCR: {last_ocr_text}", (8, 82), 0.52, (100, 230, 255))

    if location_status and location_status.get("location_state") not in (None, "idle", ""):
        ls = location_status
        state = ls.get("location_state", "")
        target = ls.get("location_target_display") or ls.get("location_target", "")
        msg    = ls.get("location_message", "")
        progress = ls.get("location_progress", "")

        arrival = ls.get("location_arrival") or {}
        sigs = arrival.get("signals") or {}

        if state == "arrived":    banner_color = (40, 140, 60)
        elif state == "approaching": banner_color = (40, 120, 200)
        elif state == "failed":   banner_color = (40, 40, 180)
        else:                     banner_color = (90, 60, 130)

        base_y = h - 160
        _fill_rect(display, (0, base_y), (w, base_y + 150), banner_color, 0.55)
        _label(display, f"TARGET: {target}  [{state.upper()}]",
               (8, base_y + 22), 0.58, C_WHITE, thickness=2)
        if progress:
            _label(display, progress, (8, base_y + 44), 0.5, C_WHITE)
        if msg:
            _label(display, msg, (8, base_y + 64), 0.5, C_WHITE)

        # M5: location danger indicator (always present when navigating)
        danger_level = ls.get("location_danger_level", "none")
        danger_msg   = ls.get("location_danger_message", "")
        if danger_level in ("stop", "warning", "caution"):
            dcolor = {
                "stop":    C_DANGER_STOP,
                "warning": C_DANGER_WARNING,
                "caution": C_DANGER_CAUTION,
            }.get(danger_level, (200, 200, 200))
            _fill_rect(display, (0, base_y + 70), (w, base_y + 96), dcolor, 0.85)
            _label(display,
                   f"⚠  {danger_level.upper()}: {danger_msg}",
                   (8, base_y + 89), 0.55, C_WHITE, thickness=2)

        chip_y = base_y + 110
        chip_w = w // 4
        chip_names = [("VIS", sigs.get("visual")),
                      ("OBJ", sigs.get("object")),
                      ("SCN", sigs.get("scene")),
                      ("PTH", sigs.get("path_metric"))]
        for i, (label, sig) in enumerate(chip_names):
            x0 = i * chip_w
            score = float(sig.get("score", 0.0)) if sig else 0.0
            agrees = bool(sig.get("agrees", False)) if sig else False
            chip_color = (60, 180, 80) if agrees else (90, 90, 90)
            cv2.rectangle(display, (x0 + 4, chip_y),
                          (x0 + chip_w - 4, chip_y + 22), chip_color, -1)
            _label(display, f"{label} {int(score*100):3d}%",
                   (x0 + 10, chip_y + 16), 0.45, C_WHITE, thickness=1)

    if last_alert_text:
        tw = cv2.getTextSize(last_alert_text, FONT, 0.52, 1)[0][0]
        ax = max(4, w // 2 - tw // 2 - 6)
        _fill_rect(display, (ax - 4, h - 38), (ax + tw + 10, h - 14), C_DARK, 0.65)
        _label(display, last_alert_text, (ax, h - 20), 0.52, C_WHITE)
    lat = (f"Det {det_ms:.0f}ms    Scene {scene_ms:.0f}ms    "
           f"OCR {ocr_ms:.0f}ms    Nav {nav_ms:.0f}ms"
           f"    {'FLIP' if frame_flip else 'noFLIP'}")   # M5
    _fill_rect(display, (0, h - 14), (w, h), C_DARK, 0.70)
    cv2.putText(display, lat, (6, h - 3), FONT, 0.38, C_GREY, 1, cv2.LINE_AA)


# =========================================================
# PIPELINE TEST
# =========================================================

def test_pipelines(device: str):
    print("\n" + "=" * 60)
    print("PIPELINE VERIFICATION MODE")
    print("=" * 60)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print("\n[1/4] Scene Recognition")
    sc = ScenePipeline(MODEL_SCENE, device=device)
    lbl, conf, ms = sc.predict(dummy)
    print(f"  {lbl}  conf={conf}  {ms}ms  OK")
    print("\n[2/4] OCR")
    oc = OCRPipeline(MODEL_OCR, device=device)
    crop, _ = get_ocr_crop(dummy)
    txt, ms = oc.read(crop)
    print(f"  '{txt}'  {ms}ms  OK")
    print("\n[3/4] Object Detection")
    ob = ObjectPipeline(MODEL_YOLO, device=device)
    dets, ms = ob.detect(dummy)
    print(f"  {len(dets)} objects  {ms}ms  OK")
    print("\n[4/4] Navigation")
    nv = NavigationPipeline(MODEL_NAV_OUTDOOR, MODEL_NAV_INDOOR, device=device)
    r = nv.predict(dummy, detections=[])
    print(f"  {r['command']}  {r['ms']}ms  OK")
    print("\n" + "=" * 60)
    print("All 4 pipelines OK.")
    print("=" * 60 + "\n")


# =========================================================
# LOCATION HELPERS
# =========================================================

def list_locations_to_terminal(mgr: LocationManager) -> None:
    locs = mgr.list_locations()
    print("\n--- Saved locations ---")
    if not locs:
        print("  (none)  Run Tools/save_location.py to add some.")
    else:
        for i, loc in enumerate(locs, 1):
            sig = "path" if loc.get("has_path_signature") else "    "
            prof = "prof" if loc.get("has_path_profile") else "    "
            print(f"  {i}. {loc['display_name']:20s}  "
                  f"refs={loc['reference_count']}  "
                  f"matches={loc['match_count']}  [{sig}] [{prof}]")
    print("-----------------------\n")


def capture_and_save_location(mgr: LocationManager,
                              frame: np.ndarray,
                              obj_pipeline,
                              scene_pipeline=None,
                              nav_pipeline=None,
                              scene_label=None,
                              scene_conf=0.0) -> None:
    """Save the current frame as a new location (multi-signal edition)."""
    try:
        name = input("Location name: ").strip()
    except EOFError:
        name = ""
    if not name:
        print("[Locations] Save cancelled (empty name).")
        return
    objects: list = []
    if obj_pipeline is not None:
        try:
            dets, _ = obj_pipeline.detect(frame)
            objects = [d["name"] for d in dets[:8]]
        except Exception:
            pass

    path_signature: list = []
    if nav_pipeline is not None:
        try:
            res = nav_pipeline.predict(frame, scene=scene_label,
                                       scene_conf=scene_conf, detections=[])
            path_signature = compute_path_signature(res.get("safe_scores", {}) or {})
        except Exception:
            pass

    try:
        loc = mgr.add_location(
            name, frame,
            objects=objects,
            scene=scene_label,
            scene_conf=scene_conf,
            path_signature=path_signature or None,
        )
        print(f"[Locations] Saved '{loc['name']}' "
              f"({len(loc['references'])} reference(s), "
              f"path_signature={'yes' if path_signature else 'no'}).")
    except Exception as exc:
        print(f"[Locations] Save failed: {exc}")


# =========================================================
# MAIN LOOP
# =========================================================

def run(camera_id: int, device: str, enable_voice: bool, frame_flip: bool):
    print(f"[System] device={device}  frame_flip={frame_flip}")

    scene_pipeline      = ScenePipeline(MODEL_SCENE,    device=device)
    ocr_pipeline        = OCRPipeline(MODEL_OCR,        device=device)
    object_pipeline     = ObjectPipeline(MODEL_YOLO,    device=device)
    navigation_pipeline = NavigationPipeline(MODEL_NAV_OUTDOOR, MODEL_NAV_INDOOR, device=device)
    decision            = DecisionEngine()
    location_manager    = LocationManager()
    location_navigator  = LocationNavigator(location_manager)

    voice_queue: queue.Queue = queue.Queue()
    voice_listener = None
    voice_silenced = False

    if enable_voice and VOICE_AVAILABLE:
        def on_voice(action: str) -> None:
            try:
                voice_queue.put_nowait(action)
            except queue.Full:
                pass
        voice_listener = VoiceListener(on_voice, location_manager.location_names())
        if voice_listener.is_available():
            voice_listener.start()
        else:
            voice_listener = None

    def _run_detection(frame):
        dets, ms = object_pipeline.detect(frame)
        return {"dets": dets, "ms": ms}

    def _run_navigation(payload):
        return navigation_pipeline.predict(
            payload["frame"],
            scene=payload.get("scene", current_scene[0]),
            scene_conf=payload.get("scene_conf", current_scene[1]),
            detections=payload.get("detections", []),
        )

    det_worker = _AsyncWorker(fn=_run_detection, default_result={"dets": [], "ms": 0})
    nav_worker = _AsyncWorker(fn=_run_navigation,
                              default_result=navigation_pipeline.default_result())

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return

    print("\nSmartVisionX running (multi-signal arrival).")
    print("Keys: [R] OCR  [L] list  [K] save-loc  [Z] cancel-nav  [V] voice  "
          "[F] flip  [1-9] goto  [Q] quit")
    print("-" * 60)
    if voice_listener:
        print('Voice (push-to-talk [V]): "go to <loc>", "stop navigation", "list locations", "repeat", "read text", "get my location"')
        print("-" * 60)

    fps_times = []
    current_scene = ("unknown", 0.0)
    last_scene_time = 0.0
    last_ocr_text = ""
    last_alert_text = ""
    last_auto_ocr_time = 0.0
    last_spoken_time = 0.0
    last_locations_reload = 0.0
    ocr_active = False
    scene_ms = ocr_ms = 0
    fps = 0.0
    location_status: dict = {"location_state": "idle"}
    prev_location_state = "idle"
    pending_ocr = False
    frame_flip_state = frame_flip   # mutable, toggled by [F]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        # M5: flip the raw frame so EVERY downstream pipeline
        # (detection, position math, decisions, display) is consistent
        # with the orientation the user actually sees.
        if frame_flip_state:
            frame = cv2.flip(frame, 1)

        t_frame = time.perf_counter()
        now     = time.time()
        key     = cv2.waitKey(1) & 0xFF

        # ── Hotkeys ───────────────────────────────────────────────────
        if key in (ord("l"), ord("L")):
            list_locations_to_terminal(location_manager)
        elif key in (ord("k"), ord("K")):
            capture_and_save_location(
                location_manager, frame.copy(), object_pipeline,
                scene_pipeline=scene_pipeline,
                nav_pipeline=navigation_pipeline,
                scene_label=current_scene[0], scene_conf=current_scene[1],
            )
            if voice_listener:
                voice_listener.set_known_locations(location_manager.location_names())
        elif key in (ord("z"), ord("Z")):
            if location_navigator.is_active() or location_status.get("location_state") not in (None, "idle", ""):
                location_status = location_navigator.stop("Navigation cancelled.")
                force_speak("Navigation cancelled.")
        elif key in (ord("f"), ord("F")):                               # M5
            frame_flip_state = not frame_flip_state
            force_speak(f"Camera flip {'on' if frame_flip_state else 'off'}.")
            print(f"[System] Frame flip → {frame_flip_state}")
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            names = location_manager.location_names()
            if 0 <= idx < len(names):
                target = names[idx]
                path_sig = None
                if nav_worker.get() and nav_worker.get().get("safe_scores"):
                    path_sig = compute_path_signature(nav_worker.get()["safe_scores"])
                location_status = location_navigator.set_target(
                    target, path_signature=path_sig)
                if location_status.get("location_message"):
                    force_speak(location_status["location_message"])
            else:
                force_speak(f"No location number {idx + 1}.")
        elif key in (ord("r"), ord("R")):
            pending_ocr = True
        elif key in (ord("v"), ord("V")):
            if voice_listener and voice_listener.is_available():
                if not getattr(voice_listener, "_ptt_active", False):
                    force_speak("I'm listening.")
                    voice_listener.begin_push_to_talk()
                else:
                    cmd = voice_listener.end_push_to_talk()
                    if cmd:
                        try:
                            voice_queue.put_nowait(cmd)
                        except queue.Full:
                            pass
                    else:
                        force_speak("I didn't hear a command.")

        # ── Periodic reload ───────────────────────────────────────────
        if (now - last_locations_reload) >= LOCATIONS_RELOAD_SEC:
            location_manager.reload()
            last_locations_reload = now
            if voice_listener:
                voice_listener.set_known_locations(location_manager.location_names())

        # ── Voice commands ───────────────────────────────────────────
        while not voice_queue.empty():
            try:
                action = voice_queue.get_nowait()
            except queue.Empty:
                break

            if action == CTRL_SILENCE:
                voice_silenced = True
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                             "-Command", "(New-Object -COMObject SAPI.SpVoice).Speak('', 2)"],
                            timeout=2, creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception:
                    pass
                continue
            if action == CTRL_UNSILENCE:
                voice_silenced = False
                continue
            if voice_silenced:
                continue

            if action == ACTION_STOP_NAV:
                location_status = location_navigator.stop("Navigation cancelled.")
                force_speak(location_status["location_message"])
            elif action == ACTION_LIST_LOCATIONS:
                names = location_manager.location_names()
                if names:
                    list_locations_to_terminal(location_manager)
                    force_speak("Saved places: " + ", ".join(
                        loc.replace("_", " ") for loc in names) + ".")
                else:
                    force_speak("No saved locations yet.")
            elif action == ACTION_REPEAT:
                if last_alert_text:
                    force_speak(last_alert_text)
            elif action == ACTION_READ_TEXT:
                pending_ocr = True
            elif action == ACTION_WHERE_AM_I:
                match = location_manager.match_location(frame)
                if match and match["match_score"] >= 0.45:
                    display_name = match["display_name"].replace("_", " ")
                    force_speak(f"You are at {display_name}.")
                    print(f"[WhereAmI] Matched '{match['name']}' score={match['match_score']}")
                else:
                    force_speak("I don't recognise this location.")
                    print("[WhereAmI] No confident match found.")
            elif action.startswith(f"{ACTION_NAVIGATE}:"):
                target = action.split(":", 1)[1].strip()
                path_sig = None
                cur_nav = nav_worker.get()
                if cur_nav and cur_nav.get("safe_scores"):
                    path_sig = compute_path_signature(cur_nav["safe_scores"])
                location_status = location_navigator.set_target(
                    target, path_signature=path_sig)
                if location_status.get("location_message"):
                    force_speak(location_status["location_message"])

        # ── Async pipelines ──────────────────────────────────────────
        det_result   = det_worker.get()
        current_dets = det_result["dets"]
        det_ms       = det_result["ms"]
        det_worker.put(frame)
        nav_worker.put({
            "frame": frame,
            "scene": current_scene[0],
            "scene_conf": current_scene[1],
            "detections": current_dets,
        })

        # ── Scene (throttled) ────────────────────────────────────────
        if (now - last_scene_time) >= SCENE_INTERVAL_SEC:
            lbl, conf, scene_ms = scene_pipeline.predict(frame)
            current_scene = (lbl, conf)
            last_scene_time = now

        nav_result = nav_worker.get()
        nav_ms = nav_result["ms"]

        # ── Location navigator (multi-signal + danger) ───────────────
        if location_navigator.is_active() or location_navigator.target:
            location_status = location_navigator.update(
                frame, nav_result, current_dets,
                scene=current_scene[0], scene_conf=current_scene[1],
            )

        # ── Arrival announcement (exact spec wording) ─────────────────
        cur_state = location_status.get("location_state", "idle")
        if cur_state == "arrived" and prev_location_state != "arrived":
            msg = location_status.get("location_message") or "You arrive to the destination."
            force_speak(msg)
            print(f"[Arrival] {msg}")
        prev_location_state = cur_state

        # ── OCR (event-based) ────────────────────────────────────────
        ocr_crop, ocr_box = get_ocr_crop(frame)
        auto_ocr_due  = AUTO_OCR_ENABLED and ((now - last_auto_ocr_time) >= OCR_AUTO_COOLDOWN_SEC)
        manual_ocr    = pending_ocr or (key in (ord("r"), ord("R")))
        ocr_triggered = False
        ocr_text      = ""
        if manual_ocr or (auto_ocr_due and _crop_has_text(ocr_crop)):
            try:
                ocr_text, ocr_ms = ocr_pipeline.read(frame)
            except Exception as e:
                print(f"[OCR] error: {e}")
            ocr_triggered      = True
            last_auto_ocr_time = now
            last_ocr_text      = ocr_text
            ocr_active         = True
            print(f"[OCR] '{ocr_text}' ({ocr_ms:.0f}ms) [manual]")
        else:
            ocr_active = False
        pending_ocr = False

        # ── FPS / latency metrics ────────────────────────────────────
        elapsed = (time.perf_counter() - t_frame) * 1000
        fps_times.append(elapsed)
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = 1000.0 / (sum(fps_times) / len(fps_times))
        total_latency_ms = (float(det_ms) + float(nav_ms) + float(scene_ms)
                            + (float(ocr_ms) if ocr_triggered else 0.0))

        # ── Decision engine ──────────────────────────────────────────
        messages = decision.decide(
            scene=current_scene[0],
            scene_conf=current_scene[1],
            detections=current_dets,
            ocr_text=ocr_text,
            ocr_triggered=ocr_triggered,
            nav_result=nav_result,
            frame_width=FRAME_WIDTH,
            ocr_source="manual" if manual_ocr else "auto",
            fps=fps, det_ms=det_ms, scene_ms=scene_ms,
            ocr_ms=ocr_ms, nav_ms=nav_ms, total_latency_ms=total_latency_ms,
            location_status=location_status,
        )

        if not voice_silenced and messages:
            speak(messages[0]["text"])
            last_alert_text = messages[0]["text"]
            last_spoken_time = now
            print(f"[Alert] {messages[0]['text']}")
        elif (not voice_silenced
              and decision.last_message
              and (now - last_spoken_time) >= AUTO_REPEAT_COOLDOWN):
            force_speak(decision.last_message)
            last_spoken_time = now

        # ── Display ──────────────────────────────────────────────────
        display = frame.copy()
        navigation_pipeline.draw(display, nav_result)
        draw_ocr_box(display, ocr_box, active=ocr_active)
        for det in current_dets:
            draw_detection(display, det)
        draw_hud(display,
                 scene=current_scene[0], scene_conf=current_scene[1],
                 fps=fps, det_ms=det_ms, scene_ms=scene_ms,
                 ocr_ms=ocr_ms, nav_ms=nav_ms,
                 last_ocr_text=last_ocr_text,
                 last_alert_text=last_alert_text,
                 location_status=location_status,
                 frame_flip=frame_flip_state)
        cv2.imshow("SmartVisionX", display)

        if key in (ord("q"), ord("Q")):
            break

    det_worker.stop()
    nav_worker.stop()
    if voice_listener:
        voice_listener.stop()
    cap.release()
    cv2.destroyAllWindows()
    _speech_queue.put(None)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",    type=int, default=0)
    parser.add_argument("--device",    type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--test-pipelines", action="store_true")
    parser.add_argument("--no-voice",  action="store_true")
    parser.add_argument("--no-flip",   action="store_true",
                        help="Disable horizontal camera flip (default: flip on)")
    args = parser.parse_args()

    if args.test_pipelines:
        test_pipelines(args.device)
    else:
        run(args.camera, args.device,
            enable_voice=not args.no_voice,
            frame_flip=not args.no_flip)