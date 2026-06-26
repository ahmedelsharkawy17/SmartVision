"""
SmartVisionX Save-Location Tool — multi-signal edition (M4)
===========================================================

In addition to the visual fingerprint and the live safe-path vector
(left/center/right), the tool captures a 3-second multi-frame
sequence and builds a 10-metric PathProfile from the navigation
output. This populates Signal 5 of ArrivalFusion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from Pipelines.location_manager   import LocationManager
from Pipelines.arrival_fusion     import compute_path_signature
from Pipelines.path_metrics       import compute_metrics, PathProfile   # M4

try:
    from Pipelines.object_pipeline     import ObjectPipeline
    _YOLO = _ROOT / "Models" / "Object Detection" / "yolov8n.pt"
    OBJECT_AVAILABLE = _YOLO.exists()
except Exception:
    OBJECT_AVAILABLE = False

try:
    from Pipelines.scene_pipeline      import ScenePipeline
    _SCENE = _ROOT / "Models" / "Scene Recognition" / "Scene Recognition.pth"
    SCENE_PIPELINE = ScenePipeline(str(_SCENE), device="cpu") if _SCENE.exists() else None
except Exception:
    SCENE_PIPELINE = None

try:
    from Pipelines.navigation_pipeline import NavigationPipeline
    _NAV_OUT = _ROOT / "Models" / "Navigation" / "best_outdoor_navigation.pt"
    _NAV_IN  = _ROOT / "Models" / "Navigation" / "best_indoor_rgbd_navigation.pt"
    if _NAV_OUT.exists():
        NAV_PIPELINE = NavigationPipeline(str(_NAV_OUT),
                                          str(_NAV_IN) if _NAV_IN.exists() else None,
                                          device="cpu")
    else:
        NAV_PIPELINE = None
except Exception:
    NAV_PIPELINE = None


WINDOW = "SmartVisionX - Save Location"


def parse_args():
    p = argparse.ArgumentParser(description="Save popular locations for SmartVisionX.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--list",   action="store_true")
    p.add_argument("--delete", type=str, default=None)
    return p.parse_args()


def list_locations(mgr: LocationManager) -> None:
    locs = mgr.list_locations()
    print("\n" + "=" * 60)
    print("Saved locations")
    print("=" * 60)
    if not locs:
        print("  (none yet — run the tool without --list to add some)")
    for loc in locs:
        sig = "✓ path" if loc.get("has_path_signature") else "  path"
        prof = "✓ profile" if loc.get("has_path_profile") else "  profile"
        print(f"  • {loc['display_name']:20s}  refs={loc['reference_count']:2d}  "
              f"matches={loc['match_count']:3d}  {sig}  {prof}")
    print("=" * 60 + "\n")


def delete_location(mgr: LocationManager, name: str) -> None:
    if mgr.remove_location(name):
        print(f"[Locations] Removed '{name}'.")
    else:
        print(f"[Locations] '{name}' not found.")


def overlay_lines(img, lines, origin=(20, 30), color=(255, 255, 255),
                  bg=(0, 0, 0), scale=0.6, thickness=2):
    x, y = origin
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 6), bg, -1)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += th + 12


def draw_test_overlay(img, mgr: LocationManager) -> None:
    if not mgr.locations:
        return
    scores = mgr.get_all_scores(img)[:5]
    lines = ["Live match scores:"]
    for s in scores:
        bar = "#" * int(round(s["score"] * 16)) + "-" * (16 - int(round(s["score"] * 16)))
        lines.append(f"  {s['display_name'][:18]:18s} {bar} {int(s['score'] * 100):3d}%")
    overlay_lines(img, lines, origin=(20, 100),
                  color=(180, 240, 255), bg=(20, 20, 60))


def prompt_name(default: str = "") -> str:
    try:
        return input(f"Location name [{default}]: " if default else "Location name: ").strip()
    except EOFError:
        return default


def main() -> int:
    args = parse_args()
    mgr = LocationManager()

    if args.list:
        list_locations(mgr)
        return 0
    if args.delete:
        delete_location(mgr, args.delete)
        return 0

    obj_pipeline = None
    if OBJECT_AVAILABLE:
        try:
            obj_pipeline = ObjectPipeline(str(_YOLO), device=args.device)
        except Exception as exc:
            print(f"[SaveLocation] Object pipeline unavailable: {exc}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[SaveLocation] Cannot open camera {args.camera}.")
        return 1

    print("\nSmartVisionX — Save Location Tool (M4 multi-signal)")
    print("=" * 50)
    print("Each saved reference stores: visual fingerprint, scene label,")
    print("detected objects, the safe-path vector (left/center/right),")
    print("AND a 10-metric path profile from a 3-second multi-frame capture.")
    print("  SPACE   capture the current frame (3s of profile follows)")
    print("  S       save (asks for a name)")
    print("  R       re-capture (discard previous)")
    print("  T       toggle test mode (live match scores)")
    print("  L       list saved locations")
    print("  D       delete a saved location")
    print("  Q/ESC   quit")
    print("=" * 50 + "\n")

    captured: np.ndarray | None = None
    captured_objects: list = []
    captured_scene = None
    captured_scene_conf = 0.0
    captured_path_signature: list = []
    captured_path_profile: dict = {}
    test_mode = False
    capturing_profile = False
    profile_start = 0.0
    PROFILE_DURATION = 3.0
    profile_frames_metrics: list = []
    profile_last_safe: dict = {"left": 0.0, "center": 0.0, "right": 0.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[SaveLocation] Camera read failed.")
            break

        display = frame.copy()
        if test_mode:
            draw_test_overlay(display, mgr)

        overlay_lines(display, [
            "SmartVisionX  -  Save Location (M4 multi-signal)",
            "SPACE: capture   S: save   R: re-capture   T: test   L: list   D: delete   Q: quit",
        ])

        # Profile-capture progress banner
        if capturing_profile:
            elapsed = time.time() - profile_start
            remaining = max(0.0, PROFILE_DURATION - elapsed)
            pct = int(100 * elapsed / PROFILE_DURATION)
            bar_w = 20
            filled = int(round((elapsed / PROFILE_DURATION) * bar_w))
            bar = "#" * filled + "-" * (bar_w - filled)
            cv2.rectangle(display, (0, display.shape[0] - 130),
                          (display.shape[1], display.shape[0] - 60), (60, 60, 0), -1)
            cv2.putText(display,
                        f"Building path profile  [{bar}] {pct}%   {remaining:.1f}s left",
                        (20, display.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display,
                        f"Hold camera steady at the destination.",
                        (20, display.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (220, 220, 180), 1, cv2.LINE_AA)

        if captured is not None and not capturing_profile:
            cv2.rectangle(display, (0, display.shape[0] - 90),
                          (display.shape[1], display.shape[0]), (0, 120, 0), -1)
            cv2.putText(display, "FRAME CAPTURED  -  press S to save, R to re-capture",
                        (20, display.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            meta = []
            if captured_scene:
                meta.append(f"scene: {captured_scene} ({captured_scene_conf:.0%})")
            if captured_objects:
                meta.append("objects: " + ", ".join(captured_objects[:6]))
            if captured_path_signature:
                meta.append("path: L={:.2f} C={:.2f} R={:.2f}".format(*captured_path_signature))
            if captured_path_profile and captured_path_profile.get("metrics"):
                n_metrics = len(captured_path_profile.get("metrics", {}))
                meta.append(f"profile: {n_metrics} metrics")
            if meta:
                cv2.putText(display, "  |  ".join(meta), (20, display.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord(" "), 13, 10):
            if capturing_profile:
                continue
            captured = frame.copy()
            captured_objects = []
            captured_scene = None
            captured_scene_conf = 0.0
            captured_path_signature = []
            captured_path_profile = {}
            if obj_pipeline is not None:
                dets, _ = obj_pipeline.detect(captured)
                captured_objects = [d["name"] for d in dets[:8]]
            if SCENE_PIPELINE is not None:
                try:
                    lbl, conf, _ = SCENE_PIPELINE.predict(captured)
                    captured_scene = lbl
                    captured_scene_conf = conf
                except Exception:
                    pass

            # Initialise the profile-capture sequence
            profile_frames_metrics = []
            profile_last_safe = {"left": 0.0, "center": 0.0, "right": 0.0}
            capturing_profile = True
            profile_start = time.time()
            print(f"[SaveLocation] Captured. objects={captured_objects} "
                  f"scene={captured_scene}. Hold steady for "
                  f"{PROFILE_DURATION:.0f}s to build the path profile...")

        elif key in (ord("s"), ord("S")):
            if capturing_profile:
                print("[SaveLocation] Still building profile — please wait.")
                continue
            if captured is None:
                print("[SaveLocation] No frame captured yet. Press SPACE first.")
                continue
            name = prompt_name()
            if not name:
                print("[SaveLocation] Save cancelled (empty name).")
                continue
            try:
                loc = mgr.add_location(
                    name, captured,
                    objects=captured_objects,
                    scene=captured_scene,
                    scene_conf=captured_scene_conf,
                    path_signature=captured_path_signature or None,
                    path_profile=captured_path_profile or None,
                )
                print(f"[SaveLocation] Saved '{loc['name']}' "
                      f"({len(loc['references'])} reference(s), "
                      f"profile={'yes' if captured_path_profile else 'no'}).")
            except Exception as exc:
                print(f"[SaveLocation] Save failed: {exc}")
            captured = None
            captured_path_profile = {}

        elif key in (ord("r"), ord("R")):
            captured = None
            captured_path_profile = {}
            capturing_profile = False
            print("[SaveLocation] Capture discarded.")

        elif key in (ord("t"), ord("T")):
            test_mode = not test_mode
            print(f"[SaveLocation] Test mode {'on' if test_mode else 'off'}.")

        elif key in (ord("l"), ord("L")):
            list_locations(mgr)

        elif key in (ord("d"), ord("D")):
            name = prompt_name()
            if name:
                delete_location(mgr, name)

        # M4: profile-capture loop — runs every frame after SPACE
        if capturing_profile:
            elapsed = time.time() - profile_start
            if elapsed >= PROFILE_DURATION:
                # Finalise the profile
                if profile_frames_metrics:
                    profile = PathProfile(profile_frames_metrics).to_dict()
                    captured_path_profile = profile
                    captured_path_signature = compute_path_signature(profile_last_safe)
                    print(f"[SaveLocation] Path profile built from "
                          f"{len(profile_frames_metrics)} frames. "
                          f"final L={profile_last_safe['left']:.2f} "
                          f"C={profile_last_safe['center']:.2f} "
                          f"R={profile_last_safe['right']:.2f}")
                else:
                    print("[SaveLocation] No frames captured for profile.")
                capturing_profile = False
            else:
                # Get safe_scores for this frame
                safe = None
                if NAV_PIPELINE is not None and captured_scene is not None:
                    try:
                        res = NAV_PIPELINE.predict(
                            frame, scene=captured_scene,
                            scene_conf=captured_scene_conf, detections=[],
                        )
                        safe = res.get("safe_scores", {}) or {}
                        profile_last_safe = safe
                    except Exception:
                        safe = None
                if safe is None:
                    safe = profile_last_safe
                m = compute_metrics(safe_mask=None, safe_scores=safe)
                profile_frames_metrics.append(m)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
