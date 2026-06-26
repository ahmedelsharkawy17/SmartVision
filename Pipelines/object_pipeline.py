from ultralytics import YOLO
import time

# FIX — Expanded DANGER_MAP with 10 additional classes important for
# visually-impaired navigation. Original had 16 entries; now 26.
DANGER_MAP = {
    "person":        "low",
    "bicycle":       "medium",
    "car":           "high",
    "motorcycle":    "high",
    "bus":           "high",
    "truck":         "high",
    "dog":           "medium",
    "chair":         "low",
    "dining table":  "low",
    "bottle":        "low",
    "fire hydrant":  "low",
    "stop sign":     "medium",
    "traffic light": "medium",
    "stairs":        "medium",
    "knife":         "high",
    "scissors":      "medium",
    "door":          "low",
    "bench":         "low",
    "potted plant":  "low",
    "suitcase":      "medium",
    "backpack":      "low",
    "umbrella":      "low",
    "pole":          "medium",
    "barrier":       "high",
    "skateboard":    "medium",
    "cow":           "high",
}
DEFAULT_DANGER = "low"


class ObjectPipeline:
    def __init__(self, model_path: str = "models/yolov8n.pt",
                 device: str = "cpu", conf: float = 0.4):
        self.model  = YOLO(model_path)
        self.device = device
        self.conf   = conf
        print(f"[Object] YOLOv8 loaded | device={device} | conf={conf}")

    def detect(self, bgr_frame):
        t0      = time.perf_counter()
        results = self.model(bgr_frame, device=self.device,
                             conf=self.conf, verbose=False)[0]
        ms      = (time.perf_counter() - t0) * 1000

        detections = []
        for box in results.boxes:
            name        = results.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area        = (x2 - x1) * (y2 - y1)
            detections.append({
                "name":       name,
                "confidence": round(float(box.conf), 2),
                "bbox":       [x1, y1, x2, y2],
                "area":       area,
                "danger":     DANGER_MAP.get(name, DEFAULT_DANGER),
            })

        danger_rank = {"high": 0, "medium": 1, "low": 2}
        detections.sort(key=lambda d: (danger_rank[d["danger"]], -d["area"]))
        return detections, round(ms, 1)
