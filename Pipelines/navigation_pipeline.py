import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque

OUT_H = 128
OUT_W = 256
ROI_START_RATIO = 0.60

MIN_SAFE_SCORE = 0.30
HYSTERESIS_RESUME = 0.40
DIRECTION_MARGIN = 0.08
SMOOTH_FRAMES = 5

# Indoor RGB-D settings from the training notebook.
INDOOR_CLOSE_THRESHOLD = 0.55
DANGER_PENALTY = 0.70

INDOOR_SCENES = {
    "bathroom", "bedroom", "corridor", "indoor_passage", "eating_place", "restaurant",
    "elevator", "learning_space", "lecture_room", "library", "hospital", "mosque",
    "shopping_mall", "staircase", "supermarket", "transport_hub", "waiting_room", "work_space",
}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MiniUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def split_three_regions(mask):
    h, w = mask.shape
    roi = mask[int(h * ROI_START_RATIO):, :]
    return roi[:, :w // 3], roi[:, w // 3:2 * w // 3], roi[:, 2 * w // 3:]


def score_safe_regions(safe_mask):
    left, center, right = split_three_regions(safe_mask)
    return {"left": float(left.mean()), "center": float(center.mean()), "right": float(right.mean())}


def score_close_regions(depth_norm):
    close_mask = (depth_norm > INDOOR_CLOSE_THRESHOLD).astype(np.float32)
    left, center, right = split_three_regions(close_mask)
    return {"left": float(left.mean()), "center": float(center.mean()), "right": float(right.mean())}


def compute_final_scores(safe_scores, close_scores=None):
    if close_scores is None:
        return safe_scores
    return {
        k: safe_scores[k] - DANGER_PENALTY * close_scores[k]
        for k in ["left", "center", "right"]
    }


def decide_command(scores: dict, in_stop_state: bool = False) -> str:
    threshold = HYSTERESIS_RESUME if in_stop_state else MIN_SAFE_SCORE
    c, l, r = scores["center"], scores["left"], scores["right"]
    best_score = max(c, l, r)

    # No direction reaches the safe threshold → path simply not recognized.
    if best_score < threshold:
        return "UNRECOGNIZED PATH"

    # MOVE LEFT: left is safe AND beats center by margin AND beats right.
    if l >= MIN_SAFE_SCORE and l > r and l > c + DIRECTION_MARGIN:
        return "MOVE LEFT"

    # MOVE RIGHT: right is safe AND beats center by margin AND beats left.
    if r >= MIN_SAFE_SCORE and r > l and r > c + DIRECTION_MARGIN:
        return "MOVE RIGHT"

    # MOVE FORWARD: center is safe and no side clearly beats it.
    if c >= MIN_SAFE_SCORE:
        return "MOVE FORWARD"

    # Center is not safe but one side is — guide to the better side.
    if l >= MIN_SAFE_SCORE and l >= r:
        return "MOVE LEFT"
    if r >= MIN_SAFE_SCORE:
        return "MOVE RIGHT"

    return "UNRECOGNIZED PATH"


def _det_position(bbox, frame_width=640):
    if not bbox or len(bbox) < 4:
        return "unknown"
    cx = (bbox[0] + bbox[2]) / 2.0
    third = frame_width / 3.0
    if cx < third:
        return "left"
    if cx > 2 * third:
        return "right"
    return "front"


def _is_close(area):
    return float(area or 0) > 30_000


def apply_detection_fusion(command, scores, detections, frame_width=640):
    """
    Fuse YOLO detections with segmentation navigation.
    Navigation is no longer alone: a dangerous object in the forward region can override MOVE FORWARD.
    """
    detections = detections or []
    front_dangers = []
    for det in detections:
        danger = str(det.get("danger", "low"))
        pos = _det_position(det.get("bbox", []), frame_width)
        area = float(det.get("area", 0) or 0)
        if pos == "front" and (danger == "high" or (danger == "medium" and _is_close(area))):
            front_dangers.append(det)

    if not front_dangers:
        return command, None

    # Highest danger + closest first.
    rank = {"high": 0, "medium": 1, "low": 2}
    front_dangers.sort(key=lambda d: (rank.get(str(d.get("danger", "low")), 2), -float(d.get("area", 0) or 0)))
    det = front_dangers[0]
    name = str(det.get("name", "object"))
    danger = str(det.get("danger", "low"))
    area = float(det.get("area", 0) or 0)

    if danger == "high" and area > 80_000:
        return "STOP / OBJECT AHEAD", f"Stop. {name} is very close in front of you."

    # If the center is blocked by object but a side has a clearly safer score, guide to that side.
    left = float(scores.get("left", 0.0))
    right = float(scores.get("right", 0.0))
    center = float(scores.get("center", 0.0))
    if left >= MIN_SAFE_SCORE and left > right + DIRECTION_MARGIN and left > center:
        return "MOVE LEFT", f"{name} ahead. Safer path is on the left."
    if right >= MIN_SAFE_SCORE and right > left + DIRECTION_MARGIN and right > center:
        return "MOVE RIGHT", f"{name} ahead. Safer path is on the right."

    if danger == "high":
        return "STOP / OBJECT AHEAD", f"Stop. {name} detected in front of you."
    return "CAUTION / OBJECT AHEAD", f"Be careful. {name} ahead. Move slowly."


def command_to_text(command):
    if command in {"STOP / OBJECT AHEAD", "CAUTION / OBJECT AHEAD"}:
        return "Obstacle ahead. Follow the warning."
    if command == "MOVE FORWARD":
        return "Path ahead is clear. Move forward."
    if command == "MOVE LEFT":
        return "Safe path is on the left. Move left."
    if command == "MOVE RIGHT":
        return "Safe path is on the right. Move right."
    if command == "STOP / NO SAFE PATH":
        return "Stop. No safe path detected."
    if command == "UNRECOGNIZED PATH":
        return "Safe path is not detected. Please move back or move around."
    return "Safe path is not detected. Please move back or move around."


class NavigationPipeline:
    """
    New navigation runtime:
    - Outdoor model: best_outdoor_navigation.pt, RGB input, Cityscapes safe path.
    - Indoor model: best_indoor_rgbd_navigation.pt, RGB-D input, SUN RGB-D pseudo safe path.

    If no real depth frame is supplied, a lightweight grayscale depth-proxy is used
    so the indoor RGB-D checkpoint can still run from a normal camera.
    """
    def __init__(self, outdoor_model_path, indoor_model_path=None, device="cpu", alert_cooldown=4.0):
        self.device = device
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = 0
        self.last_command = ""
        self._in_stop_state = False
        self._score_history = deque(maxlen=SMOOTH_FRAMES)

        self.outdoor_model = MiniUNet(in_channels=3).to(device)
        out_ckpt = torch.load(outdoor_model_path, map_location=device, weights_only=False)
        self.outdoor_model.load_state_dict(out_ckpt["model_state_dict"])
        self.outdoor_model.eval()
        print(f"[Navigation] Loaded OUTDOOR Mini U-Net from {outdoor_model_path} | device={device}")

        self.indoor_model = None
        if indoor_model_path is not None:
            self.indoor_model = MiniUNet(in_channels=4).to(device)
            in_ckpt = torch.load(indoor_model_path, map_location=device, weights_only=False)
            self.indoor_model.load_state_dict(in_ckpt["model_state_dict"])
            self.indoor_model.eval()
            print(f"[Navigation] Loaded INDOOR RGB-D Mini U-Net from {indoor_model_path} | device={device}")

    def default_result(self):
        return {
            "command": "MOVE FORWARD",
            "text": "Path ahead is clear. Move forward.",
            "safe_scores": {"left": 0.0, "center": 1.0, "right": 0.0},
            "safe_mask": None,
            "mode": "outdoor",
            "priority": 5,
            "ms": 0.0,
        }

    def _rgb_tensor(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (OUT_W, OUT_H)).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))
        return x, rgb

    def _depth_proxy(self, bgr_frame):
        # Fallback only: allows the RGB-D model to run without a physical depth camera.
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (OUT_W, OUT_H)).astype(np.float32) / 255.0
        return gray

    def _preprocess_outdoor(self, bgr_frame):
        x, _ = self._rgb_tensor(bgr_frame)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _preprocess_indoor(self, bgr_frame, depth_frame=None):
        rgb_chw, _ = self._rgb_tensor(bgr_frame)
        if depth_frame is None:
            depth_norm = self._depth_proxy(bgr_frame)
        else:
            if depth_frame.ndim == 3:
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            depth_norm = cv2.resize(depth_frame, (OUT_W, OUT_H)).astype(np.float32)
            d_min, d_max = np.percentile(depth_norm, 2), np.percentile(depth_norm, 98)
            depth_norm = np.clip(depth_norm, d_min, d_max)
            depth_norm = (depth_norm - d_min) / (d_max - d_min + 1e-6)
        x = np.concatenate([rgb_chw, depth_norm[None, :, :]], axis=0)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device), depth_norm

    def _smooth_scores(self, raw_scores):
        self._score_history.append(raw_scores)
        n = len(self._score_history)
        return {
            "left": sum(s["left"] for s in self._score_history) / n,
            "center": sum(s["center"] for s in self._score_history) / n,
            "right": sum(s["right"] for s in self._score_history) / n,
        }

    def predict(self, bgr_frame, scene=None, depth_frame=None, detections=None, scene_conf=1.0):
        t0 = time.perf_counter()
        # Avoid switching to indoor mode on weak scene predictions.
        use_indoor = self.indoor_model is not None and scene in INDOOR_SCENES and float(scene_conf or 0.0) >= 0.45

        if use_indoor:
            x, depth_norm = self._preprocess_indoor(bgr_frame, depth_frame=depth_frame)
            model = self.indoor_model
            mode = "indoor_rgbd"
        else:
            x = self._preprocess_outdoor(bgr_frame)
            depth_norm = None
            model = self.outdoor_model
            mode = "outdoor"

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        safe_mask = (prob > 0.5).astype(np.float32)
        safe_scores = score_safe_regions(safe_mask)

        if use_indoor:
            close_scores = score_close_regions(depth_norm)
            raw_scores = compute_final_scores(safe_scores, close_scores)
        else:
            raw_scores = safe_scores

        smooth_scores = self._smooth_scores(raw_scores)
        command = decide_command(smooth_scores, in_stop_state=self._in_stop_state)
        command, fusion_text = apply_detection_fusion(command, smooth_scores, detections, frame_width=bgr_frame.shape[1])
        # Only confirmed danger stops (from object fusion) lock the stop-state hysteresis.
        # UNRECOGNIZED PATH does not — the user needs to be guided to move back/around,
        # not held in a hard stop that requires a higher threshold to exit.
        self._in_stop_state = "STOP" in command
        if command == "STOP / NO SAFE PATH":
            priority = 0
        elif command == "UNRECOGNIZED PATH":
            priority = 1   # Urgent but not danger-level; lets object alerts still override
        elif command != "MOVE FORWARD":
            priority = 1
        else:
            priority = 5
        ms = (time.perf_counter() - t0) * 1000

        return {
            "command": command,
            "text": fusion_text or command_to_text(command),
            "safe_scores": smooth_scores,
            "safe_mask": safe_mask,
            "mode": mode,
            "priority": priority,
            "ms": round(ms, 1),
        }

    def should_speak(self, nav_result):
        command = nav_result["command"]
        now = time.time()
        if command != self.last_command:
            self.last_command = command
            self.last_alert_time = now
            return True
        repeat_interval = self.alert_cooldown if command != "MOVE FORWARD" else 8.0
        if now - self.last_alert_time >= repeat_interval:
            self.last_alert_time = now
            return True
        return False

    def draw(self, display, nav_result):
        h, w = display.shape[:2]
        safe_mask = nav_result.get("safe_mask")
        command = nav_result["command"]
        scores = nav_result["safe_scores"]
        mode = nav_result.get("mode", "outdoor")

        if safe_mask is not None:
            mask_full = cv2.resize(safe_mask.astype(np.float32), (w, h))
            binary_u8 = (mask_full > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(display, [max(contours, key=cv2.contourArea)], -1, (0, 220, 255), 2)

        roi_y = int(h * ROI_START_RATIO)
        cv2.line(display, (w // 3, roi_y), (w // 3, h), (60, 60, 60), 1)
        cv2.line(display, (2 * w // 3, roi_y), (2 * w // 3, h), (60, 60, 60), 1)
        cv2.line(display, (0, roi_y), (w, roi_y), (80, 80, 80), 1)

        bar_w, bar_h = 6, 60
        bar_x_start = w - 80
        bar_y_base = h - 12
        for i, (lbl, score) in enumerate([("L", scores["left"]), ("C", scores["center"]), ("R", scores["right"])]):
            bx = bar_x_start + i * 24
            fill = int(max(0, min(1, score)) * bar_h)
            cv2.rectangle(display, (bx, bar_y_base - bar_h), (bx + bar_w, bar_y_base), (40, 40, 40), -1)
            bar_color = (0, 200, 80) if score >= MIN_SAFE_SCORE else (60, 60, 200)
            cv2.rectangle(display, (bx, bar_y_base - fill), (bx + bar_w, bar_y_base), bar_color, -1)
            cv2.putText(display, lbl, (bx - 1, bar_y_base + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        if "STOP" in command:
            banner_color, text_color, short_cmd = (0, 30, 180), (255, 255, 255), "STOP"
        elif command == "UNRECOGNIZED PATH":
            banner_color, text_color, short_cmd = (60, 0, 130), (220, 180, 255), "MOVE BACK"
        elif command == "MOVE FORWARD":
            banner_color, text_color, short_cmd = (20, 100, 20), (200, 255, 200), "FORWARD"
        else:
            banner_color, text_color = (120, 80, 0), (255, 220, 100)
            short_cmd = command.replace("MOVE ", "")

        cv2.rectangle(display, (w - 220, 0), (w, 44), banner_color, -1)
        cv2.putText(display, f"NAV {short_cmd}", (w - 212, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
        cv2.putText(display, mode, (w - 212, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
        return display
