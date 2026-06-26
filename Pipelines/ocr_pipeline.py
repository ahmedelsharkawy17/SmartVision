import re
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
            batch_first=False,
            dropout=0.30,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)
        out, _ = self.rnn(f)
        return self.classifier(out)


class OCRPipeline:
    MIN_TEXT_SCORE = 0.18
    MIN_CONFIDENCE = 0.28

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        self.charset = ckpt.get("charset", "abcdefghijklmnopqrstuvwxyz0123456789 ")
        self.img_h = int(ckpt.get("img_h", 32))
        self.img_w = int(ckpt.get("img_w", 224))
        self.num_classes = int(ckpt.get("num_classes", len(self.charset) + 1))
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}

        self.model = CRNN(self.num_classes).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.last_candidates = []

        print(
            f"[OCR] Loaded enhanced CRNN runtime from {model_path} | device={device} | "
            f"charset={repr(self.charset)} | img={self.img_h}x{self.img_w}"
        )

    def _preprocess(self, crop):
        if crop is None or crop.size == 0:
            crop = np.full((self.img_h, self.img_w), 128, dtype=np.uint8)

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            gray = np.full((self.img_h, self.img_w), 128, dtype=np.uint8)
            h, w = gray.shape[:2]

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)

        new_w = int((self.img_h / max(1, h)) * w)
        new_w = max(8, min(new_w, self.img_w))
        gray = cv2.resize(gray, (new_w, self.img_h), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((self.img_h, self.img_w), 128, dtype=np.uint8)
        canvas[:, :new_w] = gray[:, :new_w]

        x = canvas.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.FloatTensor(x[None, None, :, :])

    def _make_candidates(self, frame) -> List[Tuple[str, np.ndarray]]:
        if frame is None or frame.size == 0:
            return [("empty", frame)]

        h, w = frame.shape[:2]
        if h <= 180 or w <= 260:
            return [("crop", frame)]

        candidates: List[Tuple[str, np.ndarray]] = []

        def add(name: str, x1: float, y1: float, x2: float, y2: float):
            xx1, yy1 = int(w * x1), int(h * y1)
            xx2, yy2 = int(w * x2), int(h * y2)
            xx1, yy1 = max(0, xx1), max(0, yy1)
            xx2, yy2 = min(w, xx2), min(h, yy2)
            if xx2 > xx1 and yy2 > yy1:
                candidates.append((name, frame[yy1:yy2, xx1:xx2]))

        add("center_wide", 0.05, 0.34, 0.95, 0.66)
        add("upper_wide",  0.05, 0.12, 0.95, 0.42)
        add("lower_wide",  0.05, 0.58, 0.95, 0.88)
        add("middle_line", 0.02, 0.42, 0.98, 0.58)
        add("full_inner",  0.02, 0.08, 0.98, 0.92)

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            edges = cv2.Canny(gray, 60, 160)
            row_energy = edges.mean(axis=1)
            band_h = max(80, int(h * 0.25))
            if h > band_h:
                scores = np.convolve(row_energy, np.ones(band_h), mode="valid")
                y1 = int(np.argmax(scores))
                y2 = min(h, y1 + band_h)
                candidates.append(("edge_band", frame[y1:y2, int(w*0.03):int(w*0.97)]))
        except Exception:
            pass

        return candidates

    def _decode_with_confidence(self, logits):
        probs = logits.softmax(2)
        pred = probs.argmax(2)[:, 0].detach().cpu().numpy().tolist()
        confs = probs.max(2).values[:, 0].detach().cpu().numpy().tolist()

        out = []
        used_conf = []
        prev = -1
        for idx, conf in zip(pred, confs):
            if idx != prev and idx != 0:
                ch = self.idx_to_char.get(idx, "")
                if ch:
                    out.append(ch)
                    used_conf.append(float(conf))
            prev = idx

        text = "".join(out).strip()
        avg_conf = float(np.mean(used_conf)) if used_conf else 0.0
        return text, avg_conf

    @staticmethod
    def _clean_text(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(.)\1{4,}", r"\1\1", text)
        return text

    @classmethod
    def _text_quality(cls, text: str, conf: float) -> float:
        text = cls._clean_text(text)
        if not text:
            return 0.0
        chars = [c for c in text if c != " "]
        if not chars:
            return 0.0
        alnum_ratio = sum(c.isalnum() for c in chars) / max(1, len(chars))
        unique_ratio = len(set(chars)) / max(1, len(chars))
        length_bonus = min(len(chars) / 10.0, 1.0)
        space_bonus = 0.08 if " " in text and len(chars) >= 4 else 0.0
        return (0.55 * conf) + (0.20 * alnum_ratio) + (0.15 * unique_ratio) + (0.10 * length_bonus) + space_bonus

    def _predict_one(self, crop):
        x = self._preprocess(crop).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        text, conf = self._decode_with_confidence(logits)
        text = self._clean_text(text)
        score = self._text_quality(text, conf)
        return text, conf, score

    def read_candidates(self, bgr_frame):
        results = []
        for name, crop in self._make_candidates(bgr_frame):
            try:
                text, conf, score = self._predict_one(crop)
                results.append({"region": name, "text": text, "confidence": round(conf, 3), "score": round(score, 3)})
            except Exception as exc:
                results.append({"region": name, "text": "", "confidence": 0.0, "score": 0.0, "error": str(exc)})
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results

    def read(self, bgr_frame):
        t0 = time.perf_counter()
        candidates = self.read_candidates(bgr_frame)
        self.last_candidates = candidates[:5]
        best = candidates[0] if candidates else {"text": "", "confidence": 0.0, "score": 0.0}
        text = best.get("text", "")

        if best.get("score", 0.0) < self.MIN_TEXT_SCORE or best.get("confidence", 0.0) < self.MIN_CONFIDENCE:
            text = ""

        ms = (time.perf_counter() - t0) * 1000
        return text, round(ms, 1)
