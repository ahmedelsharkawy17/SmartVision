# ============================================================
# SmartVisionX OCR - FAST STRONG LOWERCASE VERSION
# Target: CER under 30% faster on Kaggle
# TextOCR Dataset - CRNN + CTC + Realistic Augmentations
# Outputs: model + curves + confusion matrix + examples
# ============================================================

import os, sys, subprocess, json, random, time, math, string, re
from pathlib import Path

for pkg in ["editdistance", "pyarrow", "scikit-learn"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import editdistance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# =========================
# CONFIG
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

DATA_DIR = Path("/kaggle/input/datasets/robikscube/textocr-text-extraction-from-images-dataset")
IMG_PARQ = DATA_DIR / "img.parquet"
ANNOT_PARQ = DATA_DIR / "annot.parquet"
IMG_FOLDER = DATA_DIR / "train_val_images" / "train_images"

OUT_DIR = Path("/kaggle/working/smartvisionx_ocr_lowercase_strong")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = OUT_DIR / "best_smartvisionx_crnn_lowercase.pt"
LAST_MODEL = OUT_DIR / "last_smartvisionx_crnn_lowercase.pt"
RESULTS_CSV = OUT_DIR / "ocr_val_predictions.csv"

IMG_H = 32
IMG_W = 224

MAX_SAMPLES = 200_000
VAL_SPLIT = 0.08
BATCH_SIZE = 128
EPOCHS = 14

LR = 2e-4
MIN_LR = 1e-5
WARMUP_EPOCHS = 2
NUM_WORKERS = 4
GRAD_CLIP = 5.0

MIN_LABEL = 2
MAX_LABEL = 24

# مهم: lowercase + digits + space فقط
CHARSET = string.ascii_lowercase + string.digits + " "
BLANK = 0

char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1

print("Charset:", repr(CHARSET))
print("Charset length:", len(CHARSET))
print("Num classes:", NUM_CLASSES)
print("Output dir:", OUT_DIR)

# =========================
# LABEL CLEANING
# =========================

def clean_label(text):
    text = str(text).strip().lower()
    text = "".join(c if c in CHARSET else " " for c in text)
    text = " ".join(text.split())
    return text

# =========================
# DATA LOADING
# =========================

print("Reading parquet files...")
img_df = pd.read_parquet(IMG_PARQ)
annot_df = pd.read_parquet(ANNOT_PARQ)

id_to_filename = {str(r["id"]): str(r["file_name"]) for _, r in img_df.iterrows()}

def get_img_path(img_id):
    fname = id_to_filename.get(str(img_id))
    if fname is None:
        return None
    p1 = IMG_FOLDER / fname
    if p1.exists():
        return p1
    p2 = IMG_FOLDER / Path(fname).name
    if p2.exists():
        return p2
    return None

samples = []
skipped = 0

for _, row in tqdm(annot_df.iterrows(), total=len(annot_df), desc="Building samples"):
    label = clean_label(row.get("utf8_string", ""))

    if len(label) < MIN_LABEL or len(label) > MAX_LABEL:
        skipped += 1
        continue

    if not all(c in char_to_idx for c in label):
        skipped += 1
        continue

    img_path = get_img_path(row["image_id"])
    if img_path is None:
        skipped += 1
        continue

    try:
        raw_bbox = row["bbox"]
        if isinstance(raw_bbox, str):
            raw_bbox = json.loads(raw_bbox)
        bbox = [float(v) for v in raw_bbox]
        if len(bbox) != 4:
            skipped += 1
            continue
    except:
        skipped += 1
        continue

    samples.append((str(img_path), bbox, label))

    if len(samples) >= MAX_SAMPLES:
        break

random.shuffle(samples)

n_val = int(len(samples) * VAL_SPLIT)
val_samples = samples[:n_val]
train_samples = samples[n_val:]

print(f"Total samples: {len(samples):,}")
print(f"Train: {len(train_samples):,}")
print(f"Val: {len(val_samples):,}")
print(f"Skipped: {skipped:,}")

# =========================
# AUGMENTATION FOR LIVE VIDEO
# =========================

def motion_blur(img, p=0.25):
    if random.random() > p:
        return img
    k = random.choice([3, 5])
    kernel = np.zeros((k, k), np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1 / k
    else:
        kernel[:, k // 2] = 1 / k
    return cv2.filter2D(img, -1, kernel)

def perspective_aug(img, p=0.25):
    if random.random() > p:
        return img
    h, w = img.shape[:2]
    margin = max(1, int(0.06 * min(h, w)))
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst = src + np.random.uniform(-margin, margin, src.shape).astype(np.float32)
    dst[:,0] = np.clip(dst[:,0], 0, w-1)
    dst[:,1] = np.clip(dst[:,1], 0, h-1)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=128)

def brightness_aug(img, p=0.45):
    if random.random() > p:
        return img
    alpha = random.uniform(0.55, 1.55)
    beta = random.uniform(-20, 20)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

def noise_aug(img, p=0.25):
    if random.random() > p:
        return img
    noise = np.random.normal(0, random.uniform(3, 18), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def jpeg_aug(img, p=0.25):
    if random.random() > p:
        return img
    q = random.randint(35, 90)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def erase_aug(img, p=0.15):
    if random.random() > p:
        return img
    h, w = img.shape
    eh = random.randint(1, max(1, h // 4))
    ew = random.randint(1, max(1, w // 5))
    y = random.randint(0, max(0, h - eh))
    x = random.randint(0, max(0, w - ew))
    out = img.copy()
    out[y:y+eh, x:x+ew] = random.randint(0, 255)
    return out

def stroke_aug(img, p=0.15):
    if random.random() > p:
        return img
    k = np.ones((2, 2), np.uint8)
    return cv2.dilate(img, k, 1) if random.random() < 0.5 else cv2.erode(img, k, 1)

def apply_aug(gray):
    gray = brightness_aug(gray)
    gray = perspective_aug(gray)
    gray = motion_blur(gray)
    gray = noise_aug(gray)
    gray = jpeg_aug(gray)
    gray = stroke_aug(gray)
    gray = erase_aug(gray)
    return gray

# =========================
# PREPROCESS
# =========================

def preprocess_image(crop, augment=False):
    if crop is None or crop.size == 0:
        crop = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        gray = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
        h, w = gray.shape

    new_w = int((IMG_H / h) * w)
    new_w = max(8, min(new_w, IMG_W))

    gray = cv2.resize(gray, (new_w, IMG_H), interpolation=cv2.INTER_LINEAR)

    if augment:
        gray = apply_aug(gray)

    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)

    canvas = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    canvas[:, :new_w] = gray[:, :new_w]

    x = canvas.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5

    return x[np.newaxis, :, :]

# =========================
# DATASET
# =========================

class TextOCRDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("missing image")

            x, y, w, h = [int(v) for v in bbox]
            pad = 5

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img

            tensor = preprocess_image(crop, augment=self.augment)

        except:
            tensor = np.zeros((1, IMG_H, IMG_W), dtype=np.float32)
            label = "text"

        encoded = [char_to_idx[c] for c in label]

        return torch.FloatTensor(tensor), torch.IntTensor(encoded), len(encoded), label

def collate_fn(batch):
    imgs, labels, lengths, texts = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.IntTensor(lengths), list(texts)

train_ds = TextOCRDataset(train_samples, augment=True)
val_ds = TextOCRDataset(val_samples, augment=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False
)

# =========================
# MODEL
# =========================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
            batch_first=False,
            dropout=0.30
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x)
        f = f.squeeze(2)
        f = f.permute(2, 0, 1)
        out, _ = self.rnn(f)
        logits = self.classifier(out)
        return logits

model = CRNN(NUM_CLASSES).to(DEVICE)
print("Params:", round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "M")

# =========================
# DECODE + METRICS
# =========================

def greedy_decode(logits):
    preds = logits.softmax(2).argmax(2)
    results = []

    for b in range(preds.shape[1]):
        seq = preds[:, b].detach().cpu().numpy().tolist()
        decoded = []
        prev = -1

        for idx in seq:
            if idx != prev and idx != BLANK:
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx

        results.append("".join(decoded))

    return results

def unpack_targets(targets, lengths):
    out = []
    offset = 0

    for l in lengths.tolist():
        chars = targets[offset:offset+l].tolist()
        out.append("".join(idx_to_char.get(i, "") for i in chars))
        offset += l

    return out

def cer_score(preds, gts):
    total_dist, total_chars = 0, 0

    for p, g in zip(preds, gts):
        total_dist += editdistance.eval(p, g)
        total_chars += max(1, len(g))

    return total_dist / total_chars

def wer_score(preds, gts):
    total_dist, total_words = 0, 0

    for p, g in zip(preds, gts):
        pw = p.split()
        gw = g.split()

        total_dist += editdistance.eval(pw, gw)
        total_words += max(1, len(gw))

    return total_dist / total_words

def exact_match(preds, gts):
    return np.mean([p == g for p, g in zip(preds, gts)])

# =========================
# SCHEDULER
# =========================

class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            lr = self.max_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        return lr

criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = WarmupCosine(optimizer, WARMUP_EPOCHS, EPOCHS, LR, MIN_LR)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

history = {
    "epoch": [],
    "lr": [],
    "train_loss": [],
    "val_loss": [],
    "cer": [],
    "wer": [],
    "exact": []
}

best_cer = float("inf")

# =========================
# TRAIN / VALIDATE
# =========================

def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, targets, lengths, texts in tqdm(train_loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE)
        targets_cpu = targets.to(torch.int32)
        lengths_cpu = lengths.to(torch.int32)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            T, B, C = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.int32)

            loss = criterion(
                logits.log_softmax(2).cpu(),
                targets_cpu.cpu(),
                input_lengths.cpu(),
                lengths_cpu.cpu()
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def validate(save_predictions=False):
    model.eval()
    total_loss = 0
    all_preds, all_gts = [], []

    for imgs, targets, lengths, texts in tqdm(val_loader, desc="Val", leave=False):
        imgs = imgs.to(DEVICE)
        targets_cpu = targets.to(torch.int32)
        lengths_cpu = lengths.to(torch.int32)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            T, B, C = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.int32)

            loss = criterion(
                logits.log_softmax(2).cpu(),
                targets_cpu.cpu(),
                input_lengths.cpu(),
                lengths_cpu.cpu()
            )

        preds = greedy_decode(logits.float().cpu())
        gts = unpack_targets(targets.cpu(), lengths.cpu())

        all_preds.extend(preds)
        all_gts.extend(gts)
        total_loss += loss.item()

    val_loss = total_loss / len(val_loader)
    cer = cer_score(all_preds, all_gts)
    wer = wer_score(all_preds, all_gts)
    ex = exact_match(all_preds, all_gts)

    if save_predictions:
        df = pd.DataFrame({
            "ground_truth": all_gts,
            "prediction": all_preds,
            "correct": [p == g for p, g in zip(all_preds, all_gts)],
            "gt_len": [len(g) for g in all_gts],
            "pred_len": [len(p) for p in all_preds],
            "char_edit_distance": [editdistance.eval(p, g) for p, g in zip(all_preds, all_gts)]
        })
        df.to_csv(RESULTS_CSV, index=False)
        print("Saved predictions:", RESULTS_CSV)

    return val_loss, cer, wer, ex, all_preds, all_gts

# =========================
# TRAINING
# =========================

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    lr = scheduler.step(epoch)
    train_loss = train_one_epoch()
    val_loss, cer, wer, ex, _, _ = validate(save_predictions=False)

    history["epoch"].append(epoch)
    history["lr"].append(lr)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["cer"].append(cer)
    history["wer"].append(wer)
    history["exact"].append(ex)

    is_best = cer < best_cer

    if is_best:
        best_cer = cer
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "charset": CHARSET,
            "num_classes": NUM_CLASSES,
            "img_h": IMG_H,
            "img_w": IMG_W,
            "best_cer": best_cer,
            "history": history
        }, BEST_MODEL)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "charset": CHARSET,
        "num_classes": NUM_CLASSES,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "best_cer": best_cer,
        "history": history
    }, LAST_MODEL)

    elapsed = (time.time() - t0) / 60

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"lr={lr:.2e} | "
        f"train={train_loss:.4f} | val={val_loss:.4f} | "
        f"CER={cer*100:.2f}% | WER={wer*100:.2f}% | Exact={ex*100:.2f}% | "
        f"{elapsed:.1f} min"
        + ("  <-- BEST" if is_best else "")
    )

# =========================
# FINAL EVAL
# =========================

print("\nLoading best model...")
ckpt = torch.load(BEST_MODEL, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

val_loss, cer, wer, ex, all_preds, all_gts = validate(save_predictions=True)

print("\nFINAL RESULTS")
print("=" * 50)
print(f"Best CER   : {cer*100:.2f}%")
print(f"WER        : {wer*100:.2f}%")
print(f"Exact Match: {ex*100:.2f}%")
print("=" * 50)

# =========================
# SAVE HISTORY
# =========================

hist = pd.DataFrame(history)
hist.to_csv(OUT_DIR / "training_history.csv", index=False)

# =========================
# PLOTS
# =========================

plt.figure(figsize=(8, 5))
plt.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train Loss")
plt.plot(hist["epoch"], hist["val_loss"], marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("CTC Loss")
plt.title("OCR Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_1_loss_curves.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist["epoch"], hist["cer"] * 100, marker="o", label="CER (%)")
plt.plot(hist["epoch"], hist["wer"] * 100, marker="o", label="WER (%)")
plt.xlabel("Epoch")
plt.ylabel("Error Rate (%)")
plt.title("OCR CER and WER Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_2_cer_wer.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist["epoch"], hist["exact"] * 100, marker="o", label="Exact Match (%)")
plt.xlabel("Epoch")
plt.ylabel("Exact Match (%)")
plt.title("OCR Exact Match Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_3_exact_match.png", dpi=200)
plt.show()

# =========================
# CHARACTER CONFUSION MATRIX
# =========================

def align_chars(gt, pred):
    n = max(len(gt), len(pred))
    return list(gt.ljust(n, "∅")), list(pred.ljust(n, "∅"))

true_chars, pred_chars = [], []

for g, p in zip(all_gts[:5000], all_preds[:5000]):
    t, r = align_chars(g, p)
    true_chars.extend(t)
    pred_chars.extend(r)

labels = list(CHARSET) + ["∅"]
cm = confusion_matrix(true_chars, pred_chars, labels=labels)

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest")
plt.title("Character-Level Confusion Matrix")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.xlabel("Predicted Character")
plt.ylabel("Ground Truth Character")
plt.colorbar()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_4_char_confusion_matrix.png", dpi=200)
plt.show()

# =========================
# PREDICTION EXAMPLES GRID
# =========================

@torch.no_grad()
def predict_single_from_sample(sample):
    img_path, bbox, label = sample
    img = cv2.imread(img_path)

    x, y, w, h = [int(v) for v in bbox]
    pad = 5

    crop = img[
        max(0, y-pad):min(img.shape[0], y+h+pad),
        max(0, x-pad):min(img.shape[1], x+w+pad)
    ]

    tensor = preprocess_image(crop, augment=False)
    inp = torch.FloatTensor(tensor).unsqueeze(0).to(DEVICE)

    logits = model(inp)
    pred = greedy_decode(logits.cpu())[0]

    if crop is not None and crop.size:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = np.zeros((32, 224, 3), dtype=np.uint8)

    return crop_rgb, label, pred

example_samples = random.sample(val_samples, min(16, len(val_samples)))

plt.figure(figsize=(14, 10))

for i, s in enumerate(example_samples):
    crop, gt, pred = predict_single_from_sample(s)

    plt.subplot(4, 4, i + 1)
    plt.imshow(crop)
    plt.axis("off")

    status = "OK" if gt == pred else "WRONG"
    plt.title(f"{status}\nGT: {gt}\nPR: {pred}", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "plot_5_prediction_examples.png", dpi=200)
plt.show()

# =========================
# SAVE LIVE VIDEO INFERENCE PIPELINE
# =========================

pipeline_code = r'''
import cv2, time
import numpy as np
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
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
            512, 512,
            num_layers=3,
            bidirectional=True,
            batch_first=False,
            dropout=0.30
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)
        out, _ = self.rnn(f)
        return self.classifier(out)

class SmartVisionXOCR:
    def __init__(self, model_path, device="cpu"):
        self.device = device

        ckpt = torch.load(model_path, map_location=device)
        self.charset = ckpt["charset"]
        self.img_h = ckpt["img_h"]
        self.img_w = ckpt["img_w"]
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}

        self.model = CRNN(ckpt["num_classes"]).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def preprocess(self, crop):
        if crop is None or crop.size == 0:
            crop = np.full((self.img_h, self.img_w), 128, dtype=np.uint8)

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]
        new_w = int((self.img_h / max(1, h)) * w)
        new_w = max(8, min(new_w, self.img_w))

        gray = cv2.resize(gray, (new_w, self.img_h))
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)

        canvas = np.full((self.img_h, self.img_w), 128, dtype=np.uint8)
        canvas[:, :new_w] = gray[:, :new_w]

        x = canvas.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5

        return torch.FloatTensor(x[None, None, :, :])

    def decode(self, logits):
        preds = logits.softmax(2).argmax(2)
        seq = preds[:, 0].detach().cpu().numpy().tolist()

        out = []
        prev = -1

        for idx in seq:
            if idx != prev and idx != 0:
                out.append(self.idx_to_char.get(idx, ""))
            prev = idx

        return "".join(out)

    def read(self, crop):
        t0 = time.perf_counter()

        x = self.preprocess(crop).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        text = self.decode(logits)
        ms = (time.perf_counter() - t0) * 1000

        return text, round(ms, 1)

    def detect_text_regions_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grad = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8)
        )

        _, bw = cv2.threshold(
            grad,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        H, W = frame.shape[:2]
        boxes = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if w < 25 or h < 8:
                continue

            if w / max(1, h) < 1.2:
                continue

            area = w * h

            if area < 250 or area > 0.45 * W * H:
                continue

            boxes.append((x, y, x + w, y + h))

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

        return boxes

    def read_frame(self, frame, max_regions=5):
        boxes = self.detect_text_regions_opencv(frame)[:max_regions]
        results = []

        for x1, y1, x2, y2 in boxes:
            pad = 4

            crop = frame[
                max(0, y1-pad):min(frame.shape[0], y2+pad),
                max(0, x1-pad):min(frame.shape[1], x2+pad)
            ]

            text, ms = self.read(crop)

            if text.strip():
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "ms": ms
                })

        return results
'''

with open(OUT_DIR / "smartvisionx_ocr_pipeline_lowercase.py", "w", encoding="utf-8") as f:
    f.write(pipeline_code)

print("\nSaved files:")
for p in OUT_DIR.iterdir():
    print(" -", p)

print("\nDONE.")
Device: cuda
Charset: 'abcdefghijklmnopqrstuvwxyz0123456789 '
Charset length: 37
Num classes: 38
Output dir: /kaggle/working/smartvisionx_ocr_lowercase_strong
Reading parquet files...
Total samples: 200,000
Train: 184,000
Val: 16,000
Skipped: 146,458
Params: 22.39 M






# ============================================================
# SmartVisionX OCR - FAST STRONG LOWERCASE VERSION
# Target: CER under 30% faster on Kaggle
# TextOCR Dataset - CRNN + CTC + Realistic Augmentations
# Outputs: model + curves + confusion matrix + examples
# ============================================================

import os, sys, subprocess, json, random, time, math, string, re
from pathlib import Path

for pkg in ["editdistance", "pyarrow", "scikit-learn"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import editdistance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# =========================
# CONFIG
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

DATA_DIR = Path("/kaggle/input/datasets/robikscube/textocr-text-extraction-from-images-dataset")
IMG_PARQ = DATA_DIR / "img.parquet"
ANNOT_PARQ = DATA_DIR / "annot.parquet"
IMG_FOLDER = DATA_DIR / "train_val_images" / "train_images"

OUT_DIR = Path("/kaggle/working/smartvisionx_ocr_lowercase_strong")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = OUT_DIR / "best_smartvisionx_crnn_lowercase.pt"
LAST_MODEL = OUT_DIR / "last_smartvisionx_crnn_lowercase.pt"
RESULTS_CSV = OUT_DIR / "ocr_val_predictions.csv"

IMG_H = 32
IMG_W = 224

MAX_SAMPLES = 200_000
VAL_SPLIT = 0.08
BATCH_SIZE = 128
EPOCHS = 14

LR = 2e-4
MIN_LR = 1e-5
WARMUP_EPOCHS = 2
NUM_WORKERS = 4
GRAD_CLIP = 5.0

MIN_LABEL = 2
MAX_LABEL = 24

# مهم: lowercase + digits + space فقط
CHARSET = string.ascii_lowercase + string.digits + " "
BLANK = 0

char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1

print("Charset:", repr(CHARSET))
print("Charset length:", len(CHARSET))
print("Num classes:", NUM_CLASSES)
print("Output dir:", OUT_DIR)

# =========================
# LABEL CLEANING
# =========================

def clean_label(text):
    text = str(text).strip().lower()
    text = "".join(c if c in CHARSET else " " for c in text)
    text = " ".join(text.split())
    return text

# =========================
# DATA LOADING
# =========================

print("Reading parquet files...")
img_df = pd.read_parquet(IMG_PARQ)
annot_df = pd.read_parquet(ANNOT_PARQ)

id_to_filename = {str(r["id"]): str(r["file_name"]) for _, r in img_df.iterrows()}

def get_img_path(img_id):
    fname = id_to_filename.get(str(img_id))
    if fname is None:
        return None
    p1 = IMG_FOLDER / fname
    if p1.exists():
        return p1
    p2 = IMG_FOLDER / Path(fname).name
    if p2.exists():
        return p2
    return None

samples = []
skipped = 0

for _, row in tqdm(annot_df.iterrows(), total=len(annot_df), desc="Building samples"):
    label = clean_label(row.get("utf8_string", ""))

    if len(label) < MIN_LABEL or len(label) > MAX_LABEL:
        skipped += 1
        continue

    if not all(c in char_to_idx for c in label):
        skipped += 1
        continue

    img_path = get_img_path(row["image_id"])
    if img_path is None:
        skipped += 1
        continue

    try:
        raw_bbox = row["bbox"]
        if isinstance(raw_bbox, str):
            raw_bbox = json.loads(raw_bbox)
        bbox = [float(v) for v in raw_bbox]
        if len(bbox) != 4:
            skipped += 1
            continue
    except:
        skipped += 1
        continue

    samples.append((str(img_path), bbox, label))

    if len(samples) >= MAX_SAMPLES:
        break

random.shuffle(samples)

n_val = int(len(samples) * VAL_SPLIT)
val_samples = samples[:n_val]
train_samples = samples[n_val:]

print(f"Total samples: {len(samples):,}")
print(f"Train: {len(train_samples):,}")
print(f"Val: {len(val_samples):,}")
print(f"Skipped: {skipped:,}")

# =========================
# AUGMENTATION FOR LIVE VIDEO
# =========================

def motion_blur(img, p=0.25):
    if random.random() > p:
        return img
    k = random.choice([3, 5])
    kernel = np.zeros((k, k), np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1 / k
    else:
        kernel[:, k // 2] = 1 / k
    return cv2.filter2D(img, -1, kernel)

def perspective_aug(img, p=0.25):
    if random.random() > p:
        return img
    h, w = img.shape[:2]
    margin = max(1, int(0.06 * min(h, w)))
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst = src + np.random.uniform(-margin, margin, src.shape).astype(np.float32)
    dst[:,0] = np.clip(dst[:,0], 0, w-1)
    dst[:,1] = np.clip(dst[:,1], 0, h-1)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=128)

def brightness_aug(img, p=0.45):
    if random.random() > p:
        return img
    alpha = random.uniform(0.55, 1.55)
    beta = random.uniform(-20, 20)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

def noise_aug(img, p=0.25):
    if random.random() > p:
        return img
    noise = np.random.normal(0, random.uniform(3, 18), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def jpeg_aug(img, p=0.25):
    if random.random() > p:
        return img
    q = random.randint(35, 90)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def erase_aug(img, p=0.15):
    if random.random() > p:
        return img
    h, w = img.shape
    eh = random.randint(1, max(1, h // 4))
    ew = random.randint(1, max(1, w // 5))
    y = random.randint(0, max(0, h - eh))
    x = random.randint(0, max(0, w - ew))
    out = img.copy()
    out[y:y+eh, x:x+ew] = random.randint(0, 255)
    return out

def stroke_aug(img, p=0.15):
    if random.random() > p:
        return img
    k = np.ones((2, 2), np.uint8)
    return cv2.dilate(img, k, 1) if random.random() < 0.5 else cv2.erode(img, k, 1)

def apply_aug(gray):
    gray = brightness_aug(gray)
    gray = perspective_aug(gray)
    gray = motion_blur(gray)
    gray = noise_aug(gray)
    gray = jpeg_aug(gray)
    gray = stroke_aug(gray)
    gray = erase_aug(gray)
    return gray

# =========================
# PREPROCESS
# =========================

def preprocess_image(crop, augment=False):
    if crop is None or crop.size == 0:
        crop = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        gray = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
        h, w = gray.shape

    new_w = int((IMG_H / h) * w)
    new_w = max(8, min(new_w, IMG_W))

    gray = cv2.resize(gray, (new_w, IMG_H), interpolation=cv2.INTER_LINEAR)

    if augment:
        gray = apply_aug(gray)

    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)

    canvas = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    canvas[:, :new_w] = gray[:, :new_w]

    x = canvas.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5

    return x[np.newaxis, :, :]

# =========================
# DATASET
# =========================

class TextOCRDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("missing image")

            x, y, w, h = [int(v) for v in bbox]
            pad = 5

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img

            tensor = preprocess_image(crop, augment=self.augment)

        except:
            tensor = np.zeros((1, IMG_H, IMG_W), dtype=np.float32)
            label = "text"

        encoded = [char_to_idx[c] for c in label]

        return torch.FloatTensor(tensor), torch.IntTensor(encoded), len(encoded), label

def collate_fn(batch):
    imgs, labels, lengths, texts = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.IntTensor(lengths), list(texts)

train_ds = TextOCRDataset(train_samples, augment=True)
val_ds = TextOCRDataset(val_samples, augment=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False
)

# =========================
# MODEL
# =========================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
            batch_first=False,
            dropout=0.30
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x)
        f = f.squeeze(2)
        f = f.permute(2, 0, 1)
        out, _ = self.rnn(f)
        logits = self.classifier(out)
        return logits

model = CRNN(NUM_CLASSES).to(DEVICE)
print("Params:", round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "M")

# =========================
# DECODE + METRICS
# =========================

def greedy_decode(logits):
    preds = logits.softmax(2).argmax(2)
    results = []

    for b in range(preds.shape[1]):
        seq = preds[:, b].detach().cpu().numpy().tolist()
        decoded = []
        prev = -1

        for idx in seq:
            if idx != prev and idx != BLANK:
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx

        results.append("".join(decoded))

    return results

def unpack_targets(targets, lengths):
    out = []
    offset = 0

    for l in lengths.tolist():
        chars = targets[offset:offset+l].tolist()
        out.append("".join(idx_to_char.get(i, "") for i in chars))
        offset += l

    return out

def cer_score(preds, gts):
    total_dist, total_chars = 0, 0

    for p, g in zip(preds, gts):
        total_dist += editdistance.eval(p, g)
        total_chars += max(1, len(g))

    return total_dist / total_chars

def wer_score(preds, gts):
    total_dist, total_words = 0, 0

    for p, g in zip(preds, gts):
        pw = p.split()
        gw = g.split()

        total_dist += editdistance.eval(pw, gw)
        total_words += max(1, len(gw))

    return total_dist / total_words

def exact_match(preds, gts):
    return np.mean([p == g for p, g in zip(preds, gts)])

# =========================
# SCHEDULER
# =========================

class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            lr = self.max_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        return lr

criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = WarmupCosine(optimizer, WARMUP_EPOCHS, EPOCHS, LR, MIN_LR)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

history = {
    "epoch": [],
    "lr": [],
    "train_loss": [],
    "val_loss": [],
    "cer": [],
    "wer": [],
    "exact": []
}

best_cer = float("inf")

# =========================
# TRAIN / VALIDATE
# =========================

def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, targets, lengths, texts in tqdm(train_loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE)
        targets_cpu = targets.to(torch.int32)
        lengths_cpu = lengths.to(torch.int32)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            T, B, C = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.int32)

            loss = criterion(
                logits.log_softmax(2).cpu(),
                targets_cpu.cpu(),
                input_lengths.cpu(),
                lengths_cpu.cpu()
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def validate(save_predictions=False):
    model.eval()
    total_loss = 0
    all_preds, all_gts = [], []

    for imgs, targets, lengths, texts in tqdm(val_loader, desc="Val", leave=False):
        imgs = imgs.to(DEVICE)
        targets_cpu = targets.to(torch.int32)
        lengths_cpu = lengths.to(torch.int32)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            T, B, C = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.int32)

            loss = criterion(
                logits.log_softmax(2).cpu(),
                targets_cpu.cpu(),
                input_lengths.cpu(),
                lengths_cpu.cpu()
            )

        preds = greedy_decode(logits.float().cpu())
        gts = unpack_targets(targets.cpu(), lengths.cpu())

        all_preds.extend(preds)
        all_gts.extend(gts)
        total_loss += loss.item()

    val_loss = total_loss / len(val_loader)
    cer = cer_score(all_preds, all_gts)
    wer = wer_score(all_preds, all_gts)
    ex = exact_match(all_preds, all_gts)

    if save_predictions:
        df = pd.DataFrame({
            "ground_truth": all_gts,
            "prediction": all_preds,
            "correct": [p == g for p, g in zip(all_preds, all_gts)],
            "gt_len": [len(g) for g in all_gts],
            "pred_len": [len(p) for p in all_preds],
            "char_edit_distance": [editdistance.eval(p, g) for p, g in zip(all_preds, all_gts)]
        })
        df.to_csv(RESULTS_CSV, index=False)
        print("Saved predictions:", RESULTS_CSV)

    return val_loss, cer, wer, ex, all_preds, all_gts

# =========================
# TRAINING
# =========================

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    lr = scheduler.step(epoch)
    train_loss = train_one_epoch()
    val_loss, cer, wer, ex, _, _ = validate(save_predictions=False)

    history["epoch"].append(epoch)
    history["lr"].append(lr)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["cer"].append(cer)
    history["wer"].append(wer)
    history["exact"].append(ex)

    is_best = cer < best_cer

    if is_best:
        best_cer = cer
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "charset": CHARSET,
            "num_classes": NUM_CLASSES,
            "img_h": IMG_H,
            "img_w": IMG_W,
            "best_cer": best_cer,
            "history": history
        }, BEST_MODEL)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "charset": CHARSET,
        "num_classes": NUM_CLASSES,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "best_cer": best_cer,
        "history": history
    }, LAST_MODEL)

    elapsed = (time.time() - t0) / 60

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"lr={lr:.2e} | "
        f"train={train_loss:.4f} | val={val_loss:.4f} | "
        f"CER={cer*100:.2f}% | WER={wer*100:.2f}% | Exact={ex*100:.2f}% | "
        f"{elapsed:.1f} min"
        + ("  <-- BEST" if is_best else "")
    )

# =========================
# FINAL EVAL
# =========================

print("\nLoading best model...")
ckpt = torch.load(BEST_MODEL, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

val_loss, cer, wer, ex, all_preds, all_gts = validate(save_predictions=True)

print("\nFINAL RESULTS")
print("=" * 50)
print(f"Best CER   : {cer*100:.2f}%")
print(f"WER        : {wer*100:.2f}%")
print(f"Exact Match: {ex*100:.2f}%")
print("=" * 50)

# =========================
# SAVE HISTORY
# =========================

hist = pd.DataFrame(history)
hist.to_csv(OUT_DIR / "training_history.csv", index=False)

# =========================
# PLOTS
# =========================

plt.figure(figsize=(8, 5))
plt.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train Loss")
plt.plot(hist["epoch"], hist["val_loss"], marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("CTC Loss")
plt.title("OCR Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_1_loss_curves.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist["epoch"], hist["cer"] * 100, marker="o", label="CER (%)")
plt.plot(hist["epoch"], hist["wer"] * 100, marker="o", label="WER (%)")
plt.xlabel("Epoch")
plt.ylabel("Error Rate (%)")
plt.title("OCR CER and WER Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_2_cer_wer.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist["epoch"], hist["exact"] * 100, marker="o", label="Exact Match (%)")
plt.xlabel("Epoch")
plt.ylabel("Exact Match (%)")
plt.title("OCR Exact Match Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_3_exact_match.png", dpi=200)
plt.show()

# =========================
# CHARACTER CONFUSION MATRIX
# =========================

def align_chars(gt, pred):
    n = max(len(gt), len(pred))
    return list(gt.ljust(n, "∅")), list(pred.ljust(n, "∅"))

true_chars, pred_chars = [], []

for g, p in zip(all_gts[:5000], all_preds[:5000]):
    t, r = align_chars(g, p)
    true_chars.extend(t)
    pred_chars.extend(r)

labels = list(CHARSET) + ["∅"]
cm = confusion_matrix(true_chars, pred_chars, labels=labels)

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest")
plt.title("Character-Level Confusion Matrix")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.xlabel("Predicted Character")
plt.ylabel("Ground Truth Character")
plt.colorbar()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_4_char_confusion_matrix.png", dpi=200)
plt.show()

# =========================
# PREDICTION EXAMPLES GRID
# =========================

@torch.no_grad()
def predict_single_from_sample(sample):
    img_path, bbox, label = sample
    img = cv2.imread(img_path)

    x, y, w, h = [int(v) for v in bbox]
    pad = 5

    crop = img[
        max(0, y-pad):min(img.shape[0], y+h+pad),
        max(0, x-pad):min(img.shape[1], x+w+pad)
    ]

    tensor = preprocess_image(crop, augment=False)
    inp = torch.FloatTensor(tensor).unsqueeze(0).to(DEVICE)

    logits = model(inp)
    pred = greedy_decode(logits.cpu())[0]

    if crop is not None and crop.size:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = np.zeros((32, 224, 3), dtype=np.uint8)

    return crop_rgb, label, pred

example_samples = random.sample(val_samples, min(16, len(val_samples)))

plt.figure(figsize=(14, 10))

for i, s in enumerate(example_samples):
    crop, gt, pred = predict_single_from_sample(s)

    plt.subplot(4, 4, i + 1)
    plt.imshow(crop)
    plt.axis("off")

    status = "OK" if gt == pred else "WRONG"
    plt.title(f"{status}\nGT: {gt}\nPR: {pred}", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "plot_5_prediction_examples.png", dpi=200)
plt.show()

# =========================
# SAVE LIVE VIDEO INFERENCE PIPELINE
# =========================

pipeline_code = r'''
import cv2, time
import numpy as np
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
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
            512, 512,
            num_layers=3,
            bidirectional=True,
            batch_first=False,
            dropout=0.30
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)
        out, _ = self.rnn(f)
        return self.classifier(out)

class SmartVisionXOCR:
    def __init__(self, model_path, device="cpu"):
        self.device = device

        ckpt = torch.load(model_path, map_location=device)
        self.charset = ckpt["charset"]
        self.img_h = ckpt["img_h"]
        self.img_w = ckpt["img_w"]
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}

        self.model = CRNN(ckpt["num_classes"]).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def preprocess(self, crop):
        if crop is None or crop.size == 0:
            crop = np.full((self.img_h, self.img_w), 128, dtype=np.uint8)

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]
        new_w = int((self.img_h / max(1, h)) * w)
        new_w = max(8, min(new_w, self.img_w))

        gray = cv2.resize(gray, (new_w, self.img_h))
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)

        canvas = np.full((self.img_h, self.img_w), 128, dtype=np.uint8)
        canvas[:, :new_w] = gray[:, :new_w]

        x = canvas.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5

        return torch.FloatTensor(x[None, None, :, :])

    def decode(self, logits):
        preds = logits.softmax(2).argmax(2)
        seq = preds[:, 0].detach().cpu().numpy().tolist()

        out = []
        prev = -1

        for idx in seq:
            if idx != prev and idx != 0:
                out.append(self.idx_to_char.get(idx, ""))
            prev = idx

        return "".join(out)

    def read(self, crop):
        t0 = time.perf_counter()

        x = self.preprocess(crop).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        text = self.decode(logits)
        ms = (time.perf_counter() - t0) * 1000

        return text, round(ms, 1)

    def detect_text_regions_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grad = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8)
        )

        _, bw = cv2.threshold(
            grad,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        H, W = frame.shape[:2]
        boxes = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if w < 25 or h < 8:
                continue

            if w / max(1, h) < 1.2:
                continue

            area = w * h

            if area < 250 or area > 0.45 * W * H:
                continue

            boxes.append((x, y, x + w, y + h))

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

        return boxes

    def read_frame(self, frame, max_regions=5):
        boxes = self.detect_text_regions_opencv(frame)[:max_regions]
        results = []

        for x1, y1, x2, y2 in boxes:
            pad = 4

            crop = frame[
                max(0, y1-pad):min(frame.shape[0], y2+pad),
                max(0, x1-pad):min(frame.shape[1], x2+pad)
            ]

            text, ms = self.read(crop)

            if text.strip():
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "ms": ms
                })

        return results
'''

with open(OUT_DIR / "smartvisionx_ocr_pipeline_lowercase.py", "w", encoding="utf-8") as f:
    f.write(pipeline_code)

print("\nSaved files:")
for p in OUT_DIR.iterdir():
    print(" -", p)

print("\nDONE.")
Device: cuda
Charset: 'abcdefghijklmnopqrstuvwxyz0123456789 '
Charset length: 37
Num classes: 38
Output dir: /kaggle/working/smartvisionx_ocr_lowercase_strong
Reading parquet files...
Total samples: 200,000
Train: 184,000
Val: 16,000
Skipped: 146,458
Params: 22.39 M



