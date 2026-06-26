# ============================================================
# FULL NAVIGATION MODULE FOR KAGGLE (ONE CELL)
# Outdoor Mini U-Net + Indoor RGB-D Mini U-Net + Fusion + Paper Figures
# ============================================================

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Outdoor Cityscapes dataset
CITY_BASE = "/kaggle/input/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data"

# Indoor SUN RGB-D dataset
SUN_DIR = "/kaggle/input/datasets/bt14147/sun-rgb-d/MYSUN"
SUN_RGB_DIR = os.path.join(SUN_DIR, "image")
SUN_DEPTH_DIR = os.path.join(SUN_DIR, "depth_bfx")

SAVE_DIR = "/kaggle/working/navigation_full_results"
PLOT_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BATCH_SIZE = 32
OUTDOOR_EPOCHS = 25
INDOOR_EPOCHS = 15
LR = 1e-3
PATIENCE = 5

SAFE_LABELS = [0, 1]      # Cityscapes: road + sidewalk as safe path
IGNORE_LABEL = -1

IMG_H = 128
IMG_W = 256

MAX_INDOOR_TRAIN = 4000
MAX_INDOOR_VAL = 800

# Indoor pseudo-mask thresholds for SUN RGB-D inverse-depth calibration
INDOOR_SAFE_THRESHOLD = 0.35
INDOOR_CLOSE_THRESHOLD = 0.55

# Navigation logic
ROI_START_RATIO = 0.60
DANGER_PENALTY = 0.70
MIN_SAFE_SCORE = 0.18

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Device:", DEVICE)
print("Save dir:", SAVE_DIR)


# ============================================================
# MODEL: MINI U-NET
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


# ============================================================
# OUTDOOR DATASET: CITYSCAPES SAFE PATH
# ============================================================

class CityscapesSafePathDataset(Dataset):
    def __init__(self, split="train"):
        self.image_dir = os.path.join(CITY_BASE, split, "image")
        self.label_dir = os.path.join(CITY_BASE, split, "label")
        self.files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".npy")])
        print(f"Cityscapes {split} samples:", len(self.files))

    def __len__(self):
        return len(self.files)

    def label_to_safe_mask(self, label):
        mask = np.zeros_like(label, dtype=np.float32)
        for cls in SAFE_LABELS:
            mask[label == cls] = 1.0
        mask[label == IGNORE_LABEL] = 0.0
        return mask

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.load(os.path.join(self.image_dir, fname)).astype(np.float32)
        label = np.load(os.path.join(self.label_dir, fname)).astype(np.float32)
        mask = self.label_to_safe_mask(label)
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), fname


# ============================================================
# INDOOR DATASET: SUN RGB-D RGB + DEPTH WITH PSEUDO LABELS
# ============================================================

def normalize_depth(depth):
    depth = depth.astype(np.float32)
    valid = depth[depth > 0]
    if len(valid) == 0:
        return np.zeros_like(depth, dtype=np.float32)
    d_min = np.percentile(valid, 2)
    d_max = np.percentile(valid, 98)
    depth = np.clip(depth, d_min, d_max)
    return (depth - d_min) / (d_max - d_min + 1e-6)


def make_indoor_pseudo_mask(depth_norm):
    # SUN RGB-D calibrated inverse-depth rule: lower normalized depth = farther/safer
    return (depth_norm < INDOOR_SAFE_THRESHOLD).astype(np.float32)


class SUNRGBDIndoorDataset(Dataset):
    def __init__(self, split="train"):
        rgb_files = [f for f in os.listdir(SUN_RGB_DIR) if f.lower().endswith(".jpg")]
        ids = [os.path.splitext(f)[0] for f in rgb_files]
        ids = [sid for sid in ids if os.path.exists(os.path.join(SUN_DEPTH_DIR, sid + ".png"))]
        random.shuffle(ids)
        split_idx = int(len(ids) * 0.85)
        if split == "train":
            ids = ids[:split_idx][:MAX_INDOOR_TRAIN]
        else:
            ids = ids[split_idx:][:MAX_INDOOR_VAL]
        self.ids = ids
        self.split = split
        print(f"SUN RGB-D {split} samples:", len(self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        rgb_path = os.path.join(SUN_RGB_DIR, sid + ".jpg")
        depth_path = os.path.join(SUN_DEPTH_DIR, sid + ".png")
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise ValueError(f"Missing RGB/depth for sample {sid}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_W, IMG_H)).astype(np.float32) / 255.0
        depth = cv2.resize(depth, (IMG_W, IMG_H))
        depth_norm = normalize_depth(depth)
        pseudo_mask = make_indoor_pseudo_mask(depth_norm)
        rgb_chw = np.transpose(rgb, (2, 0, 1))
        depth_chw = np.expand_dims(depth_norm, axis=0)
        x = np.concatenate([rgb_chw, depth_chw], axis=0)
        y = np.expand_dims(pseudo_mask, axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), sid


# ============================================================
# METRICS
# ============================================================

def pixel_accuracy(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()


def iou_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


# ============================================================
# NAVIGATION DECISION ENGINE
# ============================================================

def split_three_regions(mask):
    h, w = mask.shape
    roi = mask[int(h * ROI_START_RATIO):, :]
    left = roi[:, :w // 3]
    center = roi[:, w // 3:2 * w // 3]
    right = roi[:, 2 * w // 3:]
    return left, center, right


def score_safe_regions(mask):
    left, center, right = split_three_regions(mask)
    return {"left": float(left.mean()), "center": float(center.mean()), "right": float(right.mean())}


def score_close_regions(depth_norm):
    close_mask = (depth_norm > INDOOR_CLOSE_THRESHOLD).astype(np.float32)
    left, center, right = split_three_regions(close_mask)
    return {"left": float(left.mean()), "center": float(center.mean()), "right": float(right.mean())}


def compute_final_scores(safe_scores, close_scores=None):
    final_scores = {}
    for key in ["left", "center", "right"]:
        if close_scores is None:
            final_scores[key] = safe_scores[key]
        else:
            final_scores[key] = safe_scores[key] - DANGER_PENALTY * close_scores[key]
    return final_scores


def decide_command(final_scores):
    best = max(final_scores, key=final_scores.get)
    if final_scores[best] < MIN_SAFE_SCORE:
        return "STOP / NO SAFE PATH"
    if best == "center":
        return "MOVE FORWARD"
    if best == "left":
        return "MOVE LEFT"
    return "MOVE RIGHT"


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer, task_name):
    model.train()
    total_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc=f"Training {task_name}"):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(model, loader, loss_fn, task_name):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc=f"Evaluating {task_name}"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()
            total_acc += pixel_accuracy(preds, masks)
            total_iou += iou_score(preds, masks)
            count += 1
    return {"loss": total_loss / count, "pixel_accuracy": total_acc / count, "miou": total_iou / count}


def train_model(model, train_loader, val_loader, epochs, task_name, save_prefix):
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_miou = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, task_name)
        metrics = evaluate(model, val_loader, loss_fn, task_name)
        row = {
            "task": task_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "miou": metrics["miou"],
        }
        history.append(row)
        print("\n" + "-" * 70)
        print(f"{task_name} Epoch {epoch}/{epochs}")
        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Val Loss:       {metrics['loss']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"mIoU:           {metrics['miou']:.4f}")

        hist_path = os.path.join(SAVE_DIR, f"{save_prefix}_history.csv")
        pd.DataFrame(history).to_csv(hist_path, index=False)

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "miou": metrics["miou"],
            "task": task_name,
        }, os.path.join(SAVE_DIR, f"last_{save_prefix}.pt"))

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            patience_counter = 0
            best_path = os.path.join(SAVE_DIR, f"best_{save_prefix}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "task": task_name,
            }, best_path)
            print("✅ Saved best model:", best_path)
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping")
            break

    return model, pd.DataFrame(history), best_miou


# ============================================================
# DATA LOADERS
# ============================================================

city_train = CityscapesSafePathDataset("train")
city_val = CityscapesSafePathDataset("val")
city_train_loader = DataLoader(city_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
city_val_loader = DataLoader(city_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

sun_train = SUNRGBDIndoorDataset("train")
sun_val = SUNRGBDIndoorDataset("val")
sun_train_loader = DataLoader(sun_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
sun_val_loader = DataLoader(sun_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# TRAIN OUTDOOR + INDOOR
# ============================================================

print("\n" + "=" * 80)
print("TRAINING OUTDOOR MINI U-NET")
print("=" * 80)
outdoor_model = MiniUNet(in_channels=3)
outdoor_model, outdoor_history, outdoor_best_miou = train_model(
    outdoor_model, city_train_loader, city_val_loader, OUTDOOR_EPOCHS,
    "Outdoor Cityscapes Safe Path", "outdoor_navigation"
)
print("Outdoor Best mIoU:", outdoor_best_miou)

print("\n" + "=" * 80)
print("TRAINING INDOOR RGB-D MINI U-NET")
print("=" * 80)
indoor_model = MiniUNet(in_channels=4)
indoor_model, indoor_history, indoor_best_miou = train_model(
    indoor_model, sun_train_loader, sun_val_loader, INDOOR_EPOCHS,
    "Indoor SUN RGB-D Pseudo Safe Path", "indoor_rgbd_navigation"
)
print("Indoor Best mIoU:", indoor_best_miou)


# ============================================================
# PAPER FIGURE 1: TRAINING CURVES
# ============================================================

combined_history = pd.concat([outdoor_history, indoor_history], ignore_index=True)
combined_history.to_csv(os.path.join(SAVE_DIR, "combined_navigation_history.csv"), index=False)

plt.figure(figsize=(10, 6))
for task_name, df in combined_history.groupby("task"):
    plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"{task_name} Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="s", label=f"{task_name} Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig1_path = os.path.join(PLOT_DIR, "01_training_curves.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fig1_path)


# ============================================================
# PAPER FIGURE 2: VALIDATION mIoU CURVE
# ============================================================

plt.figure(figsize=(10, 6))
for task_name, df in combined_history.groupby("task"):
    plt.plot(df["epoch"], df["miou"], marker="o", label=f"{task_name} mIoU")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.title("Validation mIoU Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig2_path = os.path.join(PLOT_DIR, "02_validation_miou_curve.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fig2_path)


# ============================================================
# QUALITATIVE EXAMPLES HELPERS
# ============================================================

def get_prediction_mask(model, x):
    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0).to(DEVICE)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.float32)


def overlay_mask_and_command(img, pred, command):
    overlay = img.copy()
    mask = cv2.resize(pred.astype(np.float32), (overlay.shape[1], overlay.shape[0]))
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    overlay = np.where(mask[:, :, None] > 0.5, (0.55 * overlay + 0.45 * green).astype(np.uint8), overlay)
    h, w, _ = overlay.shape
    cv2.line(overlay, (w // 3, 0), (w // 3, h), (255, 255, 255), 2)
    cv2.line(overlay, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 2)
    cv2.line(overlay, (0, int(h * ROI_START_RATIO)), (w, int(h * ROI_START_RATIO)), (255, 255, 0), 2)
    cv2.putText(overlay, command, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    return overlay


def prepare_outdoor_example(idx=0):
    x, y, sid = city_val[idx]
    pred_mask = get_prediction_mask(outdoor_model, x)
    img = np.transpose(x.numpy(), (1, 2, 0))
    img = (img * 255).clip(0, 255).astype(np.uint8)
    gt_mask = y.numpy()[0]
    safe_scores = score_safe_regions(pred_mask)
    final_scores = compute_final_scores(safe_scores)
    command = decide_command(final_scores)
    return img, gt_mask, pred_mask, command, "Outdoor"


def prepare_indoor_example(idx=0):
    x, y, sid = sun_val[idx]
    pred_mask = get_prediction_mask(indoor_model, x)
    arr = x.numpy()
    rgb = np.transpose(arr[:3], (1, 2, 0))
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    depth_norm = arr[3]
    gt_mask = y.numpy()[0]
    safe_scores = score_safe_regions(pred_mask)
    close_scores = score_close_regions(depth_norm)
    final_scores = compute_final_scores(safe_scores, close_scores)
    command = decide_command(final_scores)
    return rgb, gt_mask, pred_mask, command, "Indoor"


# ============================================================
# PAPER FIGURE 3: QUALITATIVE NAVIGATION EXAMPLES
# ============================================================

outdoor_indices = [0, min(3, len(city_val) - 1)]
indoor_indices = [0, min(3, len(sun_val) - 1)]
examples = [
    prepare_outdoor_example(outdoor_indices[0]),
    prepare_outdoor_example(outdoor_indices[1]),
    prepare_indoor_example(indoor_indices[0]),
    prepare_indoor_example(indoor_indices[1]),
]

plt.figure(figsize=(14, 12))
for i, (img, gt, pred, cmd, mode) in enumerate(examples):
    row = i
    plt.subplot(4, 4, row * 4 + 1)
    plt.imshow(img)
    plt.title(f"{mode} Original")
    plt.axis("off")

    plt.subplot(4, 4, row * 4 + 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth / Pseudo Mask")
    plt.axis("off")

    plt.subplot(4, 4, row * 4 + 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    overlay = overlay_mask_and_command(img, pred, cmd)
    plt.subplot(4, 4, row * 4 + 4)
    plt.imshow(overlay)
    plt.title(f"Decision: {cmd}")
    plt.axis("off")

plt.tight_layout()
fig3_path = os.path.join(PLOT_DIR, "03_qualitative_navigation_examples.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fig3_path)


# ============================================================
# FUSION DEMO FIGURE: OUTDOOR + INDOOR DECISION-LEVEL FUSION
# ============================================================

fusion_examples = [
    prepare_outdoor_example(0),
    prepare_indoor_example(0),
    prepare_outdoor_example(min(5, len(city_val) - 1)),
    prepare_indoor_example(min(5, len(sun_val) - 1)),
]

plt.figure(figsize=(13, 10))
for i, (img, gt, pred, cmd, mode) in enumerate(fusion_examples):
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(img)
    plt.title(f"{mode} RGB")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(pred, cmap="gray")
    plt.title("Safe Mask")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(overlay_mask_and_command(img, pred, cmd))
    plt.title(cmd)
    plt.axis("off")

plt.tight_layout()
fusion_path = os.path.join(PLOT_DIR, "04_indoor_outdoor_fusion_examples.png")
plt.savefig(fusion_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fusion_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "outdoor_best_miou": outdoor_best_miou,
    "indoor_best_miou_pseudo": indoor_best_miou,
    "outdoor_best_model": os.path.join(SAVE_DIR, "best_outdoor_navigation.pt"),
    "indoor_best_model": os.path.join(SAVE_DIR, "best_indoor_rgbd_navigation.pt"),
    "combined_history": os.path.join(SAVE_DIR, "combined_navigation_history.csv"),
    "training_curves": fig1_path,
    "miou_curve": fig2_path,
    "qualitative_examples": fig3_path,
    "fusion_examples": fusion_path,
}

print("\n" + "=" * 80)
print("FINAL NAVIGATION SUMMARY")
print("=" * 80)
for k, v in summary.items():
    print(k, ":", v)

pd.DataFrame([summary]).to_csv(os.path.join(SAVE_DIR, "navigation_summary.csv"), index=False)
print("\nSaved files in:", SAVE_DIR)
print("Saved plots in:", PLOT_DIR)
print("Plot files:", os.listdir(PLOT_DIR))
Device: cuda
Save dir: /kaggle/working/navigation_full_results
Cityscapes train samples: 2975
Cityscapes val samples: 500
SUN RGB-D train samples: 4000
SUN RGB-D val samples: 800






# ============================================================
# FULL NAVIGATION MODULE FOR KAGGLE (ONE CELL)
# Outdoor Mini U-Net + Indoor RGB-D Mini U-Net + Fusion + Paper Figures
# ============================================================

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Outdoor Cityscapes dataset
CITY_BASE = "/kaggle/input/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data"

# Indoor SUN RGB-D dataset
SUN_DIR = "/kaggle/input/datasets/bt14147/sun-rgb-d/MYSUN"
SUN_RGB_DIR = os.path.join(SUN_DIR, "image")
SUN_DEPTH_DIR = os.path.join(SUN_DIR, "depth_bfx")

SAVE_DIR = "/kaggle/working/navigation_full_results"
PLOT_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BATCH_SIZE = 32
OUTDOOR_EPOCHS = 25
INDOOR_EPOCHS = 15
LR = 1e-3
PATIENCE = 5

SAFE_LABELS = [0, 1]      # Cityscapes: road + sidewalk as safe path
IGNORE_LABEL = -1

IMG_H = 128
IMG_W = 256

MAX_INDOOR_TRAIN = 4000
MAX_INDOOR_VAL = 800

# Indoor pseudo-mask thresholds for SUN RGB-D inverse-depth calibration
INDOOR_SAFE_THRESHOLD = 0.35
INDOOR_CLOSE_THRESHOLD = 0.55

# Navigation logic
ROI_START_RATIO = 0.60
DANGER_PENALTY = 0.70
MIN_SAFE_SCORE = 0.18

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Device:", DEVICE)
print("Save dir:", SAVE_DIR)


# ============================================================
# MODEL: MINI U-NET
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


# ============================================================
# OUTDOOR DATASET: CITYSCAPES SAFE PATH
# ============================================================

class CityscapesSafePathDataset(Dataset):
    def __init__(self, split="train"):
        self.image_dir = os.path.join(CITY_BASE, split, "image")
        self.label_dir = os.path.join(CITY_BASE, split, "label")
        self.files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".npy")])
        print(f"Cityscapes {split} samples:", len(self.files))

    def __len__(self):
        return len(self.files)

    def label_to_safe_mask(self, label):
        mask = np.zeros_like(label, dtype=np.float32)
        for cls in SAFE_LABELS:
            mask[label == cls] = 1.0
        mask[label == IGNORE_LABEL] = 0.0
        return mask

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.load(os.path.join(self.image_dir, fname)).astype(np.float32)
        label = np.load(os.path.join(self.label_dir, fname)).astype(np.float32)
        mask = self.label_to_safe_mask(label)
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), fname


# ============================================================
# INDOOR DATASET: SUN RGB-D RGB + DEPTH WITH PSEUDO LABELS
# ============================================================

def normalize_depth(depth):
    depth = depth.astype(np.float32)
    valid = depth[depth > 0]
    if len(valid) == 0:
        return np.zeros_like(depth, dtype=np.float32)
    d_min = np.percentile(valid, 2)
    d_max = np.percentile(valid, 98)
    depth = np.clip(depth, d_min, d_max)
    return (depth - d_min) / (d_max - d_min + 1e-6)


def make_indoor_pseudo_mask(depth_norm):
    # SUN RGB-D calibrated inverse-depth rule: lower normalized depth = farther/safer
    return (depth_norm < INDOOR_SAFE_THRESHOLD).astype(np.float32)


class SUNRGBDIndoorDataset(Dataset):
    def __init__(self, split="train"):
        rgb_files = [f for f in os.listdir(SUN_RGB_DIR) if f.lower().endswith(".jpg")]
        ids = [os.path.splitext(f)[0] for f in rgb_files]
        ids = [sid for sid in ids if os.path.exists(os.path.join(SUN_DEPTH_DIR, sid + ".png"))]
        random.shuffle(ids)
        split_idx = int(len(ids) * 0.85)
        if split == "train":
            ids = ids[:split_idx][:MAX_INDOOR_TRAIN]
        else:
            ids = ids[split_idx:][:MAX_INDOOR_VAL]
        self.ids = ids
        self.split = split
        print(f"SUN RGB-D {split} samples:", len(self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        rgb_path = os.path.join(SUN_RGB_DIR, sid + ".jpg")
        depth_path = os.path.join(SUN_DEPTH_DIR, sid + ".png")
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise ValueError(f"Missing RGB/depth for sample {sid}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_W, IMG_H)).astype(np.float32) / 255.0
        depth = cv2.resize(depth, (IMG_W, IMG_H))
        depth_norm = normalize_depth(depth)
        pseudo_mask = make_indoor_pseudo_mask(depth_norm)
        rgb_chw = np.transpose(rgb, (2, 0, 1))
        depth_chw = np.expand_dims(depth_norm, axis=0)
        x = np.concatenate([rgb_chw, depth_chw], axis=0)
        y = np.expand_dims(pseudo_mask, axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), sid


# ============================================================
# METRICS
# ============================================================

def pixel_accuracy(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()


def iou_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


# ============================================================
# NAVIGATION DECISION ENGINE
# ============================================================

def split_three_regions(mask):
    h, w = mask.shape
    roi = mask[int(h * ROI_START_RATIO):, :]
    left = roi[:, :w // 3]
    center = roi[:, w // 3:2 * w // 3]
    right = roi[:, 2 * w // 3:]
    return left, center, right


def score_safe_regions(mask):
    left, center, right = split_three_regions(mask)
    return {"left": float(left.mean()), "center": float(center.mean()), "right": float(right.mean())}


def score_close_regions(depth_norm):
    close_mask = (depth_norm > INDOOR_CLOSE_THRESHOLD).astype(np.float32)
    left, center, right = split_three_regions(close_mask)
    return {"left": float(left.mean()), "center": float(center.mean()), "right": float(right.mean())}


def compute_final_scores(safe_scores, close_scores=None):
    final_scores = {}
    for key in ["left", "center", "right"]:
        if close_scores is None:
            final_scores[key] = safe_scores[key]
        else:
            final_scores[key] = safe_scores[key] - DANGER_PENALTY * close_scores[key]
    return final_scores


def decide_command(final_scores):
    best = max(final_scores, key=final_scores.get)
    if final_scores[best] < MIN_SAFE_SCORE:
        return "STOP / NO SAFE PATH"
    if best == "center":
        return "MOVE FORWARD"
    if best == "left":
        return "MOVE LEFT"
    return "MOVE RIGHT"


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer, task_name):
    model.train()
    total_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc=f"Training {task_name}"):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(model, loader, loss_fn, task_name):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc=f"Evaluating {task_name}"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()
            total_acc += pixel_accuracy(preds, masks)
            total_iou += iou_score(preds, masks)
            count += 1
    return {"loss": total_loss / count, "pixel_accuracy": total_acc / count, "miou": total_iou / count}


def train_model(model, train_loader, val_loader, epochs, task_name, save_prefix):
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_miou = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, task_name)
        metrics = evaluate(model, val_loader, loss_fn, task_name)
        row = {
            "task": task_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "miou": metrics["miou"],
        }
        history.append(row)
        print("\n" + "-" * 70)
        print(f"{task_name} Epoch {epoch}/{epochs}")
        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Val Loss:       {metrics['loss']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"mIoU:           {metrics['miou']:.4f}")

        hist_path = os.path.join(SAVE_DIR, f"{save_prefix}_history.csv")
        pd.DataFrame(history).to_csv(hist_path, index=False)

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "miou": metrics["miou"],
            "task": task_name,
        }, os.path.join(SAVE_DIR, f"last_{save_prefix}.pt"))

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            patience_counter = 0
            best_path = os.path.join(SAVE_DIR, f"best_{save_prefix}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "task": task_name,
            }, best_path)
            print("✅ Saved best model:", best_path)
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping")
            break

    return model, pd.DataFrame(history), best_miou


# ============================================================
# DATA LOADERS
# ============================================================

city_train = CityscapesSafePathDataset("train")
city_val = CityscapesSafePathDataset("val")
city_train_loader = DataLoader(city_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
city_val_loader = DataLoader(city_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

sun_train = SUNRGBDIndoorDataset("train")
sun_val = SUNRGBDIndoorDataset("val")
sun_train_loader = DataLoader(sun_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
sun_val_loader = DataLoader(sun_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# TRAIN OUTDOOR + INDOOR
# ============================================================

print("\n" + "=" * 80)
print("TRAINING OUTDOOR MINI U-NET")
print("=" * 80)
outdoor_model = MiniUNet(in_channels=3)
outdoor_model, outdoor_history, outdoor_best_miou = train_model(
    outdoor_model, city_train_loader, city_val_loader, OUTDOOR_EPOCHS,
    "Outdoor Cityscapes Safe Path", "outdoor_navigation"
)
print("Outdoor Best mIoU:", outdoor_best_miou)

print("\n" + "=" * 80)
print("TRAINING INDOOR RGB-D MINI U-NET")
print("=" * 80)
indoor_model = MiniUNet(in_channels=4)
indoor_model, indoor_history, indoor_best_miou = train_model(
    indoor_model, sun_train_loader, sun_val_loader, INDOOR_EPOCHS,
    "Indoor SUN RGB-D Pseudo Safe Path", "indoor_rgbd_navigation"
)
print("Indoor Best mIoU:", indoor_best_miou)


# ============================================================
# PAPER FIGURE 1: TRAINING CURVES
# ============================================================

combined_history = pd.concat([outdoor_history, indoor_history], ignore_index=True)
combined_history.to_csv(os.path.join(SAVE_DIR, "combined_navigation_history.csv"), index=False)

plt.figure(figsize=(10, 6))
for task_name, df in combined_history.groupby("task"):
    plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"{task_name} Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="s", label=f"{task_name} Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig1_path = os.path.join(PLOT_DIR, "01_training_curves.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fig1_path)


# ============================================================
# PAPER FIGURE 2: VALIDATION mIoU CURVE
# ============================================================

plt.figure(figsize=(10, 6))
for task_name, df in combined_history.groupby("task"):
    plt.plot(df["epoch"], df["miou"], marker="o", label=f"{task_name} mIoU")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.title("Validation mIoU Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig2_path = os.path.join(PLOT_DIR, "02_validation_miou_curve.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fig2_path)


# ============================================================
# QUALITATIVE EXAMPLES HELPERS
# ============================================================

def get_prediction_mask(model, x):
    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0).to(DEVICE)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.float32)


def overlay_mask_and_command(img, pred, command):
    overlay = img.copy()
    mask = cv2.resize(pred.astype(np.float32), (overlay.shape[1], overlay.shape[0]))
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    overlay = np.where(mask[:, :, None] > 0.5, (0.55 * overlay + 0.45 * green).astype(np.uint8), overlay)
    h, w, _ = overlay.shape
    cv2.line(overlay, (w // 3, 0), (w // 3, h), (255, 255, 255), 2)
    cv2.line(overlay, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 2)
    cv2.line(overlay, (0, int(h * ROI_START_RATIO)), (w, int(h * ROI_START_RATIO)), (255, 255, 0), 2)
    cv2.putText(overlay, command, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    return overlay


def prepare_outdoor_example(idx=0):
    x, y, sid = city_val[idx]
    pred_mask = get_prediction_mask(outdoor_model, x)
    img = np.transpose(x.numpy(), (1, 2, 0))
    img = (img * 255).clip(0, 255).astype(np.uint8)
    gt_mask = y.numpy()[0]
    safe_scores = score_safe_regions(pred_mask)
    final_scores = compute_final_scores(safe_scores)
    command = decide_command(final_scores)
    return img, gt_mask, pred_mask, command, "Outdoor"


def prepare_indoor_example(idx=0):
    x, y, sid = sun_val[idx]
    pred_mask = get_prediction_mask(indoor_model, x)
    arr = x.numpy()
    rgb = np.transpose(arr[:3], (1, 2, 0))
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    depth_norm = arr[3]
    gt_mask = y.numpy()[0]
    safe_scores = score_safe_regions(pred_mask)
    close_scores = score_close_regions(depth_norm)
    final_scores = compute_final_scores(safe_scores, close_scores)
    command = decide_command(final_scores)
    return rgb, gt_mask, pred_mask, command, "Indoor"


# ============================================================
# PAPER FIGURE 3: QUALITATIVE NAVIGATION EXAMPLES
# ============================================================

outdoor_indices = [0, min(3, len(city_val) - 1)]
indoor_indices = [0, min(3, len(sun_val) - 1)]
examples = [
    prepare_outdoor_example(outdoor_indices[0]),
    prepare_outdoor_example(outdoor_indices[1]),
    prepare_indoor_example(indoor_indices[0]),
    prepare_indoor_example(indoor_indices[1]),
]

plt.figure(figsize=(14, 12))
for i, (img, gt, pred, cmd, mode) in enumerate(examples):
    row = i
    plt.subplot(4, 4, row * 4 + 1)
    plt.imshow(img)
    plt.title(f"{mode} Original")
    plt.axis("off")

    plt.subplot(4, 4, row * 4 + 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth / Pseudo Mask")
    plt.axis("off")

    plt.subplot(4, 4, row * 4 + 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    overlay = overlay_mask_and_command(img, pred, cmd)
    plt.subplot(4, 4, row * 4 + 4)
    plt.imshow(overlay)
    plt.title(f"Decision: {cmd}")
    plt.axis("off")

plt.tight_layout()
fig3_path = os.path.join(PLOT_DIR, "03_qualitative_navigation_examples.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fig3_path)


# ============================================================
# FUSION DEMO FIGURE: OUTDOOR + INDOOR DECISION-LEVEL FUSION
# ============================================================

fusion_examples = [
    prepare_outdoor_example(0),
    prepare_indoor_example(0),
    prepare_outdoor_example(min(5, len(city_val) - 1)),
    prepare_indoor_example(min(5, len(sun_val) - 1)),
]

plt.figure(figsize=(13, 10))
for i, (img, gt, pred, cmd, mode) in enumerate(fusion_examples):
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(img)
    plt.title(f"{mode} RGB")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(pred, cmap="gray")
    plt.title("Safe Mask")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(overlay_mask_and_command(img, pred, cmd))
    plt.title(cmd)
    plt.axis("off")

plt.tight_layout()
fusion_path = os.path.join(PLOT_DIR, "04_indoor_outdoor_fusion_examples.png")
plt.savefig(fusion_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", fusion_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "outdoor_best_miou": outdoor_best_miou,
    "indoor_best_miou_pseudo": indoor_best_miou,
    "outdoor_best_model": os.path.join(SAVE_DIR, "best_outdoor_navigation.pt"),
    "indoor_best_model": os.path.join(SAVE_DIR, "best_indoor_rgbd_navigation.pt"),
    "combined_history": os.path.join(SAVE_DIR, "combined_navigation_history.csv"),
    "training_curves": fig1_path,
    "miou_curve": fig2_path,
    "qualitative_examples": fig3_path,
    "fusion_examples": fusion_path,
}

print("\n" + "=" * 80)
print("FINAL NAVIGATION SUMMARY")
print("=" * 80)
for k, v in summary.items():
    print(k, ":", v)

pd.DataFrame([summary]).to_csv(os.path.join(SAVE_DIR, "navigation_summary.csv"), index=False)
print("\nSaved files in:", SAVE_DIR)
print("Saved plots in:", PLOT_DIR)
print("Plot files:", os.listdir(PLOT_DIR))
Device: cuda
Save dir: /kaggle/working/navigation_full_results
Cityscapes train samples: 2975
Cityscapes val samples: 500
SUN RGB-D train samples: 4000
SUN RGB-D val samples: 800

================================================================================
TRAINING OUTDOOR MINI U-NET
================================================================================
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:43<00:00,  2.13it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:05<00:00,  2.73it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 1/25
Train Loss:     0.2751
Val Loss:       0.2137
Pixel Accuracy: 0.9302
mIoU:           0.8249
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:26<00:00,  3.52it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.96it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 2/25
Train Loss:     0.1667
Val Loss:       0.1760
Pixel Accuracy: 0.9325
mIoU:           0.8377
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.36it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.45it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 3/25
Train Loss:     0.1420
Val Loss:       0.1463
Pixel Accuracy: 0.9446
mIoU:           0.8589
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.33it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.94it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 4/25
Train Loss:     0.1308
Val Loss:       0.1452
Pixel Accuracy: 0.9480
mIoU:           0.8705
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.44it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00, 10.05it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 5/25
Train Loss:     0.1192
Val Loss:       0.1309
Pixel Accuracy: 0.9532
mIoU:           0.8836
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.40it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.93it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 6/25
Train Loss:     0.1118
Val Loss:       0.1325
Pixel Accuracy: 0.9506
mIoU:           0.8773
⚠️ No improvement. Patience 1/5
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.38it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.91it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 7/25
Train Loss:     0.1094
Val Loss:       0.1196
Pixel Accuracy: 0.9565
mIoU:           0.8903
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.39it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.97it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 8/25
Train Loss:     0.1013
Val Loss:       0.1374
Pixel Accuracy: 0.9500
mIoU:           0.8792
⚠️ No improvement. Patience 1/5
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.40it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.90it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 9/25
Train Loss:     0.0984
Val Loss:       0.1198
Pixel Accuracy: 0.9561
mIoU:           0.8913
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.38it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.69it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 10/25
Train Loss:     0.0939
Val Loss:       0.1225
Pixel Accuracy: 0.9592
mIoU:           0.8994
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.38it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.94it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 11/25
Train Loss:     0.0912
Val Loss:       0.1237
Pixel Accuracy: 0.9581
mIoU:           0.8954
⚠️ No improvement. Patience 1/5
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.40it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.86it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 12/25
Train Loss:     0.0876
Val Loss:       0.1061
Pixel Accuracy: 0.9621
mIoU:           0.9041
✅ Saved best model: /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.39it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.74it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 13/25
Train Loss:     0.0877
Val Loss:       0.1229
Pixel Accuracy: 0.9611
mIoU:           0.9034
⚠️ No improvement. Patience 1/5
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.38it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.92it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 14/25
Train Loss:     0.0843
Val Loss:       0.1138
Pixel Accuracy: 0.9599
mIoU:           0.8993
⚠️ No improvement. Patience 2/5
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.38it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.89it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 15/25
Train Loss:     0.0799
Val Loss:       0.1121
Pixel Accuracy: 0.9611
mIoU:           0.9023
⚠️ No improvement. Patience 3/5
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.39it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.99it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 16/25
Train Loss:     0.0813
Val Loss:       0.1138
Pixel Accuracy: 0.9614
mIoU:           0.9023
⚠️ No improvement. Patience 4/5
Training Outdoor Cityscapes Safe Path: 100%|██████████| 93/93 [00:27<00:00,  3.39it/s]
Evaluating Outdoor Cityscapes Safe Path: 100%|██████████| 16/16 [00:01<00:00,  9.97it/s]
----------------------------------------------------------------------
Outdoor Cityscapes Safe Path Epoch 17/25
Train Loss:     0.0759
Val Loss:       0.1165
Pixel Accuracy: 0.9594
mIoU:           0.8984
⚠️ No improvement. Patience 5/5
🛑 Early stopping
Outdoor Best mIoU: 0.9041152521967888

================================================================================
TRAINING INDOOR RGB-D MINI U-NET
================================================================================
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:49<00:00,  2.52it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:08<00:00,  3.05it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 1/15
Train Loss:     0.1612
Val Loss:       0.0989
Pixel Accuracy: 0.9709
mIoU:           0.9421
✅ Saved best model: /kaggle/working/navigation_full_results/best_indoor_rgbd_navigation.pt
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.35it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:05<00:00,  4.21it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 2/15
Train Loss:     0.0801
Val Loss:       0.0513
Pixel Accuracy: 0.9819
mIoU:           0.9620
✅ Saved best model: /kaggle/working/navigation_full_results/best_indoor_rgbd_navigation.pt
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.10it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 3/15
Train Loss:     0.0617
Val Loss:       0.0484
Pixel Accuracy: 0.9780
mIoU:           0.9537
⚠️ No improvement. Patience 1/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.04it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 4/15
Train Loss:     0.0603
Val Loss:       0.0752
Pixel Accuracy: 0.9639
mIoU:           0.9294
⚠️ No improvement. Patience 2/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  3.96it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 5/15
Train Loss:     0.0541
Val Loss:       0.0281
Pixel Accuracy: 0.9935
mIoU:           0.9864
✅ Saved best model: /kaggle/working/navigation_full_results/best_indoor_rgbd_navigation.pt
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.01it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 6/15
Train Loss:     0.0507
Val Loss:       0.0332
Pixel Accuracy: 0.9902
mIoU:           0.9796
⚠️ No improvement. Patience 1/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.01it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 7/15
Train Loss:     0.0454
Val Loss:       0.0268
Pixel Accuracy: 0.9923
mIoU:           0.9838
⚠️ No improvement. Patience 2/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.02it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 8/15
Train Loss:     0.0519
Val Loss:       0.0294
Pixel Accuracy: 0.9898
mIoU:           0.9785
⚠️ No improvement. Patience 3/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.35it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.04it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 9/15
Train Loss:     0.0500
Val Loss:       0.0293
Pixel Accuracy: 0.9937
mIoU:           0.9868
✅ Saved best model: /kaggle/working/navigation_full_results/best_indoor_rgbd_navigation.pt
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:05<00:00,  4.18it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 10/15
Train Loss:     0.0469
Val Loss:       0.0275
Pixel Accuracy: 0.9951
mIoU:           0.9897
✅ Saved best model: /kaggle/working/navigation_full_results/best_indoor_rgbd_navigation.pt
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.14it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 11/15
Train Loss:     0.0504
Val Loss:       0.0283
Pixel Accuracy: 0.9941
mIoU:           0.9876
⚠️ No improvement. Patience 1/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.02it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 12/15
Train Loss:     0.0437
Val Loss:       0.0270
Pixel Accuracy: 0.9920
mIoU:           0.9831
⚠️ No improvement. Patience 2/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  4.04it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 13/15
Train Loss:     0.0478
Val Loss:       0.0295
Pixel Accuracy: 0.9954
mIoU:           0.9903
✅ Saved best model: /kaggle/working/navigation_full_results/best_indoor_rgbd_navigation.pt
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  3.96it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 14/15
Train Loss:     0.0495
Val Loss:       0.0267
Pixel Accuracy: 0.9891
mIoU:           0.9775
⚠️ No improvement. Patience 1/5
Training Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 125/125 [00:37<00:00,  3.36it/s]
Evaluating Indoor SUN RGB-D Pseudo Safe Path: 100%|██████████| 25/25 [00:06<00:00,  3.94it/s]
----------------------------------------------------------------------
Indoor SUN RGB-D Pseudo Safe Path Epoch 15/15
Train Loss:     0.0404
Val Loss:       0.0251
Pixel Accuracy: 0.9906
mIoU:           0.9805
⚠️ No improvement. Patience 2/5
Indoor Best mIoU: 0.9903309202194214

Saved: /kaggle/working/navigation_full_results/plots/01_training_curves.png

Saved: /kaggle/working/navigation_full_results/plots/02_validation_miou_curve.png

Saved: /kaggle/working/navigation_full_results/plots/03_qualitative_navigation_examples.png

Saved: /kaggle/working/navigation_full_results/plots/04_indoor_outdoor_fusion_examples.png

================================================================================
FINAL NAVIGATION SUMMARY
================================================================================
outdoor_best_miou : 0.9041152521967888
indoor_best_miou_pseudo : 0.9903309202194214
outdoor_best_model : /kaggle/working/navigation_full_results/best_outdoor_navigation.pt
indoor_best_model : /kaggle/working/navigation_full_results/best_indoor_rgbd_navigation.pt
combined_history : /kaggle/working/navigation_full_results/combined_navigation_history.csv
training_curves : /kaggle/working/navigation_full_results/plots/01_training_curves.png
miou_curve : /kaggle/working/navigation_full_results/plots/02_validation_miou_curve.png
qualitative_examples : /kaggle/working/navigation_full_results/plots/03_qualitative_navigation_examples.png
fusion_examples : /kaggle/working/navigation_full_results/plots/04_indoor_outdoor_fusion_examples.png

Saved files in: /kaggle/working/navigation_full_results
Saved plots in: /kaggle/working/navigation_full_results/plots
Plot files: ['04_indoor_outdoor_fusion_examples.png', '02_validation_miou_curve.png', '03_qualitative_navigation_examples.png', '01_training_curves.png']