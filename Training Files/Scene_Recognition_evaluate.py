import os, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from PIL import Image

# ── CONFIG — edit these if needed ─────────────────────────────────────────────
CKPT_PATH    = "/kaggle/input/datasets/shark1717/output/sv_v9_best.pth"
VAL_ROOT     = "/kaggle/input/datasets/benjaminkz/places365/val"
OUT_DIR      = Path("/kaggle/working/eval_output")
BATCH_SIZE   = 64
NUM_WORKERS  = 4
TSNE_SAMPLES = 2000
RUN_TTA      = True
RUN_TSNE     = True
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Class map ─────────────────────────────────────────────────────────────────
CLASS_MAP = {
    "restaurant": "restaurant", "cafeteria": "restaurant",
    "classroom": "lecture_room", "lecture_room": "lecture_room",
    "airport_terminal": "transport_hub", "bus_station-indoor": "transport_hub",
    "train_station-platform": "transport_hub", "subway_station-platform": "transport_hub",
    "hospital": "hospital", "hospital_room": "hospital",
    "office": "work_space", "conference_room": "work_space",
    "park": "green_outdoor", "campus": "green_outdoor", "playground": "green_outdoor",
    "corridor": "indoor_passage", "elevator_lobby": "indoor_passage",
    "street": "street_road", "crosswalk": "street_road",
    "highway": "street_road", "parking_lot": "street_road",
    "bathroom": "bathroom", "beach": "beach", "bedroom": "bedroom",
    "library-indoor": "library", "shopping_mall-indoor": "shopping_mall",
    "supermarket": "supermarket", "waiting_room": "waiting_room",
    "staircase": "staircase", "mosque-outdoor": "mosque",
    "market-outdoor": "market", "kitchen": "kitchen",
}
LABEL_NAMES   = sorted(set(CLASS_MAP.values()))
LABEL_TO_IDX  = {n: i for i, n in enumerate(LABEL_NAMES)}
FOLDER_TO_IDX = {f: LABEL_TO_IDX[l] for f, l in CLASS_MAP.items()}
NUM_CLASSES   = len(LABEL_NAMES)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Model definition ──────────────────────────────────────────────────────────
class SceneAttentionPool(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//8, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_ch//8, 1, 1, bias=False),
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x).view(x.size(0), -1), dim=1)
        w = w.view(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([(x * w).sum(dim=[2,3]), x.amax(dim=[2,3])], dim=1)

class SceneHead(nn.Module):
    def __init__(self, in_f, n_cls, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f*2, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(1024, 256),    nn.BatchNorm1d(256),  nn.SiLU(), nn.Dropout(drop*0.4),
            nn.Linear(256, n_cls),
        )
    def forward(self, x): return self.net(x)

class SmartVisionV9(nn.Module):
    def __init__(self, n_cls, drop=0.0):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = SceneAttentionPool(2048)
        self.head = SceneHead(2048, n_cls, drop)
    def forward(self, x): return self.head(self.pool(self.backbone(x)))
    def features(self, x): return self.pool(self.backbone(x))

# ── Dataset ───────────────────────────────────────────────────────────────────
class PlacesSubset(Dataset):
    def __init__(self, root, folder_to_idx, transform=None):
        self.samples, self.transform = [], transform
        for folder, idx in folder_to_idx.items():
            fp = Path(root) / folder
            if not fp.exists(): continue
            for ext in ("*.jpg","*.jpeg","*.png","*.JPEG"):
                for p in fp.glob(ext): self.samples.append((p, idx))
        print(f"  {len(self.samples):,} images loaded")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        try: img = Image.open(path).convert("RGB")
        except: img = Image.new("RGB", (224,224))
        if self.transform: img = self.transform(img)
        return img, label

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
print(f"  Stage: {ckpt.get('stage','?')} | Epoch: {ckpt.get('epoch','?')} | Saved val acc: {ckpt.get('val_acc',0)*100:.2f}%")

model = SmartVisionV9(n_cls=NUM_CLASSES, drop=0.0)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
print("  Model loaded.")

# ── Val loader ────────────────────────────────────────────────────────────────
val_tfm = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
val_ds     = PlacesSubset(VAL_ROOT, FOLDER_TO_IDX, transform=val_tfm)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ── Inference ─────────────────────────────────────────────────────────────────
print("\nRunning inference...")
all_probs, all_preds, all_labels, all_feats = [], [], [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type=="cuda")):
            logits = model(imgs)
            feats  = model.features(imgs)
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_preds.extend(probs.argmax(1).numpy())
        all_labels.extend(labels.numpy())
        all_feats.append(feats.cpu())

all_probs  = torch.cat(all_probs).numpy()
all_feats  = torch.cat(all_feats).numpy()
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
val_acc    = (all_preds == all_labels).mean()
print(f"Val Accuracy: {val_acc*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
cm   = confusion_matrix(all_labels, all_preds)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(cm_n, annot=True, fmt=".2f", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            cmap="Blues", ax=ax, annot_kws={"size":8})
ax.set_title(f"Normalized Confusion Matrix  ({val_acc*100:.2f}%)", fontweight="bold", fontsize=13)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR/"01_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Per-class Accuracy
# ══════════════════════════════════════════════════════════════════════════════
per = cm.diagonal() / cm.sum(axis=1)
idx = np.argsort(per)
colors = ["#d32f2f" if v<0.8 else "#f57c00" if v<0.9 else "#388e3c" for v in per[idx]]
fig, ax = plt.subplots(figsize=(10, max(6, NUM_CLASSES*0.4)))
bars = ax.barh([LABEL_NAMES[i] for i in idx], per[idx]*100, color=colors)
ax.bar_label(bars, [f"{v*100:.1f}%" for v in per[idx]], padding=3, fontsize=8)
ax.axvline(80, color="red", ls="--", lw=1, alpha=0.6)
ax.axvline(90, color="orange", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 112)
ax.set_title("Per-class Accuracy (sorted)", fontweight="bold")
legend_patches = [mpatches.Patch(color="#d32f2f", label="<80%"),
                  mpatches.Patch(color="#f57c00", label="80–90%"),
                  mpatches.Patch(color="#388e3c", label="≥90%")]
ax.legend(handles=legend_patches, fontsize=8); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"02_per_class_accuracy.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Precision / Recall / F1
# ══════════════════════════════════════════════════════════════════════════════
report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES,
                                output_dict=True, zero_division=0)
p  = [report[n]["precision"] for n in LABEL_NAMES]
r  = [report[n]["recall"]    for n in LABEL_NAMES]
f1 = [report[n]["f1-score"]  for n in LABEL_NAMES]
idx = np.argsort(f1); x = np.arange(NUM_CLASSES)
fig, ax = plt.subplots(figsize=(12, max(6, NUM_CLASSES*0.45)))
ax.barh(x-0.25, [p[i]  for i in idx], 0.25, label="Precision", color="#1976D2")
ax.barh(x,      [r[i]  for i in idx], 0.25, label="Recall",    color="#43A047")
ax.barh(x+0.25, [f1[i] for i in idx], 0.25, label="F1",        color="#E53935")
ax.set_yticks(x); ax.set_yticklabels([LABEL_NAMES[i] for i in idx], fontsize=8)
ax.set_xlabel("Score"); ax.set_xlim(0, 1.12)
ax.set_title("Precision / Recall / F1 per Class (sorted by F1)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"03_precision_recall_f1.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Top-K Accuracy
# ══════════════════════════════════════════════════════════════════════════════
probs_t  = torch.tensor(all_probs)
labels_t = torch.tensor(all_labels)
ks = [1, 2, 3, 5]
topk_accs = []
for k in ks:
    hit = probs_t.topk(k, dim=1).indices.eq(labels_t.unsqueeze(1)).any(1).float().mean().item()
    topk_accs.append(hit*100)
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar([f"Top-{k}" for k in ks], topk_accs,
              color=["#1565C0","#1976D2","#42A5F5","#90CAF9"])
ax.bar_label(bars, [f"{v:.2f}%" for v in topk_accs], padding=4)
ax.set_ylim(0, 108); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y", alpha=0.3)
ax.set_title("Top-K Accuracy", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR/"04_topk_accuracy.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Confidence Distribution
# ══════════════════════════════════════════════════════════════════════════════
conf    = all_probs.max(axis=1)
correct = conf[all_preds == all_labels]
wrong   = conf[all_preds != all_labels]
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(correct, bins=40, alpha=0.6, color="#43A047", label=f"Correct (n={len(correct):,})", density=True)
ax.hist(wrong,   bins=40, alpha=0.6, color="#E53935", label=f"Wrong   (n={len(wrong):,})",   density=True)
ax.axvline(np.median(correct), color="#1B5E20", ls="--", lw=1.5, label=f"Median correct: {np.median(correct):.2f}")
ax.axvline(np.median(wrong),   color="#B71C1C", ls="--", lw=1.5, label=f"Median wrong:   {np.median(wrong):.2f}")
ax.set_xlabel("Max Softmax Confidence"); ax.set_ylabel("Density")
ax.set_title("Confidence Distribution — Correct vs Incorrect", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"05_confidence_distribution.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Calibration
# ══════════════════════════════════════════════════════════════════════════════
labels_oh = np.eye(NUM_CLASSES)[all_labels]
all_frac, all_mean_conf = [], []
for c in range(NUM_CLASSES):
    frac, mean_conf = calibration_curve(labels_oh[:,c], all_probs[:,c], n_bins=15, strategy="uniform")
    all_frac.extend(frac); all_mean_conf.extend(mean_conf)
# ECE
bins = np.linspace(0,1,16); ece = 0.0
hits = (all_preds == all_labels).astype(float)
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (conf >= lo) & (conf < hi)
    if mask.sum(): ece += mask.sum() * abs(hits[mask].mean() - conf[mask].mean())
ece /= len(all_labels)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot([0,1],[0,1],"k--",lw=1.5, label="Perfect calibration")
ax.scatter(all_mean_conf, all_frac, alpha=0.4, s=8, color="#1976D2", label="Per-class bins")
ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Fraction positive")
ax.set_title(f"Calibration Reliability Diagram  (ECE={ece:.4f})", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(OUT_DIR/"06_calibration.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
cmap = plt.cm.get_cmap("tab20", NUM_CLASSES)
fig, ax = plt.subplots(figsize=(10,8))
for c in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(labels_oh[:,c], all_probs[:,c])
    ax.plot(fpr, tpr, lw=1.2, color=cmap(c), label=f"{LABEL_NAMES[c]} ({auc(fpr,tpr):.2f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (One-vs-Rest)", fontweight="bold")
ax.legend(fontsize=6.5, ncol=2, loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"07_roc_curves.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10,8))
for c in range(NUM_CLASSES):
    prec, rec, _ = precision_recall_curve(labels_oh[:,c], all_probs[:,c])
    ap = average_precision_score(labels_oh[:,c], all_probs[:,c])
    ax.plot(rec, prec, lw=1.2, color=cmap(c), label=f"{LABEL_NAMES[c]} (AP={ap:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves", fontweight="bold")
ax.legend(fontsize=6.5, ncol=2, loc="upper right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"08_pr_curves.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 — t-SNE
# ══════════════════════════════════════════════════════════════════════════════
if RUN_TSNE:
    n   = min(TSNE_SAMPLES, len(all_feats))
    idx = np.random.choice(len(all_feats), n, replace=False)
    print(f"\nRunning t-SNE on {n} samples...")
    emb = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1).fit_transform(all_feats[idx])
    fig, ax = plt.subplots(figsize=(12,10))
    for c, name in enumerate(LABEL_NAMES):
        mask = all_labels[idx] == c
        ax.scatter(emb[mask,0], emb[mask,1], s=8, alpha=0.55, color=cmap(c), label=name)
    ax.set_title(f"t-SNE of Pooled Features  (n={n})", fontweight="bold")
    ax.legend(fontsize=7, ncol=3, markerscale=2); ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR/"09_tsne_embedding.png", dpi=150, bbox_inches="tight"); plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10 — TTA vs Single-pass
# ══════════════════════════════════════════════════════════════════════════════
if RUN_TTA:
    print("\nRunning TTA (5 augmentations)...")
    tfms = [
        transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
        transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
        transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
        transforms.Compose([transforms.Resize((256,256)), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
        transforms.Compose([transforms.Resize((288,288)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
    ]
    pa, la = [], []
    with torch.no_grad():
        for folder, idx in FOLDER_TO_IDX.items():
            fp = Path(VAL_ROOT) / folder
            if not fp.exists(): continue
            for p in list(fp.glob("*.jpg"))[:100]:
                try: img = Image.open(p).convert("RGB")
                except: continue
                ps = None
                for tfm in tfms:
                    t = tfm(img).unsqueeze(0).to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=(DEVICE.type=="cuda")):
                        q = F.softmax(model(t), dim=1).cpu()
                    ps = q if ps is None else ps + q
                pa.append(ps.argmax(1).item()); la.append(idx)
    tta_acc = float(np.mean(np.array(pa) == np.array(la)))
    print(f"TTA Accuracy: {tta_acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(5,4))
    bars = ax.bar(["Single-pass", "TTA (5×)"], [val_acc*100, tta_acc*100],
                  color=["#1976D2","#43A047"], width=0.4)
    ax.bar_label(bars, [f"{v*100:.2f}%" for v in [val_acc, tta_acc]], padding=4)
    ax.set_ylim(0,108); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Single-pass vs TTA Accuracy", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR/"10_tta_comparison.png", dpi=150, bbox_inches="tight"); plt.show()

# ── Text report ───────────────────────────────────────────────────────────────
report_str = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=3)
with open(OUT_DIR/"classification_report.txt","w") as f:
    f.write(f"SmartVisionX v9 | {NUM_CLASSES} classes | Val: {val_acc*100:.2f}%\n\n{report_str}")

print(f"\n{'='*55}")
print(f"  All done! Plots saved to {OUT_DIR}/")
print(f"  Val Accuracy : {val_acc*100:.2f}%")
if RUN_TTA: print(f"  TTA Accuracy : {tta_acc*100:.2f}%")
print(f"{'='*55}")