import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class SceneAttentionPool(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_ch // 8, 1, 1, bias=False),
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x).view(x.size(0), -1), dim=1)
        w = w.view(x.size(0), 1, x.size(2), x.size(3))
        attended = (x * w).sum(dim=[2, 3])
        maxed = x.amax(dim=[2, 3])
        return torch.cat([attended, maxed], dim=1)


class SceneHead(nn.Module):
    def __init__(self, in_f: int, n_cls: int, drop: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(drop * 0.4),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.net(x)


class SmartVisionV9(nn.Module):
    def __init__(self, n_cls: int, drop: float = 0.4):
        super().__init__()
        resnet = models.resnet50(weights=None)

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = SceneAttentionPool(2048)
        self.head = SceneHead(2048, n_cls, drop)

    def forward(self, x):
        return self.head(self.pool(self.backbone(x)))


class ScenePipeline:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    DEFAULT_LABEL_NAMES = sorted([
        "bathroom", "beach", "bedroom", "green_outdoor", "hospital",
        "indoor_passage", "kitchen", "lecture_room", "library", "market",
        "mosque", "restaurant", "shopping_mall", "staircase", "street_road",
        "supermarket", "transport_hub", "waiting_room", "work_space",
    ])

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.device = device
        self.model_path = Path(model_path)

        ckpt = torch.load(self.model_path, map_location=device, weights_only=False)

        self.label_names = ckpt.get("label_names", self.DEFAULT_LABEL_NAMES)
        self.mean = ckpt.get("mean", self.MEAN)
        self.std = ckpt.get("std", self.STD)
        num_classes = int(ckpt.get("num_classes", len(self.label_names)))

        self.model = SmartVisionV9(n_cls=num_classes, drop=0.4).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        val_acc = ckpt.get("val_acc", None)
        extra = f" | val_acc={val_acc:.4f}" if isinstance(val_acc, (float, int)) else ""
        print(
            f"[Scene] Loaded SmartVisionX Scene v9 from {self.model_path} "
            f"| classes={num_classes} | device={device}{extra}"
        )

    def predict(self, bgr_frame):
        t0 = time.perf_counter()
        rgb = Image.fromarray(bgr_frame[:, :, ::-1])
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
        ms = (time.perf_counter() - t0) * 1000
        label = self.label_names[idx]
        conf = float(probs[idx].item())
        return label, round(conf, 3), round(ms, 1)
