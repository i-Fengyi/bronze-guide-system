from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VESSEL_TYPE_NAMES = {
    "ding": "鼎",
    "gui": "簋",
    "jue": "爵",
    "gu": "觚",
    "zun": "尊",
    "you": "卣",
    "he": "盉",
    "hu": "壶",
}


class Letterbox:
    def __init__(self, size: int = 224, fill: tuple[int, int, int] = (124, 116, 104)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


def build_eval_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            Letterbox(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_mobilenet(architecture: str, num_classes: int, pretrained: bool = False):
    from torch import nn
    from torchvision.models import (
        MobileNet_V3_Large_Weights,
        MobileNet_V3_Small_Weights,
        mobilenet_v3_large,
        mobilenet_v3_small,
    )

    if architecture == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    elif architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"不支持的模型结构：{architecture}")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


class VesselClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._transform = None
        self._classes: list[str] = []
        self.device = "cpu"
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> list[str]:
        return self._classes.copy()

    def _load(self) -> None:
        if self._model is not None or self.load_error is not None:
            return
        if not self.checkpoint_path.exists():
            self.load_error = f"尚未找到训练权重：{self.checkpoint_path}"
            return
        try:
            import torch
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            class_to_idx = checkpoint["class_to_idx"]
            self._classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
            self._model = build_mobilenet(checkpoint.get("architecture", "mobilenet_v3_small"), len(self._classes))
            self._model.load_state_dict(checkpoint["model_state"])
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device).eval()
            self._transform = build_eval_transform(checkpoint.get("image_size", 224))
            self._torch = torch
        except Exception as exc:
            self._model = None
            self.load_error = str(exc)

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
        self._load()
        if self._model is None:
            return []
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            probabilities = self._model(tensor).softmax(dim=1)[0]
        values, indices = probabilities.topk(min(top_k, len(self._classes)))
        return [
            {
                "label": self._classes[index],
                "name": VESSEL_TYPE_NAMES.get(self._classes[index], self._classes[index]),
                "confidence": round(float(value), 4),
            }
            for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
        ]


class GalleryEncoder:
    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        self._model = None
        self._preprocess = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, ResNetModel

            if not self.weights_path.exists():
                raise RuntimeError(f"缺少展品特征模型目录：{self.weights_path}")
            self._preprocess = AutoImageProcessor.from_pretrained(str(self.weights_path), local_files_only=True)
            self._model = ResNetModel.from_pretrained(str(self.weights_path), local_files_only=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device).eval()
            self._torch = torch
        except Exception as exc:
            raise RuntimeError(f"无法加载展品特征模型：{exc}") from exc

    def encode(self, image: Image.Image) -> np.ndarray:
        self._load()
        inputs = {key: value.to(self.device) for key, value in self._preprocess(images=image.convert("RGB"), return_tensors="pt").items()}
        with self._torch.inference_mode():
            vector = self._model(**inputs).pooler_output.flatten(1)
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return vector[0].cpu().numpy().astype("float32")
