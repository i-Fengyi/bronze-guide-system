"""Fine-tune MobileNetV3 for bronze vessel type classification."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".model_cache" / "torch"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision import Letterbox, build_eval_transform, build_mobilenet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("models/vessel_classifier"))
    parser.add_argument("--architecture", choices=["mobilenet_v3_small", "mobilenet_v3_large"], default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--early-stopping", type=int, default=8)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    fill = (124, 116, 104)
    train_tf = transforms.Compose(
        [
            Letterbox(args.image_size, fill),
            transforms.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.9, 1.1), fill=fill),
            transforms.RandomPerspective(distortion_scale=0.08, p=0.15, fill=fill),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.12, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.12),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.18, scale=(0.02, 0.08), ratio=(0.4, 2.5), value="random"),
        ]
    )
    eval_tf = build_eval_transform(args.image_size)
    train_dir = args.data / "train"
    val_dir = args.data / "val"
    test_dir = args.data / "test"
    if not train_dir.exists():
        raise SystemExit("未找到数据集，请先运行 python scripts/prepare_dataset.py")
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf) if contains_images(val_dir) else None
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf) if contains_images(test_dir) else None
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin) if val_ds else None
    try:
        model = build_mobilenet(args.architecture, len(train_ds.classes), pretrained=True).to(device)
    except Exception as exc:
        print(f"pretrained weights unavailable, using random initialization: {exc}")
        model = build_mobilenet(args.architecture, len(train_ds.classes), pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.03)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_acc = -1.0
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = correct = running = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=pin); labels = labels.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images); loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item() * labels.size(0); correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
        train_acc = correct / max(1, total)
        train_loss = running / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion) if val_loader else (train_loss, train_acc)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc, "learning_rate": lr})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.2e}")
        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
        if improved:
            best_acc = val_acc; best_loss = val_loss; stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "architecture": args.architecture,
                    "image_size": args.image_size,
                    "best_val_accuracy": best_acc,
                    "best_val_loss": best_loss,
                    "epoch": epoch,
                },
                args.output / "best.pt",
            )
        else:
            stale_epochs += 1
        scheduler.step()
        (args.output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if stale_epochs >= args.early_stopping:
            print(f"early_stopping epoch={epoch} best_val_acc={best_acc:.4f}")
            break
    if test_ds:
        checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state"])
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
        targets, predictions = predict_all(model, test_loader, device)
        report = classification_report(targets, predictions, target_names=test_ds.classes, output_dict=True, zero_division=0)
        matrix = confusion_matrix(targets, predictions, labels=list(range(len(test_ds.classes))))
        (args.output / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savetxt(args.output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    (args.output / "classes.json").write_text(json.dumps(train_ds.class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    if loader is None: return 0.0, 0.0
    model.eval(); total = correct = 0; running = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device); labels = labels.to(device)
            logits = model(images); running += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
    return running / max(1, total), correct / max(1, total)


def predict_all(model, loader, device) -> tuple[list[int], list[int]]:
    targets: list[int] = []; predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist()); targets.extend(labels.tolist())
    return targets, predictions


def contains_images(path: Path) -> bool:
    return path.exists() and any(item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for item in path.rglob("*"))


if __name__ == "__main__":
    main()
