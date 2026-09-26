"""Fine-tune a ViT on a two-class ImageFolder dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import timm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", default="vit_small_patch16_224")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def atomic_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2))
    temp_path.replace(path)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.15)), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), normalize,
    ])
    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(args.data_dir / "val", transform=val_tf)
    if train_ds.classes != val_ds.classes or len(train_ds.classes) != 2:
        raise ValueError("train and val must contain the same two class folders")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    model = timm.create_model(args.model, pretrained=True, num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_accuracy, bad_epochs = -math.inf, 0
    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "lr"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = samples = 0
            for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * labels.size(0)
                samples += labels.size(0)
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            record = {"epoch": epoch, "train_loss": loss_sum / samples, "val_loss": val_loss,
                      "val_accuracy": val_accuracy, "lr": optimizer.param_groups[0]["lr"]}
            writer.writerow(record)
            history_file.flush()
            status = {**record, "state": "running", "classes": train_ds.classes,
                      "device": str(device), "updated_at": time.time()}
            if val_accuracy > best_accuracy:
                best_accuracy, bad_epochs = val_accuracy, 0
                torch.save({"model": args.model, "state_dict": model.state_dict(), "classes": train_ds.classes,
                            "image_size": args.image_size, "val_accuracy": val_accuracy}, output_dir / "best.pt")
                status["best_accuracy"] = best_accuracy
            else:
                bad_epochs += 1
                status["best_accuracy"] = best_accuracy
            atomic_json(output_dir / "metrics.json", status)
            scheduler.step()
            if bad_epochs >= args.patience:
                break
    atomic_json(output_dir / "metrics.json", {"state": "complete", "best_accuracy": best_accuracy,
                                                "classes": train_ds.classes, "device": str(device),
                                                "updated_at": time.time()})


if __name__ == "__main__":
    main()
