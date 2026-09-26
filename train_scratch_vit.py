"""Train a compact Vision Transformer from random initialization on Cats-vs-Dogs."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class PetDataset(Dataset):
    """Load valid cat and dog images and provide a deterministic train or validation split."""

    def __init__(self, root: Path, train: bool, transform: transforms.Compose, seed: int = 42):
        """Build the image index.

        Args:
            root: Directory containing `Cat` and `Dog` folders.
            train: Select the 90 percent training split when true.
            transform: Image transformation applied when samples are loaded.
            seed: Seed used to make the split reproducible.
        """
        files = []
        for label, folder in enumerate(("Cat", "Dog")):
            for path in (root / folder).glob("*.*"):
                try:
                    with Image.open(path) as image:
                        image.verify()
                    files.append((path, label))
                except (UnidentifiedImageError, OSError):
                    continue
        random.Random(seed).shuffle(files)
        split = int(len(files) * 0.9)
        self.samples = files[:split] if train else files[split:]
        self.transform = transform

    def __len__(self):
        """Return the number of images in this split."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load and transform one image.

        Args:
            index: Position of the sample in this split.

        Returns:
            A transformed image tensor and its integer class label.
        """
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


class VisionTransformer(nn.Module):
    """Classify images with a ViT built from patch projection and Transformer blocks."""

    def __init__(self, image_size=224, patch_size=16, dim=384, depth=8, heads=6, num_classes=2):
        """Initialize the Vision Transformer.

        Args:
            image_size: Expected square image width and height.
            patch_size: Width and height of each nonoverlapping image patch.
            dim: Token embedding width.
            depth: Number of Transformer encoder blocks.
            heads: Attention heads in each encoder block.
            num_classes: Number of classification labels.
        """
        super().__init__()
        assert image_size % patch_size == 0
        patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, patches + 1, dim))
        self.dropout = nn.Dropout(0.1)
        block = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        """Initialize a supported neural-network module.

        Args:
            module: Module whose learnable parameters may be initialized.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """Produce class logits for a batch of images.

        Args:
            images: Image batch with shape `[batch, 3, image_size, image_size]`.

        Returns:
            Unnormalized class scores with shape `[batch, num_classes]`.
        """
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(images.size(0), -1, -1)
        tokens = self.dropout(torch.cat((cls, tokens), 1) + self.position)
        return self.head(self.norm(self.encoder(tokens)[:, 0]))


def parse_args() -> argparse.Namespace:
    """Read command-line training options.

    Returns:
        Parsed configuration for dataset location, output, and training size.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing Cat/ and Dog/")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scratch_vit"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    """Atomically write job metrics to JSON.

    Args:
        path: Output JSON file.
        value: Serializable metrics dictionary.
    """
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Measure classification accuracy on a validation data loader.

    Args:
        model: Model evaluated in inference mode.
        loader: Validation image batches.
        device: Device on which inference runs.

    Returns:
        Fraction of correctly classified validation images.
    """
    model.eval(); correct = total = 0
    for images, labels in loader:
        prediction = model(images.to(device, non_blocking=True)).argmax(1).cpu()
        correct += (prediction == labels).sum().item(); total += labels.size(0)
    return correct / total


def main() -> None:
    """Train a randomly initialized ViT and save the highest-accuracy checkpoint."""
    config = parse_args()
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 9), transforms.ToTensor(), norm, transforms.RandomErasing(p=.15)])
    val_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    train_data = PetDataset(config.data_root, True, train_tf)
    val_data = PetDataset(config.data_root, False, val_tf)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=pin, persistent_workers=True)
    val_loader = DataLoader(val_data, config.batch_size * 2, shuffle=False, num_workers=config.workers, pin_memory=pin, persistent_workers=True)
    model = VisionTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=4e-4, weight_decay=.05)
    schedule = CosineAnnealingLR(optimizer, config.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = 0.0
    with (config.output_dir / "history.csv").open("w", newline="") as history:
        writer = csv.DictWriter(history, fieldnames=("epoch", "train_loss", "val_accuracy", "lr")); writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            model.train(); loss_total = count = 0
            for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}"):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                loss_total += loss.item() * labels.size(0); count += labels.size(0)
            accuracy = validate(model, val_loader, device)
            record = {"epoch": epoch, "train_loss": loss_total / count, "val_accuracy": accuracy, "lr": optimizer.param_groups[0]["lr"]}
            writer.writerow(record); history.flush()
            if accuracy > best:
                best = accuracy
                torch.save({"model": "scratch_vit_384_8_6", "state_dict": model.state_dict(), "classes": ["cat", "dog"], "image_size": 224, "val_accuracy": best}, config.output_dir / "best.pt")
            save_json(config.output_dir / "metrics.json", {**record, "best_accuracy": best, "state": "running", "device": str(device), "updated_at": time.time()})
            schedule.step()
    save_json(config.output_dir / "metrics.json", {"state": "complete", "best_accuracy": best, "device": str(device), "updated_at": time.time()})


if __name__ == "__main__":
    main()
