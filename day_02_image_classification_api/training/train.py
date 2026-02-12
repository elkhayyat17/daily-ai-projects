"""
Training Script
Fine-tunes ResNet50 on CIFAR-10 with mixed precision, OneCycleLR, and early stopping.
"""

import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from training.model import create_model
from training.transforms import get_train_transforms, get_val_transforms


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class Trainer:
    """
    Complete training pipeline with:
    - Mixed precision training (AMP)
    - OneCycleLR scheduler
    - Early stopping
    - Backbone unfreezing
    - Checkpoint saving
    - Training history logging
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
        logger.info(f"🖥️  Training on: {self.device}")

    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        raw_dir = config.DATA_DIR / "raw"

        # Load datasets with transforms
        train_dataset = datasets.CIFAR10(
            root=str(raw_dir), train=True, download=True,
            transform=get_train_transforms(),
        )
        val_full = datasets.CIFAR10(
            root=str(raw_dir), train=True, download=True,
            transform=get_val_transforms(),
        )

        # Create train/val split
        num_total = len(train_dataset)
        num_val = int(num_total * 0.1)
        num_train = num_total - num_val

        generator = torch.Generator().manual_seed(config.SEED)
        indices = torch.randperm(num_total, generator=generator).tolist()

        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_full, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        logger.info(f"📂 DataLoaders: Train={len(train_subset)} Val={len(val_subset)}")
        return train_loader, val_loader

    def _train_epoch(self, model, loader, criterion, optimizer, scaler, epoch):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            if config.USE_AMP and self.device == "cuda":
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.1f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _validate(self, model, loader, criterion, epoch):
        """Validate the model."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS} [Val]  ")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def _save_checkpoint(self, model, optimizer, epoch, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": {
                "model_name": config.MODEL_NAME,
                "num_classes": config.NUM_CLASSES,
                "class_names": config.CLASS_NAMES,
                "input_size": config.INPUT_SIZE,
            },
        }

        ckpt_dir = config.MODEL_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            path = config.MODEL_DIR / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(f"💾 Best model saved: val_acc={val_acc:.2f}%")

        path = ckpt_dir / f"epoch_{epoch + 1}.pth"
        torch.save(checkpoint, path)

    def train(self):
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info("🚀 Starting Training Pipeline")
        logger.info("=" * 60)

        # Set seed
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(config.SEED)

        # Create data
        train_loader, val_loader = self._create_dataloaders()

        # Create model
        model = create_model(self.device)

        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer — different LR for backbone vs head
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config.BACKBONE_LR},
            {"params": head_params, "lr": config.LEARNING_RATE},
        ], weight_decay=config.WEIGHT_DECAY)

        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config.BACKBONE_LR * 10, config.LEARNING_RATE],
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        # AMP scaler
        scaler = GradScaler() if config.USE_AMP and self.device == "cuda" else None

        # Early stopping
        early_stopping = EarlyStopping(patience=5)

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(config.NUM_EPOCHS):
            # Unfreeze backbone after specified epoch
            if epoch == config.UNFREEZE_AFTER_EPOCH and config.FREEZE_BACKBONE:
                model.unfreeze_backbone()
                # Re-create optimizer with all params
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.BACKBONE_LR,
                    weight_decay=config.WEIGHT_DECAY,
                )

            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler, epoch
            )

            # Validate
            val_loss, val_acc = self._validate(model, val_loader, criterion, epoch)

            # Log
            current_lr = optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            logger.info(
                f"📊 Epoch {epoch + 1}/{config.NUM_EPOCHS} — "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            # Save best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            self._save_checkpoint(model, optimizer, epoch, val_acc, is_best)

            # Early stopping
            if early_stopping(val_loss):
                logger.info(f"⏹️  Early stopping triggered at epoch {epoch + 1}")
                break

        # Save training history
        elapsed = time.time() - start_time
        history_path = config.LOGS_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"✅ Training Complete in {elapsed / 60:.1f} minutes!")
        logger.info(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
        logger.info(f"💾 Best model saved to {config.MODEL_DIR / 'best_model.pth'}")
        logger.info("=" * 60)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
