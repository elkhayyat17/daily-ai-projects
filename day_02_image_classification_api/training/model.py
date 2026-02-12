"""
ResNet50 Transfer Learning Model
Defines the classifier with a frozen/unfrozen backbone.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class ImageClassifier(nn.Module):
    """
    ResNet50-based image classifier with transfer learning.

    Architecture:
        - ResNet50 backbone (pretrained on ImageNet)
        - Custom classification head with dropout
        - Configurable backbone freezing for transfer learning

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers initially
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        freeze_backbone: bool = config.FREEZE_BACKBONE,
    ):
        super().__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Get the feature dimension from the backbone
        in_features = self.backbone.fc.in_features

        # Replace the classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

        self.num_classes = num_classes
        logger.info(f"🤖 ImageClassifier initialized: {config.MODEL_NAME}")
        logger.info(f"   Classes: {num_classes} | Pretrained: {pretrained} | Frozen: {freeze_backbone}")
        logger.info(f"   Parameters: {self.count_parameters():,} total, {self.count_trainable():,} trainable")

    def _freeze_backbone(self):
        """Freeze all backbone layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info(f"🔓 Backbone unfrozen — {self.count_trainable():,} trainable params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device: str = "cpu") -> ImageClassifier:
    """Factory function to create and initialize the model."""
    model = ImageClassifier()
    model = model.to(device)
    return model


def load_model(checkpoint_path: str, device: str = "cpu") -> ImageClassifier:
    """Load a trained model from checkpoint."""
    model = ImageClassifier(freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info(f"📦 Model loaded from {checkpoint_path}")
    return model
