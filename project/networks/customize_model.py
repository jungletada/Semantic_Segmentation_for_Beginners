"""
Minimal extension point for user-defined segmentation models.

This file intentionally contains a very small built-in custom model so the
custom-model path can be tested end to end without editing any other code:

    python train.py --data_root data/cityscapes \
        --model_source custom --model_name simple_custom

To add your own model later:

    1. Define an nn.Module or a builder function in this file.
    2. Register it with register_custom_model("your_model", builder).
    3. Run train/evaluate/visualize with --model_source custom --model_name your_model.

Every custom model must accept images shaped (N, 3, H, W) and return binary
segmentation logits shaped (N, 1, H, W). Do not apply sigmoid in the model;
the training and evaluation code handles that.
"""

from __future__ import annotations

from collections.abc import Callable
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

ModelBuilder = Callable[..., nn.Module]
CUSTOM_MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def normalize_custom_model_name(name: str) -> str:
    """Normalize registry keys so CLI names and Python names match reliably."""
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip().lower()).strip("._-")
    if not key:
        raise ValueError("Custom model name cannot be empty.")
    return key


def register_custom_model(name: str, builder: ModelBuilder) -> None:
    """Register a custom model builder under a CLI-friendly name."""
    if not callable(builder):
        raise TypeError("builder must be callable and return an nn.Module.")
    CUSTOM_MODEL_BUILDERS[normalize_custom_model_name(name)] = builder


def build_custom_model(
    model_name: str,
    in_channels: int = 3,
    num_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """Build a registered custom model."""
    key = normalize_custom_model_name(model_name)
    if key not in CUSTOM_MODEL_BUILDERS:
        available = ", ".join(sorted(CUSTOM_MODEL_BUILDERS)) or "none"
        raise ValueError(
            f"Unknown custom model '{model_name}'. "
            f"Register it in networks/customize_model.py. "
            f"Available custom models: {available}"
        )

    model = CUSTOM_MODEL_BUILDERS[key](
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"Custom model '{model_name}' builder must return an nn.Module.")
    return model


class SimpleCustomSegNet(nn.Module):
    """A tiny CNN segmentation model for testing the custom-model interface."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, width: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.encoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.decoder(x)


def build_simple_custom(
    in_channels: int = 3,
    num_classes: int = 1,
    **_: object,
) -> nn.Module:
    """Builder used by --model_source custom --model_name simple_custom."""
    return SimpleCustomSegNet(in_channels=in_channels, num_classes=num_classes)


register_custom_model("simple_custom", build_simple_custom)
register_custom_model("tiny_cnn", build_simple_custom)
