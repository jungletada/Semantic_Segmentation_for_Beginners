"""
Extension point for user-defined segmentation models.

To add a custom model later, define a builder function here and register it:

    def build_my_model(in_channels: int = 3, num_classes: int = 1) -> nn.Module:
        return MyModel(in_channels=in_channels, num_classes=num_classes)

    register_custom_model("my_model", build_my_model)

Then train with:

    python train.py --data_root data/cityscapes --model_source custom --model_name my_model

Builders must return an nn.Module whose forward pass accepts images with shape
(N, 3, H, W) and returns logits with shape (N, 1, H, W) for binary segmentation.
"""

from __future__ import annotations

from collections.abc import Callable
import re

import torch.nn as nn

ModelBuilder = Callable[..., nn.Module]
CUSTOM_MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def normalize_custom_model_name(name: str) -> str:
    """Normalize registry keys so CLI names and Python names match reliably."""
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip().lower()).strip("._-")
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
        available = sorted(CUSTOM_MODEL_BUILDERS)
        raise ValueError(
            f"Unknown custom model '{model_name}'. "
            f"Register it in networks/customize_model.py. "
            f"Available custom models: {available or 'none'}"
        )
    model = CUSTOM_MODEL_BUILDERS[key](
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"Custom model '{model_name}' builder must return an nn.Module.")
    return model
