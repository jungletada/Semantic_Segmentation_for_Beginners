"""
Phase 1 — Data Pipeline: Albumentations transform pipelines
フェーズ1 — データパイプライン：Albumentationsによる変換パイプライン

Three pipelines are provided:
  get_train_transform()     — for training (spatial + photometric augmentation)
  get_val_transform()       — for validation / test (centre-crop + normalise only)
  get_edge_case_transform() — for failure-mode analysis (rain / shadow / fog / night)

3種類のパイプラインを提供します：
  get_train_transform()     — 訓練用（空間的・光度的拡張あり）
  get_val_transform()       — 検証・テスト用（中央クロップ＋正規化のみ）
  get_edge_case_transform() — 失敗モード分析用（雨・影・霧・夜間）

Usage / 使い方:
    from transforms import get_train_transform, get_val_transform
    transform = get_train_transform(crop_size=512)
    augmented = transform(image=img_np, mask=mask_np)
    image, mask = augmented["image"], augmented["mask"]
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — the ResNet encoder backbone was pre-trained with these values.
# ResNetエンコーダバックボーンはこれらの値で事前学習されています。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(crop_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline.
    訓練用データ拡張パイプライン。

    Spatial transforms (applied identically to image AND mask):
      - RandomCrop      : keeps spatial size manageable on GPU memory
      - HorizontalFlip  : left/right symmetry of driving scenes

    Photometric transforms (applied to image only, mask is unchanged):
      - RandomBrightnessContrast : robustness to lighting variation
      - HueSaturationValue       : robustness to colour shifts

    Edge-case augmentations are listed but commented out by default.
    Uncomment them in Phase 5 to stress-test the trained model.
    エッジケース拡張はデフォルトでコメントアウトされています。
    フェーズ5でコメントを外し、訓練済みモデルをストレステストしてください。

    Args:
        crop_size: Side length (px) of the random square crop. / ランダム正方形クロップの辺長（px）。

    Returns:
        An albumentations Compose pipeline. / Albumentations Composeパイプライン。
    """
    return A.Compose([
        # ── Spatial augmentations ──────────────────────────────────────────────
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.HorizontalFlip(p=0.5),

        # ── Photometric augmentations ──────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.3,
        ),

        # ── Edge-case augmentations (Challenge A from topics.md) ───────────────
        # Uncomment any of these to simulate adverse weather / lighting.
        # 以下のいずれかをコメント解除して悪天候・照明条件をシミュレートします。

        # Rain simulation (addresses: wet road reflection, texture change)
        # 雨のシミュレーション（濡れた路面の反射・テクスチャ変化への対処）
        # A.RandomRain(
        #     slant_lower=-10, slant_upper=10,
        #     drop_length=20, drop_width=1, drop_color=(200, 200, 200),
        #     blur_value=3, brightness_coefficient=0.7, p=0.2,
        # ),

        # Shadow simulation (addresses: shadow bands mistaken for obstacles)
        # 影のシミュレーション（障害物と誤認される影への対処）
        # A.RandomShadow(
        #     shadow_roi=(0, 0.5, 1, 1),
        #     num_shadows_lower=1, num_shadows_upper=2,
        #     shadow_dimension=5, p=0.3,
        # ),

        # Fog simulation (addresses: reduced contrast / low visibility)
        # 霧のシミュレーション（コントラスト低下・視界不良への対処）
        # A.RandomFog(
        #     fog_coef_lower=0.1, fog_coef_upper=0.3,
        #     alpha_coef=0.1, p=0.2,
        # ),

        # ── Normalisation + tensor conversion ─────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),  # image (H,W,C) uint8 → tensor (C,H,W) float32
                       # mask  (H,W)         → tensor (H,W)   uint8
    ])


def get_val_transform(crop_size: int = 512) -> A.Compose:
    """
    Validation / test pipeline (no random augmentation).
    検証・テスト用パイプライン（ランダム拡張なし）。

    A centre-crop ensures a fixed spatial size without information loss at borders.
    中央クロップにより、境界部分の情報を失わずに固定の空間サイズを確保します。

    Args:
        crop_size: Side length (px) of the centre crop. / 中央クロップの辺長（px）。
    """
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_edge_case_transform(
    crop_size: int = 512,
    condition: str = "rain",
) -> A.Compose:
    """
    Deterministic single-condition pipeline for failure-mode analysis (Phase 5).
    フェーズ5の失敗モード分析用の決定論的・単一条件パイプライン。

    Args:
        crop_size : Side length (px) of the centre crop.
        condition : One of "rain", "shadow", "fog", "night".
                    "rain"（雨）, "shadow"（影）, "fog"（霧）, "night"（夜間）のいずれか。

    Returns:
        A Compose pipeline with the chosen adverse condition applied at p=1.0.
        選択した悪条件をp=1.0で適用するComposeパイプライン。
    """
    _conditions = {
        "rain": A.RandomRain(
            slant_lower=-10, slant_upper=10,
            drop_length=25, drop_width=1, drop_color=(200, 200, 200),
            blur_value=3, brightness_coefficient=0.7, p=1.0,
        ),
        "shadow": A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1),
            num_shadows_lower=2, num_shadows_upper=4,
            shadow_dimension=6, p=1.0,
        ),
        "fog": A.RandomFog(
            fog_coef_lower=0.3, fog_coef_upper=0.5,
            alpha_coef=0.15, p=1.0,
        ),
        "night": A.RandomBrightnessContrast(
            brightness_limit=(-0.6, -0.4),
            contrast_limit=(-0.2, 0.0),
            p=1.0,
        ),
    }

    if condition not in _conditions:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Choose from: {list(_conditions.keys())}"
        )

    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        _conditions[condition],
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
