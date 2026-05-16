"""
Phase 1 — Data Pipeline: CityscapesBinaryDataset
フェーズ1 — データパイプライン：CityscapesBinaryDataset

This module provides:
  CityscapesBinaryDataset  — torch Dataset for binary drivable-area segmentation
  get_dataloader()         — convenience factory that wraps the Dataset in a DataLoader

このモジュールが提供するもの：
  CityscapesBinaryDataset  — 二値走行可能領域セグメンテーション用のtorch Dataset
  get_dataloader()         — DatasetをDataLoaderでラップする便利なファクトリ関数

Binary label mapping / 二値ラベルマッピング:
  Cityscapes raw label ID 7 (road) → 1  (Drivable  / 走行可能)
  All other raw label IDs          → 0  (Background / 背景)

Expected directory layout / 期待されるディレクトリ構成:
  <root>/
    leftImg8bit/
      {split}/
        {city}/
          {city}_{seq}_{frame}_leftImg8bit.png
    gtFine/
      {split}/
        {city}/
          {city}_{seq}_{frame}_gtFine_labelIds.png

Usage / 使い方:
    from dataset import CityscapesBinaryDataset, get_dataloader
    from transforms import get_train_transform

    ds = CityscapesBinaryDataset(
        root="project/data/cityscapes",
        split="train",
        transform=get_train_transform(crop_size=512),
    )
    image, mask = ds[0]   # image: (3, 512, 512) float32
                           # mask:  (512, 512)    int64  — values 0 or 1

    loader = get_dataloader(root=..., split="train", transform=..., batch_size=8)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ── Constants ──────────────────────────────────────────────────────────────────

# Raw label ID for "road" in gtFine_labelIds.png (Cityscapes official definition).
# gtFine_labelIds.png における「道路」の生ラベルID（Cityscapes公式定義）。
#
# Why 7, not 0?
#   Cityscapes defines 34 raw label IDs (0–33).  The 19 "training IDs" (0–18)
#   are a separate mapping.  The gtFine_labelIds.png file stores raw IDs; road = 7.
#   なぜ0ではなく7か？
#   Cityscapesは34の生ラベルID（0〜33）を定義しています。19の「訓練ID」（0〜18）は
#   別のマッピングです。gtFine_labelIds.pngは生IDを格納しており、道路 = 7 です。
ROAD_LABEL_ID: int = 7


# ── Dataset ────────────────────────────────────────────────────────────────────

class CityscapesBinaryDataset(Dataset):
    """
    PyTorch Dataset for binary drivable-area segmentation on Cityscapes.
    Cityscapesを使った二値走行可能領域セグメンテーション用PyTorch Dataset。

    Every pixel with raw label ID == 7 (road) is mapped to class 1 (Drivable).
    All other pixels are mapped to class 0 (Background).
    生ラベルID == 7（道路）のすべてのピクセルをクラス1（走行可能）にマッピングします。
    その他のすべてのピクセルはクラス0（背景）にマッピングされます。

    Args:
        root      : Path to the Cityscapes root directory.
                    Cityscapesのルートディレクトリへのパス。
        split     : Dataset partition — "train", "val", or "test".
                    データセット分割 — "train"、"val"、または "test"。
        transform : An albumentations Compose pipeline that accepts keyword
                    arguments `image` (H×W×3 uint8 ndarray) and `mask`
                    (H×W uint8 ndarray) and returns a dict with the same keys.
                    `image`（H×W×3 uint8 ndarray）と `mask`（H×W uint8 ndarray）を
                    キーワード引数として受け取り、同じキーを持つdictを返す
                    Albumentations Composeパイプライン。
                    If None, raw numpy arrays are returned (useful for inspection).
                    Noneの場合、生のnumpy配列を返します（検査に便利）。
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[object] = None,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}, got '{split}'"
            )

        self.root      = Path(root)
        self.split     = split
        self.transform = transform

        self.img_paths, self.mask_paths = self._collect_pairs()

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No images found under {self.root / 'leftImg8bit' / split}.\n"
                "Make sure Cityscapes is extracted to the correct location.\n"
                "See project/topics.md §3.2 for download instructions.\n"
                f"{self.root / 'leftImg8bit' / split} 以下に画像が見つかりません。\n"
                "Cityscapesが正しい場所に解凍されているか確認してください。\n"
                "ダウンロード手順はproject/topics.md §3.2をご覧ください。"
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _collect_pairs(self) -> Tuple[List[Path], List[Path]]:
        """
        Walk leftImg8bit/{split}/ and find every image file.
        For each image, derive the path of its corresponding labelIds mask.

        leftImg8bit/{split}/ を探索してすべての画像ファイルを見つけます。
        各画像に対して、対応するlabelIdsマスクのパスを導出します。

        Naming convention / 命名規則:
          image : {city}_{seq:06d}_{frame:06d}_leftImg8bit.png
          mask  : {city}_{seq:06d}_{frame:06d}_gtFine_labelIds.png
        """
        img_root  = self.root / "leftImg8bit" / self.split
        mask_root = self.root / "gtFine"      / self.split

        img_paths  = sorted(img_root.rglob("*_leftImg8bit.png"))
        mask_paths = []

        for img_path in img_paths:
            city          = img_path.parent.name
            base_stem     = img_path.stem.replace("_leftImg8bit", "")
            mask_filename = f"{base_stem}_gtFine_labelIds.png"
            mask_path     = mask_root / city / mask_filename

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Annotation mask not found.\n"
                    f"  Image : {img_path}\n"
                    f"  Expected mask: {mask_path}\n"
                    "Ensure gtFine_trainvaltest.zip was extracted correctly."
                )
            mask_paths.append(mask_path)

        return img_paths, mask_paths

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        """
        Load one (image, binary_mask) pair.
        1つの（画像、二値マスク）ペアを読み込みます。

        Returns:
            If transform is set:
                image       : torch.Tensor  shape (3, H', W'), dtype float32
                binary_mask : torch.Tensor  shape (H', W'),    dtype int64
                              values: 0 (background) or 1 (road)
            If transform is None:
                image       : np.ndarray    shape (H, W, 3),   dtype uint8
                binary_mask : np.ndarray    shape (H, W),      dtype uint8
        """
        # Load RGB image as numpy array (H, W, 3) uint8
        # RGB画像をnumpy配列 (H, W, 3) uint8として読み込む
        image = np.array(
            Image.open(self.img_paths[idx]).convert("RGB"),
            dtype=np.uint8,
        )

        # Load raw label ID mask (H, W) int32
        # 生ラベルIDマスク (H, W) int32を読み込む
        raw_mask = np.array(
            Image.open(self.mask_paths[idx]),
            dtype=np.int32,
        )

        # Convert to binary: road label ID (7) → 1, everything else → 0
        # 二値に変換：道路ラベルID (7) → 1、それ以外 → 0
        binary_mask = (raw_mask == ROAD_LABEL_ID).astype(np.uint8)

        # Apply albumentations joint transform (spatial ops affect both)
        # Albumentationsの結合変換を適用（空間的操作は両方に影響）
        if self.transform is not None:
            augmented   = self.transform(image=image, mask=binary_mask)
            image       = augmented["image"]  # Tensor (3, H', W') float32
            binary_mask = augmented["mask"]   # Tensor (H', W')    uint8

        # Cast mask to int64 for CrossEntropyLoss compatibility
        # CrossEntropyLoss互換性のためにマスクをint64にキャスト
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = binary_mask.long()

        return image, binary_mask

    # ── Utility ────────────────────────────────────────────────────────────────

    def get_sample_info(self, idx: int) -> dict:
        """
        Return metadata for sample `idx` — useful for debugging and logging.
        サンプル `idx` のメタデータを返します — デバッグとロギングに便利です。
        """
        img_path  = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # Compute road pixel percentage on the full-resolution raw mask
        # フル解像度の生マスクで道路ピクセルのパーセンテージを計算
        raw_mask    = np.array(Image.open(mask_path), dtype=np.int32)
        binary_mask = (raw_mask == ROAD_LABEL_ID).astype(np.uint8)
        road_pct    = float(binary_mask.mean() * 100)

        return {
            "index"     : idx,
            "split"     : self.split,
            "city"      : img_path.parent.name,
            "stem"      : img_path.stem.replace("_leftImg8bit", ""),
            "image_path": str(img_path),
            "mask_path" : str(mask_path),
            "road_pct"  : road_pct,
        }

    def compute_class_weights(self, n_samples: int = 200) -> torch.Tensor:
        """
        Estimate inverse-frequency class weights over a random subset.
        Used to initialise a weighted loss function (helps with class imbalance).

        ランダムなサブセットに対して逆頻度クラス重みを推定します。
        重み付き損失関数の初期化に使用します（クラス不均衡の対処に役立ちます）。

        Args:
            n_samples: Number of images to sample for the estimate.
                       推定のためにサンプリングする画像数。

        Returns:
            Tensor of shape (2,): [weight_background, weight_road]
            形状 (2,) のテンソル：[背景の重み, 道路の重み]
        """
        indices    = np.random.choice(len(self), size=min(n_samples, len(self)), replace=False)
        road_total = 0
        bg_total   = 0

        for idx in indices:
            raw  = np.array(Image.open(self.mask_paths[idx]), dtype=np.int32)
            road = int((raw == ROAD_LABEL_ID).sum())
            bg   = raw.size - road
            road_total += road
            bg_total   += bg

        total = road_total + bg_total
        # Inverse frequency: rarer class gets higher weight
        # 逆頻度：より希少なクラスに高い重みを割り当てる
        weight_bg   = total / (2.0 * bg_total)
        weight_road = total / (2.0 * road_total)

        weights = torch.tensor([weight_bg, weight_road], dtype=torch.float32)
        print(f"Class weights — background: {weight_bg:.4f}  road: {weight_road:.4f}")
        return weights


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloader(
    root: str | Path,
    split: str,
    transform,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for the specified Cityscapes split.
    指定されたCityscapes分割のDataLoaderを構築します。

    Args:
        root        : Cityscapes root directory.
                      Cityscapesのルートディレクトリ。
        split       : "train", "val", or "test".
        transform   : Albumentations pipeline (from transforms.py).
                      transforms.py のAlbumentationsパイプライン。
        batch_size  : Samples per batch. バッチあたりのサンプル数。
        num_workers : Parallel CPU workers for data loading.
                      データ読み込み用の並列CPUワーカー数。
        pin_memory  : Pin CPU tensors for faster GPU transfer (set False if no GPU).
                      GPU転送を高速化するためにCPUテンソルをピン留め（GPU不使用時はFalse）。

    Returns:
        A configured torch.utils.data.DataLoader.
    """
    dataset = CityscapesBinaryDataset(root=root, split=split, transform=transform)

    shuffle   = (split == "train")   # shuffle only during training
    drop_last = (split == "train")   # drop incomplete final batch during training

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    print(
        f"DataLoader ready — split={split}  "
        f"samples={len(dataset)}  "
        f"batches={len(loader)}  "
        f"batch_size={batch_size}"
    )
    return loader
