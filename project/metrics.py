"""
Phase 3 — Evaluation metrics for binary segmentation
フェーズ3 — 二値セグメンテーションの評価メトリクス

Three metrics are implemented (see topics.md §6 Phase 4):
  binary_iou()        — IoU / Jaccard index      (primary metric)
  binary_dice()       — Dice coefficient / F1    (closely related to IoU)
  pixel_accuracy()    — fraction of correct pixels (can be misleading for imbalanced data)

3つのメトリクスを実装しています（topics.md §6 フェーズ4参照）：
  binary_iou()        — IoU / Jaccard指数         （主要メトリクス）
  binary_dice()       — Dice係数 / F1スコア       （IoUに密接に関連）
  pixel_accuracy()    — 正解ピクセルの割合        （クラス不均衡データには注意）

Formulas (TP = True Positive, FP = False Positive, FN = False Negative):
  IoU  = TP / (TP + FP + FN)
  Dice = 2·TP / (2·TP + FP + FN)
  PA   = (TP + TN) / N

Usage / 使い方:
    from metrics import binary_iou, binary_dice, pixel_accuracy, MetricTracker

    # Per-batch (inside the val loop)
    iou  = binary_iou(logits, masks)
    dice = binary_dice(logits, masks)
    pa   = pixel_accuracy(logits, masks)

    # Epoch-level accumulation
    tracker = MetricTracker()
    for logits, masks in val_loader:
        tracker.update({"iou": binary_iou(logits, masks),
                        "dice": binary_dice(logits, masks)},
                       n=masks.shape[0])
    print(tracker)   # iou=0.9612  dice=0.9803
"""

from __future__ import annotations

import torch


# ── Internal helper ────────────────────────────────────────────────────────────

def _threshold(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    logits (N, 1, H, W) float32  →  binary predictions (N, 1, H, W) bool

    Two steps:
      1. sigmoid squashes logits into [0, 1] probabilities.
      2. threshold converts probabilities to hard 0/1 labels.

    2つのステップ：
      1. sigmoidがロジットを [0, 1] の確率に圧縮します。
      2. 閾値が確率をハードな 0/1 ラベルに変換します。
    """
    return torch.sigmoid(logits) >= threshold


def _align(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Produce flattened boolean prediction and target tensors of the same shape.

    logits : (N, 1, H, W) float32
    targets: (N, H, W)    int64   OR  (N, 1, H, W) int64

    Returns (preds_flat, targets_flat) — both 1-D bool tensors.
    """
    preds = _threshold(logits, threshold)           # (N, 1, H, W) bool

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)              # (N, 1, H, W)

    return preds.reshape(-1), targets.bool().reshape(-1)


# ── Public metric functions ────────────────────────────────────────────────────

def binary_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Intersection over Union (Jaccard index) for binary segmentation.
    二値セグメンテーション用のIoU（Jaccard指数）。

    IoU = TP / (TP + FP + FN)

    A value of 1.0 means perfect overlap; 0.0 means no overlap at all.
    値が1.0は完全な重なり、0.0は全く重なりがないことを意味します。

    Args:
        logits   : (N, 1, H, W) float32 — raw model output (before sigmoid).
        targets  : (N, H, W) int64      — binary ground truth (0 or 1).
        threshold: Sigmoid threshold for converting probabilities to labels.
        eps      : Small constant to prevent division by zero.

    Returns:
        Scalar float in [0, 1].
    """
    preds, targets_flat = _align(logits, targets, threshold)

    intersection = (preds & targets_flat).sum().float()
    union        = (preds | targets_flat).sum().float()

    return ((intersection + eps) / (union + eps)).item()


def binary_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Dice coefficient (F1 score) for binary segmentation.
    二値セグメンテーション用のDice係数（F1スコア）。

    Dice = 2·TP / (2·TP + FP + FN)

    Relationship to IoU:  Dice = 2·IoU / (1 + IoU)
    The Dice score is always ≥ IoU for any non-trivial prediction.
    IoUとの関係：Dice = 2·IoU / (1 + IoU)
    Dice スコアは常に IoU 以上です（自明でない予測では）。

    Args:
        logits   : (N, 1, H, W) float32 — raw model output.
        targets  : (N, H, W) int64      — binary ground truth.
        threshold: Decision boundary.
        eps      : Division guard.

    Returns:
        Scalar float in [0, 1].
    """
    preds, targets_flat = _align(logits, targets, threshold)

    preds_f   = preds.float()
    targets_f = targets_flat.float()

    intersection = (preds_f * targets_f).sum()
    return ((2.0 * intersection + eps) / (preds_f.sum() + targets_f.sum() + eps)).item()


def pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Fraction of pixels classified correctly (both road and background).
    正しく分類されたピクセルの割合（道路と背景の両方）。

    PA = (TP + TN) / N

    Warning: PA can be misleading for imbalanced datasets.
    If road covers only 15% of pixels, a model that always predicts
    "background" achieves 85% PA despite being completely useless.
    Use IoU as the primary metric.

    注意：PAはクラス不均衡データセットでは誤解を招く可能性があります。
    道路がピクセルの15%しかない場合、常に「背景」を予測するモデルは
    完全に役に立たないにもかかわらず85%のPAを達成します。
    IoUを主要メトリクスとして使用してください。

    Args:
        logits  : (N, 1, H, W) float32.
        targets : (N, H, W) int64.

    Returns:
        Scalar float in [0, 1].
    """
    preds, targets_flat = _align(logits, targets, threshold)
    correct = (preds == targets_flat).sum().float()
    return (correct / targets_flat.numel()).item()


# ── Epoch-level accumulator ────────────────────────────────────────────────────

class MetricTracker:
    """
    Accumulates weighted metric values over an epoch, then returns the mean.
    エポック全体で重み付きメトリクス値を累積し、平均を返します。

    Use this inside the training and validation loops to track metrics
    across many mini-batches without storing all predictions in memory.

    多くのミニバッチにわたってメトリクスを追跡するために、訓練・検証ループ内で
    使用します。すべての予測をメモリに保持する必要がありません。

    Example / 例:
        tracker = MetricTracker()
        for images, masks in val_loader:
            logits = model(images)
            bs = images.shape[0]
            tracker.update({
                "iou" : binary_iou(logits, masks),
                "dice": binary_dice(logits, masks),
                "pa"  : pixel_accuracy(logits, masks),
            }, n=bs)
        means = tracker.mean()   # {"iou": 0.961, "dice": 0.980, "pa": 0.987}
        print(tracker)           # iou=0.9612  dice=0.9803  pa=0.9872
    """

    def __init__(self) -> None:
        self._sums:   dict[str, float] = {}
        self._counts: dict[str, int]   = {}

    def reset(self) -> None:
        """Clear all accumulated values. 累積値をすべてクリアします。"""
        self._sums.clear()
        self._counts.clear()

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        """
        Add metrics from one batch.
        1バッチのメトリクスを追加します。

        Args:
            metrics: Dict mapping metric name → scalar value for this batch.
            n      : Batch size (used for weighted averaging).
                     バッチサイズ（加重平均に使用）。
        """
        for key, value in metrics.items():
            self._sums[key]   = self._sums.get(key, 0.0)   + float(value) * n
            self._counts[key] = self._counts.get(key, 0)   + n

    def mean(self) -> dict[str, float]:
        """
        Return the weighted mean for each tracked metric.
        各追跡メトリクスの加重平均を返します。
        """
        if not self._sums:
            return {}
        return {key: self._sums[key] / self._counts[key] for key in self._sums}

    def __str__(self) -> str:
        return "  ".join(f"{k}={v:.4f}" for k, v in self.mean().items())

    def __repr__(self) -> str:
        return f"MetricTracker({self.mean()})"
