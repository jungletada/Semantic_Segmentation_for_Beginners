"""
Phase 4 — Evaluation: full evaluation of a trained segmentation model
フェーズ4 — 評価：訓練済みセグメンテーションモデルの完全な評価

Runs the best checkpoint against the val split and reports:
  • IoU / Dice / Pixel Accuracy  — overall and per-image statistics
  • Confusion matrix              — TP / FP / FN / TN counts
  • Threshold sweep               — IoU vs. threshold τ ∈ [0.1, 0.9]
  • Per-image ranking             — best / worst N samples by IoU

ベストチェックポイントをval分割に対して実行し、以下を報告します：
  • IoU / Dice / ピクセル精度  — 全体および画像ごとの統計
  • 混同行列                    — TP / FP / FN / TN カウント
  • 閾値スイープ               — IoU vs. 閾値 τ ∈ [0.1, 0.9]
  • 画像ごとのランキング       — IoU でのベスト/ワースト N サンプル

Usage / 使い方:
    # Evaluate best checkpoint (default)
    python evaluate.py --data_root project/data/cityscapes --checkpoint checkpoints/best.pth

    # Change threshold and output directory
    python evaluate.py \\
        --data_root project/data/cityscapes \\
        --checkpoint checkpoints/best.pth \\
        --threshold 0.5 \\
        --out_dir evaluation_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_factory.dataset import CityscapesBinaryDataset, get_dataloader
from utils.metrics import MetricTracker, binary_dice, binary_iou, pixel_accuracy
from networks.model import build_model
from data_factory.transforms import get_val_transform


# ── Confusion-matrix helper ────────────────────────────────────────────────────

def compute_confusion(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    """
    Compute TP, FP, FN, TN for one batch.
    1バッチの TP、FP、FN、TN を計算します。

    Args:
        logits  : (N, 1, H, W) float32
        targets : (N, H, W)    int64

    Returns:
        {"tp": int, "fp": int, "fn": int, "tn": int}
    """
    preds = (torch.sigmoid(logits) >= threshold).squeeze(1)   # (N, H, W) bool
    gt    = targets.bool()

    tp = int((preds &  gt).sum())
    fp = int((preds & ~gt).sum())
    fn = int((~preds &  gt).sum())
    tn = int((~preds & ~gt).sum())

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ── Per-image evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_image(
    model    : nn.Module,
    dataset  : CityscapesBinaryDataset,
    device   : torch.device,
    threshold: float = 0.5,
    crop_size: int   = 512,
) -> list[dict]:
    """
    Compute IoU / Dice / PA individually for every image in the dataset.
    データセット内の各画像について IoU / Dice / PA を個別に計算します。

    Returns a list of dicts sorted by IoU (ascending — worst first).
    IoU の昇順（最悪が先）でソートされた dict のリストを返します。
    """
    model.eval()
    records: list[dict] = []

    for idx in range(len(dataset)):
        img_tensor, mask_tensor = dataset[idx]

        # Add batch dimension / バッチ次元を追加
        images = img_tensor.unsqueeze(0).to(device)   # (1, 3, H, W)
        masks  = mask_tensor.unsqueeze(0).to(device)  # (1, H, W)

        logits = model(images)                         # (1, 1, H, W)

        iou  = binary_iou(logits,     masks, threshold)
        dice = binary_dice(logits,    masks, threshold)
        pa   = pixel_accuracy(logits, masks, threshold)

        info = dataset.get_sample_info(idx)

        records.append({
            "idx"     : idx,
            "city"    : info["city"],
            "stem"    : info["stem"],
            "iou"     : iou,
            "dice"    : dice,
            "pa"      : pa,
            "road_pct": info["road_pct"],
        })

        if (idx + 1) % 50 == 0:
            print(f"   Evaluated {idx + 1}/{len(dataset)} images...")

    records.sort(key=lambda r: r["iou"])
    return records


# ── Threshold sweep ────────────────────────────────────────────────────────────

@torch.no_grad()
def threshold_sweep(
    model    : nn.Module,
    loader   : torch.utils.data.DataLoader,
    device   : torch.device,
    thresholds: list[float] | None = None,
) -> dict[float, float]:
    """
    Compute mean val IoU for a range of sigmoid thresholds.
    シグモイド閾値の範囲で平均 val IoU を計算します。

    Returns:
        Dict mapping threshold → mean IoU.
        閾値 → 平均 IoU の対応辞書。
    """
    if thresholds is None:
        thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]

    model.eval()
    results: dict[float, float] = {}

    for tau in thresholds:
        tracker = MetricTracker()
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            logits = model(images)
            tracker.update({"iou": binary_iou(logits, masks, tau)},
                           n=images.shape[0])
        results[tau] = tracker.mean()["iou"]
        print(f"   τ={tau:.1f}  IoU={results[tau]:.4f}")

    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: dict[str, int], save_path: Path) -> None:
    """
    Render a 2×2 confusion matrix heatmap.
    2×2 混同行列のヒートマップを描画します。
    """
    mat   = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=np.int64)
    total = mat.sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            pct = mat[i, j] / total * 100
            ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if mat[i, j] > mat.max() * 0.5 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: BG / 背景", "Pred: Road / 道路"])
    ax.set_yticklabels(["GT: BG / 背景",   "GT: Road / 道路"])
    ax.set_title("Confusion Matrix / 混同行列", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   Saved → {save_path}")


def plot_threshold_sweep(sweep: dict[float, float], save_path: Path) -> None:
    """
    Line plot of IoU vs. sigmoid threshold.
    IoU vs. シグモイド閾値の折れ線グラフ。
    """
    taus = sorted(sweep.keys())
    ious = [sweep[t] for t in taus]
    best_tau = taus[int(np.argmax(ious))]
    best_iou = sweep[best_tau]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, ious, marker="o", color="#2980b9", linewidth=2)
    ax.axvline(best_tau, color="#e74c3c", linewidth=1.5, linestyle="--",
               label=f"Best τ={best_tau:.1f}  IoU={best_iou:.4f}")
    ax.set_xlabel("Sigmoid threshold τ / シグモイド閾値 τ")
    ax.set_ylabel("Mean val IoU")
    ax.set_title("IoU vs. Threshold / IoU vs. 閾値", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(min(taus) - 0.05, max(taus) + 0.05)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   Saved → {save_path}")


def plot_iou_distribution(records: list[dict], save_path: Path) -> None:
    """
    Histogram of per-image IoU values.
    画像ごとの IoU 値のヒストグラム。
    """
    ious = [r["iou"] for r in records]
    mean_iou = np.mean(ious)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ious, bins=30, color="#27ae60", edgecolor="white", alpha=0.85)
    ax.axvline(mean_iou, color="#e74c3c", linewidth=2,
               label=f"Mean IoU / 平均: {mean_iou:.4f}")
    ax.set_xlabel("Per-image IoU / 画像ごとの IoU")
    ax.set_ylabel("Count / 枚数")
    ax.set_title("IoU Distribution — Val Split / IoU 分布 — 検証分割", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   Saved → {save_path}")


# ── Main evaluation function ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    """
    Full evaluation pipeline.
    完全な評価パイプライン。
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice / デバイス: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    arch   = config.get("arch",     args.arch)
    enc    = config.get("encoder",  args.encoder)

    print(f"  Checkpoint epoch : {ckpt.get('epoch', 'unknown')}")
    print(f"  Checkpoint val IoU: {ckpt.get('val_iou', 'unknown'):.4f}")
    print(f"  Architecture     : {arch} / {enc}")

    # ── Build model ────────────────────────────────────────────────────────────
    model = build_model(arch=arch, encoder_name=enc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── DataLoader (val split) ────────────────────────────────────────────────
    crop_size  = config.get("crop_size", args.crop_size)
    val_transform = get_val_transform(crop_size)
    val_loader = get_dataloader(
        root        = args.data_root,
        split       = "val",
        transform   = val_transform,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )
    val_dataset = CityscapesBinaryDataset(
        root      = args.data_root,
        split     = "val",
        transform = val_transform,
    )

    # ── 1. Overall metrics ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("1. Overall Metrics / 全体メトリクス")
    print("=" * 55)

    tracker = MetricTracker()
    cm_total: dict[str, int] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        logits = model(images)
        bs = images.shape[0]

        tracker.update({
            "iou" : binary_iou(logits,     masks, args.threshold),
            "dice": binary_dice(logits,    masks, args.threshold),
            "pa"  : pixel_accuracy(logits, masks, args.threshold),
        }, n=bs)

        cm = compute_confusion(logits, masks, args.threshold)
        for k in cm_total:
            cm_total[k] += cm[k]

    means = tracker.mean()
    print(f"   IoU  (Jaccard)      : {means['iou']:.4f}")
    print(f"   Dice (F1)           : {means['dice']:.4f}")
    print(f"   Pixel Accuracy (PA) : {means['pa']:.4f}")
    print()
    print("   Confusion matrix totals / 混同行列合計:")
    print(f"     TP={cm_total['tp']:>10,}   FP={cm_total['fp']:>10,}")
    print(f"     FN={cm_total['fn']:>10,}   TN={cm_total['tn']:>10,}")

    # ── 2. Threshold sweep ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("2. Threshold Sweep / 閾値スイープ")
    print("=" * 55)
    sweep = threshold_sweep(model, val_loader, device)
    best_tau = max(sweep, key=sweep.get)
    print(f"\n   Best threshold: τ={best_tau:.1f}  IoU={sweep[best_tau]:.4f}")

    # ── 3. Per-image statistics ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("3. Per-Image Statistics / 画像ごとの統計")
    print("=" * 55)
    print(f"   Evaluating {len(val_dataset)} images individually...")
    records = evaluate_per_image(model, val_dataset, device, args.threshold, crop_size)

    ious  = [r["iou"]  for r in records]
    dices = [r["dice"] for r in records]
    pas   = [r["pa"]   for r in records]

    print(f"\n   IoU  — mean={np.mean(ious):.4f}  std={np.std(ious):.4f}"
          f"  min={np.min(ious):.4f}  max={np.max(ious):.4f}")
    print(f"   Dice — mean={np.mean(dices):.4f}  std={np.std(dices):.4f}")
    print(f"   PA   — mean={np.mean(pas):.4f}  std={np.std(pas):.4f}")

    n_show = min(args.top_n, len(records))
    print(f"\n   Worst {n_show} samples by IoU / IoU が最も低い {n_show} サンプル:")
    for r in records[:n_show]:
        print(f"     [{r['idx']:4d}] {r['city']:20s}  iou={r['iou']:.4f}  "
              f"road={r['road_pct']:.1f}%  {r['stem']}")

    print(f"\n   Best {n_show} samples by IoU / IoU が最も高い {n_show} サンプル:")
    for r in records[-n_show:][::-1]:
        print(f"     [{r['idx']:4d}] {r['city']:20s}  iou={r['iou']:.4f}  "
              f"road={r['road_pct']:.1f}%  {r['stem']}")

    # ── 4. Save plots ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("4. Saving Plots / グラフを保存中")
    print("=" * 55)

    plot_confusion_matrix(cm_total, out_dir / "confusion_matrix.png")
    plot_threshold_sweep(sweep,     out_dir / "threshold_sweep.png")
    plot_iou_distribution(records,  out_dir / "iou_distribution.png")

    # ── 5. Save JSON report ────────────────────────────────────────────────────
    report = {
        "checkpoint"      : str(ckpt_path),
        "arch"            : arch,
        "encoder"         : enc,
        "threshold"       : args.threshold,
        "best_threshold"  : best_tau,
        "overall_metrics" : means,
        "confusion_matrix": cm_total,
        "threshold_sweep" : {str(k): v for k, v in sweep.items()},
        "per_image_stats" : {
            "iou_mean" : float(np.mean(ious)),
            "iou_std"  : float(np.std(ious)),
            "iou_min"  : float(np.min(ious)),
            "iou_max"  : float(np.max(ious)),
        },
        "worst_samples"   : records[:n_show],
        "best_samples"    : records[-n_show:][::-1],
    }

    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"   Saved → {report_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Evaluation complete! / 評価完了！")
    print(f"  IoU  : {means['iou']:.4f}   (target ≥ 0.70)")
    print(f"  Dice : {means['dice']:.4f}   (target ≥ 0.80)")
    print(f"  PA   : {means['pa']:.4f}   (target ≥ 0.90)")
    print(f"  Results saved to: {out_dir}/")
    print(f"{'='*55}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained binary segmentation model on the Cityscapes val split.\n"
            "Cityscapes val 分割で訓練済み二値セグメンテーションモデルを評価します。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Cityscapes root directory / Cityscapes ルートディレクトリ")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best.pth",
                        help="Path to .pth checkpoint / チェックポイントパス")
    parser.add_argument("--arch",        type=str, default="unet",
                        help="Architecture (used if config missing in ckpt) / アーキテクチャ")
    parser.add_argument("--encoder",     type=str, default="resnet34",
                        help="Encoder backbone / エンコーダバックボーン")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Sigmoid threshold τ / シグモイド閾値 τ")
    parser.add_argument("--batch_size",  type=int, default=8,
                        help="Batch size for evaluation / 評価バッチサイズ")
    parser.add_argument("--crop_size",   type=int, default=512,
                        help="Crop size (used if config missing in ckpt) / クロップサイズ")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers / DataLoader ワーカー数")
    parser.add_argument("--top_n",       type=int, default=5,
                        help="Number of best/worst samples to report / レポートする最良/最悪サンプル数")
    parser.add_argument("--out_dir",     type=str, default="evaluation_results",
                        help="Directory for output files / 出力ファイルのディレクトリ")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
