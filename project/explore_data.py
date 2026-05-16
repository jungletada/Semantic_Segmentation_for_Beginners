"""
Phase 1 — Data Pipeline: Dataset verification and visualisation
フェーズ1 — データパイプライン：データセットの検証と可視化

Run this script AFTER placing Cityscapes under project/data/cityscapes/.
It performs four checks:
  1. Dataset size     — how many images in each split?
  2. Class balance    — what fraction of pixels are road vs. background?
  3. DataLoader test  — do batches have the right shapes and dtypes?
  4. Visualisation    — plot (image, binary mask, overlay) for random samples.

Cityscapesを project/data/cityscapes/ に配置した後にこのスクリプトを実行してください。
4つの確認を行います：
  1. データセットサイズ — 各分割の画像数は？
  2. クラスバランス    — 道路対背景のピクセル比率は？
  3. DataLoaderテスト  — バッチの形状とデータ型は正しいか？
  4. 可視化            — ランダムサンプルの（画像、バイナリマスク、オーバーレイ）を描画。

Usage / 使い方:
    python explore_data.py                        # uses default path
    python explore_data.py --root /path/to/cityscapes --split val --n 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import ROAD_LABEL_ID, CityscapesBinaryDataset, get_dataloader
from transforms import IMAGENET_MEAN, IMAGENET_STD, get_train_transform, get_val_transform

# Default path: project/data/cityscapes relative to this file
# デフォルトパス：このファイルからの相対パス project/data/cityscapes
DEFAULT_ROOT = Path(__file__).parent / "data" / "cityscapes"


# ── 1. Dataset size check ──────────────────────────────────────────────────────

def check_dataset_sizes(root: Path) -> None:
    """
    Print the number of (image, mask) pairs in each split.
    各分割の（画像、マスク）ペア数を表示します。
    """
    print("=" * 55)
    print("1. Dataset Size / データセットサイズ")
    print("=" * 55)

    total = 0
    for split in ("train", "val"):
        try:
            ds = CityscapesBinaryDataset(root=str(root), split=split)
            print(f"   {split:5s} : {len(ds):4d} images")
            total += len(ds)
        except FileNotFoundError as e:
            print(f"   {split:5s} : [NOT FOUND] {e}")

    print(f"   {'total':5s} : {total:4d} images")
    print()


# ── 2. Class balance analysis ──────────────────────────────────────────────────

def analyse_class_balance(
    root: Path,
    split: str = "train",
    n_samples: int = 100,
) -> list[float]:
    """
    Sample n_samples masks and compute the road pixel ratio for each.
    Reports mean ± std and plots a histogram.

    n_samplesのマスクをサンプリングし、各マスクの道路ピクセル比率を計算します。
    平均±標準偏差を報告し、ヒストグラムを描画します。

    This reveals class imbalance: if road occupies only ~15% of pixels on average,
    a naive model could achieve 85% pixel accuracy by predicting "all background".
    Dice Loss (see topics.md §4.3) addresses this.

    クラス不均衡を示します：道路が平均 ~15% しか占めない場合、
    ナイーブなモデルは「すべて背景」と予測するだけで85%のピクセル精度を達成できます。
    Dice Loss（topics.md §4.3参照）はこれに対処します。
    """
    print("=" * 55)
    print(f"2. Class Balance ({split}, n={n_samples}) / クラスバランス")
    print("=" * 55)

    ds      = CityscapesBinaryDataset(root=str(root), split=split)
    indices = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    road_ratios: list[float] = []
    for idx in indices:
        raw    = np.array(Image.open(ds.mask_paths[idx]), dtype=np.int32)
        binary = (raw == ROAD_LABEL_ID).astype(np.uint8)
        road_ratios.append(float(binary.mean() * 100))

    mean_road = np.mean(road_ratios)
    std_road  = np.std(road_ratios)

    print(f"   Road (class 1 / 走行可能) : {mean_road:.1f}% ± {std_road:.1f}%")
    print(f"   Background (class 0 / 背景): {100 - mean_road:.1f}% ± {std_road:.1f}%")
    print()
    print("   → If road < 30%, use Dice Loss or class-weighted BCE to avoid")
    print("     the model predicting all-background.  (topics.md §4.3)")
    print("   → 道路 < 30% の場合、モデルが全背景を予測するのを防ぐために")
    print("     Dice Lossまたはクラス重み付きBCEを使用してください。")
    print()

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(road_ratios, bins=20, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax.axvline(mean_road, color="#e74c3c", linewidth=2,
               label=f"Mean / 平均: {mean_road:.1f}%")
    ax.set_xlabel("Road pixel ratio per image (%) / 画像ごとの道路ピクセル比率 (%)")
    ax.set_ylabel("Count / 枚数")
    ax.set_title(
        f"Class Balance — {split} split (n={len(road_ratios)})\n"
        f"クラスバランス — {split}分割"
    )
    ax.legend()
    plt.tight_layout()
    out = Path(__file__).parent / "class_balance.png"
    plt.savefig(out, dpi=120)
    plt.show()
    print(f"   Saved → {out}")
    print()

    return road_ratios


# ── 3. DataLoader batch test ───────────────────────────────────────────────────

def test_dataloader(root: Path, crop_size: int = 512) -> None:
    """
    Run one batch through the val DataLoader and print tensor shapes / dtypes.
    Confirms that everything from disk → transform → tensor is correct.

    valのDataLoaderを通じて1バッチを実行し、テンソルの形状・データ型を表示します。
    ディスク → 変換 → テンソルまでのすべてが正しいことを確認します。
    """
    print("=" * 55)
    print("3. DataLoader Batch Test / DataLoaderバッチテスト")
    print("=" * 55)

    transform = get_val_transform(crop_size)
    loader    = get_dataloader(
        root=str(root), split="val",
        transform=transform, batch_size=4, num_workers=0, pin_memory=False,
    )

    images, masks = next(iter(loader))

    print(f"   Image batch shape  : {tuple(images.shape)}")
    print(f"                        expected (4, 3, {crop_size}, {crop_size})")
    print(f"   Mask batch shape   : {tuple(masks.shape)}")
    print(f"                        expected (4, {crop_size}, {crop_size})")
    print(f"   Image dtype        : {images.dtype}   (expected float32)")
    print(f"   Mask dtype         : {masks.dtype}   (expected int64)")
    print(f"   Mask unique values : {sorted(masks.unique().tolist())}  (expected [0, 1])")
    print(f"   Road pixel ratio   : {masks.float().mean().item() * 100:.1f}%")
    print()

    # Assertions
    assert images.shape == (4, 3, crop_size, crop_size), "Unexpected image shape"
    assert masks.shape  == (4, crop_size, crop_size),    "Unexpected mask shape"
    assert images.dtype == torch.float32,                "Image dtype must be float32"
    assert masks.dtype  == torch.int64,                  "Mask dtype must be int64"
    assert set(masks.unique().tolist()).issubset({0, 1}), "Mask must contain only 0 and 1"

    print("   ✓ All checks passed. / すべての確認が完了しました。")
    print()


# ── 4. Sample visualisation ────────────────────────────────────────────────────

def _denormalize(tensor: "torch.Tensor") -> np.ndarray:
    """
    Reverse ImageNet normalisation for display.
    表示用にImageNet正規化を元に戻します。

    Input : tensor (3, H, W) float32  — normalised
    Output: ndarray (H, W, 3) float32 — values clipped to [0, 1]
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
    img  = img * std + mean
    return img.clip(0.0, 1.0)


def visualise_samples(
    root: Path,
    split: str = "train",
    n: int = 4,
    crop_size: int = 512,
) -> None:
    """
    Plot n random samples as a grid of three panels each:
      Column 1 — Original image (de-normalised)
      Column 2 — Binary mask (white = road, black = background)
      Column 3 — Overlay: image blended with green road highlight

    n枚のランダムサンプルを3列グリッドで描画します：
      列1 — 元画像（逆正規化済み）
      列2 — バイナリマスク（白 = 道路、黒 = 背景）
      列3 — オーバーレイ：画像と緑の道路ハイライトのブレンド

    Args:
        root      : Cityscapes root.
        split     : "train" or "val".
        n         : Number of samples to show.
        crop_size : Crop size passed to the transform.
    """
    print("=" * 55)
    print(f"4. Visualising {n} random {split} samples")
    print(f"   {n}枚のランダムな {split} サンプルを可視化")
    print("=" * 55)

    transform = (
        get_train_transform(crop_size)
        if split == "train"
        else get_val_transform(crop_size)
    )
    ds      = CityscapesBinaryDataset(root=str(root), split=split, transform=transform)
    indices = np.random.choice(len(ds), size=min(n, len(ds)), replace=False)

    # Road overlay colour: semi-transparent green  / 道路オーバーレイ色：半透明の緑
    ROAD_RGB = np.array([0.18, 0.80, 0.44], dtype=np.float32)

    fig, axes = plt.subplots(n, 3, figsize=(14, 4.2 * n))
    fig.suptitle(
        f"Cityscapes Binary Dataset — {split} split  (crop {crop_size}×{crop_size})\n"
        f"Green = Drivable Road (class 1) / 緑 = 走行可能な道路（クラス1）",
        fontsize=12, fontweight="bold",
    )

    col_titles = [
        "Image / 画像",
        "Binary Mask / バイナリマスク\n(white=road · black=bg)",
        "Overlay / オーバーレイ\n(green=road)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9)

    for row, idx in enumerate(indices):
        img_tensor, mask_tensor = ds[idx]

        img_np  = _denormalize(img_tensor)          # (H, W, 3) float32 [0,1]
        mask_np = mask_tensor.numpy().astype(float)  # (H, W)    float64 {0,1}

        # Build overlay: blend image with green where road is predicted
        # オーバーレイ：道路部分で画像と緑をブレンド
        overlay = img_np.copy()
        road_px = mask_np == 1
        overlay[road_px] = 0.4 * overlay[road_px] + 0.6 * ROAD_RGB

        axes[row, 0].imshow(img_np)
        axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(overlay)

        # Row label: city name + road pixel percentage
        # 行ラベル：都市名と道路ピクセルパーセンテージ
        info     = ds.get_sample_info(idx)
        road_pct = mask_np.mean() * 100
        axes[row, 0].set_ylabel(
            f"idx={idx}\n{info['city']}\nroad={road_pct:.1f}%",
            fontsize=8, rotation=0, labelpad=68, va="center",
        )

        for col in range(3):
            axes[row, col].axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color="white",          label="Road / 道路 (class 1)"),
        mpatches.Patch(color="black",          label="Background / 背景 (class 0)"),
        mpatches.Patch(color=(0.18, 0.80, 0.44), label="Road highlight / 道路ハイライト"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3,
        fontsize=9, bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout()
    out = Path(__file__).parent / "sample_visualisation.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"   Saved → {out}")
    print()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and visualise the Cityscapes binary dataset.\n"
                    "Cityscapesバイナリデータセットの検証と可視化。"
    )
    parser.add_argument(
        "--root", type=str, default=str(DEFAULT_ROOT),
        help="Path to the Cityscapes root directory / Cityscapesのルートディレクトリパス",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="Dataset split to visualise / 可視化するデータセット分割",
    )
    parser.add_argument(
        "--n", type=int, default=4,
        help="Number of samples to visualise / 可視化するサンプル数",
    )
    parser.add_argument(
        "--n_balance", type=int, default=100,
        help="Samples for class balance analysis / クラスバランス分析用サンプル数",
    )
    parser.add_argument(
        "--crop_size", type=int, default=512,
        help="Crop size in pixels / クロップサイズ（ピクセル）",
    )
    args = parser.parse_args()

    root = Path(args.root)

    # Gate: check that the root exists before running any step
    # ゲート：いずれかのステップを実行する前にルートの存在を確認
    if not root.exists():
        print()
        print("[ERROR] Cityscapes root not found:")
        print(f"  {root}")
        print()
        print("Please download Cityscapes and extract it there.")
        print("See project/topics.md §3.2 for step-by-step instructions.")
        print()
        print("[エラー] Cityscapesのルートが見つかりません。")
        print("Cityscapesをダウンロードしてそこに解凍してください。")
        print("手順はproject/topics.md §3.2をご覧ください。")
        return

    print()
    check_dataset_sizes(root)
    analyse_class_balance(root, split=args.split, n_samples=args.n_balance)
    test_dataloader(root, crop_size=args.crop_size)
    visualise_samples(root, split=args.split, n=args.n, crop_size=args.crop_size)

    print("All checks complete. / すべての確認が完了しました。")


if __name__ == "__main__":
    main()
