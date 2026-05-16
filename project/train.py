"""
Phase 3 — Training: full training pipeline for binary drivable-area segmentation
フェーズ3 — 訓練：二値走行可能領域セグメンテーションの完全な訓練パイプライン

Features / 機能:
  ✓ AdamW optimiser with configurable LR
  ✓ ReduceLROnPlateau scheduler (halves LR when val IoU plateaus)
  ✓ Combined Dice + BCE loss (robust to class imbalance)
  ✓ Early stopping (saves GPU time when training has converged)
  ✓ Best-model checkpointing (saves the epoch with highest val IoU)
  ✓ Automatic Mixed Precision (AMP) for faster training on modern GPUs
  ✓ Training curve plot saved after training finishes

Usage (recommended / 推奨) / 使い方:
    # Minimal — uses all defaults from topics.md §6 Phase 3
    python train.py --data_root project/data/cityscapes

    # Custom settings
    python train.py \\
        --data_root project/data/cityscapes \\
        --arch unet \\
        --encoder resnet34 \\
        --epochs 50 \\
        --batch_size 8 \\
        --lr 1e-4 \\
        --crop_size 512 \\
        --amp

    # Resume from a saved checkpoint
    python train.py --data_root project/data/cityscapes --resume checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data_factory.dataset import get_dataloader
from utils.metrics import MetricTracker, binary_dice, binary_iou, pixel_accuracy
from networks.model import CombinedLoss, build_model, freeze_encoder, print_model_summary, unfreeze_encoder
from data_factory.transforms import get_train_transform, get_val_transform

# ── Defaults (mirrors topics.md §6 Phase 3 hyperparameter table) ──────────────
DEFAULTS = {
    "arch"        : "unet",
    "encoder"     : "resnet34",
    "epochs"      : 50,
    "batch_size"  : 8,
    "lr"          : 1e-4,
    "weight_decay": 1e-4,
    "crop_size"   : 512,
    "num_workers" : 4,
    "patience"    : 7,       # early-stopping patience (epochs without improvement)
    "warmup"      : 0,       # freeze encoder for this many epochs at the start
    "checkpoint_dir": "checkpoints",
}


# ── Training step ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    device    : torch.device,
    scaler    : GradScaler | None,
) -> dict[str, float]:
    """
    Run one full training epoch.
    1エポックの完全な訓練を実行します。

    Args:
        model     : The segmentation model.
        loader    : Training DataLoader.
        optimizer : AdamW optimiser.
        criterion : CombinedLoss (Dice + BCE).
        device    : CPU or CUDA device.
        scaler    : GradScaler for AMP, or None if AMP is disabled.

    Returns:
        Dict with mean "loss", "iou", "dice", "pa" for this epoch.
    """
    model.train()
    tracker  = MetricTracker()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)  # (N, 3, H, W)
        masks  = masks.to(device,  non_blocking=True)  # (N, H, W)

        optimizer.zero_grad()

        if scaler is not None:
            # Automatic Mixed Precision (AMP) — forward pass in float16
            # 自動混合精度（AMP） — float16でのフォワードパス
            with autocast():
                logits = model(images)          # (N, 1, H, W)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        bs = images.shape[0]
        total_loss += loss.item() * bs

        # Detach logits for metric computation (no gradient needed)
        # メトリクス計算のためにロジットをデタッチ（勾配不要）
        with torch.no_grad():
            tracker.update({
                "iou" : binary_iou(logits,       masks),
                "dice": binary_dice(logits,      masks),
                "pa"  : pixel_accuracy(logits,   masks),
            }, n=bs)

    n_total = len(loader.dataset)
    result  = tracker.mean()
    result["loss"] = total_loss / n_total
    return result


# ── Validation step ────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model    : nn.Module,
    loader   : torch.utils.data.DataLoader,
    criterion: nn.Module,
    device   : torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on the validation set.
    検証セットでモデルを評価します。

    Uses torch.no_grad() for the entire function — no gradients stored,
    which saves memory and speeds up evaluation.

    関数全体でtorch.no_grad()を使用します — 勾配を保存しないため、
    メモリを節約し評価を高速化します。

    Returns:
        Dict with mean "loss", "iou", "dice", "pa" for the validation set.
    """
    model.eval()
    tracker    = MetricTracker()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, masks)

        bs = images.shape[0]
        total_loss += loss.item() * bs
        tracker.update({
            "iou" : binary_iou(logits,     masks),
            "dice": binary_dice(logits,    masks),
            "pa"  : pixel_accuracy(logits, masks),
        }, n=bs)

    n_total = len(loader.dataset)
    result  = tracker.mean()
    result["loss"] = total_loss / n_total
    return result


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    path     : Path,
    epoch    : int,
    model    : nn.Module,
    optimizer: torch.optim.Optimizer,
    val_iou  : float,
    config   : dict,
) -> None:
    """
    Save model weights, optimizer state, and metadata to a .pth file.
    モデルの重み、オプティマイザの状態、メタデータを .pth ファイルに保存します。

    Saving the optimiser state lets us resume training exactly where we left off.
    オプティマイザの状態を保存することで、中断した場所から正確に訓練を再開できます。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_iou"             : val_iou,
        "config"              : config,
    }, path)


def load_checkpoint(
    path     : Path,
    model    : nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device   : torch.device = torch.device("cpu"),
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.
    チェックポイントを読み込み、モデル（およびオプションでオプティマイザ）の状態を復元します。

    Returns the checkpoint dict so the caller can access epoch / val_iou / config.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint: {path}  (epoch={ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f})")
    return ckpt


# ── Plot training curves ───────────────────────────────────────────────────────

def plot_history(history: dict[str, list], save_path: Path) -> None:
    """
    Plot loss and IoU curves for train and validation over all epochs.
    全エポックの訓練・検証の損失とIoU曲線を描画します。

    Args:
        history  : {"train_loss": [...], "val_loss": [...],
                    "train_iou":  [...], "val_iou":  [...]}
        save_path: Output PNG path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training History / 訓練履歴", fontsize=12, fontweight="bold")

    # Loss curve / 損失曲線
    axes[0].plot(epochs, history["train_loss"], label="Train loss / 訓練損失",
                 color="#3498db", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val loss / 検証損失",
                 color="#e74c3c", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Dice + BCE)")
    axes[0].set_title("Loss / 損失")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # IoU curve / IoU曲線
    axes[1].plot(epochs, history["train_iou"], label="Train IoU / 訓練IoU",
                 color="#3498db", linewidth=2)
    axes[1].plot(epochs, history["val_iou"],   label="Val IoU / 検証IoU",
                 color="#e74c3c", linewidth=2, linestyle="--")

    # Mark the best epoch / ベストエポックをマーク
    best_epoch = int(history["val_iou"].index(max(history["val_iou"]))) + 1
    best_iou   = max(history["val_iou"])
    axes[1].axvline(best_epoch, color="grey", linewidth=1, linestyle=":")
    axes[1].annotate(
        f"Best: {best_iou:.4f}\n(epoch {best_epoch})",
        xy=(best_epoch, best_iou),
        xytext=(best_epoch + 1, best_iou - 0.02),
        fontsize=8, color="grey",
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU (Jaccard)")
    axes[1].set_title("Validation IoU (road class) / 検証IoU（道路クラス）")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved → {save_path}")


# ── Main training loop ─────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    Full training pipeline.
    完全な訓練パイプライン。
    """
    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice / デバイス: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Config dict (saved inside checkpoints) ────────────────────────────────
    config = {
        "arch"        : args.arch,
        "encoder"     : args.encoder,
        "epochs"      : args.epochs,
        "batch_size"  : args.batch_size,
        "lr"          : args.lr,
        "weight_decay": args.weight_decay,
        "crop_size"   : args.crop_size,
    }

    # ── DataLoaders ────────────────────────────────────────────────────────────
    print("\nBuilding DataLoaders... / DataLoaderを構築中...")
    train_loader = get_dataloader(
        root        = args.data_root,
        split       = "train",
        transform   = get_train_transform(args.crop_size),
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    val_loader = get_dataloader(
        root        = args.data_root,
        split       = "val",
        transform   = get_val_transform(args.crop_size),
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\nBuilding model... / モデルを構築中...")
    model = build_model(
        arch           = args.arch,
        encoder_name   = args.encoder,
        encoder_weights= "imagenet",
    ).to(device)

    print_model_summary(model, crop_size=args.crop_size)

    # ── Loss, optimiser, scheduler, AMP ───────────────────────────────────────
    criterion = CombinedLoss(dice_weight=1.0, bce_weight=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True,
        # factor=0.5 → halve LR when val IoU does not improve for 5 epochs
        # val IoUが5エポック改善しない場合、LRを半分にする
    )
    scaler = GradScaler() if (args.amp and device.type == "cuda") else None
    if scaler:
        print("AMP enabled (float16 forward pass). / AMP有効（float16フォワードパス）。")

    # ── Resume from checkpoint (optional) ─────────────────────────────────────
    start_epoch  = 1
    best_val_iou = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt        = load_checkpoint(resume_path, model, optimizer, device)
            start_epoch = ckpt["epoch"] + 1
            best_val_iou = ckpt["val_iou"]
        else:
            print(f"[Warning] Checkpoint not found: {resume_path}. Starting fresh.")

    # ── Warm-up: freeze encoder for the first N epochs ─────────────────────────
    if args.warmup > 0:
        print(f"\nEncoder frozen for first {args.warmup} warm-up epochs.")
        print(f"最初の {args.warmup} ウォームアップエポックはエンコーダを凍結します。")
        freeze_encoder(model)

    # ── History & early-stopping state ────────────────────────────────────────
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_iou" : [], "val_iou" : [],
        "train_dice": [], "val_dice": [],
    }
    no_improve_count = 0

    ckpt_dir  = Path(args.checkpoint_dir)
    best_path = ckpt_dir / "best.pth"
    last_path = ckpt_dir / "last.pth"

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training  |  epochs={args.epochs}  |  batch={args.batch_size}")
    print(f"訓練開始           |  エポック={args.epochs}  |  バッチ={args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):

        # Unfreeze encoder after warm-up period
        # ウォームアップ期間後にエンコーダの凍結を解除
        if args.warmup > 0 and epoch == args.warmup + 1:
            unfreeze_encoder(model)

        t0 = time.time()

        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # --- Validate ---
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # --- Log ---
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"time={elapsed:.0f}s  lr={lr_now:.2e}\n"
            f"  Train — loss={train_metrics['loss']:.4f}  "
            f"iou={train_metrics['iou']:.4f}  dice={train_metrics['dice']:.4f}\n"
            f"  Val   — loss={val_metrics['loss']:.4f}  "
            f"iou={val_metrics['iou']:.4f}  dice={val_metrics['dice']:.4f}  "
            f"pa={val_metrics['pa']:.4f}"
        )

        # --- Scheduler step (based on val IoU) ---
        scheduler.step(val_metrics["iou"])

        # --- Record history ---
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])

        # --- Checkpointing ---
        save_checkpoint(last_path, epoch, model, optimizer,
                        val_metrics["iou"], config)

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            no_improve_count = 0
            save_checkpoint(best_path, epoch, model, optimizer,
                            best_val_iou, config)
            print(f"  ★ New best val IoU: {best_val_iou:.4f} — saved → {best_path}")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{args.patience}). "
                  f"Best so far: {best_val_iou:.4f}")

        # --- Early stopping ---
        if no_improve_count >= args.patience:
            print(
                f"\nEarly stopping after {epoch} epochs "
                f"(no improvement for {args.patience} epochs).\n"
                f"{args.patience} エポック改善なし — {epoch} エポックで早期終了。"
            )
            break

        print()

    # ── Final report ──────────────────────────────────────────────────────────
    total_epochs = len(history["val_iou"])
    print(f"\n{'='*60}")
    print("Training complete! / 訓練完了！")
    print(f"  Epochs run         : {total_epochs}")
    print(f"  Best val IoU       : {best_val_iou:.4f}")
    print(f"  Best model saved   : {best_path}")
    print(f"  Last model saved   : {last_path}")
    print(f"{'='*60}\n")

    # ── Save training curves ───────────────────────────────────────────────────
    curve_path = ckpt_dir / "training_curves.png"
    plot_history(history, curve_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary drivable-area segmentation model on Cityscapes.\n"
            "Cityscapesで二値走行可能領域セグメンテーションモデルを訓練します。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to Cityscapes root (contains leftImg8bit/ and gtFine/).\n"
             "Cityscapesルートへのパス（leftImg8bit/とgtFine/を含む）。",
    )

    # Model
    parser.add_argument("--arch",    type=str, default=DEFAULTS["arch"],
                        choices=["unet", "unetplusplus", "deeplabv3plus"],
                        help="Segmentation architecture. / セグメンテーションアーキテクチャ。")
    parser.add_argument("--encoder", type=str, default=DEFAULTS["encoder"],
                        help="Encoder backbone name (e.g. resnet34, resnet50).\n"
                             "エンコーダバックボーン名（例：resnet34, resnet50）。")

    # Training
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"],
                        help="Initial learning rate. / 初期学習率。")
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--crop_size",    type=int,   default=DEFAULTS["crop_size"],
                        help="Random/centre crop size (px). / クロップサイズ（px）。")
    parser.add_argument("--num_workers",  type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--patience",     type=int,   default=DEFAULTS["patience"],
                        help="Early-stopping patience (epochs). / 早期終了のパティエンス（エポック）。")
    parser.add_argument("--warmup",       type=int,   default=DEFAULTS["warmup"],
                        help="Epochs to freeze encoder at the start. / 開始時にエンコーダを凍結するエポック数。")
    parser.add_argument("--amp",          action="store_true",
                        help="Enable Automatic Mixed Precision (AMP). / 自動混合精度（AMP）を有効にする。")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULTS["checkpoint_dir"],
                        help="Directory to save checkpoints. / チェックポイントを保存するディレクトリ。")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from. / 再開するチェックポイントのパス。")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
