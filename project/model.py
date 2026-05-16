"""
Phase 2 — Model: binary segmentation model factory
フェーズ2 — モデル：二値セグメンテーションモデルのファクトリ

Wraps segmentation_models_pytorch (smp) to give students a single function
that builds any supported architecture with the correct settings for our
binary drivable-area task.

segmentation_models_pytorch (smp) をラップし、二値走行可能領域タスクに
適切な設定で任意のサポートアーキテクチャを構築する単一の関数を提供します。

Supported architectures / サポートアーキテクチャ:
  "unet"         — U-Net (recommended / 推奨) — sharp boundaries, fast training
  "unetplusplus" — U-Net++ — nested skip connections, slightly higher accuracy
  "deeplabv3plus"— DeepLabV3+ — atrous convolutions, best for complex scenes

Usage / 使い方:
    from model import build_model, print_model_summary
    model = build_model(arch="unet", encoder_name="resnet34")
    print_model_summary(model)
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    _SMP_AVAILABLE = True
except ImportError:
    _SMP_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────

# Architectures available through smp
# smpで利用可能なアーキテクチャ
SUPPORTED_ARCHS: dict[str, type] = {}

if _SMP_AVAILABLE:
    SUPPORTED_ARCHS = {
        "unet"         : smp.Unet,
        "unetplusplus" : smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }

# Encoder backbones we tested and recommend
# テスト・推奨済みのエンコーダバックボーン
RECOMMENDED_ENCODERS = [
    "resnet34",          # recommended default — fast, strong baseline
    "resnet50",          # more capacity, slower to train
    "efficientnet-b0",   # lightweight, good accuracy/speed trade-off
    "mobilenet_v2",      # smallest — useful for rapid prototyping
]


# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 1,
) -> nn.Module:
    """
    Build a binary segmentation model.
    二値セグメンテーションモデルを構築します。

    Why activation=None?
      We output raw logits and apply sigmoid only when computing metrics or
      running inference.  This is numerically more stable during training
      because BCEWithLogitsLoss and DiceLoss(from_logits=True) combine the
      sigmoid with the loss in a single, numerically stable operation.

      なぜ activation=None か？
      生のロジットを出力し、メトリクス計算や推論時のみsigmoidを適用します。
      BCEWithLogitsLossとDiceLoss(from_logits=True)がsigmoidと損失を
      数値的に安定した単一の演算に結合するため、訓練中はこのほうが
      数値的に安定しています。

    Args:
        arch           : Architecture name. One of SUPPORTED_ARCHS.
                         アーキテクチャ名。SUPPORTED_ARCHS のいずれか。
        encoder_name   : Timm/smp encoder backbone name.
                         Timm/smpのエンコーダバックボーン名。
        encoder_weights: Pre-trained weight source ("imagenet" or None).
                         事前学習済み重みのソース（"imagenet" または None）。
        in_channels    : Number of input image channels (3 for RGB).
                         入力画像チャンネル数（RGBは3）。
        num_classes    : Number of output channels.  1 for binary segmentation.
                         出力チャンネル数。二値セグメンテーションは1。

    Returns:
        nn.Module  — outputs (N, 1, H, W) float32 logits.
        (N, 1, H, W) float32 ロジットを出力するnn.Module。
    """
    if not _SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch is not installed.\n"
            "Run: pip install segmentation-models-pytorch\n"
            "segmentation_models_pytorchがインストールされていません。\n"
            "実行: pip install segmentation-models-pytorch"
        )

    arch_lower = arch.lower()
    if arch_lower not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported architecture '{arch}'.\n"
            f"Choose from: {list(SUPPORTED_ARCHS.keys())}"
        )

    model = SUPPORTED_ARCHS[arch_lower](
        encoder_name    = encoder_name,
        encoder_weights = encoder_weights,
        in_channels     = in_channels,
        classes         = num_classes,
        activation      = None,   # raw logits — more numerically stable
    )

    return model


# ── Loss function ──────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Dice Loss + Binary Cross-Entropy Loss for binary segmentation.
    二値セグメンテーション用の Dice Loss + 二値交差エントロピー損失。

    Why combine them?  (topics.md §4.3)
      BCE penalises per-pixel confidence errors uniformly.
      Dice directly maximises overlap between prediction and ground-truth —
      critical when road pixels are a small fraction of the image.

      なぜ組み合わせるか？（topics.md §4.3）
      BCEはピクセルごとの信頼度誤差を均一に罰します。
      Diceは予測と正解の重なりを直接最大化します —
      道路ピクセルが画像の小さな割合しか占めない場合に重要です。

    Args:
        dice_weight : Relative weight of the Dice loss term. Default 1.0.
        bce_weight  : Relative weight of the BCE loss term.  Default 1.0.

    Input / 入力:
        logits  : (N, 1, H, W) float32 — raw model output (before sigmoid)
        targets : (N, H, W)    int64   — binary ground truth (0 or 1)
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        if not _SMP_AVAILABLE:
            raise ImportError("Install segmentation_models_pytorch to use CombinedLoss.")

        self.dice_loss   = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_loss    = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (N, H, W) int64 → (N, 1, H, W) float32 for both losses
        # targets: (N, H, W) int64 → 両損失用に (N, 1, H, W) float32 へ変換
        targets_f = targets.unsqueeze(1).float()

        dice = self.dice_loss(logits, targets_f)
        bce  = self.bce_loss(logits, targets_f)

        return self.dice_weight * dice + self.bce_weight * bce


# ── Utilities ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Return parameter counts broken down by total / trainable / encoder / decoder.
    合計・訓練可能・エンコーダ・デコーダ別のパラメータ数を返します。
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if hasattr(model, "encoder"):
        encoder = sum(p.numel() for p in model.encoder.parameters())
    else:
        encoder = 0

    return {
        "total"    : total,
        "trainable": trainable,
        "encoder"  : encoder,
        "decoder"  : total - encoder,
    }


def print_model_summary(model: nn.Module, crop_size: int = 512) -> None:
    """
    Print a human-readable model summary and run a dry forward pass.
    人間が読めるモデルサマリーを表示し、ドライフォワードパスを実行します。
    """
    params = count_parameters(model)
    device = next(model.parameters()).device

    print("=" * 55)
    print("Model Summary / モデルサマリー")
    print("=" * 55)
    print(f"  Class           : {model.__class__.__name__}")
    if hasattr(model, "encoder"):
        print(f"  Encoder         : {model.encoder.__class__.__name__}")
    print(f"  Total params    : {params['total']:>12,}")
    print(f"  Trainable params: {params['trainable']:>12,}")
    print(f"  Encoder params  : {params['encoder']:>12,}")
    print(f"  Decoder params  : {params['decoder']:>12,}")

    # Dry forward pass / ドライフォワードパス
    dummy = torch.zeros(1, 3, crop_size, crop_size, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"  Input shape     : {tuple(dummy.shape)}")
    print(f"  Output shape    : {tuple(out.shape)}   (logits, before sigmoid)")
    print("=" * 55)


def freeze_encoder(model: nn.Module) -> None:
    """
    Freeze all encoder parameters so only the decoder is trained.
    Useful for the first few epochs when the encoder weights are already good.

    エンコーダのすべてのパラメータを凍結し、デコーダのみを訓練します。
    エンコーダの重みがすでに良好な最初の数エポックに有用です。
    """
    if not hasattr(model, "encoder"):
        print("Warning: model has no 'encoder' attribute; nothing frozen.")
        return
    for param in model.encoder.parameters():
        param.requires_grad = False
    frozen = sum(p.numel() for p in model.encoder.parameters())
    print(f"Encoder frozen ({frozen:,} params). Only decoder will be trained.")
    print(f"エンコーダを凍結しました（{frozen:,}パラメータ）。デコーダのみが訓練されます。")


def unfreeze_encoder(model: nn.Module) -> None:
    """
    Unfreeze all encoder parameters (call after warm-up epochs).
    エンコーダのすべてのパラメータの凍結を解除します（ウォームアップエポック後に呼び出します）。
    """
    if not hasattr(model, "encoder"):
        return
    for param in model.encoder.parameters():
        param.requires_grad = True
    print("Encoder unfrozen — full model will now be trained.")
    print("エンコーダの凍結を解除しました — モデル全体が訓練されます。")
