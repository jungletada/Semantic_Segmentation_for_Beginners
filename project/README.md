# Drivable Area Segmentation — Project Guide
# 走行可能領域のセグメンテーション — プロジェクトガイド

Binary semantic segmentation of drivable road areas using the Cityscapes dataset and a U-Net model.  
Cityscapesデータセットと U-Net モデルを使用した走行可能道路領域の二値セマンティックセグメンテーション。

---

## Project Structure / プロジェクト構成

```
project/
├── data/
│   └── cityscapes/            ← Dataset goes here (see Step 1)
├── data_factory/              ← Helper utilities
├── checkpoints/               ← Created automatically during training
├── dataset.py                 ← Phase 1: PyTorch Dataset & DataLoader
├── transforms.py              ← Phase 1: Albumentations augmentation pipelines
├── explore_data.py            ← Phase 1: Dataset verification & visualisation
├── model.py                   ← Phase 2: U-Net factory, CombinedLoss, utilities
├── metrics.py                 ← Phase 2: IoU, Dice, Pixel Accuracy, MetricTracker
├── train.py                   ← Phase 3: Full training loop with checkpointing
├── evaluate.py                ← Phase 4: Metrics, confusion matrix, threshold sweep
├── visualize_results.py       ← Phase 5: Prediction grids, failure modes, edge cases
├── requirements.txt
├── topics.md                  ← Detailed project reference document
└── README.md                  ← This file
```

---

## Prerequisites / 事前準備

### 1. Install dependencies / 依存関係のインストール

```bash
pip install -r requirements.txt
pip install segmentation-models-pytorch   # U-Net / UNet++ / DeepLabV3+
```

> All commands below should be run from the `project/` directory.  
> 以下のコマンドはすべて `project/` ディレクトリ内で実行してください。

---

## Step 1 — Dataset Setup / データセットのセットアップ

### 1-a. Register and download Cityscapes / 登録とダウンロード

1. Create a free account at [https://www.cityscapes-dataset.com/register/](https://www.cityscapes-dataset.com/register/)
2. Go to [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/) and download:

| File | Size | Contents |
|------|------|----------|
| `leftImg8bit_trainvaltest.zip` | ~11 GB | RGB camera images |
| `gtFine_trainvaltest.zip` | ~241 MB | Pixel-level annotations |

**Optional: command-line download / コマンドラインでのダウンロード（任意）**

```bash
pip install cityscapesscripts

export CITYSCAPES_USERNAME="your_username"
export CITYSCAPES_PASSWORD="your_password"

csDownload gtFine_trainvaltest.zip
csDownload leftImg8bit_trainvaltest.zip
```

### 1-b. Extract into the correct location / 正しい場所に解凍する

```bash
unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/
unzip gtFine_trainvaltest.zip      -d data/cityscapes/
```

After extraction the directory tree should look like this:  
解凍後のディレクトリ構造：

```
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   └── ...
│   │   └── ... (18 cities / 18都市)
│   └── val/
│       └── ... (3 cities / 3都市)
└── gtFine/
    ├── train/
    │   ├── aachen/
    │   │   ├── aachen_000000_000019_gtFine_labelIds.png
    │   │   └── ...
    │   └── ...
    └── val/
        └── ...
```

> **Important / 重要**  
> `gtFine_labelIds.png` uses **raw label IDs** (0–33). Road = raw ID **7** (not 0).  
> This is handled automatically by `dataset.py`.  
> `gtFine_labelIds.png` は**生のラベルID**（0〜33）を使います。道路 = ID **7**（0ではない）。  
> `dataset.py` が自動的に処理します。

---

## Step 2 — Explore the Dataset / データセットの探索

Verify the dataset and visualise sample images before training.  
訓練前にデータセットを検証してサンプル画像を可視化します。

```bash
python explore_data.py --root data/cityscapes
```

**Optional arguments / オプション引数:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | `data/cityscapes` | Path to Cityscapes root |
| `--split` | `train` | Split to visualise (`train` / `val`) |
| `--n` | `4` | Number of sample images to display |
| `--n_balance` | `100` | Samples for class-balance histogram |
| `--crop_size` | `512` | Crop size in pixels |

**What it checks / 確認内容:**

1. Dataset size — image count per split
2. Class balance — road vs. background pixel ratio (warns if road < 30%)
3. DataLoader test — tensor shapes and dtypes
4. Sample visualisation — saved to `sample_visualisation.png`

---

## Step 3 — Train the Model / モデルの訓練

```bash
python train.py --data_root data/cityscapes
```

**Recommended full command / 推奨コマンド（全オプション）:**

```bash
python train.py \
    --data_root  data/cityscapes \
    --arch       unet \
    --encoder    resnet34 \
    --epochs     50 \
    --batch_size 8 \
    --lr         1e-4 \
    --crop_size  512 \
    --patience   7 \
    --amp                        # enable AMP (GPU only / GPU専用)
```

**All arguments / 全引数:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | *(required)* | Cityscapes root directory |
| `--arch` | `unet` | Architecture: `unet` / `unetplusplus` / `deeplabv3plus` |
| `--encoder` | `resnet34` | Encoder: `resnet34` / `resnet50` / `efficientnet-b0` / `mobilenet_v2` |
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--crop_size` | `512` | Random crop size (pixels) |
| `--num_workers` | `4` | DataLoader worker processes |
| `--patience` | `7` | Early-stopping patience (epochs) |
| `--warmup` | `0` | Freeze encoder for first N epochs |
| `--amp` | off | Enable Automatic Mixed Precision (GPU only) |
| `--resume` | — | Resume from checkpoint, e.g. `checkpoints/last.pth` |
| `--checkpoint_dir` | `checkpoints` | Directory to save `.pth` files |

**What gets saved / 保存されるファイル:**

```
checkpoints/
├── best.pth              ← Best val IoU checkpoint
├── last.pth              ← Most recent epoch checkpoint
└── training_curves.png   ← Loss + IoU curves
```

**Resume training / 訓練の再開:**

```bash
python train.py --data_root data/cityscapes --resume checkpoints/last.pth
```

---

## Step 4 — Evaluate the Model / モデルの評価

```bash
python evaluate.py --data_root data/cityscapes --checkpoint checkpoints/best.pth
```

**All arguments / 全引数:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | *(required)* | Cityscapes root directory |
| `--checkpoint` | `checkpoints/best.pth` | Path to `.pth` checkpoint |
| `--threshold` | `0.5` | Sigmoid threshold τ for binary prediction |
| `--batch_size` | `8` | Evaluation batch size |
| `--top_n` | `5` | Best / worst samples to print |
| `--out_dir` | `evaluation_results` | Output directory |

**Output files / 出力ファイル:**

```
evaluation_results/
├── confusion_matrix.png     ← TP / FP / FN / TN heatmap
├── threshold_sweep.png      ← IoU vs. τ ∈ [0.1, 0.9]
├── iou_distribution.png     ← Per-image IoU histogram
└── evaluation_report.json   ← All metrics in JSON format
```

**Target metrics / 目標メトリクス:**

| Metric | Target |
|--------|--------|
| IoU (Jaccard) | ≥ 0.70 |
| Dice (F1) | ≥ 0.80 |
| Pixel Accuracy | ≥ 0.90 |

---

## Step 5 — Visualise Results / 結果の可視化

### Prediction grid (default) / 予測グリッド（デフォルト）

```bash
python visualize_results.py \
    --data_root  data/cityscapes \
    --checkpoint checkpoints/best.pth \
    --n          6
```

### Best vs. worst predictions / ベストとワーストの予測

```bash
python visualize_results.py \
    --data_root  data/cityscapes \
    --checkpoint checkpoints/best.pth \
    --mode       best_worst \
    --n          3
```

### Failure mode analysis / 失敗モード分析

```bash
python visualize_results.py \
    --data_root  data/cityscapes \
    --checkpoint checkpoints/best.pth \
    --mode       failure_modes \
    --n_scan     200
```

### Edge-case conditions / エッジケース条件

```bash
python visualize_results.py \
    --data_root  data/cityscapes \
    --checkpoint checkpoints/best.pth \
    --mode       edge_cases \
    --conditions rain shadow fog night
```

### Run everything at once / すべてを一度に実行

```bash
python visualize_results.py \
    --data_root  data/cityscapes \
    --checkpoint checkpoints/best.pth \
    --mode       all
```

**All arguments / 全引数:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | *(required)* | Cityscapes root directory |
| `--checkpoint` | `checkpoints/best.pth` | Path to `.pth` checkpoint |
| `--mode` | `predictions` | `predictions` / `best_worst` / `failure_modes` / `edge_cases` / `all` |
| `--n` | `6` | Number of samples to display |
| `--n_scan` | `200` | Images to scan for failure-mode analysis |
| `--threshold` | `0.5` | Sigmoid threshold τ |
| `--conditions` | all four | Edge-case conditions: `rain` `shadow` `fog` `night` |
| `--out_dir` | `visualization_results` | Output directory |

**Output files / 出力ファイル:**

```
visualization_results/
├── predictions_grid.png     ← image / GT / prediction / error map
├── best_vs_worst.png        ← highest and lowest IoU samples
├── failure_modes.png        ← FP-heavy and FN-heavy examples
└── edge_cases.png           ← rain / shadow / fog / night comparison
```

**Error map colours / エラーマップの色:**

| Colour | Meaning |
|--------|---------|
| Green | TP — Road correctly predicted / 道路を正しく予測 |
| Red | FP — Background misclassified as road / 背景を道路と誤認 |
| Amber | FN — Road missed by model / モデルが道路を見逃し |

---

## Quick-Start Checklist / クイックスタートチェックリスト

```
[ ] 1. pip install -r requirements.txt && pip install segmentation-models-pytorch
[ ] 2. Download & extract Cityscapes into data/cityscapes/
[ ] 3. python explore_data.py --root data/cityscapes       # verify data
[ ] 4. python train.py --data_root data/cityscapes --amp   # train
[ ] 5. python evaluate.py --data_root data/cityscapes      # evaluate
[ ] 6. python visualize_results.py --data_root data/cityscapes --mode all
```

---

## Further Reading / 参考資料

- Full project plan and knowledge notes: [`topics.md`](topics.md)
- Cityscapes dataset paper: [Cordts et al., 2016](https://arxiv.org/abs/1604.01685)
- segmentation-models-pytorch: [github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- Albumentations: [albumentations.ai](https://albumentations.ai/)
