# Project Deep Dive: Drivable Area Segmentation
# プロジェクト詳細：走行可能領域のセグメンテーション

> **Goal / 目標**  
> Train a binary semantic segmentation model that classifies every pixel in a driving scene as either **Drivable** (road) or **Background** (everything else), using the Cityscapes dataset.  
> Cityscapesデータセットを使い、走行シーンの各ピクセルを**走行可能**（道路）または**背景**（それ以外）に分類する二値セマンティックセグメンテーションモデルを訓練する。

---

## 1. Introduction: Why Drivable Area? / 導入：なぜ「走行可能領域」なのか？

Before an autonomous vehicle can navigate, avoid obstacles, or obey traffic lights, it must answer one fundamental question: **"Where is it safe to drive?"**

自動運転車がナビゲーションを行い、障害物を避け、信号に従う前に、まず一つの根本的な問いに答える必要があります。**「どこを走るのが安全か？」**

While Object Detection draws boxes around discrete items (cars, pedestrians, bicycles), Drivable Area Segmentation understands the **continuous, topological structure of the ground** — giving the vehicle a precise map of its operable space at every frame.

物体検出が個別のアイテム（車・歩行者・自転車）を四角い枠で囲むのに対し、走行可能領域のセグメンテーションは**地面の連続的なトポロジカル構造**を理解し、車両がフレームごとに操作可能な空間の正確なマップを取得できるようにします。

---

## 2. Project Plan Overview / プロジェクト計画の概要

| Phase | Topic | 内容 | Deliverable / 成果物 |
|-------|-------|------|----------------------|
| **1** | Dataset Setup | データセット準備 | Cityscapes downloaded & structured |
| **2** | Data Pipeline | データパイプライン | PyTorch `Dataset` + `DataLoader` |
| **3** | Model | モデル | U-Net with ResNet encoder |
| **4** | Training | 訓練 | Trained weights (`.pth`) |
| **5** | Evaluation | 評価 | IoU / F1 score report |
| **6** | Visualisation | 可視化 | Overlay plots + edge-case analysis |

---

## 3. Dataset: Cityscapes / データセット：Cityscapes

### 3.1 About the Dataset / データセットについて

Cityscapes is the benchmark dataset for urban driving scene understanding. It was collected across **50 cities** in Germany and neighbouring countries over several seasons and weather conditions.

CityscapesはUrban driving scene understandingの標準ベンチマークデータセットです。ドイツおよび近隣諸国の**50都市**において、複数の季節と天候条件下で収集されました。

| Property | Value |
|----------|-------|
| Resolution | 1024 × 2048 px |
| Fine-annotated images | 5,000 (2975 train / 500 val / 1525 test) |
| Coarse-annotated images | 20,000 (extra training data) |
| Annotation classes | 30 categories → grouped into **19 training classes** |
| Annotation type | Pixel-level polygon labels |
| Our binary mapping | `road` (class 0) → **Drivable = 1**; all others → **Background = 0** |

### 3.2 Download Instructions / ダウンロード手順

> **Registration required / 登録が必要です**  
> Cityscapes requires a free account. Sign up at the link below before downloading.  
> Cityscapesの利用には無料アカウントが必要です。ダウンロード前に以下のリンクで登録してください。

**Step 1 — Register / ステップ1：登録**  
[https://www.cityscapes-dataset.com/register/](https://www.cityscapes-dataset.com/register/)

**Step 2 — Download page / ステップ2：ダウンロードページ**  
[https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)

Download the following two packages (login required):  
以下の2つのパッケージをダウンロードしてください（要ログイン）：

| File | Size | Contents |
|------|------|----------|
| `leftImg8bit_trainvaltest.zip` | ~11 GB | RGB camera images (train / val / test) |
| `gtFine_trainvaltest.zip` | ~241 MB | Fine pixel-level annotations |

**Step 3 — Command-line download (optional) / ステップ3：コマンドラインでのダウンロード（任意）**

After logging in via your browser, you can use `wget` with your session cookie, or use the official `cityscapesscripts` helper:

ブラウザでログインした後、セッションクッキーを使った `wget`、または公式の `cityscapesscripts` ヘルパーを使用できます。

```bash
pip install cityscapesscripts

# Set your credentials as environment variables
export CITYSCAPES_USERNAME="your_username"
export CITYSCAPES_PASSWORD="your_password"

# Download fine annotations and left images
csDownload gtFine_trainvaltest.zip
csDownload leftImg8bit_trainvaltest.zip
```

### 3.3 Directory Structure / ディレクトリ構成

After extracting both zip files into `data/`, the structure should look like this.  
2つのzipファイルを `data/` に解凍すると、以下のような構成になります。

```
data/
└── cityscapes/
    ├── leftImg8bit/          ← RGB camera images / RGB画像
    │   ├── train/
    │   │   ├── aachen/
    │   │   │   ├── aachen_000000_000019_leftImg8bit.png
    │   │   │   └── ...
    │   │   └── ... (17 more cities / 他17都市)
    │   ├── val/
    │   │   └── ... (3 cities / 3都市)
    │   └── test/
    │       └── ...
    └── gtFine/               ← Pixel-level annotations / ピクセルレベルアノテーション
        ├── train/
        │   ├── aachen/
        │   │   ├── aachen_000000_000019_gtFine_labelIds.png   ← class IDs
        │   │   ├── aachen_000000_000019_gtFine_instanceIds.png
        │   │   ├── aachen_000000_000019_gtFine_color.png      ← colour-coded mask
        │   │   └── ...
        │   └── ...
        ├── val/
        └── test/
```

### 3.4 Class Mapping: 19 → Binary / クラスマッピング：19クラス → 二値

Cityscapes has 19 training classes. We collapse them into a binary task:  
Cityscapesには19の訓練クラスがあります。これらを二値タスクに統合します。

| Cityscapes class | Label ID | Our mapping | マッピング |
|-----------------|----------|-------------|-----------|
| `road` | 0 | **1 — Drivable / 走行可能** | ✅ |
| `sidewalk` | 1 | 0 — Background / 背景 | ❌ |
| `building` | 2 | 0 — Background / 背景 | ❌ |
| `wall` | 3 | 0 — Background / 背景 | ❌ |
| `fence` | 4 | 0 — Background / 背景 | ❌ |
| `pole` | 5 | 0 — Background / 背景 | ❌ |
| `traffic light` | 6 | 0 — Background / 背景 | ❌ |
| `traffic sign` | 7 | 0 — Background / 背景 | ❌ |
| `vegetation` | 8 | 0 — Background / 背景 | ❌ |
| `terrain` | 9 | 0 — Background / 背景 | ❌ |
| `sky` | 10 | 0 — Background / 背景 | ❌ |
| `person` | 11 | 0 — Background / 背景 | ❌ |
| `rider` | 12 | 0 — Background / 背景 | ❌ |
| `car` | 13 | 0 — Background / 背景 | ❌ |
| `truck` | 14 | 0 — Background / 背景 | ❌ |
| `bus` | 15 | 0 — Background / 背景 | ❌ |
| `train` | 16 | 0 — Background / 背景 | ❌ |
| `motorcycle` | 17 | 0 — Background / 背景 | ❌ |
| `bicycle` | 18 | 0 — Background / 背景 | ❌ |

> **Note / 注意**  
> The `gtFine_labelIds.png` masks use the **raw label IDs** from Cityscapes (range 0–33).  
> Road corresponds to label ID **7** in the raw file (not 0). Use `cityscapesscripts` or map manually:  
> `gtFine_labelIds.png` のマスクはCityscapesの**生のラベルID**（0〜33）を使用します。  
> 道路の生ラベルIDは **7** です（0ではありません）。`cityscapesscripts` を使うか手動でマッピングしてください：
>
> ```python
> # Raw label ID 7 = road → our binary label 1
> binary_mask = (label_id_mask == 7).astype(np.uint8)
> ```

---

## 4. Core Knowledge Points / 中核となる知識ポイント

### 4.1 Binary Classification at the Pixel Level / ピクセルレベルの二値分類

At its simplest, this project reduces the complex world into a binary decision for every pixel:  
最も単純な形として、このプロジェクトはすべてのピクセルに対する「二値の決定」に落とし込みます。

- **Class 1 — Drivable / 走行可能:** The pixel belongs to road surface the ego-vehicle can safely and legally drive on. （自車が安全かつ合法的に走行できる路面のピクセル。）
- **Class 0 — Background / 背景:** Everything else — sidewalks, buildings, sky, parked cars, pedestrians. （それ以外のすべて — 歩道・建物・空・停車中の車・歩行者。）

For an image with $N$ pixels, the model outputs a probability map $P \in [0,1]^N$. A threshold (default $\tau = 0.5$) converts probabilities to binary labels:

$N$ 個のピクセルを持つ画像に対し、モデルは確率マップ $P \in [0,1]^N$ を出力します。閾値（デフォルト $\tau = 0.5$）で確率を二値ラベルに変換します。

$$\hat{y}_i = \begin{cases} 1 & \text{if } p_i \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

### 4.2 Receptive Field and Context / 受容野とコンテキスト

To classify a grey pixel as "Road", the model cannot look at that single pixel in isolation — a grey pixel on concrete looks identical to one on asphalt. The model must learn to use **context (コンテキスト)**: it looks at surrounding pixels (the **Receptive Field / 受容野**) and reasons: *"This grey area sits below the sky and vehicles are driving on it — it must be road."*

灰色のピクセルを「道路」として分類するためには、モデルはそのピクセルだけを単独で見ることはできません — コンクリートの灰色とアスファルトの灰色は見た目が同じだからです。モデルは**コンテキスト**を使うことを学習しなければなりません：周囲のピクセル（**受容野**）を参照して、「この灰色の領域は空の下にあり、車がその上を走っているから道路に違いない」と推論するのです。

Deeper layers in the encoder have a **larger receptive field**, allowing the model to incorporate global scene context before making a pixel-level decision.

エンコーダのより深い層は**より大きな受容野**を持ち、ピクセルレベルの判断を行う前に大域的なシーン文脈を取り込むことができます。

### 4.3 Loss Function: Binary Cross-Entropy + Dice / 損失関数：BCE + Dice

For binary segmentation, we combine two complementary losses:  
二値セグメンテーションでは、補完的な2つの損失を組み合わせます。

**Binary Cross-Entropy (BCE) / 二値交差エントロピー** — penalises wrong pixel-level confidence:  
ピクセルレベルの誤った信頼度を罰します：

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i)\log(1 - p_i) \right]$$

**Dice Loss / Diceロス** — directly maximises the IoU-like overlap between prediction and ground truth (important when the road class is a small fraction of the image):  
予測と正解のIoUに近い重なりを直接最大化します（道路クラスが画像の小さな割合しか占めない場合に重要です）：

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i}{\sum_i p_i + \sum_i y_i}$$

**Combined loss / 組み合わせ損失:**

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

---

## 5. Deep Dive into Challenges / 課題の深掘り

Training a model to find the road on a sunny, well-marked highway is straightforward. The real difficulty — and what makes this a great research topic — lies in these **edge cases**:

晴れた日の白線がはっきりした高速道路で道路を見つけるモデルの訓練は容易です。本当の困難、そしてこれを優れた研究テーマにしているのは、以下の**エッジケース**にあります。

### Challenge A: Weather & Illumination / 天候と照明

| Condition | Why it's hard | 難しい理由 |
|-----------|---------------|-----------|
| **Rain / 雨** | Wet roads act as mirrors, reflecting sky and headlights — the texture of "road" looks completely different. | 濡れた路面が鏡となり空や前照灯を反射 — 「道路」のテクスチャが大きく変化します。 |
| **Night / 夜間** | Only the headlight cone is lit; the rest of the road is near-black, with little texture. | ヘッドライトの円錐部分のみが照らされ、道路の大部分がほぼ黒でテクスチャがほとんどありません。 |
| **Shadows / 影** | Sharp shadows from trees or buildings create dark bands that the model mistakes for obstacles or gaps. | 木や建物からの鋭い影が暗い帯を作り、モデルが障害物や隙間と誤認します。 |
| **Glare / 逆光** | Direct sunlight saturates the camera sensor, washing out road markings and colour cues. | 直射日光がカメラセンサーを飽和させ、道路標示や色の手がかりを消します。 |

### Challenge B: Unstructured Roads / 非構造化道路

In cities (Cityscapes), roads have sharp curbs and painted lines. In rural areas, the transition from "road" to "dirt shoulder" or "grass" is gradual and ambiguous — even human annotators disagree on the exact boundary.

都市部（Cityscapes）では道路に明確な縁石と白線があります。しかし田舎道では、「道路」から「土の路肩」や「草地」への移行は緩やかで曖昧です — 人間のアノテーターでさえ正確な境界について意見が分かれます。

### Challenge C: Dynamic Occlusion / 動的な遮蔽

Heavy traffic means the road surface is partially blocked by other vehicles. The model must:
1. Segment only the **visible** road portions.
2. **Infer** the likely drivable space hidden beneath and between vehicles.

交通量が多い場合、路面は他の車両によって部分的に隠されます。モデルは以下を行う必要があります。
1. **見えている**道路部分のみをセグメンテーションする。
2. 車両の下や間に隠れている走行可能空間を**推論**する。

---

## 6. Implementation Plan / 実装計画

### Phase 1 — Data Pipeline / フェーズ1：データパイプライン

**Goal:** Build a `torch.utils.data.Dataset` that loads Cityscapes images and converts the 19-class masks to binary.  
**目標：** Cityscapes画像を読み込み、19クラスのマスクを二値に変換する `torch.utils.data.Dataset` を構築する。

```python
# Pseudocode outline / 疑似コードの概要
class CityscapesBinaryDataset(Dataset):
    def __init__(self, root, split, transform):
        # Collect (image_path, mask_path) pairs for the given split
        # 指定されたsplitの (image_path, mask_path) ペアを収集する
        ...

    def __getitem__(self, idx):
        img  = load_rgb_image(self.img_paths[idx])    # (H, W, 3)
        mask = load_label_mask(self.mask_paths[idx])  # (H, W)  raw label IDs
        binary_mask = (mask == 7).long()              # road label ID = 7 → 1, rest → 0
        return self.transform(img), binary_mask
```

Key transforms to implement / 実装すべき主要な変換：
- `RandomHorizontalFlip` — data augmentation / データ拡張
- `RandomCrop(512, 512)` — crop to manageable size / 扱いやすいサイズにクロップ
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — ImageNet stats

### Phase 2 — Model Selection / フェーズ2：モデルの選定

We recommend **U-Net with a pre-trained ResNet-34 encoder** as the starting point. It balances accuracy and training speed well for a binary task.  
出発点として**ResNet-34エンコーダを持つ事前学習済みU-Net**を推奨します。二値タスクに対して精度と訓練速度のバランスが取れています。

| Model | Encoder | Pre-trained? | Notes | 備考 |
|-------|---------|--------------|-------|------|
| **U-Net (recommended)** | ResNet-34 | ✅ ImageNet | Fast to train, sharp boundaries | 訓練が速く境界が鮮明 |
| DeepLabV3+ | ResNet-50 | ✅ ImageNet | Better for complex scenes | 複雑なシーンに強い |
| SegFormer-B0 | MiT-B0 | ✅ ImageNet | Transformer-based, lightweight | Transformer系・軽量 |

```python
# Using the segmentation_models_pytorch library (recommended)
# segmentation_models_pytorchライブラリを使用（推奨）
pip install segmentation-models-pytorch

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,           # binary: single output channel
    activation="sigmoid" # output in [0, 1]
)
```

### Phase 3 — Training / フェーズ3：訓練

**Recommended hyperparameters / 推奨ハイパーパラメータ:**

| Parameter | Value | 説明 |
|-----------|-------|------|
| Optimizer | `AdamW` | Weight decay prevents overfitting |
| Learning rate | `1e-4` | Start here; reduce on plateau |
| LR scheduler | `ReduceLROnPlateau` | Halve LR when val IoU stops improving |
| Batch size | 8–16 | Adjust to GPU memory |
| Epochs | 30–50 | Use early stopping (patience=7) |
| Input crop size | 512 × 512 | Balanced memory / context trade-off |

```python
# Training loop skeleton / 訓練ループのスケルトン
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
criterion = smp.losses.DiceLoss("binary") + smp.losses.SoftBCEWithLogitsLoss()

for epoch in range(NUM_EPOCHS):
    # --- train ---
    model.train()
    for images, masks in train_loader:
        pred   = model(images)
        loss   = criterion(pred, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # --- validate ---
    val_iou = evaluate(model, val_loader)
    scheduler.step(val_iou)
    print(f"Epoch {epoch+1}  val IoU: {val_iou:.4f}")
```

### Phase 4 — Evaluation / フェーズ4：評価

We evaluate using three complementary metrics:  
3つの補完的なメトリクスで評価します。

| Metric | Formula | What it measures | 測定内容 |
|--------|---------|-----------------|---------|
| **IoU** (Jaccard) | $\frac{TP}{TP+FP+FN}$ | Overlap quality | 重なりの品質 |
| **Dice / F1** | $\frac{2TP}{2TP+FP+FN}$ | Harmonic mean of precision & recall | 適合率・再現率の調和平均 |
| **Pixel Accuracy** | $\frac{TP+TN}{N}$ | Overall correct pixels | 全体的な正解ピクセル率 |

**Target benchmarks on Cityscapes road class / Cityscapesの道路クラスの目標ベンチマーク:**

| Model | IoU (road) |
|-------|-----------|
| Baseline FCN | ~94% |
| U-Net ResNet-34 | ~96–97% |
| DeepLabV3+ ResNet-50 | ~97–98% |
| State-of-the-art | ~98.5%+ |

### Phase 5 — Visualisation & Analysis / フェーズ5：可視化と分析

Plot predictions overlaid on the original images for:  
以下の条件で、元画像に重ねた予測結果を描画します。

1. **Best-case examples / ベストケース例** — sunny day, clear road markings
2. **Worst-case examples / ワーストケース例** — rain, night, heavy traffic
3. **Failure modes / 失敗モード** — where does the model make mistakes? Why?
4. **Threshold analysis / 閾値の分析** — how does IoU change as $\tau$ varies from 0.3 to 0.7?

---

## 7. Challenges (Edge Cases) Summary / 課題（エッジケース）のまとめ

| Edge Case | Expected model behaviour | 期待されるモデルの挙動 | Mitigation strategy | 対策 |
|-----------|--------------------------|------------------------|---------------------|------|
| Rain / 雨 | Mask fragments, specular noise | マスクが断片化・鏡面ノイズ | Add rain augmentation (albumentations) | 雨のデータ拡張を追加 |
| Night / 夜間 | Misses dark road, over-segments lit areas | 暗い道路を見落とし、明るい領域を過剰検出 | Include night images; brightness jitter | 夜間画像を含める・輝度ジッター |
| Shadows / 影 | Creates "holes" in the drivable mask | 走行可能マスクに「穴」ができる | Shadow augmentation; CRF post-processing | 影の拡張・CRF後処理 |
| Occluded road / 遮蔽された道路 | Under-segments (misses area under cars) | 過少検出（車の下の領域を見逃す） | Instance-aware training; context features | インスタンス認識訓練・文脈特徴 |

---

## 8. References & Further Reading / 参考文献と参考資料

1. **Cityscapes Dataset Paper:**  
   Cordts, M., et al. (2016). "The Cityscapes Dataset for Semantic Urban Scene Understanding." *CVPR 2016*.  
   * *Link:* [https://arxiv.org/abs/1604.01685](https://arxiv.org/abs/1604.01685)

2. **Cityscapes Official Website / 公式サイト:**  
   * *Download:* [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)  
   * *Benchmark:* [https://www.cityscapes-dataset.com/benchmarks/](https://www.cityscapes-dataset.com/benchmarks/)

3. **segmentation_models_pytorch Library:**  
   Iakubovskii, P. (2019). Segmentation Models PyTorch.  
   * *GitHub:* [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

4. **U-Net Paper:**  
   Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015*.  
   * *Link:* [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

5. **Albumentations (data augmentation library):**  
   * *Docs:* [https://albumentations.ai/docs/](https://albumentations.ai/docs/)  
   * *Note:* Provides rain, fog, shadow, and brightness augmentations essential for this project. / このプロジェクトに不可欠な雨・霧・影・輝度の拡張を提供します。
