---

# Chapter 4: Hands-on Practice with PyTorch
# 第4章：PyTorchを用いたハンズオン実践

## 4.1 Environment & Setup / 環境構築と準備

The easiest way to get started with PyTorch — without installing anything locally — is **Google Colab**. It runs entirely in your browser and provides free access to a GPU, which dramatically speeds up neural network calculations.

PyTorchを使い始める最も簡単な方法は**Google Colab**です。ローカルへのインストールは不要で、ブラウザ上で完結します。また、ニューラルネットワークの計算を大幅に高速化するGPUへの無料アクセスも提供されています。

**How to enable GPU in Colab / ColabでGPUを有効にする方法**

> Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save

**Importing Libraries / ライブラリのインポート**

Once the environment is ready, import the libraries we will use throughout this chapter.
環境が整ったら、この章で使用するライブラリをインポートします。

```python
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Check if GPU is available; fall back to CPU if not
# GPUが利用可能かチェックし、利用できない場合はCPUを使用する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

> **Tip:** If the output says `cuda`, the GPU is active and your code will run much faster. If it says `cpu`, the code still works — just a bit slower.
>
> **ヒント：** 出力が `cuda` であればGPUが有効で、コードはより速く実行されます。`cpu` でもコードは問題なく動きます — 少し遅くなるだけです。

---

## 4.2 Loading Pre-trained Models / 学習済みモデルの読み込み

Training a segmentation model from scratch requires days of computation and millions of labeled images. Instead, we will use a **pre-trained model** — a model that experts have already trained on a large dataset called **COCO** (Common Objects in Context), which contains over 118,000 photos with pixel-level annotations.

セグメンテーションモデルをゼロから学習させるには、何日もの計算時間と数百万枚のラベル付き画像が必要です。代わりに、**学習済みモデル**を使用します。これは、**COCO**（Common Objects in Context）と呼ばれる大規模データセット（11万8,000枚以上のピクセルレベルアノテーション付き写真）で専門家がすでに学習させたモデルです。

PyTorch provides pre-trained segmentation models out-of-the-box through `torchvision.models.segmentation`. We will use **FCN with a ResNet-50 backbone** — the architecture we studied in Chapter 3, backed by a powerful feature extractor.

PyTorchは `torchvision.models.segmentation` を通じて学習済みセグメンテーションモデルを標準で提供しています。ここでは第3章で学んだ**FCN（ResNet-50バックボーン）**を使用します。

```python
# Step 1: Select the recommended pre-trained weights (trained on COCO with 21 classes)
# ステップ1: 推奨の学習済み重みを選択（COCOデータセット・21クラス）
weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT

# Step 2: Build the model and load the pre-trained weights into it
# ステップ2: モデルを構築し、学習済み重みを読み込む
model = models.segmentation.fcn_resnet50(weights=weights)

# Step 3: Move the model to the GPU (if available) and switch to evaluation mode
# ステップ3: モデルをGPUに移動し、評価モードに切り替える
model = model.to(device)
model.eval()  # IMPORTANT: disables dropout / batch-norm training behaviour
              # 重要: ドロップアウト・バッチ正規化の学習時の挙動を無効化します
```

> **Why `model.eval()`?**  
> During training, layers like Dropout randomly zero out values to prevent overfitting. During inference we want consistent, deterministic outputs, so we call `eval()` to disable this behaviour.
>
> **なぜ `model.eval()` が必要か？**  
> 学習中、Dropoutなどの層は過学習を防ぐためにランダムに値をゼロにします。推論時は一貫した決定論的な出力が必要なため、`eval()` を呼び出してこの挙動を無効にします。

**What can this model detect? / このモデルは何を検出できるか？**

The COCO-trained FCN recognises 20 object categories plus background (21 classes total):
COCOで学習したFCNは、20のオブジェクトカテゴリと背景（計21クラス）を認識します。

```python
# The 21 PASCAL VOC / COCO class names (index 0 = background)
# 21クラスの名前（インデックス0 = 背景）
VOC_CLASSES = [
    "background",   # 0
    "aeroplane",    # 1
    "bicycle",      # 2
    "bird",         # 3
    "boat",         # 4
    "bottle",       # 5
    "bus",          # 6
    "car",          # 7
    "cat",          # 8
    "chair",        # 9
    "cow",          # 10
    "diningtable",  # 11
    "dog",          # 12
    "horse",        # 13
    "motorbike",    # 14
    "person",       # 15
    "pottedplant",  # 16
    "sheep",        # 17
    "sofa",         # 18
    "train",        # 19
    "tvmonitor",    # 20
]
print(f"Total classes: {len(VOC_CLASSES)}")
```

---

## 4.3 Inference and Visualization / 推論と可視化

Now for the exciting part. We will load a real image, preprocess it, pass it through the model, and visualize the resulting segmentation mask.

いよいよお楽しみの部分です。実際の画像を読み込み、前処理し、モデルに通して、セグメンテーションマスクを可視化します。

```
Raw Image → Preprocess → Model → Output (21 channels) → argmax → Predicted Mask
画像       →  前処理   → モデル → 出力（21チャンネル） → argmax → 予測マスク
```

---

**Step 1: Preprocessing the Image / ステップ1：画像の前処理**

Deep learning models are strict about the exact format of their inputs. The image must be:
1. Resized to a size the model expects
2. Converted from a PIL image to a PyTorch tensor (`[C, H, W]` with values in `[0, 1]`)
3. Normalised using the mean and standard deviation of the training dataset

ディープラーニングモデルは入力形式に非常に厳格です。画像は以下の処理が必要です。
1. モデルが期待するサイズにリサイズ
2. PIL画像からPyTorchテンソル（値が `[0, 1]` の `[C, H, W]` 形式）に変換
3. 学習データセットの平均と標準偏差で正規化

Luckily, the `weights` object already knows the exact transformation the model was trained with, so we do not need to hard-code any numbers.

幸い、`weights` オブジェクトにはモデルの学習時に使用された正確な変換が含まれているため、数値をハードコードする必要はありません。

```python
# Load your image (upload an image file to your Colab workspace first)
# 画像を読み込む（先にColabのワークスペースに画像ファイルをアップロードしてください）
IMAGE_PATH = "dog.jpg"  # <-- change this to your image file name
img = Image.open(IMAGE_PATH).convert("RGB")

# Get the exact preprocessing pipeline the model was trained with
# モデルの学習時に使われた前処理パイプラインを取得する
preprocess = weights.transforms()

# Apply preprocessing and add a batch dimension: (C, H, W) → (1, C, H, W)
# 前処理を適用し、バッチ次元を追加する
input_tensor = preprocess(img).unsqueeze(0).to(device)
print("Input tensor shape:", input_tensor.shape)
# e.g. torch.Size([1, 3, 520, 520])
```

---

**Step 2: The Forward Pass / ステップ2：順伝播（推論）**

We pass the preprocessed tensor into the model. Wrapping the call in `torch.no_grad()` tells PyTorch not to store gradient information — we are only doing inference, not training, so this saves memory and time.

前処理済みテンソルをモデルに通します。`torch.no_grad()` でラップすることで、PyTorchに勾配情報を保持しないよう指示します。推論のみを行うため（学習ではないため）、これによりメモリと時間を節約できます。

```python
with torch.no_grad():
    raw_output = model(input_tensor)

# torchvision segmentation models return a dict; 'out' is the main prediction
# torchvisionのセグメンテーションモデルはdictを返す。'out' がメインの予測
output = raw_output["out"][0]  # shape: (21, H, W) — one channel per class
                               # 形状: (21, H, W) — クラスごとに1チャンネル

print("Output tensor shape:", output.shape)
# e.g. torch.Size([21, 520, 520])
# 21 channels = 20 object classes + 1 background class
# 21チャンネル = 20のオブジェクトクラス + 1の背景クラス
```

> **What does each channel represent?**  
> Each of the 21 channels holds a "confidence score" (logit) for every pixel. A high value in channel 12 means the model is confident that pixel belongs to class 12 (`dog`). The final step is to pick the winning channel for each pixel.
>
> **各チャンネルは何を表しているか？**  
> 21チャンネルのそれぞれが、全ピクセルに対する「信頼スコア」（ロジット）を持っています。チャンネル12の値が高い場合、モデルはそのピクセルがクラス12（`dog`）に属すると確信しているということです。最後のステップは各ピクセルの最大スコアのチャンネルを選択することです。

---

**Step 3: Extracting the Predicted Mask / ステップ3：予測マスクの抽出**

`argmax(dim=0)` selects the channel index with the highest score for each pixel, collapsing the 21 channels into a single 2D map of class indices.

`argmax(dim=0)` は各ピクセルで最高スコアのチャンネルインデックスを選択し、21チャンネルをクラスインデックスの単一の2Dマップに集約します。

```python
# For each pixel, find the class with the highest confidence score
# 各ピクセルについて、最も信頼スコアの高いクラスを選択する
predicted_mask = output.argmax(dim=0).cpu().numpy()  # shape: (H, W)

print("Predicted mask shape:", predicted_mask.shape)
print("Unique class indices found:", np.unique(predicted_mask))
print("Detected classes:", [VOC_CLASSES[i] for i in np.unique(predicted_mask)])
```

---

**Step 4: Visualizing the Results / ステップ4：結果の可視化**

```python
# Build a colour palette: one distinct colour per class
# クラスごとに異なる色のカラーパレットを作成する
np.random.seed(42)
palette = np.random.randint(50, 220, size=(len(VOC_CLASSES), 3), dtype=np.uint8)
palette[0] = [0, 0, 0]  # class 0 (background) = black / クラス0（背景）= 黒

# Convert the integer mask to an RGB colour image
# 整数マスクをRGBカラー画像に変換する
colored_mask = palette[predicted_mask]  # shape: (H, W, 3)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(img)
axes[0].set_title("Original Image / 元画像")
axes[0].axis("off")

axes[1].imshow(colored_mask)
axes[1].set_title("Segmentation Mask / セグメンテーションマスク")
axes[1].axis("off")

# Overlay: blend original image with the mask (alpha = 0.5)
# オーバーレイ：元画像とマスクをブレンド（alpha = 0.5）
img_array = np.array(img.resize((colored_mask.shape[1], colored_mask.shape[0])))
blended = (0.5 * img_array + 0.5 * colored_mask).astype(np.uint8)
axes[2].imshow(blended)
axes[2].set_title("Overlay / オーバーレイ")
axes[2].axis("off")

# Add a legend for detected classes
# 検出されたクラスの凡例を追加する
detected_ids = np.unique(predicted_mask)
legend_patches = [
    mpatches.Patch(color=palette[i] / 255.0, label=VOC_CLASSES[i])
    for i in detected_ids
]
axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1),
               loc="upper left", fontsize=9)

plt.suptitle("FCN-ResNet50 Semantic Segmentation (COCO Weights)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("segmentation_result.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved as segmentation_result.png")
```

> **Reading the output / 出力の読み方**
>
> * Each colour in the mask corresponds to one object class.  
>   マスクの各色は1つのオブジェクトクラスに対応しています。
> * Pixels coloured **black** are classified as "background."  
>   **黒**に色付けされたピクセルは「背景」に分類されています。
> * The **overlay panel** makes it easy to see how well the model boundaries align with the real object edges.  
>   **オーバーレイパネル**では、モデルの境界が実際のオブジェクトのエッジとどれだけ一致しているかを確認しやすくなっています。

---

**Congratulations! / おめでとうございます！**

You have just run a state-of-the-art semantic segmentation model in fewer than 30 lines of code. The key takeaways from this chapter are:

わずか30行未満のコードで、最先端のセマンティックセグメンテーションモデルを実行することができました。この章の重要なポイントは以下の通りです。

| Step | What we did | 何をしたか |
|------|-------------|------------|
| 1 | Loaded pre-trained FCN weights from `torchvision` | `torchvision` からFCNの学習済み重みを読み込んだ |
| 2 | Preprocessed the image using the model's built-in transforms | モデル組み込みの変換で画像を前処理した |
| 3 | Ran a forward pass inside `torch.no_grad()` | `torch.no_grad()` 内で順伝播を実行した |
| 4 | Applied `argmax` to get per-pixel class predictions | `argmax` でピクセルごとのクラス予測を取得した |
| 5 | Visualized the result with a colour-coded overlay | カラーコードオーバーレイで結果を可視化した |

---

## References & Further Reading / 参考文献と参考資料

1. **PyTorch Torchvision Segmentation Models:**
   * *Link:* [pytorch.org/vision/stable/models.html#semantic-segmentation](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
   * *Note:* Official documentation for all available pre-trained models (FCN, DeepLabV3, LRASPP) and their weight options.（利用可能なすべての学習済みモデルと重みオプションの公式ドキュメント。）

2. **Google Colaboratory:**
   * *Link:* [colab.research.google.com](https://colab.research.google.com/)
   * *Note:* Free cloud Jupyter notebook environment with GPU support — the recommended place to run this chapter's code.（GPU対応の無料クラウドJupyter Notebook環境 — この章のコードを実行するための推奨環境。）

3. **COCO Dataset:**
   * *Link:* [cocodataset.org](https://cocodataset.org/)
   * *Note:* The dataset the model was trained on. Browse it to understand the scale and variety of training data behind the pre-trained weights.（モデルの学習に使用されたデータセット。学習済み重みの背景にある学習データの規模と多様性を理解するために参照してください。）
