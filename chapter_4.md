***

# Chapter 4: Hands-on Practice with PyTorch
# 第4章：PyTorchを用いたハンズオン実践

## 4.1 Environment & Setup / 環境構築と準備

The easiest way to start writing PyTorch code without worrying about local installations is by using **Google Colab**. It provides a free environment with access to a GPU, which speeds up neural network calculations.

ローカル環境の構築を気にせずにPyTorchのコードを書き始める最も簡単な方法は、**Google Colab**を使用することです。ニューラルネットワークの計算を高速化するGPUにアクセスできる無料の環境を提供しています。

**Importing Libraries / ライブラリのインポート**

First, we need to import the core libraries.
まず、コアとなるライブラリをインポートする必要があります。

```python
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available / GPUが利用可能かチェック
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} / 使用デバイス: {device}")
```

## 4.2 Loading Pre-trained Models / 学習済みモデルの読み込み

Training a segmentation model from scratch takes days and massive amounts of data. Instead, we will use a **Pre-trained Model** (a model that has already been trained by experts on large datasets like COCO or PASCAL VOC).

セグメンテーションモデルをゼロから学習させるには、何日もの時間と膨大なデータが必要です。代わりに、**学習済みモデル**（COCOやPASCAL VOCなどの大規模データセットで専門家によってすでに学習されたモデル）を使用します。

PyTorch provides these out-of-the-box via `torchvision.models.segmentation`. We will use the **FCN (Fully Convolutional Network)** with a ResNet50 backbone.

PyTorchは `torchvision.models.segmentation` を通じてこれらを標準で提供しています。ここでは、ResNet50をバックボーンに持つ**FCN（完全畳み込みネットワーク）**を使用します。

```python
# 1. Load the recommended weights (trained on COCO dataset)
# 1. 推奨される重み（COCOデータセットで学習済み）を読み込む
weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT

# 2. Initialize the model with these weights
# 2. これらの重みでモデルを初期化する
model = models.segmentation.fcn_resnet50(weights=weights)

# 3. Move model to GPU (if available) and set to Evaluation Mode
# 3. モデルをGPU（利用可能な場合）に移動し、評価モードに設定する
model = model.to(device)
model.eval() # VERY IMPORTANT! Disable training behavior like dropout
             # 非常に重要！ドロップアウトのような学習時の挙動を無効化します
```

## 4.3 Inference and Visualization / 推論と可視化



Now for the exciting part! We will load an image, process it, feed it to our model, and visualize the output mask.
さあ、お楽しみの部分です！画像を読み込み、処理し、モデルに入力して、出力マスクを可視化します。

**Step 1: Preprocessing the Image / ステップ1：画像の前処理**
Deep learning models are very strict about the input format. The image must be resized, converted to a tensor, and normalized mathematically. Luckily, the `weights` object we loaded automatically provides the exact transformations the model expects.
ディープラーニングモデルは入力形式に非常に厳格です。画像はリサイズされ、テンソルに変換され、数学的に正規化される必要があります。幸いなことに、読み込んだ `weights` オブジェクトには、モデルが期待する正確な変換（前処理）が自動的に用意されています。

```python
# Load your image (ensure you upload 'dog.jpg' to your Colab workspace)
# 画像を読み込む（Colabのワークスペースに 'dog.jpg' をアップロードしてください）
img = Image.open('dog.jpg').convert('RGB')

# Use the automatic transforms from our weights
# 重みから自動変換（前処理）を使用する
preprocess = weights.transforms()
input_tensor = preprocess(img).unsqueeze(0).to(device) # Add batch dimension (B, C, H, W)
                                                       # バッチ次元を追加
```

**Step 2: The Forward Pass / ステップ2：順伝播（推論）**
We pass the tensor into the model. We use `torch.no_grad()` to tell PyTorch not to calculate gradients, which saves memory and speeds up the process.
テンソルをモデルに入力します。 `torch.no_grad()` を使用してPyTorchに勾配を計算しないように指示します。これにより、メモリが節約され、処理が高速化されます。

```python
# Perform inference / 推論を実行
with torch.no_grad():
    output = model(input_tensor)['out'][0] # We only want the main output / メインの出力のみを取得
    
print("Output shape / 出力の形状:", output.shape) 
# Example Output: torch.Size([21, 520, 520]) 
# 21 represents the number of classes (20 objects + 1 background)
# 21はクラスの数（20の物体 + 1の背景）を表します
```

**Step 3: Extracting Classes and Visualizing / ステップ3：クラスの抽出と可視化**
The output has 21 channels (one for each class probability). We use `argmax` to find which channel has the highest probability for every single pixel.
出力は21チャンネル（各クラスの確率ごとに1つ）あります。 `argmax` を使用して、すべてのピクセルについて最も確率が高いチャンネルを見つけます。

```python
# Get the class index with the highest probability for each pixel
# 各ピクセルに対して最も確率の高いクラスのインデックスを取得
predicted_mask = output.argmax(0).cpu().numpy()

# Plot the original image and the predicted segmentation mask
# 元の画像と予測されたセグメンテーションマスクを描画
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img)
ax1.set_title('Original Image / 元画像')
ax1.axis('off')

# Use a colormap ('jet' or 'nipy_spectral') to colorize the integer classes
# カラーマップ（'jet' または 'nipy_spectral'）を使用して整数のクラスを色付け
ax2.imshow(predicted_mask, cmap='nipy_spectral') 
ax2.set_title('Segmentation Mask / セグメンテーションマスク')
ax2.axis('off')

plt.show()
```

By completing this code, your students will have successfully run a state-of-the-art semantic segmentation model in just a few lines of code!
このコードを完了することで、学生たちはわずか数行のコードで最先端のセマンティックセグメンテーションモデルを正常に実行できたことになります！

***

## References & Further Reading / 参考文献と参考資料

1.  **PyTorch Torchvision Models Documentation:**
    * *Web:* [pytorch.org/vision/stable/models.html#semantic-segmentation](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
    * *Note:* The official documentation detailing all available pre-trained segmentation models (FCN, DeepLabV3, LRASPP) and their weights. (利用可能なすべての学習済みセグメンテーションモデルと重みが詳しく記載された公式ドキュメント。)
2.  **Google Colaboratory:**
    * *Web:* [colab.research.google.com](https://colab.research.google.com/)
    * *Note:* Cloud-based Jupyter notebook environment for easy PyTorch execution. (PyTorchを簡単に実行できるクラウドベースのJupyter Notebook環境。)
