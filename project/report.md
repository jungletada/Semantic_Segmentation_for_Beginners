# Cityscapes における走行可能領域セグメンテーションの比較評価

## 概要

本研究では，自動運転シーン理解の基礎課題として，Cityscapes データセットにおける走行可能領域の二値セマンティックセグメンテーションを扱う。具体的には，Cityscapes の `gtFine_labelIds` に含まれる road の生ラベル ID 7 を正例，その他の画素を背景として二値化し，U-Net + ResNet34 と DeepLabV3+ + ResNet50 の二つのモデルを比較した。両モデルはいずれも ImageNet 事前学習済みエンコーダを用い，Dice loss と Binary Cross Entropy loss の和を目的関数として AdamW により学習した。検証分割 500 枚に対する正式評価の結果，U-Net + ResNet34 は IoU 0.9377，Dice 0.9585，Pixel Accuracy 0.9754 を達成し，DeepLabV3+ + ResNet50 は IoU 0.9365，Dice 0.9581，Pixel Accuracy 0.9738 を達成した。両者の差は小さいが，本実験条件では U-Net + ResNet34 がわずかに高い総合性能を示した。一方で，混同行列の比較から DeepLabV3+ は偽陰性を抑える傾向を示し，運用目的に応じた閾値選択の重要性が確認された。

キーワード: セマンティックセグメンテーション，Cityscapes，走行可能領域，U-Net，DeepLabV3+

## 1. はじめに

自動運転システムにおいて，道路領域を画素単位で推定することは，走行可能空間の把握，経路計画，障害物回避の前処理として重要である。一般的な都市環境では，車線，歩道，建物，車両，影，路面反射などが複雑に混在するため，単純な色やエッジに基づく規則では安定した道路抽出が困難である。このため，畳み込みニューラルネットワークに基づくセマンティックセグメンテーションが広く用いられている。

本研究の目的は，Cityscapes における道路画素を対象とした二値セグメンテーション課題に対し，代表的なエンコーダ・デコーダ型モデルである U-Net と，空洞畳み込みを利用する DeepLabV3+ を同一データ分割および同一評価手順で比較することである。多クラスのシーン解析ではなく，road クラスに限定した二値化問題として定式化することにより，走行可能領域抽出におけるモデル特性を明確に観察する。

本稿の貢献は以下の三点である。第一に，Cityscapes の road 生ラベル ID 7 を用いた二値マスク生成から学習・評価までの一貫した実験系を構築した。第二に，U-Net + ResNet34 と DeepLabV3+ + ResNet50 を同一条件で学習し，IoU，Dice，Pixel Accuracy，閾値探索，混同行列に基づき比較した。第三に，検証結果から実用上の閾値選択および偽陽性・偽陰性の傾向について考察した。

## 2. 関連研究

Cityscapes は，都市走行環境における高品質な画素単位アノテーションを提供する代表的なベンチマークであり，セマンティックセグメンテーション研究において広く用いられている [1]。本研究では，Cityscapes の多クラスラベルをそのまま用いるのではなく，road クラスを走行可能領域として抽出し，背景との二値分類問題に変換する。

U-Net は，エンコーダで抽出した特徴をデコーダへスキップ接続する構造により，局所的な境界情報と大域的な文脈情報を統合する代表的なセグメンテーションモデルである [2]。本来は医用画像解析を想定して提案されたが，少数クラスや境界復元が重要な課題において有効である。本研究では，エンコーダとして ResNet34 を用いる U-Net を採用する。

DeepLabV3+ は，空洞畳み込みと Atrous Spatial Pyramid Pooling により複数スケールの文脈を獲得し，さらにデコーダによって境界精度を改善するモデルである [3]。道路領域は画像下部に広く連続して現れる一方で，遠方領域や交差点では形状が複雑になるため，多スケール文脈の利用は有効であると考えられる。本研究では，ResNet50 をエンコーダとする DeepLabV3+ を比較対象とする。

また，両モデルでは ImageNet で事前学習された ResNet 系エンコーダを用いる。深い残差ネットワークは，残差接続によって深層化に伴う最適化の困難を緩和し，画像認識における汎用的特徴抽出器として広く利用されている [4]。本実験では，`segmentation_models_pytorch` に実装されたモデル群を利用し，アーキテクチャおよびエンコーダを引数で切り替えられる実装とした [5]。

## 3. 方法

### 3.1 データ前処理

入力データには Cityscapes を用いた。画像は `leftImg8bit`，教師ラベルは `gtFine_labelIds` から読み込み，`gtFine_labelIds` における road の生ラベル ID 7 を 1，その他すべてのラベルを 0 とする二値マスクへ変換した。この処理により，問題は「道路」対「背景」の画素単位二値分類として定式化される。

学習時には 512 x 512 のランダムクロップ，水平反転，輝度・コントラスト変換，色相・彩度・明度変換を適用した。検証時には 512 x 512 の中央クロップのみを用い，ランダムな拡張は適用しなかった。入力画像は ImageNet の平均と標準偏差により正規化した。

### 3.2 モデル

比較対象は U-Net + ResNet34 と DeepLabV3+ + ResNet50 の二モデルである。両モデルとも入力チャネル数は 3，出力チャネル数は 1 とし，出力は sigmoid 適用前のロジットとして扱った。U-Net はスキップ接続により局所的な空間情報を復元しやすい構造であり，道路境界の再構成に適している。DeepLabV3+ は空洞畳み込みにより受容野を拡大し，多スケール文脈を利用できる構造である。

### 3.3 学習目的関数と最適化

損失関数には Dice loss と Binary Cross Entropy loss の和を用いた。Dice loss は予測マスクと正解マスクの重なりを直接最適化するため，背景画素が多い二値セグメンテーションに適している。一方，Binary Cross Entropy loss は各画素の確率推定を安定化する。両者を組み合わせることで，領域全体の重なりと画素単位の分類精度を同時に改善することを狙った。

最適化には AdamW を用い，初期学習率を 1e-4，weight decay を 1e-4，バッチサイズを 8，最大エポック数を 50 とした。検証 IoU が停滞した場合には ReduceLROnPlateau により学習率を 0.5 倍にし，7 エポック改善が見られない場合には早期終了した。各モデルについて，検証 IoU が最大となった epoch の checkpoint を `best.pth` として保存し，正式評価ではこの checkpoint を用いた。

## 4. 実験

### 4.1 実験設定

Cityscapes の学習分割 2,975 枚を学習に用い，検証分割 500 枚を評価に用いた。テスト分割 1,525 枚は本実験の定量評価には用いなかった。これは，教師ラベルを用いた再現可能な比較を行うためである。評価指標には IoU，Dice，Pixel Accuracy を用い，混同行列として TP，FP，FN，TN を集計した。また，sigmoid 出力の二値化閾値を 0.1 から 0.9 まで 0.1 刻みで変化させ，検証 IoU が最大となる閾値を探索した。

学習曲線は [図1: U-Net + ResNet34 の学習曲線](checkpoints/unet_resnet34/training_curves.pdf) および [図2: DeepLabV3+ + ResNet50 の学習曲線](checkpoints/deeplab_resnet50/training_curves.pdf) に示す。U-Net は 19 epoch，DeepLabV3+ は 18 epoch で早期終了した。DeepLabV3+ の学習ログ上の best validation IoU は 0.9365 であるが，本稿の比較表では正式評価スクリプトが出力した `evaluation_report.json` の値を採用する。

### 4.2 定量結果

表1に，検証分割に対する正式評価結果を示す。IoU，Dice，Pixel Accuracy は標準閾値 0.5 における値であり，best threshold は閾値探索により得られた最良閾値である。

| モデル | アーキテクチャ | エンコーダ | IoU | Dice | Pixel Accuracy | best threshold | best-threshold IoU |
|---|---|---|---:|---:|---:|---:|---:|
| `unet_resnet34` | U-Net | ResNet34 | 0.9377 | 0.9585 | 0.9754 | 0.3 | 0.9394 |
| `deeplab_resnet50` | DeepLabV3+ | ResNet50 | 0.9365 | 0.9581 | 0.9738 | 0.5 | 0.9365 |

U-Net + ResNet34 は，標準閾値 0.5 において DeepLabV3+ + ResNet50 より IoU で 0.0012，Dice で 0.0003，Pixel Accuracy で 0.0016 高い値を示した。差は非常に小さいものの，本実験の設定では U-Net + ResNet34 が最良の総合性能を示した。また，U-Net は閾値を 0.3 に下げた場合に IoU が 0.9394 まで改善しており，標準閾値が必ずしも最適ではないことが分かる。一方，DeepLabV3+ は標準閾値 0.5 がそのまま最良閾値となった。

### 4.3 閾値探索と混同行列

閾値探索の結果は [図3: U-Net + ResNet34 の閾値探索](evaluation_results/unet_resnet34/threshold_sweep.pdf) および [図4: DeepLabV3+ + ResNet50 の閾値探索](evaluation_results/deeplab_resnet50/threshold_sweep.pdf) に示す。U-Net は 0.3 付近で最良となり，閾値を高くしすぎると道路画素の取りこぼしが増える傾向が見られる。DeepLabV3+ は 0.4 から 0.6 の範囲で比較的安定し，0.5 で最大値を示した。

混同行列の可視化は [図5: U-Net + ResNet34 の混同行列](evaluation_results/unet_resnet34/confusion_matrix.pdf) および [図6: DeepLabV3+ + ResNet50 の混同行列](evaluation_results/deeplab_resnet50/confusion_matrix.pdf) に示す。表2に集計値を示す。

| モデル | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| `unet_resnet34` | 63,455,221 | 1,866,883 | 1,362,529 | 64,387,367 |
| `deeplab_resnet50` | 63,687,242 | 2,304,846 | 1,130,508 | 63,949,404 |

DeepLabV3+ は U-Net より TP が多く，FN が少ない。これは道路画素をより積極的に検出する傾向を示す。一方で，FP も多く，背景を道路と誤検出する画素が増加している。走行可能領域抽出では，偽陰性を抑えることが安全側に働く場合もあるが，歩道や非走行領域を道路として過大推定する偽陽性は経路計画に悪影響を及ぼす可能性がある。したがって，実運用では単一の総合指標だけでなく，FP と FN の許容度に応じて閾値を調整する必要がある。

### 4.4 画像単位評価

画像単位 IoU の分布は [図7: U-Net + ResNet34 の画像単位 IoU 分布](evaluation_results/unet_resnet34/iou_distribution.pdf) および [図8: DeepLabV3+ + ResNet50 の画像単位 IoU 分布](evaluation_results/deeplab_resnet50/iou_distribution.pdf) に示す。画像単位の平均 IoU は U-Net が 0.9290，DeepLabV3+ が 0.9288 であり，標準閾値の全体 IoU と同様に差は小さい。標準偏差は U-Net が 0.1939，DeepLabV3+ が 0.1901 であり，DeepLabV3+ の方がわずかに分散が小さい。

| モデル | 画像単位 IoU 平均 | 標準偏差 | 最小値 | 最大値 |
|---|---:|---:|---:|---:|
| `unet_resnet34` | 0.9290 | 0.1939 | 0.0000 | 1.0000 |
| `deeplab_resnet50` | 0.9288 | 0.1901 | 0.0000 | 1.0000 |

最小 IoU が 0 に近いサンプルは，主として road 画素の割合が 0.0% の画像である。このような画像では，わずかな道路誤検出でも IoU が大きく低下するため，通常の道路画像とは異なる評価挙動を示す。したがって，平均指標に加えて，空クラス画像や稀なシーンに対する個別分析を行うことが今後重要である。

### 4.5 考察

本実験では，U-Net + ResNet34 が最も高い IoU を示した。U-Net のスキップ接続は，中央クロップされた道路境界や画像下部の連続領域を復元するうえで有効に働いたと考えられる。特に best threshold を 0.3 とした場合，より広く道路候補を採用することで IoU が改善した。

DeepLabV3+ + ResNet50 は，ResNet50 エンコーダと多スケール文脈モジュールを持つため，構造的には U-Net + ResNet34 より計算量およびメモリ使用量が大きくなる傾向がある。本実験では推論時間や FLOPs を直接測定していないため，計算効率に関する結論は限定的である。しかし，性能差が小さいことを踏まえると，軽量性や実装単純性を重視する場面では U-Net + ResNet34 が有力な選択肢となる。一方，偽陰性を抑えることを重視する場面では，DeepLabV3+ の出力傾向や閾値調整を検討する価値がある。

## 5. 結論

本研究では，Cityscapes における走行可能領域の二値セマンティックセグメンテーションに対して，U-Net + ResNet34 と DeepLabV3+ + ResNet50 を比較した。検証分割 500 枚に対する正式評価の結果，U-Net + ResNet34 は IoU 0.9377，Dice 0.9585，Pixel Accuracy 0.9754 を達成し，DeepLabV3+ + ResNet50 は IoU 0.9365，Dice 0.9581，Pixel Accuracy 0.9738 を達成した。両モデルはいずれも高い性能を示したが，本実験では U-Net + ResNet34 がわずかに優位であった。

混同行列の分析から，DeepLabV3+ は偽陰性を抑える一方で偽陽性が増える傾向を示した。また，閾値探索により，U-Net では標準閾値 0.5 よりも 0.3 の方が高い IoU を示すことが分かった。したがって，走行可能領域抽出モデルを実用化する際には，アーキテクチャの選択だけでなく，運用上のリスクに応じた二値化閾値の調整が重要である。

今後の課題として，第一に推論速度，メモリ使用量，FLOPs を含む計算量評価を行う必要がある。第二に，雨，霧，夜間，強い影などの悪条件に対する頑健性評価が必要である。第三に，road のみではなく lane，sidewalk，terrain など周辺クラスを含めた多クラス評価へ拡張することで，より実運用に近い走行環境理解へ発展させることができる。

## 参考文献

[1] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[2] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” Medical Image Computing and Computer-Assisted Intervention, 2015.

[3] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation,” European Conference on Computer Vision, 2018.

[4] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[5] P. Yakubovskiy, “Segmentation Models PyTorch,” GitHub repository, 2020. Available: https://github.com/qubvel-org/segmentation_models.pytorch
