# Projects for Semantic Segmentation
# セマンティックセグメンテーション プロジェクト

This guide provides three practical, beginner-friendly projects to help you start your journey in semantic segmentation. Each project focuses on a distinct application area with manageable datasets.

このガイドでは、セマンティックセグメンテーションの学習を始めるのに役立つ、実践的で初心者向けの3つのプロジェクトを紹介します。各プロジェクトは、扱いやすいデータセットを用いた明確な応用分野に焦点を当てています。

---

## 1. Portrait Matting & Background Swap / 人物セグメンテーションと背景の置き換え

**Background / 背景:** With the rise of remote work, features like "virtual backgrounds" in video conferencing are essential. The goal of this task is to precisely extract the "person" class from complex indoor or outdoor backgrounds (a binary classification task: Person vs. Background).

リモートワークの普及に伴い、ビデオ会議での「バーチャル背景」のような機能は不可欠です。このタスクの目標は、複雑な屋内または屋外の背景から「人」のクラスを正確に抽出することです（二値分類タスク：人 vs 背景）。

**Challenges / 課題:**
* Handling fine details like hair edges. / 髪の毛などの微細なエッジの処理。
* Adapting to variable lighting and dynamic backgrounds. / 変化する照明や動的な背景への適応。

**Applications / 応用:**
* Privacy protection in video calls. / ビデオ通話におけるプライバシー保護。
* Special effects for short videos (e.g., TikTok filters). / ショート動画（TikTokフィルターなど）の特殊効果。

**Recommended Datasets / 推奨データセット:**
* **Supervisely Person Dataset:** Contains high-quality annotated human images, perfect for binary classification. / 高品質な注釈付き人物画像が含まれており、二値分類に最適です。
* **AISegment (Matting Human Datasets):** Focuses on portrait matting with highly detailed edge annotations. / 人物の切り抜きに特化しており、エッジの注釈が非常に詳細です。

---

## 2. Drivable Area Segmentation (Autonomous Driving) / 可視領域の抽出（自動運転）

**Background / 背景:**
Autonomous driving systems must first identify safe road areas (the "drivable area") for path planning. Rather than detecting all vehicles, this project focuses on understanding the topological structure of the environment.

自動運転システムは、経路計画のためにまず安全な道路領域（「可視領域」または「走行可能領域」）を特定する必要があります。すべての車両を検出するのではなく、このプロジェクトでは環境のトポロジー構造を理解することに焦点を当てます。

**Challenges / 課題:**
* Weather conditions (e.g., rain reflections, snow). / 天候条件（雨の反射、雪など）。
* Uncertain road edges (e.g., country roads without clear lane markings or guardrails). / 不確実な道路の境界（明確な車線やガードレールのない田舎道など）。

**Applications / 応用:**
* Advanced Driver Assistance Systems (ADAS). / 高度運転支援システム (ADAS)。
* Path navigation for delivery robots. / 配送ロボットの経路ナビゲーション。

**Recommended Datasets / 推奨データセット:**
* **BDD100K (Drivable Area subset):** A large-scale driving dataset from UC Berkeley, specifically providing drivable area labels. / UC Berkeleyによる大規模な運転データセットで、可視領域のラベルが提供されています。
* **Cityscapes:** You can simplify this dataset by extracting only the "Road" class to create a beginner-friendly binary task. / 「Road（道路）」クラスのみを抽出して単純化することで、初心者向けの二値分類タスクを作成できます。

---

## 3. Building and Water Extraction (Remote Sensing) / 建物と水域の抽出（リモートセンシング）

**Background / 背景:**
By analyzing top-down views from satellites or drones, we can automatically outline building footprints or water bodies. This is highly valuable for urban planning and environmental monitoring.

衛星やドローンからの俯瞰（ふかん）画像を分析することで、建物の輪郭や水域を自動的に抽出できます。これは都市計画や環境モニタリングにおいて非常に価値があります。

**Challenges / 課題:**
* Extreme scale variations (e.g., skyscrapers vs. small houses). / 極端なスケールの変化（高層ビル vs 小さな家屋）。
* Interference from cloud cover or building shadows. / 雲による遮蔽や建物の影による干渉。

**Applications / 応用:**
* Automated illegal building detection. / 違法建築物の自動検出。
* Disaster assessment (e.g., calculating flooded areas). / 災害評価（浸水面積の計算など）。

**Recommended Datasets / 推奨データセット:**
* **Massachusetts Buildings Dataset:** A classic dataset for building extraction with relatively clean imagery. / 画像が比較的鮮明な、建物抽出のための古典的なデータセット。
* **WHU Building Dataset:** A high-resolution dataset perfect for practicing precise edge segmentation. / 正確なエッジセグメンテーションの練習に最適な高解像度データセット。

------

***

# Project Deep Dive: Drivable Area Segmentation
# プロジェクトの詳細：走行可能領域の抽出

## 1. Introduction: Why Drivable Area? / 導入：なぜ「走行可能領域」なのか？

Before an autonomous vehicle can navigate, avoid obstacles, or obey traffic lights, it must answer one fundamental question: **"Where is it safe to drive?"**
自動運転車がナビゲーションを行い、障害物を避け、信号に従う前に、まず一つの根本的な問いに答える必要があります。**「どこを走るのが安全か？」**

While Object Detection focuses on drawing boxes around discrete items (like cars, pedestrians, and bicycles), Drivable Area Segmentation is about understanding the continuous, topological structure of the ground. It tells the vehicle the boundaries of its operable space.
物体検出が個別のアイテム（車、歩行者、自転車など）を四角い枠で囲むことに焦点を当てるのに対し、走行可能領域の抽出は、地面の連続的かつトポロジカルな構造を理解することに関するものです。これにより、車両は操作可能な空間の境界を把握します。

## 2. Core Knowledge Points / 中核となる知識ポイント

To tackle this project, we need to understand a few key theoretical concepts:
このプロジェクトに取り組むために、いくつかの重要な理論的概念を理解する必要があります。

### 2.1 Binary Classification at the Pixel Level / ピクセルレベルの二値分類
At its simplest, this project reduces the complex world into a binary decision for every single pixel:
最も単純な形として、このプロジェクトは複雑な世界を、すべてのピクセルに対する「二値の決定（バイナリ分類）」に落とし込みます。
* **Class 1 (Drivable / 走行可能):** The pixel belongs to the road surface where our car (the ego-vehicle) can legally and safely drive. (自車が合法的かつ安全に走行できる路面に属するピクセル)
* **Class 0 (Background / 背景):** Everything else. This includes sidewalks, buildings, sky, and even other cars or pedestrians currently occupying the road. (それ以外のすべて。歩道、建物、空、さらには現在道路を占有している他の車や歩行者も含まれます)

Mathematically, for an image with $N$ pixels, the model outputs a probability map where each pixel value $p_i \in [0, 1]$. We apply a threshold (usually $0.5$) to determine the final class.
数学的には、$N$ 個のピクセルを持つ画像に対して、モデルは各ピクセル値が $p_i \in [0, 1]$ となる確率マップを出力します。しきい値（通常は $0.5$）を適用して最終的なクラスを決定します。

### 2.2 Receptive Field and Context / 受容野とコンテキスト
To classify a grey pixel as "Road", the model cannot just look at that single pixel (because a grey pixel on a concrete building looks exactly the same as a grey pixel on the asphalt). The neural network must learn to use the **Context (周囲の文脈)**. It looks at the surrounding pixels (Receptive Field) to realize, "This grey area is below the sky and cars are driving on it, so it must be a road."
灰色のピクセルを「道路」として分類するためには、モデルはその単一のピクセルだけを見ることはできません（コンクリートの建物の灰色と、アスファルトの灰色は全く同じに見えるためです）。ニューラルネットワークは**コンテキスト（周囲の文脈）**を使うことを学習しなければなりません。周囲のピクセル（受容野）を見て、「この灰色の領域は空の下にあり、車がその上を走っているから道路に違いない」と認識するのです。

## 3. Deep Dive into Challenges (The "Edge Cases") / 課題の深掘り（エッジケース）

Training a model to find the road on a sunny, perfectly painted highway is easy. The real challenge—and what makes this a great research topic—lies in the "Edge Cases" (困難な状況):
晴れた日の、白線が完璧に引かれた高速道路で道路を見つけるモデルを訓練するのは簡単です。本当の課題、そしてこれを素晴らしい研究テーマにしているのは、「エッジケース（困難な状況）」にあります。

* **Challenge A: Weather & Illumination (天候と照明)**
  * **Rain (雨):** Wet roads act like mirrors, reflecting the sky, traffic lights, and other cars. This drastically changes the texture of the "road" class, confusing the model. (濡れた路面は鏡のようになり、空や信号機、他の車を反射します。これにより「道路」クラスのテクスチャが大きく変わり、モデルを混乱させます。)
  * **Night/Shadows (夜間・影):** Sharp shadows from trees or buildings can look like physical obstacles, causing the model to predict "holes" in the drivable area. (木や建物からのくっきりとした影は物理的な障害物のように見え、モデルが走行可能領域に「穴」があると予測する原因になります。)
* **Challenge B: Unstructured Roads (非構造化道路)**
  * In city centers (Cityscapes), roads are defined by sharp curbs and painted lines. In rural areas, the transition from "road" to "dirt shoulder" or "grass" is blurry and gradual. Defining the exact boundary becomes highly subjective, even for humans.
  * 都市部（Cityscapesなど）では、道路は明確な縁石や白線で定義されます。しかし田舎道では、「道路」から「土の路肩」や「草むら」への境界は曖昧で段階的です。正確な境界を定義することは、人間にとっても非常に主観的になります。
* **Challenge C: Dynamic Occlusion (動的な遮蔽)**
  * Heavy traffic means the actual road surface is blocked by other cars. The model must learn to segment only the visible road parts, and infer the drivable space between vehicles.
  * 交通量が多い場合、実際の路面は他の車によって隠されてしまいます。モデルは、見えている道路部分のみをセグメンテーションし、車両間の走行可能な空間を推論することを学ばなければなりません。

## 4. Dataset Strategy / データセットの戦略

For this project, we will use subsets of industry-standard datasets.
このプロジェクトでは、業界標準のデータセットのサブセットを使用します。

* **BDD100K (Berkeley DeepDrive):**
  * *Why it's great:* It is incredibly diverse. It contains data from different times of day (dawn, day, night) and different weather conditions (clear, overcast, rainy, snowy). It forces the model to generalize well.
  * *素晴らしい理由:* 非常に多様性に富んでいます。1日の様々な時間帯（夜明け、昼、夜）や天候（晴れ、曇り、雨、雪）のデータが含まれています。モデルに高い汎化性能を強要します。
* **Cityscapes:**
  * *Why it's great:* It provides extremely high-resolution images ($1024 \times 2048$) of European streets. While it has 19 classes, we will map everything that isn't "Road" to "Background" to create our binary task.
  * *素晴らしい理由:* ヨーロッパの市街地の非常に高解像度な画像（$1024 \times 2048$）を提供します。19のクラスがありますが、「道路」以外のすべてを「背景」にマッピングすることで、二値分類タスクを作成します。

---
### Interactive Concept Explorer / インタラクティブ概念エクスプローラー

To truly understand how weather and environment affect the model's confidence, try adjusting the parameters in the simulator below. Notice how the segmentation mask degrades (gets noisy or loses sharp edges) under challenging conditions.
天候や環境がモデルの信頼度にどのように影響するかを真に理解するために、以下のシミュレーターでパラメータを調整してみてください。困難な条件下で、セグメンテーションマスクがどのように劣化する（ノイズが乗る、またはシャープなエッジが失われる）かに注目してください。
