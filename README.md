# DCCT（Color Matters）再現実装

AI生成画像を見分けるモデル **DCCT**（Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*, arXiv:2601.22778）を再現したコードと、学習済みモデルです。卒業研究チームで使うために、必要なものだけを置いています。

- **学習済みモデル**：[`weights/dcct_A.pt`](weights/dcct_A.pt) と [`weights/dcct_B.pt`](weights/dcct_B.pt)。どちらも1ファイルだけで判定できます
- **ほかのモデルとつなぐ**：`DCCTDetector.embedding()` で、画像1枚から 256次元の特徴が取れます

---

## しくみ（ひとことで）

カメラのセンサーは1画素につき1色しか測らず、残りの2色は周りから補って作ります。DCCT はこの「色の補い方の癖」を見ます。

| 部品 | 役割 |
|---|---|
| pθ | 本物の写真で学習。見えている1色から、隠れた2色（を高周波フィルタに通した値）の出やすさを予測する |
| qφ | AI画像で学習。同じ予測を AI画像の癖で行う |
| 分類器 gψ | pθ・qφ の予測（特徴マップ 60チャンネル）を見て、本物か AI かを判定する |

画像1枚から 64×64 のパッチを16か所切り出し、パッチごとの点数を平均します（論文の Algorithm 2）。

---

## 配布しているモデル：`weights/dcct_A.pt` と `weights/dcct_B.pt`

どちらも、pθ（・qφ）・分類器の重み、設定、判定の線（閾値）を1ファイルにまとめたものです。

| | A（`dcct_A.pt`） | B（`dcct_B.pt`） |
|---|---|---|
| 作り | 論文どおり：±t = 7、pθ ＋ qφ | ±t = 2、pθ だけ（qφ なし） |
| 分類器に入る特徴 | 60チャンネル | 30チャンネル |
| 判定の線（閾値） | 0.4017 | 0.3284 |
| 元の実行 | run42（`configs/stage2_classifier_aijpeg.yaml`） | run43（`configs/stage2_classifier_t2_photo_only_aijpeg.yaml`） |

共通：

- 30種のハイパスフィルタ、混合数 K = 10、64×64 のパッチを16か所
- 学習データ：GenImage の SD v1.4（AI画像）と、対応する ImageNet の写真
- **論文と違う点**：分類器の学習で、AI画像を JPEG（品質 70〜95 からランダム）にしてから使った（理由は下の「大事な注意」）

**どちらを使う？** 迷ったら **B** がおすすめです。成績はほぼ同じで（下の表）、qφ がいらないぶん作りが簡単で速いです。A から qφ だけを外したモデルでも成績は変わらなかったので（AUC .928）、この条件では qφ は効いていません。論文と同じ形で比べたいときは A を使ってください。

### 成績（GenImage のテスト用、7つの生成AI）

AUC（本物と AI画像の点数がどれだけきれいに分かれるか。1.0 で完全、0.5 で当てずっぽう）

| モデル・評価のしかた | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | 平均 |
|---|---|---|---|---|---|---|---|---|
| **A：AI画像も JPEG（品質90）にして評価** | .987 | .988 | .873 | .957 | .960 | .863 | .872 | **.929** |
| **B：AI画像も JPEG（品質90）にして評価** | .985 | .986 | .876 | .918 | .951 | .894 | .896 | **.929** |
| A：AI画像は PNG のまま評価 | .961 | .963 | .781 | .963 | .891 | .691 | .767 | .860 |
| B：AI画像は PNG のまま評価 | .938 | .939 | .745 | .899 | .846 | .703 | .770 | .834 |
| A：本物も AI画像も PIL で JPEG（品質90）にかけ直して評価 | .970 | .973 | .704 | .902 | .906 | .698 | .694 | .835 |
| B：本物も AI画像も PIL で JPEG（品質90）にかけ直して評価 | .966 | .967 | .683 | .802 | .876 | .747 | .735 | .825 |

- A と B の差は、1回ずつしか学習していないので、たまたまの範囲かもしれません
- いちばん厳しくそろえた条件（最後の2行。本物も AI画像も最後に同じ PIL の JPEG を通す）では、AUC は .83 前後まで下がり、ADM・VQDM・BigGAN が苦手です。本物は2回圧縮になるので、その影響もあるかもしれません

---

## 大事な注意：GenImage は本物が JPEG、AI画像が PNG

GenImage の評価用画像は、**本物の写真が全部 JPEG、AI画像が全部 PNG** です。そのまま学習すると、モデルは「カメラの色の癖」ではなく「JPEG かどうか」で見分けてしまいます。

- 普通に学習したモデル（AI画像は PNG のまま）は、AI画像を JPEG にして評価すると AUC .97 → .79 まで落ちました
- そこでこのモデルは、**学習のときから AI画像も JPEG にして**、形式の差を使えないようにしました。形式をそろえた評価で AUC .929 です

**使うときのお願い**

- 評価や比べっこをするときは、**AI画像も JPEG（品質90 くらい）にしてから入れてください**。PNG のまま入れると、このモデルには見慣れない入力になり、点数が下がります（上の表の2行目）
- 判定の線（閾値。A は 0.4017、B は 0.3284）は、PNG のままの AI画像で決めたものです。JPEG にそろえた条件では線がずれるので、**正解率より AUC で比べる**のがおすすめです
- 本物も AI画像も同じ PIL の JPEG にかけ直すと、AUC は .93 → .83 に下がります（成績の表の最後の2行）。PIL の JPEG と ImageNet の元の JPEG の違いを少し手がかりにしている可能性があります。いちばん公平に比べたいときは、この条件の数字も見てください

---

## 使い方

### 1. 準備

```bash
git clone https://github.com/sotuken6001/DCCT.git
cd DCCT
python -m venv .venv
source .venv/bin/activate          # Windows は .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # dcct コマンドが使えるようになる
```

GPU（CUDA）や Apple Silicon（mps）があれば自動で使います。CPU でも動きます。

### 2. コマンドで判定する

```bash
dcct predict --bundle weights/dcct_B.pt photo1.jpg photo2.png
```

出力の形（数字は例）：

```
画像                                        AIらしさ  判定
photo1.jpg                                  0.0312  real
photo2.png                                  0.9475  ai
（判定の線: 0.3284。AIらしさがこれより大きければ ai）
```

### 3. Python から使う（ほかのモデルとつなぐとき）

```python
from src.eval.bundle import DCCTDetector

detector = DCCTDetector.from_bundle("weights/dcct_B.pt")   # A なら dcct_A.pt。device="cpu" なども指定できる

detector.score("photo.jpg")           # AIらしさ（0〜1）。16パッチの平均
detector.predict("photo.jpg")         # 例: {"prob_ai": 0.93, "label": "ai", "threshold": 0.3284}
detector.embedding("photo.jpg")       # 分類器の最後の層の手前の特徴。torch.Tensor [256]（16パッチの平均）
detector.patch_features("photo.jpg")  # 分類器に入る特徴マップ。torch.Tensor [16, C, 64, 64]（C は A が 60、B が 30）
```

- 画像は、ファイルのパス・PIL の画像・`[3, H, W]`（値は 0〜255）のテンソルのどれでも渡せます
- 画像は 64×64 以上が必要です
- 同じ画像なら、毎回同じ場所のパッチを切り出すので、結果は毎回同じです
- ほかのモデルの特徴とつなぐ（3本目の枝として足す）なら、`embedding()` の 256次元（A も B も同じ）を使うのがいちばん簡単です。もっと細かく使いたいときは `patch_features()` を使ってください

---

## 学習し直したいとき

GenImage（外付けSSDに置く想定）と、メタデータを入れる SQLite のデータベースを使います。

```bash
dcct init-db  --config configs/base.yaml                       # データベースを作る
dcct ingest   --config configs/base.yaml                       # GenImage を走査して登録する（configs/base.yaml の paths.dataset_root を自分の環境に合わせる）
dcct train-stage1 --target photo --config configs/stage1_photo.yaml   # pθ
dcct train-stage1 --target ai    --config configs/stage1_ai.yaml      # qφ
dcct train-stage2 --config configs/stage2_classifier_aijpeg.yaml      # A の分類器（AI画像も JPEG にして学習）

# B を作るとき：pθ を t=2 で学習してから、qφ なしで分類器を学習する
dcct train-stage1 --target photo --config configs/stage1_photo_t2.yaml
dcct train-stage2 --config configs/stage2_classifier_t2_photo_only_aijpeg.yaml
```

評価と配布用ファイルの作成：

```bash
dcct evaluate --mode cross_generator --config configs/stage2_classifier_aijpeg.yaml --run-id <ID>
dcct evaluate --mode ai_jpeg --ai-jpeg-quality 90 --config configs/stage2_classifier_aijpeg.yaml --run-id <ID>
dcct evaluate-one-class --config configs/stage2_classifier.yaml              # pθ だけで判定（論文の付録B）
dcct export-bundle --config configs/stage2_classifier_aijpeg.yaml --run-id <ID> --threshold <τ> --out weights/my_model.pt
```

`configs/stage2_classifier.yaml` は、論文どおり AI画像を PNG のまま学習する設定です（比べるとき用）。

学習の目安（Apple M 系の Mac Studio）：pθ・qφ は1本数時間、分類器はひと晩。

---

## フォルダの中身

| 場所 | 中身 |
|---|---|
| `weights/` | 配布する学習済みモデル（`dcct_A.pt`、`dcct_B.pt`） |
| `src/preprocess/` | Bayer 配列で色を分ける、ハイパスフィルタ、パッチの切り出し |
| `src/model/` | pθ・qφ（条件付き U-Net）、分類器、損失 |
| `src/train/` | 学習（Stage I：pθ・qφ、Stage II：分類器） |
| `src/eval/` | 評価、配布用ファイルの作成と読み込み（`bundle.py`） |
| `src/data/` | GenImage の走査、データベース、データの読み込み |
| `configs/` | 設定（`base.yaml` が全部の土台） |
| `migrations/` | データベースの表の定義 |
| `tests/` | テスト（`pytest`） |

---

## 出典

- 論文：Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*. arXiv:2601.22778。公式のコードは公開されていないので、論文の記述から実装しています
- データセット：GenImage（Zhu et al., 2023）
- ハイパスフィルタ：Fridrich & Kodovsky, *Rich Models for Steganalysis of Digital Images*, 2012
