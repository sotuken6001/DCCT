# DCCT（Color Matters）再現実装

AI生成画像を見分けるモデル **DCCT**（Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*, arXiv:2601.22778）を再現したコードと、学習済みモデルです。卒業研究チームで使うために、必要なものだけを置いています。

- **学習済みモデル**：[`weights/dcct_C.pt`](weights/dcct_C.pt)（10/21 追加、いちばん新しい）、[`weights/dcct_A.pt`](weights/dcct_A.pt)、[`weights/dcct_B.pt`](weights/dcct_B.pt)。どれも1ファイルだけで判定できます
- **ほかのモデルとつなぐ**：`DCCTDetector.embedding()` で、画像1枚から 256次元の特徴が取れます（A・B・C 共通）

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

## 配布しているモデル：`weights/dcct_A.pt`・`dcct_B.pt`・`dcct_C.pt`

どれも、pθ（・qφ）・分類器の重み、設定、判定の線（閾値）を1ファイルにまとめたものです。

| | A（`dcct_A.pt`） | B（`dcct_B.pt`） | **C（`dcct_C.pt`）** |
|---|---|---|---|
| 作り | 論文どおり：±t = 7、pθ ＋ qφ | ±t = 2、pθ だけ | ±t = 7、pθ だけ |
| 分類器に入る特徴 | 60チャンネル | 30チャンネル | 30チャンネル |
| 分類器の学習での AI画像 | JPEG（品質 70〜95） | JPEG（品質 70〜95） | **JPEG（品質 96、色の間引き方も本物にそろえる）** |
| 判定の線（閾値） | 0.4017 | 0.3284 | 0.4077 |
| 線の決め方 | PNG のままの val で、本物と AI画像の正解率の和が最大 | 同じ | **本物の val だけで、上から 5%**（論文の DCCT† と同じ決め方） |
| 元の実行 | run42（`configs/stage2_classifier_aijpeg.yaml`） | run43（`configs/stage2_classifier_t2_photo_only_aijpeg.yaml`） | run52（`configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml`） |

共通：

- 30種のハイパスフィルタ、混合数 K = 10、64×64 のパッチを16か所
- 学習データ：GenImage の SD v1.4（AI画像）と、対応する ImageNet の写真
- **論文と違う点**：分類器の学習で、AI画像を JPEG にしてから使った（理由は下の「大事な注意」）

**どれを使う？** ふだんは **C** がおすすめです。A・B のあとに見つかった偏り（JPEG の品質と色の間引き方）までそろえて学習したモデルで、偏りを使えない評価でいちばん成績が良く（AUC .951）、どの評価のしかたでも .95 前後で安定しています。**論文と同じ形で比べたいときは A を使ってください。**

### 成績：C（GenImage のテスト用、7つの生成AI）

AUC（本物と AI画像の点数がどれだけきれいに分かれるか。1.0 で完全、0.5 で当てずっぽう）。AI画像だけ JPEG（品質96・色の間引き方 4:4:4）にして評価した値です。

| | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | 平均 |
|---|---|---|---|---|---|---|---|---|
| **C** | .998 | .999 | .889 | .973 | .983 | .907 | .909 | **.951** |

正解率（判定の線 0.4077 で）：86.1%（本物 96.1%・AI画像 76.2%）。見たことのない生成器（ADM・VQDM・BigGAN）の AI画像を本物側にこぼしやすいのが弱点です。

評価のしかたを変えたときの C の成績：

| 評価のしかた | AUC |
|---|---|
| AI画像は PNG のまま | .953 |
| AI画像だけ JPEG（品質96・4:2:0） | .962 |
| AI画像だけ JPEG（品質96・4:4:4） | .951 |
| 本物も AI画像も PIL で JPEG（品質96）にかけ直す（いちばん厳しい） | .948 |
| 種（乱数）だけ変えて学習し直したモデル（4:4:4） | .943 |

### 成績：A・B（10/7 時点）

| モデル・評価のしかた | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | 平均 |
|---|---|---|---|---|---|---|---|---|
| **A：AI画像も JPEG（品質90）にして評価** | .987 | .988 | .873 | .957 | .960 | .863 | .872 | **.929** |
| **B：AI画像も JPEG（品質90）にして評価** | .985 | .986 | .876 | .918 | .951 | .894 | .896 | **.929** |
| A：AI画像は PNG のまま評価 | .961 | .963 | .781 | .963 | .891 | .691 | .767 | .860 |
| B：AI画像は PNG のまま評価 | .938 | .939 | .745 | .899 | .846 | .703 | .770 | .834 |
| A：本物も AI画像も PIL で JPEG（品質90）にかけ直して評価 | .970 | .973 | .704 | .902 | .906 | .698 | .694 | .835 |
| B：本物も AI画像も PIL で JPEG（品質90）にかけ直して評価 | .966 | .967 | .683 | .802 | .876 | .747 | .735 | .825 |

- A・B・C の差は、1回ずつしか学習していないので、.02 より小さい差はたまたまの範囲と考えてください（種だけ変えても .006〜.015 動きました）
- C の表と A・B の表は、評価のしかた（JPEG の品質・色の間引き方）が違うので、そのまま並べて比べないでください

---

## 大事な注意：GenImage の本物と AI画像は、画像の形式がそろっていない

GenImage の評価用画像は、**本物の写真が全部 JPEG、AI画像が全部 PNG** です。そのまま学習すると、モデルは「カメラの色の癖」ではなく「JPEG かどうか」で見分けてしまいます。

- 普通に学習したモデル（AI画像は PNG のまま）は、AI画像を JPEG にして評価すると AUC .97 → .79 まで落ちました
- そこで A・B・C は、**学習のときから AI画像も JPEG にして**、形式の差を使えないようにしました

JPEG にするときは、品質と**色の間引き方**（JPEG が色の情報をどれだけ減らすか。4:4:4 は減らさない、4:2:0 は縦横半分に減らす）も本物にそろえる必要がありました。

- 本物（ImageNet）の JPEG は、多くが品質 96 で、約8割が 4:4:4
- PIL で普通に JPEG にすると全部 4:2:0 になる。4:2:0 の画像は色が予測しやすいので、モデルはこの差も手がかりにしていました
- C は、AI画像を品質 96 で、4:4:4 を8割・4:2:0 を2割にして学習しています

**使うときのお願い**

- 評価や比べっこをするときは、**AI画像も JPEG にしてから入れてください**。C なら品質 96・`subsampling=0`（4:4:4）がいちばん公平です。PNG のまま入れると形式の差が混ざります
  ```python
  img.save("ai.jpg", quality=96, subsampling=0)   # PIL。subsampling=0 が 4:4:4、2 が 4:2:0
  ```
- 判定の線は、A・B は PNG のままの AI画像で、C は本物の val だけで決めたものです。条件をそろえて比べるときは、**正解率より AUC で比べる**のがおすすめです

---

## 使い方（Jupyter ノートブック・ipynb から使う）

ノートブックのセルから `DCCTDetector` を読み込んで使うのが基本です。GPU（CUDA）や Apple Silicon（mps）があれば自動で使います。CPU でも動きます（1枚あたり数秒かかることがあります）。

### 1. 準備

**自分のPC（Jupyter / VS Code）で使うとき**：ターミナルで1回だけ次を実行し、ノートブックのカーネルに「DCCT (.venv)」を選びます。

```bash
git clone https://github.com/sotuken6001/DCCT.git
cd DCCT
python -m venv .venv
source .venv/bin/activate          # Windows は .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # どのフォルダのノートブックからでも `from src...` で読み込めるようになる
pip install ipykernel
python -m ipykernel install --user --name dcct --display-name "DCCT (.venv)"
```

すでに clone している人は `git pull` で C とコードの更新が入ります（A・B はそのまま残っています）。

**Google Colab で使うとき**：ノートブックの最初のセルで次を実行します。`%cd DCCT` でこのフォルダに入ると、`pip install -e .` なしで `from src...` が使えます。

```python
!git clone https://github.com/sotuken6001/DCCT.git
%cd DCCT
!pip install -q -r requirements.txt
```

### 2. モデルを読み込んで、1枚判定する

```python
from pathlib import Path
from src.eval.bundle import DCCTDetector

REPO = Path("/path/to/DCCT")    # clone した場所（Colab なら Path("/content/DCCT")）
detector = DCCTDetector.from_bundle(REPO / "weights/dcct_C.pt")   # A・B なら dcct_A.pt・dcct_B.pt。device="cpu" なども指定できる
print(detector.device, detector.threshold)   # 使う装置と判定の線（C は 0.4077）

detector.score("photo.jpg")      # AIらしさ（0〜1）。16パッチの平均
detector.predict("photo.jpg")    # 例: {"prob_ai": 0.93, "label": "ai", "threshold": 0.4077}
```

- 重みのファイルは、ノートブックの置き場所から見た相対パスで探されます。上のように **clone した場所からのパス**で渡すと迷いません
- 画像は、ファイルのパス・PIL の画像・`[3, H, W]`（値は 0〜255）のテンソルのどれでも渡せます。64×64 以上が必要です
- 画像から16か所を切り出して点数を平均します。切り出す場所は、パスで渡すとパスの文字列で決まり、PIL の画像やテンソルで渡すと毎回同じ決め方になります。同じ画像を同じ渡し方で渡せば、結果は毎回同じです

### 3. フォルダの画像をまとめて判定し、AUC を出す

本物と AI画像を比べるときは、**AI画像も JPEG（品質96・4:4:4）にしてから**入れてください（理由は上の「大事な注意」）。下の `to_jpeg` は、ファイルを書き出さずにノートブックの中で JPEG にします。

```python
import io
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

def to_jpeg(path, quality=96, subsampling=0):    # subsampling=0 が 4:4:4、2 が 4:2:0
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    return Image.open(io.BytesIO(buf.getvalue()))

rows = []
for label, folder, make_jpeg in [(0, "data/real", False), (1, "data/ai", True)]:   # 0 = 本物、1 = AI画像
    for path in tqdm(sorted(Path(folder).glob("*"))):
        image = to_jpeg(path) if make_jpeg else Image.open(path)
        rows.append({"path": str(path), "label": label, "prob_ai": detector.score(image)})

df = pd.DataFrame(rows)
df["pred"] = (df["prob_ai"] > detector.threshold).astype(int)
print("AUC:", roc_auc_score(df["label"], df["prob_ai"]))
print("正解率:", (df["pred"] == df["label"]).mean())
```

### 4. 特徴を取り出す（ほかのモデルとつなぐとき）

```python
detector.embedding("photo.jpg")       # 分類器の最後の層の手前の特徴。torch.Tensor [256]（16パッチの平均）
detector.patch_features("photo.jpg")  # 分類器に入る特徴マップ。torch.Tensor [16, C, 64, 64]（C は A が 60、B・C が 30）
```

- ほかのモデルの特徴とつなぐ（3本目の枝として足す）なら、`embedding()` の 256次元（A・B・C 共通）を使うのがいちばん簡単です。もっと細かく使いたいときは `patch_features()` を使ってください
- 返ってくるテンソルは `detector.device` の上にあります。numpy にするときは `.cpu().numpy()` を付けてください

### おまけ：ターミナルで判定する

ノートブックを使わずに、数枚だけ確かめたいとき用です（`pip install -e .` が必要）。

```bash
dcct predict --bundle weights/dcct_C.pt photo1.jpg photo2.png
```

```
画像                                        AIらしさ  判定
photo1.jpg                                  0.0312  real
photo2.png                                  0.9475  ai
（判定の線: 0.4077。AIらしさがこれより大きければ ai）
```

---

## 学習し直したいとき

GenImage（外付けSSDに置く想定）と、メタデータを入れる SQLite のデータベースを使います。

```bash
dcct init-db  --config configs/base.yaml                       # データベースを作る
dcct ingest   --config configs/base.yaml                       # GenImage を走査して登録する（configs/base.yaml の paths.dataset_root を自分の環境に合わせる）

# C を作るとき：pθ を学習してから、qφ なしで分類器を学習する（AI画像は品質96・色の間引き方を本物にそろえた JPEG）
dcct train-stage1 --target photo --config configs/stage1_photo.yaml   # pθ
dcct train-stage2 --config configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml

# A を作るとき：pθ と qφ を学習してから分類器を学習する
dcct train-stage1 --target photo --config configs/stage1_photo.yaml   # pθ
dcct train-stage1 --target ai    --config configs/stage1_ai.yaml      # qφ
dcct train-stage2 --config configs/stage2_classifier_aijpeg.yaml

# B を作るとき：pθ を t=2 で学習してから、qφ なしで分類器を学習する
dcct train-stage1 --target photo --config configs/stage1_photo_t2.yaml
dcct train-stage2 --config configs/stage2_classifier_t2_photo_only_aijpeg.yaml
```

評価と配布用ファイルの作成（C の例）：

```bash
dcct evaluate --mode cross_generator --config configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml --run-id <ID>
dcct threshold-check --config configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml --run-id <ID> --ai-jpeg-quality 96 --subsampling 4:4:4 --real-false-alarm 5   # 線を本物の val の上から 5% に決めて、生成器ごとの正解率を出す
dcct evaluate-one-class --config configs/stage2_classifier.yaml              # pθ だけで判定（論文の付録B）
dcct export-bundle --config configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml --run-id <ID> --threshold <τ> --out weights/my_model.pt
```

`configs/stage2_classifier.yaml` は、論文どおり AI画像を PNG のまま学習する設定です（比べるとき用）。評価のしかたを変える細かい設定（AI画像の JPEG の品質・色の間引き方など）は `configs/base.yaml` の `evaluate` と、`dcct evaluate --help` を見てください。

学習の目安（Apple M 系の Mac Studio）：pθ・qφ は1本数時間、分類器はひと晩。

---

## フォルダの中身

| 場所 | 中身 |
|---|---|
| `weights/` | 配布する学習済みモデル（`dcct_A.pt`、`dcct_B.pt`、`dcct_C.pt`） |
| `src/preprocess/` | Bayer 配列で色を分ける、ハイパスフィルタ、パッチの切り出し |
| `src/model/` | pθ・qφ（条件付き U-Net）、分類器、損失 |
| `src/train/` | 学習（Stage I：pθ・qφ、Stage II：分類器） |
| `src/eval/` | 評価、配布用ファイルの作成と読み込み（`bundle.py`）、線の引き方や JPEG の作りを調べる道具 |
| `src/data/` | GenImage の走査、データベース、データの読み込み |
| `configs/` | 設定（`base.yaml` が全部の土台） |
| `migrations/` | データベースの表の定義 |
| `tests/` | テスト（`pytest`） |

---

## 出典

- 論文：Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*. arXiv:2601.22778。公式のコードは公開されていないので、論文の記述から実装しています
- データセット：GenImage（Zhu et al., 2023）
- ハイパスフィルタ：Fridrich & Kodovsky, *Rich Models for Steganalysis of Digital Images*, 2012
- JPEG の品質の偏り：Grommelt et al., *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, 2024
