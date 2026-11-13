# DCCT（Color Matters）再現実装

AI生成画像を見分けるモデル **DCCT**（Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*, arXiv:2601.22778）を再現したコードと、学習済みモデルです。卒業研究チームで使うために置いています。

カメラのセンサーは1画素につき1色しか測らず、残りの2色は周りから補って作ります。DCCT はこの「色の補い方の癖」を見て、本物の写真か AI画像かを判定します。

---

## メインのモデル：`weights/dcct_Cmix8.pt`

[`weights/dcct_Cmix8.pt`](weights/dcct_Cmix8.pt) は、1ファイルだけで判定できる学習済みモデルです（11/13 追加）。

| 項目 | 内容 |
|---|---|
| 学習データ | GenImage の8つの生成AI（Midjourney・SD1.4・SD1.5・ADM・GLIDE・Wukong・VQDM・BigGAN）の train を 1/8 ずつ混ぜたもの（合計は生成AI 1つぶん）と、対応する ImageNet の写真 |
| 学習での AI画像 | JPEG（品質 96、色の間引き方も本物にそろえる。理由は下の「使うときの注意」） |
| 判定の線（閾値） | 0.5900（本物の val で、上から 5% が AI と判定される位置） |
| 取り出せる特徴 | `embedding()` で画像1枚から 256次元 |

**成績**（AUC：本物と AI画像の点数がどれだけきれいに分かれるか。1.0 で完全、0.5 で当てずっぽう。AI画像だけ JPEG 品質96・4:4:4 にして評価）

| 評価データ | AUC | 正解率 |
|---|---|---|
| GenImage の8つ（どれも学習で見ている） | .984 | 93.9% |
| Misinfo の4つ（FLUX.1-dev・PixArt-α・SDXL-1.0・Imagen3。どれも学習で見ていない） | .928 | 71.8% |

- Misinfo は、AI画像の JPEG の品質を本物と同じ割合（品質96 が約2/3・90 が約1/3）にそろえて評価しました
- 1回だけの学習です。同じ設定でも乱数の種だけで AUC が .006〜.015 動くので、.02 未満の差は「同じ」と考えてください

---

## 使い方（Jupyter ノートブック・ipynb から使う）

GPU（CUDA）や Apple Silicon（mps）があれば自動で使います。CPU でも動きます（1枚あたり数秒）。

### 1. 準備

**自分のPC（Jupyter / VS Code）**：ターミナルで1回だけ次を実行し、ノートブックのカーネルに「DCCT (.venv)」を選びます。

```bash
git clone https://github.com/sotuken6001/DCCT.git
cd DCCT
python -m venv .venv
source .venv/bin/activate          # Windows は .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pip install ipykernel
python -m ipykernel install --user --name dcct --display-name "DCCT (.venv)"
```

すでに clone している人は `git pull` で最新になります。

**Google Colab**：ノートブックの最初のセルで次を実行します。

```python
!git clone https://github.com/sotuken6001/DCCT.git
%cd DCCT
!pip install -q -r requirements.txt
```

### 2. 1枚判定する

```python
from pathlib import Path
from src.eval.bundle import DCCTDetector

REPO = Path("/path/to/DCCT")    # clone した場所（Colab なら Path("/content/DCCT")）
detector = DCCTDetector.from_bundle(REPO / "weights/dcct_Cmix8.pt")

detector.score("photo.jpg")      # AIらしさ（0〜1）
detector.predict("photo.jpg")    # 例: {"prob_ai": 0.93, "label": "ai", "threshold": 0.59}
```

- 画像は、ファイルのパス・PIL の画像・`[3, H, W]`（値は 0〜255）のテンソルのどれでも渡せます。64×64 以上が必要です
- 画像から 64×64 を16か所切り出して点数を平均します。同じ画像を同じ渡し方で渡せば、結果は毎回同じです

### 3. フォルダの画像をまとめて判定し、AUC を出す

AI画像は **JPEG（品質96・4:4:4）にしてから**入れてください（理由は下の「使うときの注意」）。

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
detector.embedding("photo.jpg")       # 256次元の特徴。torch.Tensor [256]
detector.patch_features("photo.jpg")  # 分類器に入る特徴マップ。torch.Tensor [16, 30, 64, 64]
```

- 返ってくるテンソルは `detector.device` の上にあります。numpy にするときは `.cpu().numpy()` を付けてください

---

## 使うときの注意：AI画像も JPEG にしてから入れる

GenImage は **本物の写真が全部 JPEG、AI画像が全部 PNG** です。そのまま学習すると、モデルは色の癖ではなく「JPEG かどうか」で見分けてしまいます。そこで学習のときから AI画像も JPEG にしています。

- 評価や比べっこをするときは、**AI画像も JPEG（品質96・`subsampling=0`）にしてから**入れてください。PNG のままだと形式の差が混ざります
- 本物の JPEG の品質がまちまちなデータでは、**AI画像も本物と同じ割合の品質にしてください**。強く圧縮した画像ほど本物らしく見えるので、品質がそろっていないと AUC が良く見えすぎたり悪く見えすぎたりします
- 条件をそろえて比べるときは、**正解率より AUC で比べてください**

---

## ほかのモデル（A・B・C・Cmix）

`weights/` には、Cmix8 の前に作った `dcct_A.pt`・`dcct_B.pt`・`dcct_C.pt`・`dcct_Cmix.pt` も残しています。使い方は Cmix8 と同じで、ファイル名を変えるだけです。

<details>
<summary>作りと成績（クリックで開く）</summary>

| | A（`dcct_A.pt`） | B（`dcct_B.pt`） | C（`dcct_C.pt`） | Cmix（`dcct_Cmix.pt`） | Cmix8（`dcct_Cmix8.pt`） |
|---|---|---|---|---|---|
| 作り | 論文どおり：±t = 7、pθ ＋ qφ | ±t = 2、pθ だけ | ±t = 7、pθ だけ | C と同じ | C と同じ |
| 分類器に入る特徴 | 60チャンネル | 30チャンネル | 30チャンネル | 30チャンネル | 30チャンネル |
| 分類器の学習データ | GenImage の SD v1.4 と対応する ImageNet の写真 | 同じ | 同じ | GenImage の7つ（Midjourney 以外）を 1/7 ずつ | GenImage の8つを 1/8 ずつ |
| 学習での AI画像 | JPEG（品質 70〜95） | JPEG（品質 70〜95） | JPEG（品質 96、色の間引き方も本物にそろえる） | C と同じ | C と同じ |
| 判定の線（閾値） | 0.4017 | 0.3284 | 0.4077 | 0.5322 | 0.5900 |
| 線の決め方 | PNG のままの val で、本物と AI画像の正解率の和が最大 | 同じ | 本物の val だけで、上から 5%（論文の DCCT† と同じ決め方） | C と同じ | C と同じ |
| 元の実行 | run42 | run43 | run52 | run66 | run67 |

- pθ・qφ：本物の写真（pθ）・AI画像（qφ）で学習した、隠れた色を予測する部品。±t：予測する値の範囲
- **論文と違う点**：分類器の学習で AI画像を JPEG にしてから使った。Cmix・Cmix8 は、学習に使う生成AIも論文（SD v1.4 だけ）と違う

**成績：GenImage の8つ**（AI画像だけ JPEG 品質96・4:4:4、線は本物の val の上から 5%。生成AIごとに本物・AI画像 各1000枚。Cmix8 はどれも学習で見ている）

| AUC | Midjourney | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | 平均 | 正解率 |
|---|---|---|---|---|---|---|---|---|---|---|
| C | .966 | .998 | .998 | .916 | .976 | .987 | .895 | .913 | .956 | 86.7% |
| Cmix | .946 | .992 | .994 | .982 | .995 | .984 | .963 | .987 | .980 | 93.2% |
| Cmix8 | .972 | .992 | .993 | .984 | .994 | .985 | .969 | .986 | .984 | 93.9% |

**成績：Misinfo の4つ**（Pelekh（2025）のデータセット。本物は COCO。どれも学習で見ていない。AI画像の品質を本物と同じ割合にそろえた評価）

| AUC | FLUX.1-dev | PixArt-α | SDXL-1.0 | Imagen3 | 平均 | 正解率 |
|---|---|---|---|---|---|---|
| C | .922 | .889 | .941 | .914 | .916 | 73.3% |
| Cmix | .935 | .898 | .903 | .892 | .907 | 71.0% |
| Cmix8 | .942 | .921 | .936 | .915 | .928 | 71.8% |

- C は学習で見た生成AIが SD1.4 だけなので、ADM・VQDM・BigGAN の AI画像を本物側にこぼしやすい
- Cmix・Cmix8 は学習で見た生成AIで強い。GenImage の生成AIで「見たことのない生成AIへの強さ」を比べたいときは、比べる相手として使えない

**成績：A・B（10/7 時点。評価のしかたが C 以降と違うので、上の表とは並べて比べないでください）**

| モデル・評価のしかた | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | 平均 |
|---|---|---|---|---|---|---|---|---|
| A：AI画像も JPEG（品質90）にして評価 | .987 | .988 | .873 | .957 | .960 | .863 | .872 | .929 |
| B：AI画像も JPEG（品質90）にして評価 | .985 | .986 | .876 | .918 | .951 | .894 | .896 | .929 |

</details>

---

## 学習し直したいとき

GenImage（外付けSSDに置く想定）と、メタデータを入れる SQLite のデータベースを使います。C の例：

```bash
dcct init-db  --config configs/base.yaml        # データベースを作る（paths.dataset_root を自分の環境に合わせる）
dcct ingest   --config configs/base.yaml        # GenImage を登録する
dcct train-stage1 --target photo --config configs/stage1_photo.yaml                 # pθ
dcct train-stage2 --config configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml  # 分類器
dcct export-bundle --config configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml --run-id <ID> --threshold <τ> --out weights/my_model.pt
```

- Cmix・Cmix8 の学習の設定は aotaku160601/DCCT にあります
- 学習の目安（Apple M 系の Mac Studio）：pθ は数時間、分類器はひと晩

主なフォルダ：`weights/`（学習済みモデル）、`src/`（前処理・モデル・学習・評価）、`configs/`（設定。`base.yaml` が土台）。

---

## 出典

- 論文：Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*. arXiv:2601.22778。公式のコードは公開されていないので、論文の記述から実装しています
- データセット：GenImage（Zhu et al., 2023）、Misinfo（Pelekh, 2025）
- JPEG の品質の偏り：Grommelt et al., *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, 2024
