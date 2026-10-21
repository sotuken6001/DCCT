"""pθ と qφ の出力がどれだけ似ているかを測る（案ウ、学習なし）。

確認エ・オで、pθ だけ・qφ だけ・両方の分類器が同じくらい見分けられるとわかった。
「2本が同じ情報を持っている」という見立てを、出力を直接比べて確かめる。

同じ画像（AI画像と本物）を pθ と qφ に通し、生成器ごとに3つを出す。

1. 特徴の似かた（linear CKA）：分類器に入る混合分布パラメータ（各 30ch）どうしが、
   回転・拡大・ずらしを除いてどれだけ同じ形か。0〜1、1 なら同じ情報。
   画素ごとのベクトルを、パッチごとに決まった数だけ無作為に取り出して比べる
2. 点数のつながり（スピアマンの順位相関）：画像ごとの mean(NLL − entropy) が、
   2本でどれだけ連動しているか。本物だけ・AI画像だけ・全部の3通り
3. 差の点数の力：pθ の点数 − qφ の点数 だけで見分けたときの AUC。
   pθ だけ・qφ だけの AUC より高ければ、2本の違いに使える情報がある（案エへの手がかり）

案カ：qφ のチェックポイントに "random" を渡すと、学習していない（種で決まる乱数の重みの）
モデルを置く。「学習で似たのか、入力が同じだから似て見えるだけか」の基準になる。

案キ：画像ごとの細かい模様の量（予測する値 y′ の絶対値の平均）も出し、2本の点数との
順位相関と、それだけで見分けたときの AUC（AI画像のほうがなめらか＝小さい向き）を出す。
2本とも「なめらかさ」を見ているなら、点数はこの量と強く連動する。

点数はどちらも「そのモデルにとって予測しにくいほど高い」。pθ は本物で学習したので
高いほど AI らしく、qφ は AI画像で学習したので低いほど AI らしい。
qφ の AUC は向きをそろえて（点数を裏返して）出す。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from ..data.dataset import MultiPatchDataset, skip_invalid_collate
from ..data.repository import MetadataRepository
from ..experiment import builders
from ..experiment.config import Config
from ..experiment.paths import resolve
from .evaluator import jpeg_transform
from .metrics import compute_metrics
from .one_class import _REAL_GENERATOR_PREFIX, nll_minus_entropy

logger = logging.getLogger(__name__)

# 案カ：チェックポイントの代わりに渡すと、学習していないモデルを置く
RANDOM = "random"


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """linear CKA（Kornblith ほか 2019）。`x: [N, D1]`、`y: [N, D2]` → 0〜1。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(cross / denom) if denom > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """スピアマンの順位相関。どちらかが一定なら NaN。"""
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    return compute_metrics(scores, labels, threshold=0.0)["auc"]


class ModelSimilarity:
    """pθ と qφ を同じ画像に通して、出力の似かたを生成器ごとに測る。"""

    def __init__(
        self,
        config: Config,
        photo_checkpoint: str | Path | None = None,
        ai_checkpoint: str | Path | None = None,
        device: torch.device | None = None,
        ai_jpeg_quality: int | None = None,
        max_images: int | None = None,
        pixels_per_patch: int = 16,
    ) -> None:
        self.config = config
        self.device = device or builders.build_device(config)
        self.preprocessor = builders.build_preprocessor(config, self.device)
        self.sampler = builders.build_patch_sampler(config)
        self.num_patches = int(config.get("inference.num_patches", 16))
        self.truncation_t = float(config.get("preprocess.truncation_t", 7))
        self.batch_size = int(config.get("evaluate.one_class.batch_size", 4))
        self.max_images = max_images if max_images is not None else config.get("evaluate.max_images_per_generator", None)
        self.pixels_per_patch = int(pixels_per_patch)
        if ai_jpeg_quality is not None and not 1 <= int(ai_jpeg_quality) <= 100:
            raise ValueError(f"ai_jpeg_quality は 1〜100: {ai_jpeg_quality!r}")
        self.ai_jpeg_quality = None if ai_jpeg_quality is None else int(ai_jpeg_quality)

        self.photo_checkpoint = photo_checkpoint or config.get("stage2.photo_model_checkpoint", None)
        self.ai_checkpoint = ai_checkpoint or config.get("stage2.ai_model_checkpoint", None)
        if self.photo_checkpoint is None or self.ai_checkpoint is None:
            raise ValueError("pθ と qφ のチェックポイントを指定してください（--photo-checkpoint / --ai-checkpoint）")
        self.photo_model = self._load(self.photo_checkpoint)
        self.ai_model = self._load(self.ai_checkpoint)
        self.ai_checkpoint = str(self.ai_checkpoint)
        logger.info("pθ: %s / qφ: %s", self.photo_checkpoint, self.ai_checkpoint)

        self.repo = MetadataRepository(config.get("paths.db_path"))

    def _load(self, checkpoint: str | Path):
        if str(checkpoint) == RANDOM:
            # 案カ：学習していないモデル。種を固定して、毎回同じ重みにする
            torch.manual_seed(int(self.config.get("project.seed", 42)))
            model = builders.build_conditional_unet(self.config, self.preprocessor).to(self.device)
            logger.info("学習していないモデル（乱数の重み）を使います")
            return model.freeze()
        model = builders.build_conditional_unet(self.config, self.preprocessor).to(self.device)
        state = torch.load(resolve(checkpoint), map_location=self.device, weights_only=False)
        model.load_state_dict(state["modules"]["conditional_unet"])
        return model.freeze()

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "ModelSimilarity":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------

    def rows_for(self, generator: str, split: str) -> list[dict[str, Any]]:
        """生成器の AI画像と、それに対応する本物を集める（一クラス評価と同じ選び方）。"""
        if self.config.get("dataset.real_generator_naming", "per_source") == "shared":
            real_name = _REAL_GENERATOR_PREFIX
        else:
            real_name = f"{_REAL_GENERATOR_PREFIX}@{generator}"
        rows: list[dict[str, Any]] = []
        for label, name in (("ai", generator), ("real", real_name)):
            found = self.repo.list_images(label=label, split=split, generator_names=[name], limit=self.max_images)
            rows.extend(dict(row) for row in found)
        return rows

    @torch.no_grad()
    def _process(self, patches: torch.Tensor, generator: torch.Generator):
        """`[B, P, 3, s, s]` → (pθ の点数 `[B]`, qφ の点数 `[B]`, 模様の量 `[B]`, pθ の画素特徴 `[M, D]`, qφ の画素特徴 `[M, D]`)。"""
        batch, num_patches = patches.shape[:2]
        flat = patches.reshape(batch * num_patches, *patches.shape[2:]).to(self.device)
        x_prime, y_prime = self.preprocessor.prepare(flat)
        texture = y_prime.abs().flatten(1).mean(dim=1).reshape(batch, num_patches).mean(dim=1)

        scores, features = [], []
        for model in (self.photo_model, self.ai_model):
            params = model(x_prime)
            scores.append(nll_minus_entropy(y_prime, params, self.truncation_t).reshape(batch, num_patches).mean(dim=1))
            features.append(params.as_feature_map())                         # [B*P, D, H, W]

        # 画素をパッチごとに同じ位置で取り出す（2本で同じ画素を比べる）
        n, d, h, w = features[0].shape
        index = torch.stack(
            [torch.randperm(h * w, generator=generator)[: self.pixels_per_patch] for _ in range(n)]
        ).to(self.device)                                                     # [N, k]
        sampled = []
        for feature in features:
            flat_feature = feature.reshape(n, d, h * w)
            picked = flat_feature.gather(2, index.unsqueeze(1).expand(-1, d, -1))   # [N, D, k]
            sampled.append(picked.permute(0, 2, 1).reshape(-1, d).cpu().numpy())
        return scores[0].cpu().numpy(), scores[1].cpu().numpy(), texture.cpu().numpy(), sampled[0], sampled[1]

    def _analyze_part(self, rows: Sequence[dict[str, Any]], transform: Any, generator: torch.Generator):
        dataset = MultiPatchDataset(rows, self.sampler, self.num_patches, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(self.config.get("train.num_workers", 0)),
            collate_fn=skip_invalid_collate,
        )
        parts: dict[str, list[np.ndarray]] = {"photo": [], "ai": [], "texture": [], "photo_feat": [], "ai_feat": [], "label": []}
        for batch in loader:
            if batch is None:
                continue
            patches, targets, _ = batch
            photo, ai, texture, photo_feat, ai_feat = self._process(patches, generator)
            parts["photo"].append(photo)
            parts["ai"].append(ai)
            parts["texture"].append(texture)
            parts["photo_feat"].append(photo_feat)
            parts["ai_feat"].append(ai_feat)
            # 共有メモリ上のテンソルを参照したままにすると fd が積み上がるのでコピーする
            parts["label"].append(targets.numpy().copy())
        return parts

    def analyze(self, rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        """画像ごとの点数と、取り出した画素の特徴を返す。AI画像だけ JPEG にする指定があれば通す。"""
        generator = torch.Generator().manual_seed(int(self.config.get("project.seed", 42)))
        ai_rows = [row for row in rows if row["label"] == "ai"]
        real_rows = [row for row in rows if row["label"] != "ai"]
        ai_transform = None if self.ai_jpeg_quality is None else jpeg_transform(self.ai_jpeg_quality)

        merged: dict[str, list[np.ndarray]] = {"photo": [], "ai": [], "texture": [], "photo_feat": [], "ai_feat": [], "label": []}
        for part_rows, transform in ((ai_rows, ai_transform), (real_rows, None)):
            if not part_rows:
                continue
            for key, values in self._analyze_part(part_rows, transform, generator).items():
                merged[key].extend(values)
        if not merged["photo"]:
            raise ValueError(f"{len(rows)}枚すべてが読み込めませんでした")
        result = {key: np.concatenate(values) for key, values in merged.items()}
        skipped = len(rows) - len(result["label"])
        if skipped:
            logger.warning("%d枚を対象から除外しました（%d枚中）。64px未満か破損した画像です", skipped, len(rows))
        return result

    def summarize(self, generator: str, data: dict[str, np.ndarray]) -> dict[str, Any]:
        labels = data["label"]
        real, ai = labels == 0, labels == 1
        photo, ai_score, texture = data["photo"], data["ai"], data["texture"]
        return {
            "generator": generator,
            "cka": linear_cka(data["photo_feat"], data["ai_feat"]),
            "spearman_real": spearman(photo[real], ai_score[real]) if real.sum() > 1 else None,
            "spearman_ai": spearman(photo[ai], ai_score[ai]) if ai.sum() > 1 else None,
            "spearman_all": spearman(photo, ai_score),
            "auc_photo": _auc(photo, labels),
            "auc_ai": _auc(-ai_score, labels),
            "auc_diff": _auc(photo - ai_score, labels),
            "spearman_texture_photo": spearman(texture, photo),
            "spearman_texture_ai": spearman(texture, ai_score),
            "auc_texture": _auc(-texture, labels),
            "num_samples": int(len(labels)),
        }

    def run(self, split: str = "test") -> list[dict[str, Any]]:
        excluded = set(self.config.get("dataset.excluded_generators", []))
        results = []
        for generator in self.config.get("dataset.eval_generators", []):
            if generator in excluded:
                continue
            rows = self.rows_for(generator, split)
            if not rows:
                logger.warning("対象の画像がありません: %s（split=%s）", generator, split)
                continue
            result = self.summarize(generator, self.analyze(rows))
            logger.info(
                "%s: CKA=%.4f 順位相関 本物=%s AI=%s 全部=%.4f AUC pθ=%s qφ=%s 差=%s",
                generator, result["cka"], _fmt(result["spearman_real"]), _fmt(result["spearman_ai"]),
                result["spearman_all"], _fmt(result["auc_photo"]), _fmt(result["auc_ai"]), _fmt(result["auc_diff"]),
            )
            logger.info(
                "%s: 模様の量との順位相関 pθ=%s qφ=%s 模様の量だけの AUC=%s",
                generator, _fmt(result["spearman_texture_photo"]), _fmt(result["spearman_texture_ai"]),
                _fmt(result["auc_texture"]),
            )
            results.append(result)
        return results


def _fmt(value: float | None) -> str:
    return "N/A" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4f}"


COLUMNS = [
    "generator", "cka", "spearman_real", "spearman_ai", "spearman_all", "auc_photo", "auc_ai", "auc_diff",
    "spearman_texture_photo", "spearman_texture_ai", "auc_texture", "num_samples",
]


def with_mean_row(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return results
    row: dict[str, Any] = {"generator": "平均", "num_samples": sum(r["num_samples"] for r in results)}
    for key in COLUMNS[1:-1]:
        values = [r[key] for r in results if r[key] is not None and not np.isnan(r[key])]
        row[key] = float(np.mean(values)) if values else None
    return [*results, row]


def write_report(
    results: list[dict[str, Any]],
    out_dir: str | Path,
    photo_checkpoint: str | Path,
    ai_checkpoint: str | Path,
    ai_jpeg_quality: int | None = None,
) -> Path:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = with_mean_row(results)
    with open(out / "model_similarity.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in COLUMNS})

    lines = [
        "# pθ と qφ の出力の似かた（案ウ）",
        "",
        f"- pθ: `{photo_checkpoint}`",
        f"- qφ: `{ai_checkpoint}`",
        "- CKA: 混合分布パラメータ（分類器の入力）の似かた。1 なら同じ情報",
        "- 順位相関: 画像ごとの mean(NLL − entropy) が2本でどれだけ連動しているか",
        "- AUC: pθ は点数のまま、qφ は裏返して、差は pθ − qφ。どれも 0.5 が当てずっぽう",
        "- 模様の量: y′ の絶対値の平均。順位相関は点数との連動、AUC は模様が少ないほど AI とみなしたとき",
        *(["- qφ の代わりに学習していないモデル（乱数の重み）を置いた（案カ：CKA の基準）"] if str(ai_checkpoint) == RANDOM else []),
        *([f"- AI画像だけ JPEG（品質 {ai_jpeg_quality}）にしてから測った（本物はそのまま）"] if ai_jpeg_quality is not None else []),
        "",
        "| generator | CKA | 順位相関（本物） | 順位相関（AI） | 順位相関（全部） | AUC pθ | AUC qφ | AUC 差 "
        "| 模様と pθ | 模様と qφ | AUC 模様 | 枚数 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['generator']} | " + " | ".join(_fmt(row[key]) for key in COLUMNS[1:-1]) + f" | {row['num_samples']} |"
        )
    (out / "model_similarity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
