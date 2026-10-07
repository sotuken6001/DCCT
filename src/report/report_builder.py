"""結果レポートの出力（FR-9 / 02_基本設計書 `report.builder`）。

DBに保存された評価結果を集計し、CSV・Markdown表・グラフとして出力する。
図中のラベルは日本語フォントに依存しないよう英語で描画する（本文は日本語）。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..data.repository import MetadataRepository  # noqa: E402
from ..experiment.config import Config  # noqa: E402
from ..experiment.paths import resolve  # noqa: E402

logger = logging.getLogger(__name__)

# dataviz の検証済みカテゴリカルパレット（light）。系列は必ずこの順で固定的に割り当てる
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID_COLOR = "#d8d7d2"


def _fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(_fmt(value) if isinstance(value, float) else ("" if value is None else str(value)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- 集計クエリ


def fetch_cross_generator(repo: MetadataRepository, run_id: int) -> list[dict[str, Any]]:
    rows = repo.conn.execute(
        """
        SELECT g.name AS generator, er.accuracy, er.auc, er.ap, er.num_samples
        FROM evaluation_results er
        JOIN generators g ON g.generator_id = er.generator_id
        WHERE er.run_id = ? AND er.eval_type = 'cross_generator'
          -- 同じ生成器を再評価した場合は最新の1件だけを採る
          AND er.result_id = (
              SELECT MAX(e2.result_id) FROM evaluation_results e2
              WHERE e2.run_id = er.run_id AND e2.eval_type = er.eval_type
                AND e2.generator_id = er.generator_id
          )
        ORDER BY g.name
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_robustness(repo: MetadataRepository, run_id: int) -> list[dict[str, Any]]:
    rows = repo.conn.execute(
        """
        SELECT g.name AS generator, er.perturbation_type, er.perturbation_level,
               er.accuracy, er.auc, er.ap
        FROM evaluation_results er
        JOIN generators g ON g.generator_id = er.generator_id
        WHERE er.run_id = ? AND er.eval_type = 'robustness'
          -- 同じ条件を再評価した場合は最新の1件だけを採る
          AND er.result_id = (
              SELECT MAX(e2.result_id) FROM evaluation_results e2
              WHERE e2.run_id = er.run_id AND e2.eval_type = er.eval_type
                AND e2.generator_id = er.generator_id
                AND e2.perturbation_type IS er.perturbation_type
                AND e2.perturbation_level IS er.perturbation_level
          )
        ORDER BY er.perturbation_type, er.perturbation_level DESC, g.name
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_ablation(repo: MetadataRepository) -> list[dict[str, Any]]:
    """アブレーション条件ごとの平均Accuracy（04_DB設計書 6節のクエリ例に対応）。"""
    rows = repo.conn.execute(
        """
        SELECT ac.ablation_group, e.name AS experiment, ac.parameter_name, ac.parameter_value,
               AVG(er.accuracy) AS mean_accuracy, COUNT(DISTINCT er.generator_id) AS num_generators
        FROM ablation_configs ac
        JOIN experiments e     ON e.experiment_id = ac.experiment_id
        JOIN training_runs r   ON r.experiment_id = e.experiment_id
        JOIN evaluation_results er ON er.run_id = r.run_id AND er.eval_type = 'ablation'
        GROUP BY ac.ablation_group, e.name, ac.parameter_name, ac.parameter_value
        ORDER BY ac.ablation_group, e.name, ac.parameter_name
        """
    ).fetchall()
    return [dict(row) for row in rows]


def add_mean_row(rows: list[dict[str, Any]], label_column: str = "generator") -> list[dict[str, Any]]:
    """生成器平均の行を末尾に足す（論文Table 1の mAcc に相当）。"""
    if not rows:
        return rows

    def mean(column: str) -> float | None:
        values = [row[column] for row in rows if row.get(column) is not None]
        return sum(values) / len(values) if values else None

    return rows + [
        {
            label_column: f"平均({len(rows)}生成器)",
            "accuracy": mean("accuracy"),
            "auc": mean("auc"),
            "ap": mean("ap"),
            "num_samples": sum(row.get("num_samples") or 0 for row in rows),
        }
    ]


# -------------------------------------------------------------------- グラフ


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)


def plot_robustness(
    rows: Sequence[dict[str, Any]],
    perturbation_type: str,
    out_path: Path,
    xlabel: str,
    metric: str = "accuracy",
) -> Path | None:
    """劣化強度ごとの指標の推移を生成器別の折れ線で描く（Figure 6相当）。

    Accuracy は固定した閾値τに対する値なので、劣化でスコア分布がずれるだけでも
    偶然と同じ0.5まで落ちる。閾値に依存しないAUCも並べて描き、識別力そのものが
    失われたのか、閾値がずれただけなのかを見分けられるようにする。
    """
    subset = [
        row
        for row in rows
        if row["perturbation_type"] == perturbation_type and row.get(metric) is not None
    ]
    if not subset:
        return None

    generators = sorted({row["generator"] for row in subset})
    baseline = [
        row[metric]
        for row in rows
        if row["perturbation_type"] == "none" and row.get(metric) is not None
    ]

    figure, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    if baseline:
        # 劣化なしの平均を参照線として1本だけ引く（系列色は使わない）
        ax.axhline(
            sum(baseline) / len(baseline),
            color=TEXT_SECONDARY, linestyle="--", linewidth=1.2, alpha=0.7,
            label="No perturbation (mean)",
        )

    for index, generator in enumerate(generators):
        points = sorted(
            [(row["perturbation_level"], row[metric]) for row in subset if row["generator"] == generator],
            reverse=True,
        )
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
            markersize=7,
            linewidth=2,
            # マーカーが重なったときに見分けがつくよう、地色のリングを付ける
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=generator,
        )

    _style_axes(
        ax, xlabel, metric.upper() if metric == "auc" else metric.capitalize(),
        f"Robustness to {perturbation_type} ({metric.upper() if metric == 'auc' else metric})",
    )
    ax.set_ylim(0.0, 1.02)
    levels = sorted({row["perturbation_level"] for row in subset}, reverse=True)
    ax.set_xticks(levels)   # 実在する劣化強度だけを刻む
    ax.invert_xaxis()       # 左ほど劣化が小さい（QF・倍率は大きいほど劣化が小さい）
    # 凡例はプロット領域の外（下）に置き、データと重ならないようにする
    ax.legend(
        frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=min(len(generators) + 1, 4),
    )

    figure.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return out_path


# ------------------------------------------------------------ レポート生成


def build_report(config: Config, run_id: int, out_dir: str | Path) -> Path:
    """run_id の評価結果をCSV・グラフ・Markdownにまとめる。"""
    out_path = resolve(out_dir) / f"run{run_id}"
    out_path.mkdir(parents=True, exist_ok=True)

    repo = MetadataRepository(config.get("paths.db_path"))
    try:
        cross = fetch_cross_generator(repo, run_id)
        robustness = fetch_robustness(repo, run_id)
        ablation = fetch_ablation(repo)
        experiment = repo.conn.execute(
            """
            SELECT e.name, e.stage, e.git_commit, e.seed, r.started_at, r.completed_at, r.status
            FROM training_runs r JOIN experiments e ON e.experiment_id = r.experiment_id
            WHERE r.run_id = ?
            """,
            (run_id,),
        ).fetchone()
    finally:
        repo.close()

    sections: list[str] = ["# DCCT 再現実装 評価レポート", ""]
    if experiment is not None:
        sections += [
            f"- 実行ID: {run_id}",
            f"- 実験名: {experiment['name']}（stage={experiment['stage']}, status={experiment['status']}）",
            f"- gitコミット: {experiment['git_commit'] or 'N/A'}",
            f"- seed: {experiment['seed']}",
            f"- 期間: {experiment['started_at']} 〜 {experiment['completed_at'] or '（未完了）'}",
            "",
        ]

    # --- Table 1相当 ---
    if cross:
        columns = ["generator", "accuracy", "auc", "ap", "num_samples"]
        with_mean = add_mean_row(cross)
        _write_csv(out_path / "cross_generator.csv", with_mean, columns)
        sections += [
            "## Cross-generator評価（Table 1相当）",
            "",
            _markdown_table(with_mean, columns),
            "",
        ]

    # --- Table 4相当 ---
    if ablation:
        columns = ["ablation_group", "experiment", "parameter_name", "parameter_value", "mean_accuracy", "num_generators"]
        _write_csv(out_path / "ablation.csv", ablation, columns)
        sections += ["## アブレーション（Table 4相当）", "", _markdown_table(ablation, columns), ""]

    # --- Figure 6相当 ---
    if robustness:
        columns = ["generator", "perturbation_type", "perturbation_level", "accuracy", "auc", "ap"]
        _write_csv(out_path / "robustness.csv", robustness, columns)
        sections += [
            "## ロバスト性評価（Figure 6相当）",
            "",
            "Accuracy は固定した閾値τに対する値のため、劣化でスコア分布がずれるだけでも"
            "0.5（偶然と同じ）まで落ちる。閾値に依存しないAUCを併記してあるので、"
            "識別力そのものが失われたのかを併せて確認すること。",
            "",
        ]

        for perturbation_type, xlabel in (("jpeg", "JPEG quality factor"), ("downsample", "Downsample ratio")):
            for metric in ("accuracy", "auc"):
                figure_path = plot_robustness(
                    robustness,
                    perturbation_type,
                    out_path / f"robustness_{perturbation_type}_{metric}.png",
                    xlabel,
                    metric=metric,
                )
                if figure_path is not None:
                    sections.append(f"![{perturbation_type} {metric}]({figure_path.name})")
                    sections.append("")

        sections += [_markdown_table(robustness, columns), ""]

    if not (cross or ablation or robustness):
        sections += ["（この実行に紐づく評価結果がまだありません）", ""]

    report_path = out_path / "report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("レポートを出力しました: %s", report_path)
    return report_path
