"""DCCT再現実装のコマンドラインエントリポイント。

02_基本設計書 6節「実行フロー（CLIコマンド遷移）」に対応する。

    python -m src.cli init-db       --config configs/base.yaml
    python -m src.cli ingest        --config configs/base.yaml
    python -m src.cli status        --config configs/base.yaml
    python -m src.cli train-stage1  --target photo --config configs/stage1_photo.yaml
    python -m src.cli train-stage1  --target ai    --config configs/stage1_ai.yaml
    python -m src.cli train-stage2  --config configs/stage2_classifier.yaml
    python -m src.cli evaluate      --mode cross_generator --run-id <ID>
    python -m src.cli evaluate-one-class --config configs/stage2_classifier.yaml
    python -m src.cli jpeg-stats    --config configs/base.yaml
    python -m src.cli subsampling-scores --config configs/stage2_classifier_aijpeg_q96.yaml
    python -m src.cli threshold-check --config configs/stage2_classifier_aijpeg_q96_mix_photo_only.yaml --run-id 52 --ai-jpeg-quality 96 --subsampling 4:4:4
    python -m src.cli compare-models --config configs/stage2_classifier_aijpeg_q96_qjpeg.yaml --ai-jpeg-quality 96
    python -m src.cli report        --run-id <ID>
    python -m src.cli export-bundle --config configs/stage2_classifier_aijpeg.yaml --run-id <ID> --out weights/dcct_A.pt
    python -m src.cli predict       --bundle weights/dcct_A.pt photo.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .data import db as db_module
from .data import ingest as ingest_module
from .experiment.config import Config
from .experiment.paths import resolve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcct", description="DCCT (Color Matters) 再現実装")
    sub = parser.add_subparsers(dest="command", required=True)

    p_initdb = sub.add_parser("init-db", help="メタデータDBを作成する（04_DB設計書 4節のDDLを適用）")
    p_initdb.add_argument("--config", default="configs/base.yaml")
    p_initdb.add_argument("--force", action="store_true", help="既存DBを削除して作り直す")

    p_ingest = sub.add_parser("ingest", help="外部SSD上のGenImageを走査しDBへ登録する")
    p_ingest.add_argument("--config", default="configs/base.yaml")
    p_ingest.add_argument(
        "--generator", action="append", dest="generators",
        help="この生成器のみ走査する（複数指定可）。失敗した生成器だけの再実行に使う",
    )
    p_ingest.add_argument("--limit", type=int, help="1生成器あたりの走査枚数上限（スモークテスト用）")
    p_ingest.add_argument("--dry-run", action="store_true", help="DBへ書き込まずに走査結果だけ表示する")
    p_ingest.add_argument("--no-progress", action="store_true", help="進捗バーを表示しない")
    probe_group = p_ingest.add_mutually_exclusive_group()
    probe_group.add_argument(
        "--no-probe", action="store_true",
        help="すべてのソースで画像ヘッダを読まない（width/heightと破損判定を行わない）。"
             "既定ではZIPのみ読み、ディレクトリは読まない",
    )
    probe_group.add_argument(
        "--probe", action="store_true",
        help="すべてのソースで画像ヘッダを読む（ディレクトリ形式では非常に遅くなることがある）",
    )
    p_ingest.add_argument(
        "--cleanup", action="store_true",
        help="登録済みのOS管理ファイル（._xxx 等）の行を削除する（--dry-run と併用可）",
    )

    p_status = sub.add_parser("status", help="取り込み状況と直近の実行をまとめて表示する")
    p_status.add_argument("--config", default="configs/base.yaml")

    p_s1 = sub.add_parser("train-stage1", help="条件付き分布モデル pθ / qφ を学習する")
    p_s1.add_argument("--target", choices=["photo", "ai"], required=True)
    p_s1.add_argument("--config", required=True)
    p_s1.add_argument("--epochs", type=int, help="configの train.num_epochs を上書きする")
    p_s1.add_argument("--max-steps", type=int, help="1エポックあたりのステップ数上限（スモークテスト用）")
    p_s1.add_argument("--resume", help="再開するチェックポイントのパス")

    p_s2 = sub.add_parser("train-stage2", help="二値分類器 gψ を学習する")
    p_s2.add_argument("--config", default="configs/stage2_classifier.yaml")
    p_s2.add_argument("--epochs", type=int, help="configの train.num_epochs を上書きする")
    p_s2.add_argument("--max-steps", type=int, help="1エポックあたりのステップ数上限（スモークテスト用）")
    p_s2.add_argument("--resume", help="再開するチェックポイントのパス")

    p_eval = sub.add_parser("evaluate", help="評価を実行する")
    p_eval.add_argument("--mode", choices=["cross_generator", "ablation", "robustness", "ai_jpeg"], required=True)
    p_eval.add_argument("--config", default="configs/stage2_classifier.yaml")
    p_eval.add_argument("--run-id", type=int, help="評価対象のStage II実行ID（省略時は最新の完了実行）")
    p_eval.add_argument("--experiment-id", type=int, help="実験IDから実行IDを解決する")
    p_eval.add_argument("--checkpoint", help="分類器チェックポイントを明示的に指定する")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--epochs", type=int, help="ablationモード: 各variantの学習エポック数")
    p_eval.add_argument("--max-steps", type=int, help="ablationモード: 1エポックのステップ数上限")
    p_eval.add_argument("--variant", action="append", dest="variants", help="ablationモード: 実行するvariantのlabel")
    p_eval.add_argument(
        "--ai-jpeg-quality", type=int, action="append", dest="ai_jpeg_qualities",
        help="ai_jpegモード: AI画像だけをこの品質の JPEG にする（複数回指定できる。既定 90 と 75）",
    )

    p_one = sub.add_parser(
        "evaluate-one-class",
        help="pθ だけで判定する一クラス検出（論文 Appendix B の DCCT†）を評価し、表5と比べる",
    )
    p_one.add_argument("--config", default="configs/stage2_classifier.yaml")
    p_one.add_argument("--photo-checkpoint", help="使う pθ（省略時は config の stage2.photo_model_checkpoint）")
    p_one.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_one.add_argument(
        "--ai-jpeg-quality", type=int,
        help="AI画像だけをこの品質の JPEG にしてから評価する（本物は JPEG・AI画像は PNG という形式の差を消す確認用）",
    )
    p_one.add_argument("--out", help="出力先（既定: <report_root>/one_class_<pθの保存フォルダ名>）")

    p_cmp = sub.add_parser(
        "compare-models",
        help="pθ と qφ の出力の似かた（特徴の CKA・点数の順位相関・差の点数の AUC）を生成器ごとに測る（案ウ）",
    )
    p_cmp.add_argument("--config", default="configs/stage2_classifier.yaml")
    p_cmp.add_argument("--photo-checkpoint", help="使う pθ（省略時は config の stage2.photo_model_checkpoint）")
    p_cmp.add_argument(
        "--ai-checkpoint",
        help="使う qφ（省略時は config の stage2.ai_model_checkpoint）。random で学習していないモデル（案カ：CKA の基準）",
    )
    p_cmp.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_cmp.add_argument("--ai-jpeg-quality", type=int, help="AI画像だけをこの品質の JPEG にしてから測る")
    p_cmp.add_argument("--max-images", type=int, default=1000, help="生成器ごと・ラベルごとの枚数の上限（既定 1000）")
    p_cmp.add_argument("--out", help="出力先（既定: <report_root>/compare_<pθの保存フォルダ名>_<qφの保存フォルダ名>）")

    p_js = sub.add_parser(
        "jpeg-stats",
        help="本物の JPEG の作り（色の間引き方・推定品質）を数え、AI画像にかけている PIL の JPEG と比べる（案ク）",
    )
    p_js.add_argument("--config", default="configs/base.yaml")
    p_js.add_argument("--max-images", type=int, default=1000, help="まとまりごとに調べる枚数の上限（既定 1000）")
    p_js.add_argument("--pil-quality", type=int, default=96, help="比べる PIL の JPEG の品質（既定 96）")
    p_js.add_argument("--out", help="出力先（既定: <report_root>/jpeg_stats）")

    p_ss = sub.add_parser(
        "subsampling-scores",
        help="色の間引き方（4:4:4 / 4:2:0）ごとに pθ / qφ の予測しにくさの点数を比べる（確認コ）",
    )
    p_ss.add_argument("--config", default="configs/stage2_classifier.yaml")
    p_ss.add_argument("--checkpoint", help="使う条件付きモデル（省略時は config の stage2.photo_model_checkpoint）")
    p_ss.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_ss.add_argument("--max-images", type=int, default=1000, help="まとまりごとの枚数の上限（既定 1000）")
    p_ss.add_argument("--ai-jpeg-quality", type=int, default=96, help="AI画像にかける JPEG の品質（既定 96）")
    p_ss.add_argument("--out", help="出力先（既定: <report_root>/subsampling_<モデルの保存フォルダ名>）")

    p_tc = sub.add_parser(
        "threshold-check",
        help="生成器ごとに本物・AI画像それぞれの正解率と、線をいちばん良く引いたときの正解率を出す（確認セ、DB に残さない）",
    )
    p_tc.add_argument("--config", default="configs/stage2_classifier.yaml")
    p_tc.add_argument("--run-id", type=int, help="評価対象のStage II実行ID（省略時は最新の完了実行）")
    p_tc.add_argument("--experiment-id", type=int, help="実験IDから実行IDを解決する")
    p_tc.add_argument("--checkpoint", help="分類器チェックポイントを明示的に指定する")
    p_tc.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_tc.add_argument("--ai-jpeg-quality", type=int, help="AI画像だけをこの品質の JPEG にする（省略時は AI画像は元のまま）")
    p_tc.add_argument("--subsampling", choices=["4:4:4", "4:2:2", "4:2:0"], help="その JPEG の色の間引き方（省略時は PIL の初期設定 4:2:0）")
    p_tc.add_argument(
        "--real-false-alarm", type=float,
        help="線を本物の val だけで決め、本物を AI と間違える割合をこの値（%%）にする（確認ソ。例 5）",
    )
    p_tc.add_argument("--out", help="出力先（既定: <report_root>/threshold_check_run<ID>_<条件>）")

    p_report = sub.add_parser("report", help="結果レポートを出力する")
    p_report.add_argument("--config", default="configs/base.yaml")
    p_report.add_argument("--run-id", type=int, help="対象の実行ID（省略時は最新の完了実行）")
    p_report.add_argument("--experiment-id", type=int, help="実験IDから実行IDを解決する")
    p_report.add_argument("--out", help="出力先（既定: configの paths.report_root）")

    p_export = sub.add_parser(
        "export-bundle", help="学習済みの pθ / qφ / 分類器と設定を1つのファイルにまとめる（配布用）"
    )
    p_export.add_argument("--config", required=True, help="分類器を学習したときの設定")
    p_export.add_argument("--run-id", type=int, help="分類器の実行ID（best.pt を探す）")
    p_export.add_argument("--checkpoint", help="分類器のチェックポイントを直接指定する")
    p_export.add_argument("--threshold", type=float, help="判定の線（評価で出た閾値τ）。predict の ai/real に使う")
    p_export.add_argument("--out", required=True, help="書き出すファイル（例: weights/dcct_A.pt）")

    p_predict = sub.add_parser("predict", help="まとめたファイル1つで画像を判定する")
    p_predict.add_argument("--bundle", required=True, help="export-bundle で作ったファイル")
    p_predict.add_argument("--device", help="cpu / cuda / mps（省略時はバンドルの設定どおり auto）")
    p_predict.add_argument("images", nargs="+", help="判定する画像のパス")

    return parser


def cmd_init_db(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    db_path = config.get("paths.db_path")

    applied = db_module.init_db(db_path, force=args.force)
    tables = db_module.table_names(db_path)

    print(f"メタデータDB: {resolve(db_path)}")
    if applied:
        print(f"適用したマイグレーション: {', '.join(applied)}")
    else:
        print("適用したマイグレーション: なし（すべて適用済み）")
    print(f"テーブル({len(tables)}): {', '.join(tables)}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    config = Config.load(args.config)

    if args.cleanup:
        count, samples = ingest_module.cleanup_junk(config, dry_run=args.dry_run)
        verb = "削除対象" if args.dry_run else "削除しました"
        print(f"OS管理ファイルの行: {count:,} 件を{verb}")
        for sample in samples:
            print(f"  例: {sample}")
        return 0

    results = ingest_module.ingest(
        config,
        only_generators=args.generators,
        limit=args.limit,
        dry_run=args.dry_run,
        progress=not args.no_progress,
        probe_header=True if args.probe else (False if args.no_probe else None),
    )

    print()
    print(f"{'生成器':<14}{'走査':>10}{'新規登録':>10}{'登録済み':>10}{'破損':>8}{'64px未満':>10}  備考")
    for st in results:
        note = st.skipped_reason or ", ".join(f"{k}={v}" for k, v in sorted(st.by_split.items()))
        print(
            f"{st.generator:<14}{st.scanned:>10,}{st.inserted:>10,}"
            f"{max(st.already_registered, 0):>10,}{st.invalid:>8,}{st.too_small:>10,}  {note}"
        )
    if args.dry_run:
        print("\n（--dry-run のためDBへは書き込んでいません）")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .data.repository import MetadataRepository

    config = Config.load(args.config)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        dataset = repo.conn.execute("SELECT name, root_path FROM datasets LIMIT 1").fetchone()
        summary = repo.image_summary()
        empty_generators = [
            row["name"]
            for row in repo.conn.execute(
                """
                SELECT g.name FROM generators g
                LEFT JOIN images i ON i.generator_id = g.generator_id
                GROUP BY g.generator_id HAVING COUNT(i.image_id) = 0
                ORDER BY g.name
                """
            )
        ]
        runs = repo.conn.execute(
            """
            SELECT r.run_id, e.stage, e.name, r.status, r.started_at
            FROM training_runs r JOIN experiments e ON e.experiment_id = r.experiment_id
            ORDER BY r.run_id DESC LIMIT 5
            """
        ).fetchall()

    print(f"メタデータDB: {resolve(config.get('paths.db_path'))}")
    if dataset is not None:
        print(f"データセット: {dataset['name']}  ({dataset['root_path']})")

    if not summary:
        print("\n取り込み済みの画像はありません。`ingest` を実行してください。")
    else:
        print()
        print(f"{'生成器':<26}{'split':>8}{'ラベル':>8}{'枚数':>12}{'破損':>9}{'64px未満':>10}")
        totals = {"num_images": 0, "num_invalid": 0, "num_too_small": 0}
        for row in summary:
            print(
                f"{row['generator']:<26}{row['split']:>8}{row['label']:>8}"
                f"{row['num_images']:>12,}{row['num_invalid'] or 0:>9,}{row['num_too_small'] or 0:>10,}"
            )
            for key in totals:
                totals[key] += row[key] or 0
        print(f"{'合計':<26}{'':>8}{'':>8}{totals['num_images']:>12,}"
              f"{totals['num_invalid']:>9,}{totals['num_too_small']:>10,}")

    excluded = config.get("dataset.excluded_generators", [])
    if excluded:
        print(f"\n評価・学習から除外中: {', '.join(excluded)}")
    if empty_generators:
        print(f"画像が1枚も登録されていない生成器: {', '.join(empty_generators)}")

    if runs:
        print()
        print(f"{'run_id':>7}  {'stage':<20}{'status':<12}{'開始':<20}実験名")
        for row in runs:
            print(
                f"{row['run_id']:>7}  {row['stage']:<20}{row['status']:<12}"
                f"{row['started_at']:<20}{row['name']}"
            )
    return 0


def _run_trainer(trainer, args: argparse.Namespace) -> int:
    with trainer:
        resume_from = args.resume or trainer.config.get("train.resume_from", None)
        if resume_from:
            trainer.load_checkpoint(resume_from, resume=True)

        run_id = trainer.fit(num_epochs=args.epochs, max_steps_per_epoch=args.max_steps)

    print(f"学習が完了しました: run_id={run_id}")
    print(f"チェックポイント: {trainer.checkpoint_dir}")
    return 0


def cmd_train_stage1(args: argparse.Namespace) -> int:
    from .train.train_stage1 import TrainerStage1

    config = Config.load(args.config)
    return _run_trainer(TrainerStage1(config, args.target), args)


def cmd_train_stage2(args: argparse.Namespace) -> int:
    from .train.train_stage2 import TrainerStage2

    config = Config.load(args.config)
    return _run_trainer(TrainerStage2(config), args)


def _resolve_run(repo, run_id: int | None, experiment_id: int | None) -> int:
    """評価・レポート対象の run_id を決める。"""
    if run_id is not None:
        return run_id

    if experiment_id is not None:
        row = repo.conn.execute(
            "SELECT run_id FROM training_runs WHERE experiment_id = ? ORDER BY run_id DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise SystemExit(f"experiment_id={experiment_id} に紐づく実行が見つかりません")
        return int(row["run_id"])

    row = repo.conn.execute(
        """
        SELECT r.run_id FROM training_runs r
        JOIN experiments e ON e.experiment_id = r.experiment_id
        WHERE e.stage = 'stage2_classifier' AND r.status = 'completed'
        ORDER BY r.run_id DESC LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise SystemExit("完了済みのStage II実行が見つかりません。--run-id を指定してください")
    return int(row["run_id"])


def _resolve_classifier_checkpoint(repo, run_id: int) -> str | None:
    """その実行が保存したチェックポイント（best.pt優先）を探す。"""
    rows = repo.conn.execute(
        "SELECT filepath FROM model_checkpoints WHERE run_id = ? ORDER BY checkpoint_id DESC", (run_id,)
    ).fetchall()
    for row in rows:
        if row["filepath"].endswith("best.pt"):
            return row["filepath"]
    return rows[0]["filepath"] if rows else None


def cmd_export_bundle(args: argparse.Namespace) -> int:
    from .data.repository import MetadataRepository
    from .eval.bundle import export_bundle

    config = Config.load(args.config)
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.run_id is None:
            raise SystemExit("--run-id か --checkpoint を指定してください")
        with MetadataRepository(config.get("paths.db_path")) as repo:
            checkpoint = _resolve_classifier_checkpoint(repo, args.run_id)
        if checkpoint is None:
            raise SystemExit(f"run_id={args.run_id} のチェックポイントが見つかりません")

    metadata = {"run_id": args.run_id, "source_config": str(args.config)}
    out = export_bundle(config, checkpoint, args.out, threshold=args.threshold, metadata=metadata)
    print(f"書き出しました: {out}（{out.stat().st_size / 1e6:.1f} MB）")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .eval.bundle import DCCTDetector

    detector = DCCTDetector.from_bundle(args.bundle, device=args.device)
    print(f"{'画像':<40}{'AIらしさ':>10}  判定")
    for image in args.images:
        result = detector.predict(image)
        print(f"{image:<40}{result['prob_ai']:>10.4f}  {result['label']}")
    print(f"（判定の線: {result['threshold']:.4f}。AIらしさがこれより大きければ ai）")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .data.repository import MetadataRepository
    from .eval.evaluator import Evaluator

    config = Config.load(args.config)

    if args.mode == "ablation":
        from .eval.ablation import run_ablation

        results = run_ablation(
            config,
            num_epochs=args.epochs,
            max_steps_per_epoch=args.max_steps,
            only_labels=args.variants,
        )
        print()
        print(f"{'variant':<16}{'run_id':>8}{'mAcc':>10}  条件")
        for result in results:
            accuracy = f"{result.mean_accuracy:.4f}" if result.mean_accuracy is not None else "N/A"
            params = ", ".join(f"{k}={v}" for k, v in result.parameters.items())
            print(f"{result.label:<16}{result.run_id:>8}{accuracy:>10}  {params}")
        return 0

    with MetadataRepository(config.get("paths.db_path")) as repo:
        run_id = _resolve_run(repo, args.run_id, args.experiment_id)
        checkpoint = args.checkpoint or _resolve_classifier_checkpoint(repo, run_id)

    with Evaluator(config, run_id, checkpoint) as evaluator:
        if args.mode == "cross_generator":
            results = evaluator.run_cross_generator(split=args.split)
        elif args.mode == "ai_jpeg":
            results = evaluator.run_ai_jpeg(args.ai_jpeg_qualities or [90, 75], split=args.split)
        else:
            results = evaluator.run_robustness(split=args.split)
        threshold = evaluator.threshold
        ai_jpeg_thresholds = dict(evaluator.ai_jpeg_thresholds)

    print()
    print(f"run_id={run_id} 閾値τ={threshold:.4f}")
    for quality, value in sorted(ai_jpeg_thresholds.items(), reverse=True):
        print(f"  AI画像も JPEG 品質 {quality} にした val で決め直した閾値τ={value:.4f}")
    print(f"{'生成器':<14}{'劣化':>12}{'強度':>8}{'Accuracy':>10}{'AUC':>10}{'AP':>10}{'枚数':>8}")
    for result in results:
        metrics = result.metrics
        print(
            f"{result.generator:<14}{result.perturbation_type:>12}"
            f"{'' if result.perturbation_level is None else result.perturbation_level:>8}"
            f"{metrics['accuracy']:>10.4f}"
            f"{(metrics['auc'] if metrics['auc'] is not None else float('nan')):>10.4f}"
            f"{(metrics['ap'] if metrics['ap'] is not None else float('nan')):>10.4f}"
            f"{metrics['num_samples']:>8}"
        )
    if results and args.mode == "ai_jpeg":
        # 閾値は圧縮しない val で決めているので、見るのは AUC
        for level in sorted({r.perturbation_level for r in results}, reverse=True):
            aucs = [r.metrics["auc"] for r in results if r.perturbation_level == level and r.metrics["auc"] is not None]
            if aucs:
                print(f"\n平均AUC（AI画像だけ JPEG 品質 {level:g}）: {sum(aucs) / len(aucs):.4f}")
            if int(level) in ai_jpeg_thresholds:
                # 線を同じ条件の val で決め直したときだけ、正解率にも意味がある
                accs = [r.metrics["accuracy"] for r in results if r.perturbation_level == level]
                print(f"平均Accuracy（同じ条件で閾値を決め直し）: {sum(accs) / len(accs):.4f}")
    elif results:
        mean_accuracy = sum(r.metrics["accuracy"] for r in results) / len(results)
        print(f"\n平均Accuracy: {mean_accuracy:.4f}")
    return 0


def cmd_jpeg_stats(args: argparse.Namespace) -> int:
    from .eval.jpeg_stats import JpegStats, write_report

    config = Config.load(args.config)
    with JpegStats(config, max_images=args.max_images, pil_quality=args.pil_quality) as stats:
        results = stats.run()
    out_dir = args.out or Path(config.get("paths.report_root", "reports")) / "jpeg_stats"
    path = write_report(results, out_dir)

    def fmt(value) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    print()
    print(f"{'まとまり':<34}{'枚数':>6}{'JPEG':>7}{'4:4:4':>7}{'4:2:2':>7}{'4:2:0':>7}{'白黒':>7}{'品質96':>8}{'標準表':>8}{'中央値':>7}  よく出る品質")
    for row in results:
        median = "N/A" if row["median_quality"] is None else f"{row['median_quality']:.0f}"
        print(
            f"{row['group']:<34}{row['num_images']:>6}{fmt(row['jpeg_share']):>7}{fmt(row['share_444']):>7}"
            f"{fmt(row['share_422']):>7}{fmt(row['share_420']):>7}{fmt(row['share_gray']):>7}{fmt(row['share_q96']):>8}"
            f"{fmt(row['share_standard']):>8}{median:>7}  {row['top_qualities']}"
        )
    print(f"\nレポートを出力しました: {path}")
    return 0


def cmd_subsampling_scores(args: argparse.Namespace) -> int:
    from .eval.subsampling_scores import SubsamplingScores, write_report

    config = Config.load(args.config)
    with SubsamplingScores(
        config, args.checkpoint, max_images=args.max_images, ai_jpeg_quality=args.ai_jpeg_quality, split=args.split
    ) as scores:
        result = scores.run()
        checkpoint = scores.checkpoint
    out_dir = args.out or Path(config.get("paths.report_root", "reports")) / f"subsampling_{Path(checkpoint).parent.name}"
    path = write_report(result, out_dir, checkpoint, args.ai_jpeg_quality)

    print()
    print(f"モデル={checkpoint}")
    print(f"{'まとまり':<22}{'枚数':>6}{'平均':>10}{'中央値':>10}")
    for g in result["groups"]:
        print(f"{g['group']:<22}{g['num_images']:>6}{g['mean']:>10.4f}{g['median']:>10.4f}")
    print("AUC（後ろを当てる向き。0.5 より下なら後ろのほうが予測しやすい）")
    for p in result["pairs"]:
        auc = "N/A" if p["auc"] is None else f"{p['auc']:.4f}"
        print(f"  {p['first']} → {p['second']}: {auc}")
    print(f"\nレポートを出力しました: {path}")
    return 0


def cmd_threshold_check(args: argparse.Namespace) -> int:
    from .data.repository import MetadataRepository
    from .eval.evaluator import Evaluator, jpeg_transform
    from .eval.threshold_check import run_threshold_check, with_mean_row, write_report

    config = Config.load(args.config)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        run_id = _resolve_run(repo, args.run_id, args.experiment_id)
        checkpoint = args.checkpoint or _resolve_classifier_checkpoint(repo, run_id)

    if args.ai_jpeg_quality is None:
        ai_transform, condition, tag = None, "AI画像は元のまま（PNG）", "none"
    else:
        subsampling = args.subsampling or "4:2:0"
        ai_transform = jpeg_transform(args.ai_jpeg_quality, args.subsampling)
        condition = f"AI画像だけ JPEG 品質 {args.ai_jpeg_quality}・{subsampling}"
        tag = f"aijpeg{args.ai_jpeg_quality}_s{subsampling.replace(':', '')}"
    rule = ""
    if args.real_false_alarm is not None:
        rule = f"本物の val だけで、本物を AI と間違える割合が {args.real_false_alarm:g}% になる位置"
        tag += f"_fa{args.real_false_alarm:g}"
    with Evaluator(config, run_id, checkpoint) as evaluator:
        results = run_threshold_check(evaluator, ai_transform, split=args.split, real_false_alarm=args.real_false_alarm)

    out_dir = args.out or Path(config.get("paths.report_root", "reports")) / f"threshold_check_run{run_id}_{tag}"
    path = write_report(results, out_dir, run_id, condition, rule)

    def fmt(value) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    print()
    print(f"run_id={run_id} 条件: {condition}" + (f" 線: {rule}" if rule else ""))
    print(f"{'生成器':<10}{'正解率':>9}{'本物':>9}{'AI画像':>9}{'上限':>9}{'AUC':>9}{'枚数':>8}")
    for row in with_mean_row(results):
        print(
            f"{row['generator']:<10}{fmt(row['accuracy']):>9}{fmt(row['real_accuracy']):>9}{fmt(row['ai_accuracy']):>9}"
            f"{fmt(row['best_accuracy']):>9}{fmt(row['auc']):>9}{row['num_samples']:>8}"
        )
    print(f"\nレポートを出力しました: {path}")
    return 0


def cmd_compare_models(args: argparse.Namespace) -> int:
    from .eval.model_similarity import ModelSimilarity, with_mean_row, write_report

    config = Config.load(args.config)
    with ModelSimilarity(
        config, args.photo_checkpoint, args.ai_checkpoint,
        ai_jpeg_quality=args.ai_jpeg_quality, max_images=args.max_images,
    ) as similarity:
        results = similarity.run(split=args.split)
        photo, ai, quality = similarity.photo_checkpoint, similarity.ai_checkpoint, similarity.ai_jpeg_quality

    suffix = "" if quality is None else f"_aijpeg{quality}"
    ai_name = "random" if ai == "random" else Path(ai).parent.name
    out_dir = args.out or (
        Path(config.get("paths.report_root", "reports")) / f"compare_{Path(photo).parent.name}_{ai_name}{suffix}"
    )
    path = write_report(results, out_dir, photo, ai, quality)

    def fmt(value) -> str:
        return "N/A" if value is None or value != value else f"{value:.4f}"

    print()
    print(f"pθ={photo}")
    print(f"qφ={ai}")
    print(
        f"{'生成器':<10}{'CKA':>8}{'相関本物':>10}{'相関AI':>10}{'相関全部':>10}{'AUC pθ':>9}{'AUC qφ':>9}{'AUC差':>9}"
        f"{'模様pθ':>9}{'模様qφ':>9}{'AUC模様':>9}{'枚数':>7}"
    )
    for row in with_mean_row(results):
        print(
            f"{row['generator']:<10}{fmt(row['cka']):>8}{fmt(row['spearman_real']):>10}{fmt(row['spearman_ai']):>10}"
            f"{fmt(row['spearman_all']):>10}{fmt(row['auc_photo']):>9}{fmt(row['auc_ai']):>9}{fmt(row['auc_diff']):>9}"
            f"{fmt(row['spearman_texture_photo']):>9}{fmt(row['spearman_texture_ai']):>9}{fmt(row['auc_texture']):>9}"
            f"{row['num_samples']:>7}"
        )
    print(f"\nレポートを出力しました: {path}")
    return 0


def cmd_evaluate_one_class(args: argparse.Namespace) -> int:
    from .eval.one_class import OneClassEvaluator, write_report, with_mean_row

    config = Config.load(args.config)
    with OneClassEvaluator(config, args.photo_checkpoint, ai_jpeg_quality=args.ai_jpeg_quality) as evaluator:
        results = evaluator.run(split=args.split)
        checkpoint = evaluator.checkpoint
        ai_jpeg_quality = evaluator.ai_jpeg_quality

    suffix = "" if ai_jpeg_quality is None else f"_aijpeg{ai_jpeg_quality}"
    out_dir = args.out or (
        Path(config.get("paths.report_root", "reports")) / f"one_class_{Path(checkpoint).parent.name}{suffix}"
    )
    path = write_report(results, out_dir, checkpoint, ai_jpeg_quality)

    print()
    print(f"pθ={checkpoint}")
    print(f"{'生成器':<14}{'Accuracy':>10}{'論文':>8}{'AUC':>10}{'実写正解':>10}{'AI正解':>10}{'枚数':>8}")
    for row in with_mean_row(results):
        paper = "" if row["paper_accuracy"] is None else f"{row['paper_accuracy'] / 100:.4f}"
        print(
            f"{row['generator']:<14}{row['accuracy']:>10.4f}{paper:>8}"
            f"{(row['auc'] if row['auc'] is not None else float('nan')):>10.4f}"
            f"{(row['real_accuracy'] if row['real_accuracy'] is not None else float('nan')):>10.4f}"
            f"{(row['ai_accuracy'] if row['ai_accuracy'] is not None else float('nan')):>10.4f}"
            f"{row['num_samples']:>8}"
        )
    print(f"\nレポートを出力しました: {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .data.repository import MetadataRepository
    from .report.report_builder import build_report

    config = Config.load(args.config)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        run_id = _resolve_run(repo, args.run_id, args.experiment_id)

    out_dir = args.out or config.get("paths.report_root", "reports")
    path = build_report(config, run_id, out_dir)
    print(f"レポートを出力しました: {path}")
    return 0


def _setup_logging(args: argparse.Namespace) -> Path | None:
    """標準出力に加えて、configの `paths.log_root` 配下にも実行ログを残す。

    取り込みや学習は数時間走ることがあるため、ターミナルを閉じた後でも
    経過を追えるようにしておく。

    ファイル名にはプロセスIDを含める。pθとqφを並列に流すと2つのプロセスが同じ秒に
    起動することがあり、秒までのタイムスタンプだけでは同じファイルへ両方が
    書き込んでしまうため。
    """
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 同一プロセスで複数回呼ばれてもハンドラが増えないようにする
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_root = "logs"
    config_path = getattr(args, "config", None)
    if config_path:
        try:
            log_root = Config.load(config_path).get("paths.log_root", "logs")
        except Exception:  # configが読めない場合はログだけ既定の場所に出す
            pass

    try:
        directory = resolve(log_root)
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"{args.command}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        return log_path
    except OSError:
        return None


def _configure_multiprocessing() -> None:
    """ワーカー間のテンソル共有をファイルシステム方式にする。

    既定の `file_descriptor` 方式は共有テンソル1つにつきfdを1つ使う。評価では
    条件ごとにDataLoaderを作り直す（7生成器 × 9条件 = 63回）ため、fdが積み上がって
    `OSError: [Errno 24] Too many open files` で落ちる。macOSは既定の上限が
    小さいので特に起きやすい。
    """
    import torch.multiprocessing as mp

    try:
        mp.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError):  # その環境で使えなければ既定のまま進める
        logging.getLogger(__name__).debug("共有方式を変更できませんでした", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = _setup_logging(args)
    _configure_multiprocessing()
    if log_path is not None:
        logging.getLogger(__name__).info("ログファイル: %s", log_path)

    handlers = {
        "init-db": cmd_init_db,
        "ingest": cmd_ingest,
        "status": cmd_status,
        "train-stage1": cmd_train_stage1,
        "train-stage2": cmd_train_stage2,
        "evaluate": cmd_evaluate,
        "evaluate-one-class": cmd_evaluate_one_class,
        "compare-models": cmd_compare_models,
        "jpeg-stats": cmd_jpeg_stats,
        "subsampling-scores": cmd_subsampling_scores,
        "threshold-check": cmd_threshold_check,
        "report": cmd_report,
        "export-bundle": cmd_export_bundle,
        "predict": cmd_predict,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse が先に弾く
        raise NotImplementedError(f"コマンド '{args.command}' は未実装です。")

    try:
        return handler(args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("中断されました")
        return 130
    except Exception:
        # nohup + `> /dev/null 2>&1` で流すと標準エラーが捨てられ、失敗した理由が
        # どこにも残らない。ログファイルにはトレースバックを必ず書いておく。
        logging.getLogger(__name__).exception("コマンド '%s' が異常終了しました", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
