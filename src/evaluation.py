"""Cross-day (Day 1 -> Day 2) evaluation engine for OpenBMI SSVEP & P300 speller."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import (
    SessionEpochData,
    load_openbmi_p300_subject,
    load_openbmi_ssvep_subject,
    slice_ssvep_window,
)
from src.p300_models import (
    EEGNetP300Decoder,
    XdawnRiemannianDecoder,
    evaluate_p300_flash_metrics,
    evaluate_speller_character_accuracy,
    extract_p300_erp_stats,
)
from src.ssvep_models import (
    CCADecoder,
    ChebyshevFilterBank,
    FBCCADecoder,
    eTRCADecoder,
    extract_ssvep_harmonic_stats,
)
from src.utils import compute_itr, save_json

logger = logging.getLogger("openbmi_bci")


def evaluate_ssvep_subject(
    data: SessionEpochData, cfg: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run Day 1 -> Day 2 SSVEP cross-day evaluation across window lengths."""
    scfg = cfg["ssvep"]
    fb_cfg = scfg["filter_bank"]
    freqs = [float(f) for f in scfg["frequencies"]]
    n_classes = len(freqs)
    gaze_time = float(scfg["gaze_shift_time"])
    visual_delay = float(scfg["visual_delay"])

    filter_bank = ChebyshevFilterBank(
        sfreq=data.sfreq,
        subband_lows=fb_cfg["subband_lows"],
        subband_high=fb_cfg["subband_high"],
        order=fb_cfg["cheby_order"],
        rp=fb_cfg["cheby_rp"],
        weight_a=fb_cfg["weight_a"],
        weight_b=fb_cfg["weight_b"],
    )

    cca = CCADecoder(frequencies=freqs, sfreq=data.sfreq, n_harmonics=scfg["n_harmonics"])
    fbcca = FBCCADecoder(
        frequencies=freqs,
        sfreq=data.sfreq,
        filter_bank=filter_bank,
        n_harmonics=scfg["n_harmonics"],
    )

    records: List[Dict[str, Any]] = []
    for win_sec in scfg["window_lengths"]:
        win_sec = float(win_sec)
        X_tr = slice_ssvep_window(data.X_day1, data.sfreq, data.tmin, win_sec, visual_delay)
        X_te = slice_ssvep_window(data.X_day2, data.sfreq, data.tmin, win_sec, visual_delay)

        etrca = eTRCADecoder(
            n_classes=n_classes,
            filter_bank=filter_bank,
            regularization=scfg["etrca"]["regularization"],
        )
        etrca.fit(X_tr, data.y_day1)

        for model_name, model in [("CCA", cca), ("FBCCA", fbcca), ("eTRCA", etrca)]:
            y_pred = model.predict(X_te)
            acc = float(np.mean(y_pred == data.y_day2))
            itr_with_gaze = compute_itr(n_classes, acc, win_sec + gaze_time)
            itr_no_gaze = compute_itr(n_classes, acc, win_sec)
            records.append(
                {
                    "subject": data.subject,
                    "model": model_name,
                    "window_sec": win_sec,
                    "accuracy_pct": round(acc * 100.0, 2),
                    "itr_bits_per_min": round(itr_with_gaze, 2),
                    "itr_no_gaze_bits_per_min": round(itr_no_gaze, 2),
                    "n_train_day1": int(len(data.y_day1)),
                    "n_test_day2": int(len(data.y_day2)),
                }
            )

    # Extract numerical harmonic SNR stats on 2.0s Day 2 window
    X_spec = slice_ssvep_window(data.X_day2, data.sfreq, data.tmin, 2.0, visual_delay)
    harmonic_stats = extract_ssvep_harmonic_stats(X_spec, data.y_day2, freqs, data.sfreq, data.ch_names)
    for row in harmonic_stats:
        row["subject"] = data.subject

    return records, harmonic_stats


def evaluate_p300_subject(data: SessionEpochData, cfg: Dict[str, Any]):
    """Run Day 1 -> Day 2 P300 cross-day evaluation for xDAWN+Riemannian and EEGNet."""
    pcfg = cfg["p300"]
    sp_cfg = pcfg["speller"]
    xd_cfg = pcfg["xdawn"]
    eg_cfg = pcfg["eegnet"]
    seed = int(cfg["project"]["seed"])

    models = {
        "xDAWN_Riemannian": XdawnRiemannianDecoder(
            n_filters=xd_cfg["n_filters"],
            cov_estimator=xd_cfg["cov_estimator"],
            xdawn_estimator=xd_cfg["xdawn_estimator"],
            classifier_c=xd_cfg["classifier_c"],
        ),
        "EEGNet_WeightedCE": EEGNetP300Decoder(
            f1=eg_cfg["f1"],
            d=eg_cfg["d"],
            f2=eg_cfg["f2"],
            kernel_length=eg_cfg["kernel_length"],
            dropout=eg_cfg["dropout"],
            epochs=eg_cfg["epochs"],
            batch_size=eg_cfg["batch_size"],
            lr=eg_cfg["lr"],
            weight_decay=eg_cfg["weight_decay"],
            val_ratio=eg_cfg["val_ratio"],
            patience=eg_cfg["patience"],
            seed=seed,
        ),
    }

    records: List[Dict[str, Any]] = []
    for model_name, model in models.items():
        model.fit(data.X_day1, data.y_day1)
        y_prob = model.predict_proba(data.X_day2)
        flash_metrics = evaluate_p300_flash_metrics(data.y_day2, y_prob)
        speller_accs = evaluate_speller_character_accuracy(
            y_true=data.y_day2,
            y_prob=y_prob,
            n_rows=sp_cfg["n_rows"],
            n_cols=sp_cfg["n_cols"],
            n_repetitions=sp_cfg["n_repetitions"],
        )

        row: Dict[str, Any] = {
            "subject": data.subject,
            "model": model_name,
            "roc_auc": round(flash_metrics["roc_auc"], 4),
            "balanced_acc_pct": round(flash_metrics["balanced_acc"] * 100.0, 2),
            "f1_target": round(flash_metrics["f1_target"], 4),
            "n_train_day1": int(len(data.y_day1)),
            "n_test_day2": int(len(data.y_day2)),
            "day2_target_ratio": round(float(np.mean(data.y_day2 == 1)), 4),
        }
        for rep, acc in speller_accs.items():
            row[f"speller_rep{rep}_acc_pct"] = round(acc * 100.0, 2)
        records.append(row)

    erp_stats = extract_p300_erp_stats(data.X_day2, data.y_day2, data.sfreq, data.tmin, data.ch_names)
    for row in erp_stats:
        row["subject"] = data.subject

    return records, erp_stats


def _write_benchmark_report_md(
    ssvep_summary: pd.DataFrame,
    p300_summary: pd.DataFrame,
    n_subjects: int,
    out_path: Path,
) -> None:
    """Write a concise Markdown benchmark report summarizing cross-day metrics."""
    lines: List[str] = [
        "# OpenBMI (Lee2019) 双范式跨天评估基准报告",
        "",
        f"- 评估被试数：`{n_subjects}` 名（Day 1 训练 -> Day 2 测试，严格跨天无泄露）",
        "",
    ]

    if not ssvep_summary.empty:
        lines.extend(["## 1. SSVEP 跨天评估汇总（不同窗长 Accuracy 与 ITR）", ""])
        lines.append(ssvep_summary.to_markdown(index=False))
        lines.append("")
        sub_1s = ssvep_summary[ssvep_summary["window_sec"] == 1.0].sort_values("accuracy_pct_mean", ascending=False)
        if not sub_1s.empty:
            best = sub_1s.iloc[0]
            lines.append(
                f"> **SSVEP 1.0s 窗最优结果 ({best['model']})**："
                f"准确率 **{best['accuracy_pct_mean']:.1f}%**，"
                f"ITR **{best['itr_bits_per_min_mean']:.1f} bits/min**（含 0.5s 视觉转移时间；"
                f"不含转移时间 **{best['itr_no_gaze_bits_per_min_mean']:.1f} bits/min**）。"
            )
            lines.append("")

    if not p300_summary.empty:
        lines.extend(["## 2. P300 跨天目标检测与多轮拼写汇总", ""])
        lines.append(p300_summary.to_markdown(index=False))
        lines.append("")
        best_p = p300_summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
        lines.append(
            f"> **P300 跨天最优结果 ({best_p['model']})**："
            f"单试次跨天 AUC **{best_p['roc_auc_mean']:.3f}**，"
            f"5 轮叠加字符拼写准确率 **{best_p['speller_rep5_acc_pct_mean']:.1f}%**。"
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_full_benchmark(
    subjects: Sequence[int],
    cfg: Dict[str, Any],
    run_ssvep: bool = True,
    run_p300: bool = True,
) -> Dict[str, Any]:
    """Run complete cross-day benchmark across all requested subjects and save all outputs."""
    results_dir = Path(cfg["project"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    ssvep_rows: List[Dict[str, Any]] = []
    ssvep_harm_rows: List[Dict[str, Any]] = []
    p300_rows: List[Dict[str, Any]] = []
    p300_erp_rows: List[Dict[str, Any]] = []

    if run_ssvep:
        logger.info("=== Starting SSVEP Cross-Day Evaluation (%d subjects) ===", len(subjects))
        for subj in subjects:
            logger.info("[SSVEP] Loading Subject %02d (Day 1 & Day 2)...", subj)
            try:
                data = load_openbmi_ssvep_subject(subj, cfg)
                recs, harm = evaluate_ssvep_subject(data, cfg)
                ssvep_rows.extend(recs)
                ssvep_harm_rows.extend(harm)
                # Checkpoint save after every subject so Colab disconnect never loses progress
                pd.DataFrame(ssvep_rows).to_csv(results_dir / "ssvep_per_subject.csv", index=False)
                logger.info("[SSVEP] Subject %02d completed.", subj)
            except Exception as exc:
                logger.error("[SSVEP] Subject %02d failed: %s", subj, exc)

    if run_p300:
        logger.info("=== Starting P300 Cross-Day Evaluation (%d subjects) ===", len(subjects))
        for subj in subjects:
            logger.info("[P300] Loading Subject %02d (Day 1 & Day 2)...", subj)
            try:
                data = load_openbmi_p300_subject(subj, cfg)
                recs, erp = evaluate_p300_subject(data, cfg)
                p300_rows.extend(recs)
                p300_erp_rows.extend(erp)
                pd.DataFrame(p300_rows).to_csv(results_dir / "p300_per_subject.csv", index=False)
                logger.info("[P300] Subject %02d completed.", subj)
            except Exception as exc:
                logger.error("[P300] Subject %02d failed: %s", subj, exc)

    ssvep_summary = pd.DataFrame()
    if ssvep_rows:
        df_ssvep = pd.DataFrame(ssvep_rows)
        df_ssvep.to_csv(results_dir / "ssvep_per_subject.csv", index=False)
        ssvep_summary = (
            df_ssvep.groupby(["model", "window_sec"], as_index=False)
            .agg(
                accuracy_pct_mean=("accuracy_pct", "mean"),
                accuracy_pct_std=("accuracy_pct", "std"),
                itr_bits_per_min_mean=("itr_bits_per_min", "mean"),
                itr_no_gaze_bits_per_min_mean=("itr_no_gaze_bits_per_min", "mean"),
            )
            .round(2)
        )
        ssvep_summary.to_csv(results_dir / "ssvep_summary.csv", index=False)
        if ssvep_harm_rows:
            pd.DataFrame(ssvep_harm_rows).to_csv(results_dir / "ssvep_harmonic_snr.csv", index=False)

    p300_summary = pd.DataFrame()
    if p300_rows:
        df_p300 = pd.DataFrame(p300_rows)
        df_p300.to_csv(results_dir / "p300_per_subject.csv", index=False)
        agg_dict = {
            "roc_auc_mean": ("roc_auc", "mean"),
            "roc_auc_std": ("roc_auc", "std"),
            "balanced_acc_pct_mean": ("balanced_acc_pct", "mean"),
            "f1_target_mean": ("f1_target", "mean"),
        }
        for r in range(1, int(cfg["p300"]["speller"]["n_repetitions"]) + 1):
            col = f"speller_rep{r}_acc_pct"
            if col in df_p300.columns:
                agg_dict[f"{col}_mean"] = (col, "mean")
        p300_summary = df_p300.groupby("model", as_index=False).agg(**agg_dict).round(3)
        p300_summary.to_csv(results_dir / "p300_summary.csv", index=False)
        if p300_erp_rows:
            pd.DataFrame(p300_erp_rows).to_csv(results_dir / "p300_erp_stats.csv", index=False)

    summary_payload = {
        "n_subjects_requested": len(subjects),
        "ssvep_summary": ssvep_summary.to_dict(orient="records"),
        "p300_summary": p300_summary.to_dict(orient="records"),
    }
    save_json(summary_payload, results_dir / "experiment_summary.json")
    _write_benchmark_report_md(ssvep_summary, p300_summary, len(subjects), results_dir / "benchmark_report.md")
    logger.info("All results saved to %s", results_dir)
    return summary_payload
