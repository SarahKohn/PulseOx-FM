"""Figures-only runner for external dataset evaluations."""

from __future__ import annotations

import os
import re

import pandas as pd

import evaluate_external_datasets as analysis
from refactored_io import EMBEDDINGS_ROOT, figures_dir, redirect_figure_writes, results_dir, write_numerical_results_markdown

AGE_TRUE_MIN = 0
AGE_TRUE_MAX = 100


def _result_records(path):
    if not os.path.isfile(path):
        return []
    return pd.read_csv(path).to_dict("records")


def _filter_age_true_range(df: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
    if df.empty or "age_true" not in df.columns:
        return df.copy()
    age_true = pd.to_numeric(df["age_true"], errors="coerce")
    mask = age_true.between(min_age, max_age, inclusive="both")
    filtered = df.loc[mask].copy()
    if "age" not in filtered.columns:
        filtered["age"] = pd.to_numeric(filtered["age_true"], errors="coerce")
    return filtered


def _feature_family_name(row: dict) -> str:
    family = row.get("feature_family") or row.get("feature_set") or row.get("feature_kind") or ""
    family = str(family)
    return {
        "mae_embeddings": "mae_pretrained",
        "mae_random_init": "mae_random_init",
        "pyppg_mean_only": "pyppg_mean_only",
    }.get(family, family)


def _recompute_age_results(filtered_df: pd.DataFrame, results: list[dict]) -> list[dict]:
    if filtered_df.empty or not results or "age_true" not in filtered_df.columns:
        return []

    y_true_all = pd.to_numeric(filtered_df["age_true"], errors="coerce")
    recomputed = []
    for row in results:
        feature_family = _feature_family_name(row)
        prediction_mode = str(row.get("prediction_mode", ""))
        pred_col = f"age_pred_{feature_family}_{prediction_mode}"
        if pred_col not in filtered_df.columns:
            continue

        y_pred_all = pd.to_numeric(filtered_df[pred_col], errors="coerce")
        valid = y_true_all.notna() & y_pred_all.notna()
        if int(valid.sum()) < 2:
            continue

        metrics = analysis._compute_regression_metrics(
            y_true_all.loc[valid].to_numpy(),
            y_pred_all.loc[valid].to_numpy(),
            "age",
            min_samples=2,
        )
        if not metrics:
            continue

        updated = dict(row)
        updated.update(
            {
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "r": metrics["r"],
                "N": metrics["N"],
            }
        )
        recomputed.append(updated)
    return recomputed


def _append_pretrained_pyppg_ensemble_results(filtered_df: pd.DataFrame, results: list[dict]) -> list[dict]:
    if filtered_df.empty or "age_true" not in filtered_df.columns:
        return list(results)

    base_results = list(results)
    y_true_all = pd.to_numeric(filtered_df["age_true"], errors="coerce")
    seen_modes = {str(r.get("prediction_mode", "")) for r in base_results if r.get("prediction_mode")}
    for mode in sorted(seen_modes):
        seed_pattern = re.compile(rf"^age_pred_mae_pretrained_{re.escape(mode)}_seed_(\d+)$")
        seed_ids = []
        for col in filtered_df.columns:
            m = seed_pattern.match(str(col))
            if m is None:
                continue
            seed = int(m.group(1))
            pyppg_seed_col = f"age_pred_pyppg_mean_only_{mode}_seed_{seed}"
            if pyppg_seed_col in filtered_df.columns:
                seed_ids.append(seed)
        seed_ids = sorted(set(seed_ids))

        metrics = None
        repeat_metrics = []
        if seed_ids:
            seed_ensemble_predictions = []
            for seed in seed_ids:
                mae_seed_col = f"age_pred_mae_pretrained_{mode}_seed_{seed}"
                pyppg_seed_col = f"age_pred_pyppg_mean_only_{mode}_seed_{seed}"
                y_pred_mae_seed = pd.to_numeric(filtered_df[mae_seed_col], errors="coerce")
                y_pred_pyppg_seed = pd.to_numeric(filtered_df[pyppg_seed_col], errors="coerce")
                y_pred_ensemble_seed = (y_pred_mae_seed + y_pred_pyppg_seed) / 2.0
                seed_ensemble_predictions.append(y_pred_ensemble_seed)

                valid_seed = y_true_all.notna() & y_pred_ensemble_seed.notna()
                if int(valid_seed.sum()) < 2:
                    continue
                seed_metrics = analysis._compute_regression_metrics(
                    y_true_all.loc[valid_seed].to_numpy(),
                    y_pred_ensemble_seed.loc[valid_seed].to_numpy(),
                    f"ensemble_repeat_seed_{seed}",
                    min_samples=2,
                )
                if seed_metrics:
                    seed_metrics["cv_repeat_seed"] = int(seed)
                    repeat_metrics.append(seed_metrics)

            if not seed_ensemble_predictions:
                continue

            y_pred_ensemble = pd.concat(seed_ensemble_predictions, axis=1).mean(axis=1)
            valid = y_true_all.notna() & y_pred_ensemble.notna()
            if int(valid.sum()) < 2:
                continue
            metrics = analysis._compute_regression_metrics(
                y_true_all.loc[valid].to_numpy(),
                y_pred_ensemble.loc[valid].to_numpy(),
                "age",
                min_samples=2,
            )
            if metrics and repeat_metrics:
                r2s = [m["R2"] for m in repeat_metrics]
                maes = [m["MAE"] for m in repeat_metrics]
                rs = [m["r"] for m in repeat_metrics]
                metrics["R2_repeat_mean"] = float(sum(r2s) / len(r2s))
                metrics["R2_repeat_sd"] = float(pd.Series(r2s).std(ddof=0))
                metrics["MAE_repeat_mean"] = float(sum(maes) / len(maes))
                metrics["MAE_repeat_sd"] = float(pd.Series(maes).std(ddof=0))
                metrics["r_repeat_mean"] = float(sum(rs) / len(rs))
                metrics["r_repeat_sd"] = float(pd.Series(rs).std(ddof=0))
                metrics["cv_n_repeats"] = int(len(repeat_metrics))
                metrics["cv_repeat_seeds"] = [int(s) for s in seed_ids]
        else:
            pred_mae_col = f"age_pred_mae_pretrained_{mode}"
            pred_pyppg_col = f"age_pred_pyppg_mean_only_{mode}"
            if pred_mae_col not in filtered_df.columns or pred_pyppg_col not in filtered_df.columns:
                continue
            y_pred_mae = pd.to_numeric(filtered_df[pred_mae_col], errors="coerce")
            y_pred_pyppg = pd.to_numeric(filtered_df[pred_pyppg_col], errors="coerce")
            y_pred_ensemble = (y_pred_mae + y_pred_pyppg) / 2.0
            valid = y_true_all.notna() & y_pred_ensemble.notna()
            if int(valid.sum()) < 2:
                continue
            metrics = analysis._compute_regression_metrics(
                y_true_all.loc[valid].to_numpy(),
                y_pred_ensemble.loc[valid].to_numpy(),
                "age",
                min_samples=2,
            )

        if not metrics:
            continue

        template = next((r for r in base_results if str(r.get("prediction_mode", "")) == mode), {})
        ensemble_row = dict(template)
        ensemble_row.update(
            {
                "feature_family": "pretrained+PYPPG ensemble",
                "prediction_mode": mode,
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "r": metrics["r"],
                "N": metrics["N"],
            }
        )
        for key in ("R2_repeat_mean", "R2_repeat_sd", "r_repeat_mean", "r_repeat_sd", "MAE_repeat_mean", "MAE_repeat_sd"):
            if key in metrics:
                ensemble_row[key] = metrics[key]
        for key in ("cv_n_repeats", "cv_repeat_seeds"):
            if key in metrics:
                ensemble_row[key] = metrics[key]
        for key in ("R2_sd", "r_sd", "MAE_sd"):
            ensemble_row.pop(key, None)
        base_results.append(ensemble_row)

    return base_results


def main() -> None:
    result_dir = results_dir("external_datasets")
    fig_dir = figures_dir("external_datasets")
    analysis.OUTPUT_DIR = str(result_dir)
    analysis.EMBEDDINGS_CACHE_DIR = str(EMBEDDINGS_ROOT)

    detailed_path = result_dir / "external_dataset_case_predictions_with_embeddings.csv"
    summary_path = result_dir / "external_dataset_predictions.csv"
    if not detailed_path.is_file() and not summary_path.is_file():
        raise FileNotFoundError(f"No cached external dataset results found in {result_dir}")

    detailed = (
        analysis.read_external_case_predictions_csv(str(detailed_path))
        if detailed_path.is_file()
        else pd.DataFrame()
    )
    results = _result_records(summary_path)
    vital_df = detailed[detailed["dataset"] == "VitalDB"].copy() if "dataset" in detailed else pd.DataFrame()
    gold_df = detailed[detailed["dataset"] == "Gold_test"].copy() if "dataset" in detailed else pd.DataFrame()
    vital_df = _filter_age_true_range(vital_df, AGE_TRUE_MIN, AGE_TRUE_MAX)
    gold_df = _filter_age_true_range(gold_df, AGE_TRUE_MIN, AGE_TRUE_MAX)
    vital_results = [r for r in results if r.get("dataset") == "VitalDB"]
    gold_results = [r for r in results if r.get("dataset") == "Gold_test"]
    vital_results = _recompute_age_results(vital_df, vital_results)
    gold_results = _recompute_age_results(gold_df, gold_results)
    vital_results = _append_pretrained_pyppg_ensemble_results(vital_df, vital_results)
    gold_results = _append_pretrained_pyppg_ensemble_results(gold_df, gold_results)
    write_numerical_results_markdown(
        fig_dir / "external_datasets_numerical_results.md",
        "External Dataset Numerical Results",
        [
            summary_path,
            detailed_path,
            result_dir / "vitaldb_age_crossval_metrics_cross_validation.csv",
            result_dir / "gold_test_age_crossval_metrics_cross_validation.csv",
            result_dir / "external_datasets_segment_lengths.json",
        ],
    )

    with redirect_figure_writes(fig_dir):
        if not vital_df.empty and "age_pred_mae_pretrained_cross_validation" in vital_df:
            analysis.plot_mae_age_scatter_pdf(
                vital_df["age_true"],
                vital_df["age_pred_mae_pretrained_cross_validation"],
                title="VitalDB age prediction",
                color=analysis.SCATTER_COLOR_VITALDB_MAE,
                out_path=str(result_dir / "vitaldb_mae_age_scatter.pdf"),
            )
        if not gold_df.empty and "age_pred_mae_pretrained_cross_validation" in gold_df:
            analysis.plot_mae_age_scatter_pdf(
                gold_df["age_true"],
                gold_df["age_pred_mae_pretrained_cross_validation"],
                title="HPP held-out age prediction",
                color=analysis.SCATTER_COLOR_GOLD_TEST_MAE,
                out_path=str(result_dir / "gold_test_mae_age_scatter.pdf"),
            )
        if vital_results or gold_results:
            analysis.plot_summary(
                vital_df,
                vital_results,
                vital_df,
                gold_emb_df=gold_df,
                gold_results=gold_results,
                embeddings_cache_path=str(analysis.EMBEDDINGS_CACHE_DIR),
                include_pearson_comparison=True,
            )


if __name__ == "__main__":
    main()

