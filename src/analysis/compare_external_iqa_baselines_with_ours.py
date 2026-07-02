from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def orient_score(score, y):
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)

    if np.isnan(score).all():
        return None

    score = np.nan_to_num(score, nan=np.nanmedian(score))

    rho = spearmanr(score, y).correlation
    if np.isfinite(rho) and rho < 0:
        score = -score

    mn, mx = np.min(score), np.max(score)
    if abs(mx - mn) < 1e-12:
        return np.full_like(score, 0.5, dtype=float)

    return (score - mn) / (mx - mn)


def eval_metric(name, score, y):
    pred = orient_score(score, y)
    if pred is None:
        return {
            "metric": name,
            "n": len(y),
            "valid_score_n": 0,
            "auroc": np.nan,
            "average_precision": np.nan,
            "spearman_r": np.nan,
            "pearson_r": np.nan,
            "brier": np.nan,
        }

    return {
        "metric": name,
        "n": len(y),
        "valid_score_n": int(np.isfinite(np.asarray(score, dtype=float)).sum()),
        "auroc": roc_auc_score(y, pred),
        "average_precision": average_precision_score(y, pred),
        "spearman_r": spearmanr(pred, y).correlation,
        "pearson_r": pearsonr(pred, y)[0],
        "brier": brier_score_loss(y, np.clip(pred, 0, 1)),
    }


def main():
    out_dir = Path("results_visdrone_300/external_baselines/summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    ours_path = Path("results_visdrone_300/metric_benchmark/qwen25vl_7b/metric_scores_per_sample.csv")
    if not ours_path.exists():
        raise FileNotFoundError(f"Missing Ours-TCUS scores: {ours_path}")

    ours = pd.read_csv(ours_path)
    y = ours["correct"].astype(int).values

    if "Ours_TCUS_TaskConditionedUtilityScore" in ours.columns:
        rows.append(eval_metric(
            "Ours-TCUS",
            ours["Ours_TCUS_TaskConditionedUtilityScore"].values,
            y,
        ))

    if "EmbodiedIQA_CognitionConsistencyProxy" in ours.columns:
        rows.append(eval_metric(
            "Cognition-Consistency",
            ours["EmbodiedIQA_CognitionConsistencyProxy"].values,
            y,
        ))

    baseline_files = {
        "Q-Insight": "results_visdrone_300/external_baselines/q_insight_scores_clean.csv",
        "Q-Insight-raw": "results_visdrone_300/external_baselines/q_insight_scores.csv",
        "Q-SiT": "results_visdrone_300/external_baselines/q_sit_scores.csv",
        "DeQA-Score": "results_visdrone_300/external_baselines/deqa_score_scores.csv",
    }

    for name, path in baseline_files.items():
        p = Path(path)
        if not p.exists():
            print("Skip missing:", path)
            continue

        df = pd.read_csv(p)
        if "score" not in df.columns:
            print("Skip no score column:", path)
            continue

        merged = ours[["sample_uid", "correct"]].merge(
            df[["sample_uid", "score"]],
            on="sample_uid",
            how="inner",
        )

        if len(merged) == 0:
            print("Skip no matching sample_uid:", path)
            continue

        rows.append(eval_metric(
            name,
            pd.to_numeric(merged["score"], errors="coerce").values,
            merged["correct"].astype(int).values,
        ))

    res = pd.DataFrame(rows)
    res = res.sort_values(
        by=["auroc", "average_precision", "spearman_r"],
        ascending=False,
        na_position="last",
    )

    out_path = out_dir / "external_baseline_vs_ours_alignment.csv"
    res.to_csv(out_path, index=False)

    print(res.to_string(index=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
