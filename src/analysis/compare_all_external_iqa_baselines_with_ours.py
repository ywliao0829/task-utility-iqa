from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def orient_score(score, y):
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)

    valid = np.isfinite(score)
    if valid.sum() == 0:
        return None

    median = np.nanmedian(score)
    score = np.nan_to_num(score, nan=median)

    rho = spearmanr(score, y).correlation
    if np.isfinite(rho) and rho < 0:
        score = -score

    mn, mx = np.min(score), np.max(score)
    if abs(mx - mn) < 1e-12:
        return np.full_like(score, 0.5, dtype=float)

    return (score - mn) / (mx - mn)


def eval_metric(name, score, y):
    raw = np.asarray(score, dtype=float)
    pred = orient_score(raw, y)

    if pred is None:
        return {
            "metric": name,
            "n": len(y),
            "valid_score_n": int(np.isfinite(raw).sum()),
            "auroc": np.nan,
            "average_precision": np.nan,
            "spearman_r": np.nan,
            "pearson_r": np.nan,
            "brier": np.nan,
        }

    return {
        "metric": name,
        "n": len(y),
        "valid_score_n": int(np.isfinite(raw).sum()),
        "auroc": roc_auc_score(y, pred),
        "average_precision": average_precision_score(y, pred),
        "spearman_r": spearmanr(pred, y).correlation,
        "pearson_r": pearsonr(pred, y)[0],
        "brier": brier_score_loss(y, np.clip(pred, 0, 1)),
    }


def main():
    out_dir = Path("results_visdrone_300/external_baselines/summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    ours_path = Path("results_visdrone_300/metric_benchmark/qwen25vl_7b/metric_scores_per_sample.csv")
    if not ours_path.exists():
        raise FileNotFoundError(ours_path)

    ours = pd.read_csv(ours_path)
    y = ours["correct"].astype(int).values

    rows = []

    metric_cols = {
        "Ours-TCUS": "Ours_TCUS_TaskConditionedUtilityScore",
        "Cognition-Consistency": "EmbodiedIQA_CognitionConsistencyProxy",
    }

    for name, col in metric_cols.items():
        if col in ours.columns:
            rows.append(eval_metric(name, ours[col].values, y))

    baseline_files = {
        "Q-SiT": "results_visdrone_300/external_baselines/q_sit_scores.csv",
        "Q-Insight-raw": "results_visdrone_300/external_baselines/q_insight_scores.csv",
        "Q-Insight": "results_visdrone_300/external_baselines/q_insight_scores_clean.csv",
        "DeQA-Score-Mix3": "results_visdrone_300/external_baselines/deqa_scores.csv",
    }

    for name, path in baseline_files.items():
        p = Path(path)
        if not p.exists():
            print("Skip missing:", name, path)
            continue

        df = pd.read_csv(p)
        if "sample_uid" not in df.columns or "score" not in df.columns:
            print("Skip malformed:", name, path)
            continue

        merged = ours[["sample_uid", "correct"]].merge(
            df[["sample_uid", "score"]],
            on="sample_uid",
            how="inner",
        )

        print(name, "matched rows:", len(merged))

        if len(merged) == 0:
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

    out_path = out_dir / "external_baseline_vs_ours_alignment_all.csv"
    res.to_csv(out_path, index=False)

    print(res.to_string(index=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
