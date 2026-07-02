from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


def safe_eval(y, s):
    y = np.asarray(y).astype(int)
    s = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy()
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or len(set(y)) < 2:
        return np.nan, np.nan, np.nan, len(y), len(set(y))
    return (
        roc_auc_score(y, s),
        average_precision_score(y, s),
        spearmanr(s, y).correlation,
        len(y),
        len(set(y)),
    )


def add_scores(df):
    # Traditional metrics
    if "brightness_quality" in df.columns:
        bq = pd.to_numeric(df["brightness_quality"], errors="coerce")
    elif "brightness" in df.columns:
        bq_raw = pd.to_numeric(df["brightness"], errors="coerce")
        bq = -abs(bq_raw - bq_raw.median())
        df["brightness_quality"] = bq
    else:
        bq = pd.Series(np.nan, index=df.index)

    # Normalize brightness quality
    bq_min, bq_max = bq.min(), bq.max()
    bq_norm = (bq - bq_min) / (bq_max - bq_min + 1e-8)

    cog = pd.to_numeric(df.get("cognition_consistency_proxy", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0)

    # Cross-family task-conditioned utility proxy
    df["tcus_proxy"] = 0.7 * cog + 0.3 * bq_norm.fillna(0)
    return df


metric_specs = [
    ("PSNR", "psnr"),
    ("SSIM", "ssim"),
    ("Sharpness", "sharpness"),
    ("Brightness-quality", "brightness_quality"),
    ("Cognition-Consistency", "cognition_consistency_proxy"),
    ("TCUS-proxy", "tcus_proxy"),
]

rows = []
for p in sorted(Path("results_api_vlm/all_strict300").glob("*.csv")):
    df = pd.read_csv(p)
    df = add_scores(df)

    provider = df["provider"].iloc[0] if "provider" in df.columns and len(df) else ""
    model = df["api_model"].iloc[0] if "api_model" in df.columns and len(df) else p.stem
    n_total = len(df)
    err_count = int((df["api_error"].fillna("") != "").sum()) if "api_error" in df.columns else 0
    nonempty = int((df["api_pred_raw"].fillna("") != "").sum()) if "api_pred_raw" in df.columns else 0
    success_rate = float(df["api_correct"].mean()) if "api_correct" in df.columns else np.nan

    if "api_correct" not in df.columns:
        continue

    for metric_name, col in metric_specs:
        if col not in df.columns:
            rows.append({
                "Provider": provider,
                "Model": model,
                "Metric": metric_name,
                "Score column": col,
                "N_total": n_total,
                "N_eval": 0,
                "Error count": err_count,
                "Nonempty pred": nonempty,
                "Task success rate": success_rate,
                "AUROC": np.nan,
                "Average Precision": np.nan,
                "Spearman": np.nan,
                "Note": "missing score column",
                "File": str(p),
            })
            continue

        auroc, ap, sp, n_eval, n_classes = safe_eval(df["api_correct"], df[col])
        rows.append({
            "Provider": provider,
            "Model": model,
            "Metric": metric_name,
            "Score column": col,
            "N_total": n_total,
            "N_eval": n_eval,
            "Error count": err_count,
            "Nonempty pred": nonempty,
            "Task success rate": success_rate,
            "AUROC": auroc,
            "Average Precision": ap,
            "Spearman": sp,
            "Note": "" if n_classes >= 2 else "only one class in api_correct",
            "File": str(p),
        })

out = pd.DataFrame(rows)
out_path = Path("results_api_vlm/all_strict300_alignment_summary.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)

print("Saved:", out_path)

print("\n==== TCUS-proxy only ====")
print(out[out["Metric"].eq("TCUS-proxy")].to_string(index=False))
