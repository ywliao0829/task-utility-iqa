import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DEGRADATIONS = ["original", "blur", "dark", "jpeg_low", "noise"]


def load_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "sample_uid": r.get("sample_uid", ""),
                "question_type_manifest": r.get("question_type", ""),
                "answer_type_manifest": r.get("answer_type", ""),
                "question_manifest": r.get("question", ""),
                "answer_manifest": r.get("answer", ""),
                "image_path_manifest": r.get("image_path", ""),
            })
    return pd.DataFrame(rows)


def first_existing(row, candidates, default=""):
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return row[c]
    return default


def normalize_columns(df):
    mapping = {
        "degradation_show": ["degradation", "degradation_vlm", "degradation_iqa"],
        "question_show": ["question", "question_vlm", "question_manifest"],
        "gold_show": ["gold_answer", "answer", "answer_vlm", "answer_manifest"],
        "pred_show": ["pred_answer", "prediction", "response"],
        "question_type_show": ["question_type", "question_type_vlm", "question_type_manifest"],
        "answer_type_show": ["answer_type", "answer_type_vlm", "answer_type_manifest"],
    }
    for new_col, candidates in mapping.items():
        df[new_col] = [first_existing(row, candidates, "") for _, row in df.iterrows()]
    return df


def make_base_uid(sample_uid, degradation):
    s = str(sample_uid)
    d = str(degradation)
    # remove suffix forms such as _original, _blur, -dark, etc.
    for deg in DEGRADATIONS:
        s = re.sub(rf"([_\-]){deg}$", "", s)
    # fallback: remove degradation anywhere at end
    s = s.replace(f"_{d}", "").replace(f"-{d}", "")
    return s


def token_f1(a, b):
    a_tokens = str(a).lower().strip().split()
    b_tokens = str(b).lower().strip().split()
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    common = {}
    for t in a_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in b_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(b_tokens)
    recall = overlap / len(a_tokens)
    return 2 * precision * recall / (precision + recall)


def char_f1(a, b):
    a = str(a).lower().strip()
    b = str(b).lower().strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca = {}
    for ch in a:
        ca[ch] = ca.get(ch, 0) + 1
    overlap = 0
    for ch in b:
        if ca.get(ch, 0) > 0:
            overlap += 1
            ca[ch] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(b)
    recall = overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def minmax_from_train(train_values, values):
    train_values = np.asarray(train_values, dtype=float)
    values = np.asarray(values, dtype=float)
    mn = np.nanmin(train_values)
    mx = np.nanmax(train_values)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - mn) / (mx - mn), 0, 1)


def orient_single_metric_cv(score, y, n_splits=5, seed=42):
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)

    score = np.nan_to_num(score, nan=np.nanmedian(score))
    pred = np.zeros_like(score, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in skf.split(score.reshape(-1, 1), y):
        train_score = score[train_idx]
        train_y = y[train_idx]

        rho = spearmanr(train_score, train_y).correlation
        sign = 1.0
        if np.isfinite(rho) and rho < 0:
            sign = -1.0

        oriented_train = sign * train_score
        oriented_test = sign * score[test_idx]
        pred[test_idx] = minmax_from_train(oriented_train, oriented_test)

    return pred


def evaluate_score(name, pred, y):
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=int)

    out = {"metric": name}

    try:
        out["auroc"] = roc_auc_score(y, pred)
    except Exception:
        out["auroc"] = np.nan

    try:
        out["average_precision"] = average_precision_score(y, pred)
    except Exception:
        out["average_precision"] = np.nan

    try:
        out["spearman_r"] = spearmanr(pred, y).correlation
    except Exception:
        out["spearman_r"] = np.nan

    try:
        out["pearson_r"] = pearsonr(pred, y)[0]
    except Exception:
        out["pearson_r"] = np.nan

    try:
        out["brier"] = brier_score_loss(y, np.clip(pred, 0, 1))
    except Exception:
        out["brier"] = np.nan

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--iqa", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", default="results_visdrone_300/metric_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vlm = pd.read_csv(args.vlm)
    iqa = pd.read_csv(args.iqa)
    meta = load_manifest(args.manifest)

    df = vlm.merge(iqa, on="sample_uid", how="left", suffixes=("_vlm", "_iqa"))
    df = df.merge(meta, on="sample_uid", how="left")
    df = normalize_columns(df)

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)

    for c in ["psnr", "ssim", "sharpness", "brightness"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df["base_uid"] = [
        make_base_uid(s, d) for s, d in zip(df["sample_uid"], df["degradation_show"])
    ]

    original_pred = (
        df[df["degradation_show"] == "original"]
        .set_index("base_uid")["pred_show"]
        .to_dict()
    )
    df["original_pred_show"] = df["base_uid"].map(original_pred).fillna("")

    df["token_consistency"] = [
        token_f1(a, b) for a, b in zip(df["original_pred_show"], df["pred_show"])
    ]
    df["char_consistency"] = [
        char_f1(a, b) for a, b in zip(df["original_pred_show"], df["pred_show"])
    ]
    df["cognition_consistency_proxy"] = (
        0.5 * df["token_consistency"] + 0.5 * df["char_consistency"]
    )

    # Brightness quality: too dark or too bright are both bad.
    # Use original-image median brightness as the neutral target.
    orig_brightness_median = df[df["degradation_show"] == "original"]["brightness"].median()
    df["brightness_quality"] = -np.abs(df["brightness"] - orig_brightness_median)

    y = df["correct"].values

    candidate_scores = {
        "PSNR": df["psnr"].values,
        "SSIM": df["ssim"].values,
        "Sharpness": df["sharpness"].values,
        "BrightnessQuality": df["brightness_quality"].values,
        "EmbodiedIQA_CognitionConsistencyProxy": df["cognition_consistency_proxy"].values,
    }

    rows = []
    score_outputs = pd.DataFrame({
        "sample_uid": df["sample_uid"],
        "correct": df["correct"],
        "degradation": df["degradation_show"],
        "question_type": df["question_type_show"],
    })

    for name, score in candidate_scores.items():
        pred = orient_single_metric_cv(score, y, n_splits=5, seed=args.seed)
        score_outputs[name] = pred
        rows.append(evaluate_score(name, pred, y))

    # Ours: task-conditioned utility score.
    # This is a learned utility predictor using no gold answer text, only task/degradation/quality/consistency features.
    feature_df = pd.DataFrame({
        "psnr": df["psnr"],
        "ssim": df["ssim"],
        "sharpness": df["sharpness"],
        "brightness_quality": df["brightness_quality"],
        "cognition_consistency_proxy": df["cognition_consistency_proxy"],
        "degradation": df["degradation_show"],
        "question_type": df["question_type_show"],
        "answer_type": df["answer_type_show"],
    })
    X = pd.get_dummies(feature_df, columns=["degradation", "question_type", "answer_type"], dummy_na=True)
    X = X.fillna(0.0)

    pred_ours = np.zeros(len(df), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for train_idx, test_idx in skf.split(X, y):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X.iloc[train_idx], y[train_idx])
        pred_ours[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    score_outputs["Ours_TCUS_TaskConditionedUtilityScore"] = pred_ours
    rows.append(evaluate_score("Ours_TCUS_TaskConditionedUtilityScore", pred_ours, y))

    result = pd.DataFrame(rows)

    # rank: higher AUROC/AP/Spearman better; lower Brier better.
    result = result.sort_values(by=["auroc", "average_precision", "spearman_r"], ascending=False)

    df.to_csv(out_dir / "merged_metric_benchmark_inputs.csv", index=False)
    score_outputs.to_csv(out_dir / "metric_scores_per_sample.csv", index=False)
    result.to_csv(out_dir / "metric_task_alignment_comparison.csv", index=False)

    print("\nMetric-task alignment comparison:")
    print(result.to_string(index=False))
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
