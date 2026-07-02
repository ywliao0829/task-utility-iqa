import json
from pathlib import Path

import pandas as pd


def load_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "sample_uid": r.get("sample_uid", ""),
                "image_path": r.get("image_path", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "question_type": r.get("question_type", ""),
                "degradation": r.get("degradation", ""),
            })
    return pd.DataFrame(rows)


def main():
    manifest_path = Path("data/processed/manifest_visdrone_degraded_300.jsonl")
    vlm_path = Path("results_visdrone_300/vlm/qwen25vl_7b_results.csv")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    if not vlm_path.exists():
        raise FileNotFoundError(f"Missing VLM result: {vlm_path}")

    meta = load_manifest(manifest_path)
    vlm = pd.read_csv(vlm_path)

    if "sample_uid" not in vlm.columns or "correct" not in vlm.columns:
        raise RuntimeError(f"VLM result must contain sample_uid and correct columns. Current columns: {list(vlm.columns)}")

    df = meta.merge(vlm[["sample_uid", "correct"]], on="sample_uid", how="left")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)

    df["image_exists"] = df["image_path"].map(lambda p: Path(str(p)).exists())
    missing = df[~df["image_exists"]]

    print("Total rows before filtering:", len(df))
    print("Missing images:", len(missing))

    if len(missing) > 0:
        print(missing[["sample_uid", "image_path"]].head(10).to_string(index=False))

    df = df[df["image_exists"]].copy()

    out_path = Path("results_visdrone_300/external_baselines/visdrone_external_baseline_inputs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print(df[["sample_uid", "degradation", "question_type", "correct", "image_path"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
