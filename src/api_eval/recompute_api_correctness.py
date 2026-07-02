from pathlib import Path
import argparse
import pandas as pd

from src.api_eval.run_viviai_vlm_visdrone import normalize_answer


def process_one(path: Path, gold_col: str = "gold_answer"):
    df = pd.read_csv(path)

    if "api_pred_raw" not in df.columns:
        print("[SKIP] no api_pred_raw:", path)
        return

    if gold_col not in df.columns:
        if "gold_col_used" in df.columns and len(df):
            candidate = str(df["gold_col_used"].dropna().iloc[0])
            if candidate in df.columns:
                gold_col = candidate

    if gold_col not in df.columns:
        print("[SKIP] no gold column:", path)
        return

    backup = path.with_suffix(path.suffix + ".bak_norm")
    if not backup.exists():
        df.to_csv(backup, index=False)

    df["api_pred_norm"] = df["api_pred_raw"].apply(normalize_answer)
    df["gold_norm"] = df[gold_col].apply(normalize_answer)
    df["api_correct"] = (df["api_pred_norm"] == df["gold_norm"]).astype(int)

    df.to_csv(path, index=False)

    err = int((df["api_error"].fillna("") != "").sum()) if "api_error" in df.columns else 0
    nonempty = int((df["api_pred_raw"].fillna("") != "").sum())
    acc = float(df["api_correct"].mean()) if len(df) else 0.0

    print(f"[OK] {path} rows={len(df)} errors={err} nonempty={nonempty} acc={acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--gold_col", default="gold_answer")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.glob))
    print("Matched files:", len(files))
    for p in files:
        process_one(p, args.gold_col)


if __name__ == "__main__":
    main()
