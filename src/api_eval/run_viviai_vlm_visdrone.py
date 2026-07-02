import argparse
import base64
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


def find_col(cols, candidates, exclude_substrings=None):
    exclude_substrings = exclude_substrings or []
    lower = {c.lower(): c for c in cols}

    # exact match first
    for cand in candidates:
        if cand.lower() in lower:
            c = lower[cand.lower()]
            if not any(x in c.lower() for x in exclude_substrings):
                return c

    # partial match later
    for c in cols:
        lc = c.lower()
        if any(x in lc for x in exclude_substrings):
            continue
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def encode_image(path: Path):
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def normalize_answer(x):
    """
    Normalize short VQA answers.

    Important:
    - Do NOT map numeric "1" to "yes" or "0" to "no", because this benchmark
      contains counting questions.
    - yes/no answers and integer counting answers must remain separable.
    """
    if pd.isna(x):
        return ""

    s = str(x).strip().lower()
    s = s.replace("，", ",").replace("。", ".")
    s = re.sub(r"^[\"'`]+|[\"'`]+$", "", s)
    s = re.sub(r"[\s\.\,\;\:\!\?]+$", "", s)

    # Extract JSON-like {"answer": "..."} first.
    m_ans = re.search(r'"?answer"?\s*[:=]\s*"?([^\"\}\n]+)"?', s)
    if m_ans:
        s = m_ans.group(1).strip().lower()
        s = re.sub(r"[\s\.\,\;\:\!\?]+$", "", s)

    # Yes/no normalization. Do not include numeric 1/0 here.
    if s in ["yes", "y", "true"]:
        return "yes"
    if s in ["no", "n", "false"]:
        return "no"

    # Counting answers: preserve integers.
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))

    # If the model gives a short phrase containing a single integer, use it.
    m = re.search(r"-?\d+", s)
    if m and len(s) <= 80:
        return str(int(m.group(0)))

    # Short class/color/category answers.
    s = re.sub(r"[^a-z0-9\- ]", "", s)
    return s.strip()


def build_prompt(question):
    return (
        "You are answering a visual question for a UAV/drone-view image.\n"
        "Answer with ONLY the final answer, no explanation.\n"
        "If the question is yes/no, answer only yes or no.\n"
        "If the question asks for a count, answer only an integer.\n"
        "If the question asks for an object/color/category, answer only the short label.\n\n"
        f"Question: {question}\n"
        "Final answer:"
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def call_viviai_chat(model, key, image_path, question, base_url, max_tokens=16):
    url = base_url.rstrip("/") + "/chat/completions"
    image_url = encode_image(image_path)
    prompt = build_prompt(question)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1200]}")

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def resolve_image_path(raw_path, project_root):
    s = str(raw_path)
    p = Path(s)
    if p.exists():
        return p

    candidates = [
        project_root / s,
        project_root / s.lstrip("./"),
        Path("/root/autodl-tmp/projects/task_utility_iqa") / s.lstrip("./"),
    ]
    for c in candidates:
        if c.exists():
            return c

    name = Path(s).name
    if name:
        for root in [
            project_root / "data",
            project_root / "data/processed/images",
            project_root / "results_visdrone_300",
            project_root / "final_delivery_integrated_report_external_baseline_mixed_package",
        ]:
            if root.exists():
                hits = list(root.rglob(name))
                if hits:
                    return hits[0]

    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key_env", required=True)
    ap.add_argument("--base_url", default=os.environ.get("VIVIAI_BASE_URL", "https://api.viviai.cc/v1"))
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--project_root", default=".")

    # Explicit column override
    ap.add_argument("--image_col", default=None)
    ap.add_argument("--question_col", default=None)
    ap.add_argument("--gold_col", default=None)

    args = ap.parse_args()

    key = os.environ.get(args.key_env)
    if not key:
        raise SystemExit(f"Missing env var: {args.key_env}")

    project_root = Path(args.project_root).resolve()
    df = pd.read_csv(args.input_csv)

    image_col = args.image_col or find_col(
        df.columns,
        ["image_path_vlm", "image_path_iqa", "image_path_manifest", "image_path", "img_path", "file_path", "path"],
        exclude_substrings=["image_id", "question_id", "base_uid"],
    )
    question_col = args.question_col or find_col(
        df.columns,
        ["question_vlm", "question_show", "question_manifest", "question_iqa", "question", "query", "prompt"],
        exclude_substrings=["question_id", "question_type"],
    )
    gold_col = args.gold_col or find_col(
        df.columns,
        ["gold_answer", "gold_norm", "gold_show", "answer_manifest", "answer", "label", "gt_answer", "target"],
        exclude_substrings=["answer_type"],
    )

    if image_col is None or question_col is None or gold_col is None:
        print("Columns:")
        for c in df.columns:
            print(" ", c)
        raise SystemExit(f"Cannot infer columns: image={image_col}, question={question_col}, gold={gold_col}")

    print("Using columns:")
    print(" image_col   =", image_col)
    print(" question_col=", question_col)
    print(" gold_col    =", gold_col)

    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    done = {}
    if out_path.exists():
        old = pd.read_csv(out_path)
        if "row_id" in old.columns:
            done = {int(r["row_id"]): dict(r) for _, r in old.iterrows()}
            print(f"Resume from {out_path}: done={len(done)}")

    for i, row in tqdm(list(df.iterrows()), total=len(df), desc=args.model):
        row_id = int(i)
        if row_id in done:
            rows.append(done[row_id])
            continue

        image_path = resolve_image_path(row[image_col], project_root)
        question = str(row[question_col])
        gold = row[gold_col]

        if not image_path.exists():
            pred = ""
            err = f"image not found: {image_path}"
        else:
            try:
                pred = call_viviai_chat(args.model, key, image_path, question, args.base_url)
                err = ""
            except Exception as e:
                pred = ""
                err = repr(e)

        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)
        correct = int(pred_norm == gold_norm)

        out_row = row.to_dict()
        out_row.update({
            "row_id": row_id,
            "provider": args.provider,
            "api_model": args.model,
            "api_pred_raw": pred,
            "api_pred_norm": pred_norm,
            "gold_norm": gold_norm,
            "api_correct": correct,
            "api_error": err,
            "image_col_used": image_col,
            "question_col_used": question_col,
            "gold_col_used": gold_col,
            "resolved_image_path": str(image_path),
        })
        rows.append(out_row)

        pd.DataFrame(rows).to_csv(out_path, index=False)

        if err:
            print(f"[ERR] row={row_id} model={args.model}: {err[:260]}")
        else:
            print(f"[OK] row={row_id} correct={correct} gold={gold_norm} pred={pred_norm}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
