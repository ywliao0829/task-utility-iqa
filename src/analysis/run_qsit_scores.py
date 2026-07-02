import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration


TOKS = ["Excellent", "Good", "Fair", "Poor", "Bad"]
WEIGHTS = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float64)


def weighted_average_5(logits_dict):
    logits = np.array([logits_dict[t] for t in TOKS], dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    return float(np.inner(probs, WEIGHTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results_visdrone_300/external_baselines/visdrone_external_baseline_inputs.csv")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/iqa_baselines/q-sit")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Loading Q-SiT...")
    print("Model:", args.model_path)
    print("Device:", device)
    print("Rows:", len(df))

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    tokenized = tokenizer(TOKS)["input_ids"]
    rating_ids = [x[0] for x in tokenized]
    print("Rating token IDs:", dict(zip(TOKS, rating_ids)))

    prompt_text = (
        "Assume you are an image quality evaluator.\n"
        "Your rating should be chosen from the following five categories: "
        "Excellent, Good, Fair, Poor, and Bad (from high to low).\n"
        "How would you rate the quality of this image?"
    )

    prefix_text = "The quality of this image is "

    rows = []
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, r in tqdm(df.iterrows(), total=len(df), desc="Q-SiT scoring"):
        image_path = Path(str(r["image_path"]))
        raw_output = ""
        score = np.nan

        try:
            image = Image.open(image_path).convert("RGB")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(device, dtype)

            prefix_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            last_logits = output.scores[-1][0]
            logits_dict = {tok: float(last_logits[i].detach().cpu()) for tok, i in zip(TOKS, rating_ids)}
            score = weighted_average_5(logits_dict)

            pred_token_id = int(torch.argmax(last_logits).detach().cpu())
            pred_token = tokenizer.decode([pred_token_id])
            raw_output = f"pred_token={pred_token}; logits={logits_dict}"

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            raw_output = "ERROR: " + repr(e)
            score = np.nan
        except Exception as e:
            raw_output = "ERROR: " + repr(e)
            score = np.nan

        rows.append({
            "sample_uid": r.get("sample_uid", ""),
            "baseline": "Q-SiT",
            "score": score,
            "correct": int(r.get("correct", 0)),
            "degradation": r.get("degradation", ""),
            "question_type": r.get("question_type", ""),
            "image_path": str(image_path),
            "raw_output": raw_output,
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
