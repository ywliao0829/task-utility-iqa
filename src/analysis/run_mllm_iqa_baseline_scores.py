import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch


def parse_rating(text):
    text = str(text)

    # Prefer JSON-style output.
    m = re.search(r'"rating"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if m:
        return float(m.group(1))

    # Then search content inside answer tag.
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if m:
        inside = m.group(1)
        m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)", inside)
        if m2:
            return float(m2.group(1))

    # Last fallback: first number in the decoded text.
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    return np.nan


def run_qinsight(df, model_path, out_csv, max_new_tokens=128, max_pixels=401408):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
    from qwen_vl_utils import process_vision_info

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    subfolder = "score_degradation"

    print("Loading Q-Insight...")
    print("Model path:", model_path)
    print("Subfolder:", subfolder)
    print("Device:", device)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        device_map=device if torch.cuda.is_available() else None,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        subfolder=subfolder,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels,
    )

    system_prompt = (
        "You are an image quality assessment expert. "
        "Evaluate the visual quality of the given image."
    )

    score_prompt = (
        "What is your overall rating on the quality of this picture? "
        "The rating should be a float between 1 and 5, rounded to two decimal places, "
        "with 1 representing very poor quality and 5 representing excellent quality. "
        'Return the final answer in JSON format with the following key: "rating".'
    )

    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    rows = []
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, r in tqdm(df.iterrows(), total=len(df), desc="Q-Insight scoring"):
        image_path = str(Path(r["image_path"]).resolve())

        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": score_prompt},
                ],
            },
        ]

        try:
            text = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info([message])

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    use_cache=True,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            score = parse_rating(output_text)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            output_text = f"ERROR: {repr(e)}"
            score = np.nan

        rows.append({
            "sample_uid": r.get("sample_uid", ""),
            "baseline": "Q-Insight",
            "score": score,
            "correct": int(r.get("correct", 0)),
            "degradation": r.get("degradation", ""),
            "question_type": r.get("question_type", ""),
            "image_path": r.get("image_path", ""),
            "raw_output": output_text,
        })

        # Incremental saving, safer for long full runs.
        if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["qinsight"])
    parser.add_argument("--input_csv", default="results_visdrone_300/external_baselines/visdrone_external_baseline_inputs.csv")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_pixels", type=int, default=401408)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    if args.baseline == "qinsight":
        run_qinsight(
            df=df,
            model_path=args.model_path,
            out_csv=args.out_csv,
            max_new_tokens=args.max_new_tokens,
            max_pixels=args.max_pixels,
        )


if __name__ == "__main__":
    main()
