import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from src.utils.io import read_jsonl

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def resize_like(src, ref):
    if src.shape[:2] == ref.shape[:2]:
        return src
    return cv2.resize(src, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

def calc_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calc_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/iqa/iqa_scores.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    rows = read_jsonl(cfg["data"]["degraded_manifest"])

    out_rows = []

    for row in tqdm(rows, desc="Computing IQA metrics"):
        orig = load_rgb(row["original_path"])
        deg = load_rgb(row["degraded_image_path"])
        deg = resize_like(deg, orig)

        if row["degradation"] == "original":
            psnr = 60.0
            ssim = 1.0
        else:
            psnr = float(peak_signal_noise_ratio(orig, deg, data_range=255))
            ssim = float(structural_similarity(orig, deg, channel_axis=2, data_range=255))

        out_rows.append({
            "sample_uid": row["sample_uid"],
            "question_id": row["question_id"],
            "image_id": row["image_id"],
            "degradation": row["degradation"],
            "psnr": psnr,
            "ssim": ssim,
            "sharpness": calc_sharpness(deg),
            "brightness": calc_brightness(deg),
            "question": row["question"],
            "answer": row["answer"],
            "image_path": row["degraded_image_path"],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"Saved IQA scores to {out_path}")

if __name__ == "__main__":
    main()
