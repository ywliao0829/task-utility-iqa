import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

from src.utils.io import read_jsonl, write_jsonl

def add_noise(img, sigma=25):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    rows = read_jsonl(cfg["data"]["subset_manifest"])

    out_dir = Path("data/processed/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    degraded_rows = []

    for row in tqdm(rows, desc="Making degraded images"):
        image_id = row["image_id"]
        qid = row["question_id"]
        img = Image.open(row["image_path"]).convert("RGB")

        variants = {
            "original": img,
            "blur": img.filter(ImageFilter.GaussianBlur(radius=3)),
            "dark": ImageEnhance.Brightness(img).enhance(0.35),
            "noise": add_noise(img, sigma=25),
        }

        for deg_type, deg_img in variants.items():
            out_path = out_dir / f"{image_id}_{qid}_{deg_type}.jpg"
            deg_img.save(out_path, quality=95)

            new_row = dict(row)
            new_row["degradation"] = deg_type
            new_row["original_path"] = row["image_path"]
            new_row["degraded_image_path"] = str(out_path)
            new_row["sample_uid"] = f"{image_id}_{qid}_{deg_type}"
            degraded_rows.append(new_row)

        jpeg_path = out_dir / f"{image_id}_{qid}_jpeg_low.jpg"
        img.save(jpeg_path, quality=15)

        new_row = dict(row)
        new_row["degradation"] = "jpeg_low"
        new_row["original_path"] = row["image_path"]
        new_row["degraded_image_path"] = str(jpeg_path)
        new_row["sample_uid"] = f"{image_id}_{qid}_jpeg_low"
        degraded_rows.append(new_row)

    write_jsonl(degraded_rows, cfg["data"]["degraded_manifest"])
    print(f"Saved {len(degraded_rows)} degraded samples to {cfg['data']['degraded_manifest']}")

if __name__ == "__main__":
    main()
