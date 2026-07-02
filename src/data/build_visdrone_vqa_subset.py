import argparse
import random
from pathlib import Path

import yaml
from tqdm import tqdm

from src.utils.io import write_jsonl

CATEGORY_ID_TO_NAME = {
    1: "pedestrian",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning tricycle",
    9: "bus",
    10: "motorcycle",
}

PLURAL = {
    "pedestrian": "pedestrians",
    "person": "people",
    "bicycle": "bicycles",
    "car": "cars",
    "van": "vans",
    "truck": "trucks",
    "tricycle": "tricycles",
    "awning tricycle": "awning tricycles",
    "bus": "buses",
    "motorcycle": "motorcycles",
}

def parse_annotation(path):
    counts = {name: 0 for name in CATEGORY_ID_TO_NAME.values()}
    boxes = []

    if not Path(path).exists():
        return counts, boxes

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            try:
                x, y, w, h = map(float, parts[:4])
                cat_id = int(parts[5])
            except Exception:
                continue

            if cat_id not in CATEGORY_ID_TO_NAME:
                continue
            if w <= 1 or h <= 1:
                continue

            name = CATEGORY_ID_TO_NAME[cat_id]
            counts[name] += 1
            boxes.append({"category": name, "bbox": [x, y, w, h]})

    return counts, boxes

def build_question(image_path, ann_path, idx):
    counts, boxes = parse_annotation(ann_path)
    present = [k for k, v in counts.items() if v > 0]
    absent = [k for k, v in counts.items() if v == 0]

    if not present:
        return None

    mode = idx % 3

    if mode == 0:
        cat = max(present, key=lambda k: counts[k])
        question = f"Is there at least one {cat} in this drone-view image?"
        answer = "yes"
        qtype = "positive_existence"

    elif mode == 1 and absent:
        cat = random.choice(absent)
        question = f"Is there at least one {cat} in this drone-view image?"
        answer = "no"
        qtype = "negative_existence"

    else:
        valid = [k for k in present if counts[k] <= 20]
        cat = random.choice(valid) if valid else max(present, key=lambda k: counts[k])
        plural = PLURAL.get(cat, cat + "s")
        question = f"How many {plural} are visible in this drone-view image?"
        answer = str(counts[cat])
        qtype = "counting"

    return {
        "question": question,
        "answer": answer,
        "question_type": qtype,
        "counts": counts,
        "boxes": boxes,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visdrone_300.yaml")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    random.seed(cfg.get("seed", 42))

    root = Path(cfg["data"]["visdrone_root"])
    image_dir = root / "images"
    ann_dir = root / "annotations"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")

    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    random.shuffle(images)

    num_samples = args.num_samples or cfg.get("num_samples", 300)

    rows = []
    for img_path in tqdm(images, desc="Building VisDrone VQA-style manifest"):
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.txt"

        item = build_question(img_path, ann_path, len(rows))
        if item is None:
            continue

        row = {
            "question_id": f"visdrone_{len(rows):06d}",
            "image_id": stem,
            "image_path": str(img_path),
            "question": item["question"],
            "answer": item["answer"],
            "answer_type": "yesno" if item["question_type"].endswith("existence") else "number",
            "question_type": item["question_type"],
            "counts": item["counts"],
            "boxes": item["boxes"],
        }
        rows.append(row)

        if len(rows) >= num_samples:
            break

    out_path = cfg["data"]["subset_manifest"]
    write_jsonl(rows, out_path)
    print(f"Saved {len(rows)} samples to {out_path}")

if __name__ == "__main__":
    main()
