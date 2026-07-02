import argparse
import random
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from src.utils.io import read_json, write_jsonl

def image_name(image_id):
    return f"COCO_val2014_{int(image_id):012d}.jpg"

def download_image(image_id, out_dir, retries=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = image_name(image_id)
    out_path = out_dir / name

    if out_path.exists() and out_path.stat().st_size > 1000:
        return str(out_path)

    url = f"http://images.cocodataset.org/val2014/{name}"

    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                out_path.write_bytes(r.content)
                return str(out_path)
            print(f"Bad response {r.status_code} for {url}")
        except Exception as e:
            print(f"Retry {i+1}/{retries} failed for {url}: {e}")
            time.sleep(2)

    raise RuntimeError(f"Failed to download image: {url}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed = cfg.get("seed", 42)
    num_samples = args.num_samples or cfg.get("num_samples", 100)

    random.seed(seed)

    q_data = read_json(cfg["data"]["vqa_question_file"])
    a_data = read_json(cfg["data"]["vqa_annotation_file"])

    qid_to_question = {
        q["question_id"]: q["question"]
        for q in q_data["questions"]
    }

    candidates = []
    for ann in a_data["annotations"]:
        answer = ann.get("multiple_choice_answer", "")
        if not answer:
            continue
        if len(str(answer).split()) > 3:
            continue
        qid = ann["question_id"]
        if qid not in qid_to_question:
            continue
        candidates.append(ann)

    random.shuffle(candidates)
    selected = candidates[:num_samples]

    rows = []
    for ann in tqdm(selected, desc="Downloading COCO images"):
        image_id = ann["image_id"]
        image_path = download_image(image_id, cfg["data"]["coco_image_dir"])

        row = {
            "question_id": ann["question_id"],
            "image_id": image_id,
            "image_path": image_path,
            "question": qid_to_question[ann["question_id"]],
            "answer": ann["multiple_choice_answer"],
            "answer_type": ann.get("answer_type", ""),
            "question_type": ann.get("question_type", ""),
            "all_answers": [x["answer"] for x in ann.get("answers", [])],
        }
        rows.append(row)

    write_jsonl(rows, cfg["data"]["subset_manifest"])
    print(f"Saved {len(rows)} samples to {cfg['data']['subset_manifest']}")

if __name__ == "__main__":
    main()
