import argparse
import json
import os
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--key_env", required=True)
    ap.add_argument("--base_url", default=os.environ.get("VIVIAI_BASE_URL", "https://api.viviai.cc/v1"))
    ap.add_argument("--out_dir", default="results_api_vlm/model_lists")
    args = ap.parse_args()

    key = os.environ.get(args.key_env)
    if not key:
        raise SystemExit(f"Missing env var: {args.key_env}")

    url = args.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {key}"}

    print("Request:", url)
    r = requests.get(url, headers=headers, timeout=60)
    print("HTTP status:", r.status_code)
    print("Content-Type:", r.headers.get("content-type"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{args.name}_models_raw.json"
    raw_path.write_text(r.text, encoding="utf-8")

    if r.status_code != 200:
        print(r.text[:2000])
        raise SystemExit(f"Model list request failed for {args.name}")

    data = r.json()
    models = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("model") or item.get("name")
                if mid:
                    models.append(str(mid))
            elif isinstance(item, str):
                models.append(item)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("model") or item.get("name")
                if mid:
                    models.append(str(mid))
            elif isinstance(item, str):
                models.append(item)

    models = sorted(set(models))
    txt_path = out_dir / f"{args.name}_models.txt"
    txt_path.write_text("\n".join(models) + "\n", encoding="utf-8")

    print(f"Saved raw JSON to: {raw_path}")
    print(f"Saved model ids to: {txt_path}")
    print(f"Model count: {len(models)}")
    print("First 80 models:")
    for m in models[:80]:
        print(" ", m)


if __name__ == "__main__":
    main()
