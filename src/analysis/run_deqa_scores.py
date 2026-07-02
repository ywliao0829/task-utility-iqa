import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig


def to_float_score(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    arr = np.array(x).reshape(-1)
    return float(arr[0])



def patch_deqa_runtime_flags(model):
    """
    DeQA-Score-Mix3 is implemented with an older custom mPLUG-Owl2/LLaMA stack.
    Newer transformers versions may expect attention implementation flags such as
    _use_flash_attention_2 and _use_sdpa to exist on LLaMA-like modules.
    """
    patched = 0

    def patch_obj(obj):
        nonlocal patched
        if obj is None:
            return

        cls_name = obj.__class__.__name__.lower()
        should_patch = (
            "llama" in cls_name
            or "mplug" in cls_name
            or "attention" in cls_name
            or hasattr(obj, "config")
        )

        if not should_patch:
            return

        if not hasattr(obj, "_use_flash_attention_2"):
            try:
                setattr(obj, "_use_flash_attention_2", False)
                patched += 1
            except Exception:
                pass

        if not hasattr(obj, "_use_sdpa"):
            try:
                setattr(obj, "_use_sdpa", False)
                patched += 1
            except Exception:
                pass

        cfg = getattr(obj, "config", None)
        if cfg is not None:
            for attr, value in [
                ("_attn_implementation", "eager"),
                ("attn_implementation", "eager"),
                ("mlp_bias", False),
                ("attention_bias", False),
            ]:
                if not hasattr(cfg, attr):
                    try:
                        setattr(cfg, attr, value)
                    except Exception:
                        pass

    patch_obj(model)

    if hasattr(model, "modules"):
        for module in model.modules():
            patch_obj(module)

    # Also patch common nested holders explicitly.
    for name in ["model", "language_model", "llm", "vision_model"]:
        patch_obj(getattr(model, name, None))

    print(f"Runtime patch: added/checked DeQA attention flags on modules; patched_count={patched}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results_visdrone_300/external_baselines/visdrone_external_baseline_inputs.csv")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/iqa_baselines/DeQA-Score-Mix3")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    print("Loading DeQA-Score-Mix3...")
    print("Model:", args.model_path)
    print("Rows:", len(df))

    # DeQA-Score-Mix3 uses an older custom MPLUGOwl2Config.
    # Newer transformers LlamaMLP expects config.mlp_bias, which may be missing.
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if not hasattr(config, "mlp_bias"):
        print("Patch config: setting missing mlp_bias=False")
        config.mlp_bias = False

    if not hasattr(config, "attention_bias"):
        print("Patch config: setting missing attention_bias=False")
        config.attention_bias = False

    if not hasattr(config, "_attn_implementation"):
        print("Patch config: setting missing _attn_implementation='eager'")
        config._attn_implementation = "eager"

    if not hasattr(config, "attn_implementation"):
        print("Patch config: setting missing attn_implementation='eager'")
        config.attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    rows = []
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, r in tqdm(df.iterrows(), total=len(df), desc="DeQA scoring"):
        image_path = Path(str(r["image_path"]))
        raw_output = ""
        score = np.nan

        try:
            image = Image.open(image_path).convert("RGB")

            with torch.inference_mode():
                out = model.score([image])

            score = to_float_score(out)
            raw_output = str(out)

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
            "baseline": "DeQA-Score-Mix3",
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
