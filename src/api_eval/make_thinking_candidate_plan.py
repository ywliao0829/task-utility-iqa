from pathlib import Path
import re
import pandas as pd

provider_key = {
    "gpt": "VIVIAI_GPT_KEY",
    "gemini": "VIVIAI_GEMINI_KEY",
    "claude": "VIVIAI_CLAUDE_KEY",
    "grok": "VIVIAI_GROK_KEY",
}

pat = re.compile(r"(thinking|reasoning|reason|qwen|qwq)", re.I)

# 已经跑过 300 条的模型，不再进入候选 smoke test。
already_done = {
    "gemini-2.5-flash-thinking",
}

def provider_model_consistent(provider: str, model: str) -> bool:
    m = model.lower()

    # gpt provider 下如果出现 gemini 模型，说明只是通道可见，不作为 GPT family 模型。
    if provider == "gpt" and not (m.startswith("gpt") or m.startswith("o")):
        return False

    if provider == "gemini" and not m.startswith("gemini"):
        return False

    if provider == "claude" and not m.startswith("claude"):
        return False

    if provider == "grok" and not m.startswith("grok"):
        return False

    return True

rows = []
for p in sorted(Path("results_api_vlm/model_lists").glob("*_models.txt")):
    provider = p.name.replace("_models.txt", "")
    key_env = provider_key.get(provider)
    if key_env is None:
        continue

    models = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    for m in models:
        lower = m.lower()

        if m in already_done:
            continue

        if not pat.search(m):
            continue

        if not provider_model_consistent(provider, m):
            continue

        # 排除明显图像生成/编辑 endpoint。
        if any(x in lower for x in ["image-preview", "flash-image", "pro-image-preview"]):
            continue

        if "thinking" in lower:
            scale = "thinking"
        elif "reasoning" in lower or "reason" in lower:
            scale = "reasoning"
        elif "qwen" in lower or "qwq" in lower:
            scale = "qwen_candidate"
        else:
            scale = "candidate"

        rows.append({
            "provider": provider,
            "key_env": key_env,
            "model": m,
            "scale": scale,
            "group": f"{provider}_thinking_or_reasoning_candidate",
        })

out = pd.DataFrame(rows).drop_duplicates()

out_path = "results_api_vlm/api_model_plan_thinking_candidates.tsv"
out.to_csv(out_path, sep="\t", index=False)

print(out.to_string(index=False))
print("\nCandidate count:", len(out))
print("Saved:", out_path)
