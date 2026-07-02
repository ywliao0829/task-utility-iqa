import subprocess
import pandas as pd

plan_path = "results_api_vlm/api_model_plan_thinking300.tsv"
plan = pd.read_csv(plan_path, sep="\t")

input_csv = "results_visdrone_300/metric_benchmark/qwen25vl_7b/merged_metric_benchmark_inputs.csv"
project_root = "/root/autodl-tmp/projects/task_utility_iqa"

for _, r in plan.iterrows():
    provider = r["provider"]
    model = r["model"]
    key_env = r["key_env"]
    scale = r["scale"]

    safe_model = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
    out_csv = f"results_api_vlm/all_strict300/{provider}_{scale}_{safe_model}_visdrone300.csv"

    cmd = [
        "python", "-m", "src.api_eval.run_viviai_vlm_visdrone",
        "--input_csv", input_csv,
        "--out_csv", out_csv,
        "--provider", provider,
        "--model", model,
        "--key_env", key_env,
        "--max_samples", "300",
        "--sleep", "0.2",
        "--project_root", project_root,
        "--image_col", "image_path_vlm",
        "--question_col", "question_vlm",
        "--gold_col", "gold_answer",
    ]

    print("\n==== Running thinking300 experiment ====")
    print("provider:", provider)
    print("scale:", scale)
    print("model:", model)
    print("out:", out_csv)
    subprocess.run(cmd, check=False)
