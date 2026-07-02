from pathlib import Path
import pandas as pd

src = Path("results_visdrone_300/external_baselines/summary/external_baseline_vs_ours_alignment_all.csv")
out_tex = Path("emnlp_task_utility_iqa_paper/tables/external_baseline_visdrone_all.tex")
preview_tex = Path("emnlp_task_utility_iqa_paper/preview_tables/preview_external_baseline_visdrone_all.tex")

out_tex.parent.mkdir(parents=True, exist_ok=True)
preview_tex.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)

order = [
    "Q-SiT",
    "Q-Insight",
    "DeQA-Score-Mix3",
    "Cognition-Consistency",
    "Ours-TCUS",
]

df = df[df["metric"].isin(order)].copy()
df["metric"] = pd.Categorical(df["metric"], categories=order, ordered=True)
df = df.sort_values("metric")

def fmt(x):
    return f"{float(x):.3f}"

lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{External IQA baseline comparison on VisDrone for predicting downstream task success. All reported methods are evaluated on the same 1,500 samples. Higher is better for AUROC, AP, and Spearman correlation.}")
lines.append(r"\label{tab:external_baseline_visdrone_all}")
lines.append(r"\begin{tabular}{lccc}")
lines.append(r"\toprule")
lines.append(r"Method & AUROC $\uparrow$ & AP $\uparrow$ & Spearman $\rho$ $\uparrow$ \\")
lines.append(r"\midrule")

for _, row in df.iterrows():
    method = str(row["metric"])
    auroc = fmt(row["auroc"])
    ap = fmt(row["average_precision"])
    spr = fmt(row["spearman_r"])

    if method == "Ours-TCUS":
        lines.append(r"\rowcolor{gray!12}")
        lines.append(rf"\textbf{{{method}}} & \textbf{{{auroc}}} & \textbf{{{ap}}} & \textbf{{{spr}}} \\")
    else:
        lines.append(rf"{method} & {auroc} & {ap} & {spr} \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\vspace{2pt}")
lines.append(r"\begin{minipage}{0.95\linewidth}")
lines.append(r"\footnotesize")
lines.append(r"\textit{Note:} Q-Insight-raw is omitted from the main table because only 225/1500 raw numerical outputs were validly parsed; the cleaned Q-Insight score is reported instead.")
lines.append(r"\end{minipage}")
lines.append(r"\end{table*}")

out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

preview = []
preview.append(r"\documentclass{article}")
preview.append(r"\IfFileExists{newtxtext.sty}{\usepackage{newtxtext}\usepackage{newtxmath}}{\usepackage{times}}")
preview.append(r"\usepackage[margin=1in]{geometry}")
preview.append(r"\usepackage{booktabs}")
preview.append(r"\usepackage{xcolor}")
preview.append(r"\usepackage{colortbl}")
preview.append(r"\begin{document}")
preview.append(r"\thispagestyle{empty}")
preview.extend(lines)
preview.append(r"\end{document}")

preview_tex.write_text("\n".join(preview) + "\n", encoding="utf-8")

print("Generated:")
print(" -", out_tex)
print(" -", preview_tex)
