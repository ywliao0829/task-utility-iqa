from pathlib import Path
import pandas as pd

summary_path = Path("results_api_vlm/all_strict300_alignment_summary.csv")
summary = pd.read_csv(summary_path)

# Clean validity rule:
# - completed 300-row evaluation
# - no more than 5 API errors
# - at least 290 non-empty short-form outputs
# - valid AUROC
# - not one-class success labels
clean = summary[
    (summary["N_total"] >= 300) &
    (summary["Error count"] <= 5) &
    (summary["Nonempty pred"] >= 290) &
    (summary["AUROC"].notna()) &
    (~summary["Note"].fillna("").str.contains("only one class", case=False, regex=False))
].copy()

excluded = summary[~summary.index.isin(clean.index)].copy()

clean_path = Path("results_api_vlm/all_strict300_alignment_summary_clean.csv")
excluded_path = Path("results_api_vlm/all_strict300_alignment_summary_excluded.csv")

clean.to_csv(clean_path, index=False)
excluded.to_csv(excluded_path, index=False)

print("Saved clean summary:", clean_path)
print("Saved excluded summary:", excluded_path)

print("\n===== Clean TCUS-proxy results =====")
tcus = clean[clean["Metric"].eq("TCUS-proxy")].sort_values(["Provider", "AUROC"], ascending=[True, False])
print(tcus[[
    "Provider", "Model", "N_total", "Error count", "Nonempty pred",
    "Task success rate", "AUROC", "Average Precision", "Spearman"
]].to_string(index=False))

print("\n===== Gemini rows in clean summary =====")
print(clean[clean["Provider"].astype(str).str.lower().eq("gemini")][[
    "Provider", "Model", "Metric", "N_total", "Error count", "Nonempty pred",
    "Task success rate", "AUROC", "Average Precision", "Spearman"
]].to_string(index=False))
