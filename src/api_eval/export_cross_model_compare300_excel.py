from pathlib import Path
import pandas as pd

summary_path = Path("results_api_vlm/cross_model_compare300/cross_model_compare300_alignment_summary.csv")
summary = pd.read_csv(summary_path)

out_dir = Path("export_for_senior_cross_model_compare300")
out_dir.mkdir(parents=True, exist_ok=True)

tcus = summary[summary["Metric"].eq("TCUS-compare")].copy()
tcus = tcus.sort_values(["Provider", "AUROC"], ascending=[True, False])

# Pivot for a compact visualization table.
pivot_auroc = summary.pivot_table(
    index=["Provider", "Model"],
    columns="Metric",
    values="AUROC",
    aggfunc="first"
).reset_index()

pivot_ap = summary.pivot_table(
    index=["Provider", "Model"],
    columns="Metric",
    values="Average Precision",
    aggfunc="first"
).reset_index()

pivot_sp = summary.pivot_table(
    index=["Provider", "Model"],
    columns="Metric",
    values="Spearman",
    aggfunc="first"
).reset_index()

xlsx = out_dir / "cross_model_compare300_results.xlsx"

with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    tcus.to_excel(writer, index=False, sheet_name="Ours-like TCUS Compare")
    summary.to_excel(writer, index=False, sheet_name="Long Summary")
    pivot_auroc.to_excel(writer, index=False, sheet_name="AUROC Pivot")
    pivot_ap.to_excel(writer, index=False, sheet_name="AP Pivot")
    pivot_sp.to_excel(writer, index=False, sheet_name="Spearman Pivot")

    wb = writer.book
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.style = "Headline 4"
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 42)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000"

print("Saved:", xlsx)
