from pathlib import Path
import shutil
import pandas as pd

out_dir = Path("export_for_senior_api_all_strict300")
raw_dir = out_dir / "raw_api_outputs"
out_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv("results_api_vlm/all_strict300_alignment_summary.csv")
status = pd.read_csv("results_api_vlm/all_strict300_run_status.csv") if Path("results_api_vlm/all_strict300_run_status.csv").exists() else pd.DataFrame()
tagged = pd.read_csv("results_api_vlm/smoke_candidates_summary_tagged.csv")

tcus = summary[summary["Metric"].eq("TCUS-proxy")].copy()

source_rows = []
for p in sorted(Path("results_api_vlm/all_strict300").glob("*.csv")):
    dst = raw_dir / p.name
    shutil.copy2(p, dst)
    df = pd.read_csv(p)
    source_rows.append({
        "file": str(p),
        "copied_to": str(dst),
        "rows": len(df),
        "provider": df["provider"].iloc[0] if "provider" in df.columns and len(df) else "",
        "model": df["api_model"].iloc[0] if "api_model" in df.columns and len(df) else "",
        "success_rate": df["api_correct"].mean() if "api_correct" in df.columns else "",
        "error_count": int((df["api_error"].fillna("") != "").sum()) if "api_error" in df.columns else "",
        "nonempty_count": int((df["api_pred_raw"].fillna("") != "").sum()) if "api_pred_raw" in df.columns else "",
    })

source = pd.DataFrame(source_rows)

xlsx = out_dir / "api_all_strict300_cross_family_results.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    tcus.to_excel(writer, index=False, sheet_name="TCUS-proxy Summary")
    summary.to_excel(writer, index=False, sheet_name="All Metrics Summary")
    source.to_excel(writer, index=False, sheet_name="API Output Files")
    tagged.to_excel(writer, index=False, sheet_name="Smoke Test Tagged")
    if len(status):
        status.to_excel(writer, index=False, sheet_name="Run Status")

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
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 50)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000"

print("Saved:", xlsx)
