from pathlib import Path
import runpy
import os

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "reproduce" / "figure"
os.chdir(FIG_DIR)
runpy.run_path(str(FIG_DIR / "make_task_utility_composite.py"), run_name="__main__")
