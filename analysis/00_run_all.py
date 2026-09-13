"""Reproduce the article's aggregate tables and figures from local TurkStat data."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

for name in (
    "01_age_analysis.py",
    "03_pretrend_analysis.py",
    "06_cohort_analysis.py",
    "09_education_composition_time.py",
    "02_age_figures.py",
    "07_cohort_figures.py",
):
    result = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE)
    if result.returncode:
        raise SystemExit(result.returncode)

print("Core and supplementary analysis outputs reproduced.")
