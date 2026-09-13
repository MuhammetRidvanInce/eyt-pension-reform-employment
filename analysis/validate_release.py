"""Validate aggregate release contents without requiring restricted microdata."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def main():
    required_tables = [
        *(f"table{i:02d}_" for i in range(1, 12)),
        *(f"table{i:02d}_" for i in range(15, 19)),
    ]
    names = [p.name for p in TABLES.glob("*.csv")]
    for prefix in required_tables:
        assert sum(name.startswith(prefix) for name in names) == 1, prefix

    age = pd.read_csv(TABLES / "table02_age_did.csv")
    cohort = pd.read_csv(TABLES / "table15_cohort_did.csv")
    extension = pd.read_csv(TABLES / "extension_education_effects.csv")
    temporal = pd.read_csv(TABLES / "extension_year_contrasts.csv")
    support = pd.read_csv(TABLES / "extension_support.csv")

    assert len(age) == 60 and len(cohort) == 20
    assert len(extension) == 24 and len(temporal) == 8 and len(support) == 216
    assert support["n"].min() == 124
    assert np.allclose(support.groupby(["year", "treated"])["share"].sum(), 1)
    assert np.allclose(temporal["coefficient_2025"] - temporal["coefficient_2024"], temporal["coef"])

    for (_, group), rows in extension.groupby(["year", "group"]):
        rows = rows.set_index("outcome")
        assert np.isclose(rows.loc["total", "coef"], rows.loc["formal", "coef"] + rows.loc["informal", "coef"])
    for (_, _), rows in extension.groupby(["year", "outcome"]):
        rows = rows.set_index("group")
        assert np.isclose(rows.loc["difference", "coef"], rows.loc["high", "coef"] - rows.loc["low", "coef"])

    expected = {
        "figure1_cohort_means", "figure2_cohort_did", "figure3_age_did",
        "figure4_informality_components", "extension_education",
        "figureS1_long_run_age_means", "figureS2_alternative_age_cutoffs",
        "figureS3_stock_accounting", "figureS4_age_event_study",
    }
    stems = {p.stem for p in FIGURES.iterdir() if p.is_file()}
    assert expected <= stems

    prohibited = ["mikro_veri_", ".dta", ".sav", ".sas7bdat", "paper_organized", "Asik_LaborEconomics"]
    released = [p for p in ROOT.rglob("*") if p.is_file()]
    paths = "\n".join(str(p.relative_to(ROOT)) for p in released)
    for token in prohibited:
        assert token.lower() not in paths.lower(), token

    code = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (ROOT / "analysis").glob("*.py")
        if p.name != Path(__file__).name
    )
    assert not re.search(r"C:\\\\Users\\\\|/home/|api[_-]?key|password\s*=|secret\s*=", code, re.I)

    report = {
        "status": "passed",
        "csv_tables": len(names),
        "figure_files": len([p for p in FIGURES.iterdir() if p.is_file()]),
        "education_models": 8,
        "temporal_contrasts": 8,
        "composition_support_cells": 216,
        "restricted_microdata_in_release": False,
        "absolute_user_paths_or_secrets_detected": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
