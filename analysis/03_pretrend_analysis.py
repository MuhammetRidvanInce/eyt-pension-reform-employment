"""Pretrends and placebo diagnostics. Treat the pre-2021 and post-2021 HLFS regimes separately."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = Path(os.environ.get("EYT_DATA_ROOT", PROJECT.parent)).expanduser().resolve()
OUT = PROJECT / "tables"
YEARS = (2018, 2019, 2020, 2022, 2024, 2025)
RAW = {y: WORKSPACE / str(y) / "HHIA_Isgucu" / f"mikro_veri_{y}.csv" for y in YEARS}
USE = ["BIRIMNO", "CINSIYET", "YAS", "AGIRLIK_KATSAYISI", "KAYITLILIK", "DURUM", "IDO_NEDEN"]
LABELS = {
    "employment": "Total employment", "formal": "Formal employment",
    "informal": "Informal employment", "retired_inactive": "Retired and outside the labor force",
}


def load_sample(year: int) -> pd.DataFrame:
    d = pd.read_csv(RAW[year], sep=";", encoding="iso-8859-9", usecols=USE, low_memory=False)
    for c in USE:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["YAS"].between(40, 49) & d["CINSIYET"].isin([1, 2])].copy()
    employed = d["DURUM"].eq(1)
    d["employment"] = employed.astype(float) * 100
    d["formal"] = (employed & d["KAYITLILIK"].eq(1)).astype(float) * 100
    d["informal"] = (employed & d["KAYITLILIK"].eq(2)).astype(float) * 100
    d["retired_inactive"] = (d["DURUM"].eq(3) & d["IDO_NEDEN"].eq(36)).astype(float) * 100
    d["year"] = year
    d["exposed"] = d["YAS"].between(43, 49).astype(int)
    d["male"] = d["CINSIYET"].eq(1).astype(int)
    d["household_year_id"] = str(year) + "_" + d["BIRIMNO"].astype(str)
    return d


def wmean(d: pd.DataFrame, col: str) -> float:
    ok = d[col].notna() & d["AGIRLIK_KATSAYISI"].notna()
    return float(np.average(d.loc[ok, col], weights=d.loc[ok, "AGIRLIK_KATSAYISI"]))


def fit(formula: str, d: pd.DataFrame, term: str) -> dict:
    model = smf.wls(formula, data=d, weights=d["AGIRLIK_KATSAYISI"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["household_year_id"]}
    )
    ci = model.conf_int().loc[term]
    return {
        "coef": float(model.params[term]), "se_household": float(model.bse[term]),
        "p_household": float(model.pvalues[term]), "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]), "n": int(model.nobs), "r2": float(model.rsquared),
    }


def long_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, sex, treat), z in df.groupby(["year", "male", "exposed"]):
        row = {
            "Year": int(year), "Sample": "Men" if sex else "Women",
            "Group": "43-49" if treat else "40-42", "N": len(z),
            "Weighted population (thousands)": float(z["AGIRLIK_KATSAYISI"].sum()),
            "Survey regime": "Pre-2021 HLFS regime" if year <= 2020 else "Post-2021 HLFS regime",
        }
        row.update({label: wmean(z, var) for var, label in LABELS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def male_event_study(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["male"].eq(1)].copy()
    event_years = [2018, 2019, 2022, 2024, 2025]
    for year in event_years:
        d[f"event_{year}"] = d["exposed"] * d["year"].eq(year).astype(int)
    terms = " + ".join(f"event_{year}" for year in event_years)
    rows = []
    for var, label in LABELS.items():
        formula = f"{var} ~ C(YAS) + C(year) + {terms}"
        model = smf.wls(formula, data=d, weights=d["AGIRLIK_KATSAYISI"]).fit(
            cov_type="cluster", cov_kwds={"groups": d["household_year_id"]}
        )
        for year in event_years:
            term = f"event_{year}"
            ci = model.conf_int().loc[term]
            rows.append({
                "Outcome": label, "Year": year, "Reference year": 2020,
                "Methodological comparability": "Same regime" if year <= 2020 else "Across regimes; interpret cautiously",
                "coef": float(model.params[term]), "se_household": float(model.bse[term]),
                "p_household": float(model.pvalues[term]), "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]), "n": int(model.nobs), "r2": float(model.rsquared),
            })
        rows.append({
            "Outcome": label, "Year": 2020, "Reference year": 2020,
            "Methodological comparability": "Reference", "coef": 0.0,
            "se_household": 0.0, "p_household": np.nan, "ci_low": 0.0,
            "ci_high": 0.0, "n": int(model.nobs), "r2": float(model.rsquared),
        })
    result = pd.DataFrame(rows)
    # Preserve VI's published row order rather than sorting translated labels.
    order = {name: i for i, name in enumerate([
        "Informal employment", "Formal employment", "Total employment",
        "Retired and outside the labor force"])}
    return result.assign(_order=result["Outcome"].map(order)).sort_values(
        ["_order", "Year"]).drop(columns="_order")


def pretrend_tests(df: pd.DataFrame) -> pd.DataFrame:
    old = df[df["year"].isin([2018, 2019, 2020])].copy()
    old["trend"] = old["year"] - 2018
    rows = []
    for sex_name, sample in [("Men", old[old["male"].eq(1)]), ("Women", old[old["male"].eq(0)])]:
        for var, label in LABELS.items():
            est = fit(f"{var} ~ exposed * trend", sample, "exposed:trend")
            rows.append({"Test": "2018-2020 linear differential trend", "Sample": sex_name,
                         "Outcome": label, "Parameter": "Annual higher-exposure minus comparison trend", **est})
    # Linear male-female age differential within the old survey regime.
    for var, label in LABELS.items():
        est = fit(f"{var} ~ exposed * trend * male", old, "exposed:trend:male")
        rows.append({"Test": "2018-2020 linear DDD pretrend", "Sample": "All",
                     "Outcome": label, "Parameter": "Male-female differential trend", **est})
    # Pairwise placebos; 2020-2022 is a diagnostic across survey regimes.
    for y0, y1, kind in [(2018, 2019, "Within-regime placebo"), (2019, 2020, "Within-regime placebo"),
                         (2020, 2022, "Survey-break diagnostic")]:
        z = df[df["male"].eq(1) & df["year"].isin([y0, y1])].copy()
        z["post"] = z["year"].eq(y1).astype(int)
        for var, label in LABELS.items():
            est = fit(f"{var} ~ exposed * post", z, "exposed:post")
            rows.append({"Test": f"{y1} vs {y0}: {kind}", "Sample": "Men",
                         "Outcome": label, "Parameter": "DiD", **est})
    return pd.DataFrame(rows)


def main() -> None:
    missing = [str(path) for path in RAW.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing raw data: " + ", ".join(missing))
    df = pd.concat([load_sample(year) for year in YEARS], ignore_index=True)
    outputs = {
        "table09_long_run_age_means.csv": long_cells(df),
        "table10_age_event_study.csv": male_event_study(df),
        "table11_pretrends_and_placebos.csv": pretrend_tests(df),
    }
    for name, table in outputs.items():
        table.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    print(f"Pretrend module completed: {len(df):,} observations; {len(outputs)} tables.")


if __name__ == "__main__":
    main()
