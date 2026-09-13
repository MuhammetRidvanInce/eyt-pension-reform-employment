"""Fixed-age DiD, gender DDD, local-age diagnostics and exploratory decompositions. Raw TurkStat columns retain their official names."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
import os

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = Path(os.environ.get("EYT_DATA_ROOT", PROJECT.parent)).expanduser().resolve()
OUT = PROJECT / "tables"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = (2022, 2024, 2025)
RAW = {y: WORKSPACE / str(y) / "HHIA_Isgucu" / f"mikro_veri_{y}.csv" for y in YEARS}
COLS = [
    "REFERANS_YIL", "BIRIMNO", "FERTNO", "CINSIYET", "YAS", "DOGUM_YIL",
    "IBBS_1", "OKUL_BITEN_K", "AGIRLIK_KATSAYISI", "CALISMA", "KAYITLILIK",
    "IS_BASLAMA_YIL", "IS_AYRIL_YIL", "IS_AYRIL_NEDEN", "DURUM", "IDO_NEDEN",
    "OZEL_KAMU", "ISCO08_ESAS_K", "NACE2_ESAS_K", "CALISAN_SAYI_HH",
    "CALISMA_SEKLI", "ISTEKI_DURUM_K",
]
NUMERIC = [c for c in COLS if c not in {"IBBS_1"}]


def load_year(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="iso-8859-9") as fh:
        header = [x.strip(' \t\r\n"') for x in fh.readline().strip().split(";")]
    use = [c for c in COLS if c in header]
    d = pd.read_csv(path, sep=";", names=header, skiprows=1, usecols=use,
                    encoding="iso-8859-9", low_memory=False)
    for c in NUMERIC:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["IBBS_1"] = d["IBBS_1"].astype(str).str.strip()
    return d


def weighted_mean(d: pd.DataFrame, col: str) -> float:
    ok = d[col].notna() & d["AGIRLIK_KATSAYISI"].notna()
    return float(np.average(d.loc[ok, col], weights=d.loc[ok, "AGIRLIK_KATSAYISI"]))


def fit_model(formula: str, d: pd.DataFrame, term: str) -> dict:
    m = smf.wls(formula, data=d, weights=d["AGIRLIK_KATSAYISI"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["household_year_id"]}
    )
    ci = m.conf_int().loc[term]
    return {
        "coef": float(m.params[term]), "se_household": float(m.bse[term]),
        "p_household": float(m.pvalues[term]), "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]), "n": int(m.nobs), "r2": float(m.rsquared),
    }


def wild_signflip_p(formula_u: str, formula_r: str, d: pd.DataFrame,
                    term: str, cluster: pd.Series, draws: int = 9999) -> float:
    """Restricted-residual wild cluster sign-flip coefficient test.

    This is deliberately reported as a sensitivity p-value, not as an exact
    design-based test. For <=12 clusters every Rademacher sign vector is used.
    """
    yu, xu = patsy.dmatrices(formula_u, d, return_type="dataframe")
    yr, xr = patsy.dmatrices(formula_r, d, return_type="dataframe")
    idx = yu.index.intersection(yr.index)
    yu, xu, xr = yu.loc[idx], xu.loc[idx], xr.loc[idx]
    w = d.loc[idx, "AGIRLIK_KATSAYISI"].to_numpy(float)
    y = yu.iloc[:, 0].to_numpy(float)
    x = xu.to_numpy(float)
    z = xr.to_numpy(float)
    inv_x = np.linalg.pinv(x.T @ (w[:, None] * x))
    b_u = inv_x @ (x.T @ (w * y))
    b_r = np.linalg.pinv(z.T @ (w[:, None] * z)) @ (z.T @ (w * y))
    resid = y - z @ b_r
    j = list(xu.columns).index(term)
    labels = cluster.loc[idx].astype(str).to_numpy()
    uniq, codes = np.unique(labels, return_inverse=True)
    scores = np.zeros((len(uniq), x.shape[1]))
    np.add.at(scores, codes, x * (w * resid)[:, None])
    influence = scores @ inv_x[j, :]
    observed = abs(float(b_u[j]))
    if len(uniq) <= 12:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(uniq))))
    else:
        rng = np.random.default_rng(7438)
        signs = rng.choice((-1.0, 1.0), size=(draws, len(uniq)))
    boot = np.abs(signs @ influence)
    return float((np.sum(boot >= observed - 1e-12) + 1) / (len(boot) + 1))


def add_outcomes(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    employed = d["DURUM"].eq(1)
    d["employment"] = employed.astype(float) * 100
    d["formal"] = (employed & d["KAYITLILIK"].eq(1)).astype(float) * 100
    d["informal"] = (employed & d["KAYITLILIK"].eq(2)).astype(float) * 100
    d["retired_inactive"] = (d["DURUM"].eq(3) & d["IDO_NEDEN"].eq(36)).astype(float) * 100
    d["other_status"] = 100 - d["formal"] - d["informal"] - d["retired_inactive"]
    d["male"] = d["CINSIYET"].eq(1).astype(int)
    d["female"] = d["CINSIYET"].eq(2).astype(int)
    edu = d["OKUL_BITEN_K"]
    d["secondary_education"] = edu.isin([41, 42]).astype(int)
    d["tertiary_education"] = edu.isin([511, 512, 52]).astype(int)
    d["household_year_id"] = d["REFERANS_YIL"].astype(int).astype(str) + "_" + d["BIRIMNO"].astype(str)
    return d


def main_did(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = {"employment": "Total employment", "formal": "Formal employment",
              "informal": "Informal employment", "retired_inactive": "Retired and outside the labor force",
              "other_status": "Unemployed or otherwise inactive"}
    cells, results = [], []
    for post_year in (2024, 2025):
        base = df[df["YAS"].between(40, 49) & df["REFERANS_YIL"].isin([2022, post_year])].copy()
        base["exposed"] = base["YAS"].between(43, 49).astype(int)
        base["post"] = base["REFERANS_YIL"].eq(post_year).astype(int)
        base["age_year"] = base["YAS"].astype(int).astype(str) + "_" + base["REFERANS_YIL"].astype(int).astype(str)
        for sex_name, sample in [("All", base), ("Men", base[base["male"].eq(1)]),
                                 ("Women", base[base["female"].eq(1)])]:
            for (year, treat), z in sample.groupby(["REFERANS_YIL", "exposed"]):
                row = {"Period": f"{post_year} vs 2022", "Sample": sex_name,
                       "Year": int(year), "Group": "43-49" if treat else "40-42",
                       "N": len(z), "Weighted population (thousands)": z["AGIRLIK_KATSAYISI"].sum()}
                row.update({labels[v]: weighted_mean(z, v) for v in labels})
                cells.append(row)
            for var, lab in labels.items():
                specs = [
                    ("Unadjusted 2x2 DiD", f"{var} ~ exposed * post", "exposed:post"),
                    ("Age FE + demographics + region", f"{var} ~ C(YAS) + C(REFERANS_YIL) + exposed:post + secondary_education + tertiary_education + C(IBBS_1)" + (" + female" if sex_name == "All" else ""), "exposed:post"),
                ]
                for spec, formula, term in specs:
                    est = fit_model(formula, sample, term)
                    row = {"Period": f"{post_year} vs 2022", "Sample": sex_name,
                           "Specification": spec, "Outcome": lab, **est}
                    if spec == "Unadjusted 2x2 DiD":
                        row["p_wild_age"] = wild_signflip_p(
                            f"{var} ~ exposed * post", f"{var} ~ exposed + post", sample,
                            term, sample["YAS"].astype(int).astype(str))
                        row["p_wild_age_year"] = wild_signflip_p(
                            f"{var} ~ exposed * post", f"{var} ~ exposed + post", sample,
                            term, sample["age_year"])
                    results.append(row)
    return pd.DataFrame(cells), pd.DataFrame(results)


def sex_difference(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for post_year in (2024, 2025):
        d = df[df["YAS"].between(40, 49) & df["REFERANS_YIL"].isin([2022, post_year])].copy()
        d["exposed"] = d["YAS"].between(43, 49).astype(int)
        d["post"] = d["REFERANS_YIL"].eq(post_year).astype(int)
        for var, lab in [("formal", "Formal employment"), ("informal", "Informal employment"),
                         ("retired_inactive", "Retired and outside the labor force")]:
            formula = f"{var} ~ exposed * post * male"
            est = fit_model(formula, d, "exposed:post:male")
            rows.append({"Period": f"{post_year} vs 2022", "Outcome": lab,
                         "Parameter": "Male DiD - female DiD", **est})
    return pd.DataFrame(rows)


def rdid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for post_year in (2024, 2025):
        males = df[df["male"].eq(1) & df["REFERANS_YIL"].isin([2022, post_year])].copy()
        males["post"] = males["REFERANS_YIL"].eq(post_year).astype(int)
        for lo, hi in [(40, 45), (38, 47), (35, 50)]:
            d = males[males["YAS"].between(lo, hi)].copy()
            d["above"] = d["YAS"].ge(43).astype(int)
            d["run"] = d["YAS"] - 42.5
            d["age_year"] = d["YAS"].astype(int).astype(str) + "_" + d["REFERANS_YIL"].astype(int).astype(str)
            for var, lab in [("formal", "Formal employment"), ("informal", "Informal employment"),
                             ("retired_inactive", "Retired and outside the labor force")]:
                fu = f"{var} ~ post + above + run + above:run + post:run + post:above + post:above:run"
                fr = f"{var} ~ post + above + run + above:run + post:run + post:above:run"
                est = fit_model(fu, d, "post:above")
                est["p_wild_age_year"] = wild_signflip_p(fu, fr, d, "post:above", d["age_year"])
                rows.append({"Period": f"{post_year} vs 2022", "Sample": "Men",
                             "Window": f"{lo}-{hi}", "Outcome": lab, **est})
    return pd.DataFrame(rows)


def placebo_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[df["male"].eq(1) & df["REFERANS_YIL"].isin([2022, 2024])].copy()
    base["post"] = base["REFERANS_YIL"].eq(2024).astype(int)
    for cutoff in range(36, 49):
        d = base[base["YAS"].between(cutoff - 3, cutoff + 2)].copy()
        d["above"] = d["YAS"].ge(cutoff).astype(int)
        d["run"] = d["YAS"] - (cutoff - 0.5)
        for var, lab in [("formal", "Formal"), ("informal", "Informal"), ("retired_inactive", "Retired, outside labor force")]:
            f = f"{var} ~ post + above + run + above:run + post:run + post:above + post:above:run"
            est = fit_model(f, d, "post:above")
            rows.append({"Cutoff": cutoff, "Candidate cutoff at age 43": cutoff == 43, "Outcome": lab, **est})
    return pd.DataFrame(rows)


def mechanism_components(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for post_year in (2024, 2025):
        d = df[df["YAS"].between(40, 49) & df["REFERANS_YIL"].isin([2022, post_year])].copy()
        d["exposed"] = d["YAS"].between(43, 49).astype(int)
        d["post"] = d["REFERANS_YIL"].eq(post_year).astype(int)
        informal = d["DURUM"].eq(1) & d["KAYITLILIK"].eq(2)
        size, nace, status = d["CALISAN_SAYI_HH"], d["NACE2_ESAS_K"], d["ISTEKI_DURUM_K"]
        components = {
            "Firm: 1-9": informal & size.eq(1), "Firm: 10-49": informal & size.isin([2, 3]),
            "Firm: 50+ (known size)": informal & size.isin([4, 5]),
            "Firm: 10+ (size unknown)": informal & size.eq(6),
            "Sector: agriculture": informal & nace.le(3), "Sector: non-agriculture": informal & nace.ge(5),
            "Status: wage/casual employee": informal & status.isin([11, 12]),
            "Status: employer": informal & status.eq(2), "Status: own-account worker": informal & status.eq(3),
            "Status: unpaid family worker": informal & status.eq(4),
            "Narrow component: private non-agricultural wage employment": informal & d["OZEL_KAMU"].eq(2) & nace.ge(5) & status.isin([11, 12]),
        }
        for name, mask in components.items():
            col = "component"
            d[col] = mask.astype(float) * 100
            for sex_name, z in [("All", d), ("Men", d[d["male"].eq(1)])]:
                est = fit_model(f"{col} ~ exposed * post", z, "exposed:post")
                pre = z[z["exposed"].eq(1) & z["post"].eq(0)]
                rows.append({"Period": f"{post_year} vs 2022", "Component": name,
                             "Sample": sex_name,
                             "2022 higher-exposure mean": weighted_mean(pre, col), **est})
    return pd.DataFrame(rows)


def stock_accounting(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["DURUM"].eq(1) & df["YAS"].between(40, 55)].copy()
    d["cohort_exact"] = d["REFERANS_YIL"] - d["YAS"]
    d["long_current_job"] = d["IS_BASLAMA_YIL"].le(1999)
    d["formal"] = d["KAYITLILIK"].eq(1)
    rows = []
    groups = {
        "All ages 40-55": pd.Series(True, index=d.index),
        "Common cohort 1970-1982": d["cohort_exact"].between(1970, 1982),
        "1967-1969 cohorts leaving the 2022 age window": d["cohort_exact"].between(1967, 1969),
        "1983-1985 cohorts entering the 2025 age window": d["cohort_exact"].between(1983, 1985),
    }
    statuses = {
        "Long current-job tenure, formal": d["long_current_job"] & d["formal"],
        "Short current-job tenure, formal": ~d["long_current_job"] & d["formal"],
        "Total formal": d["formal"], "Total informal": ~d["formal"],
    }
    for gname, gm in groups.items():
        for sname, sm in statuses.items():
            z = d[gm & sm]
            vals = {y: float(z.loc[z["REFERANS_YIL"].eq(y), "AGIRLIK_KATSAYISI"].sum()) for y in YEARS}
            rows.append({"Cohort/window": gname, "Status": sname,
                         "2022 stock (thousands)": vals[2022], "2024 stock (thousands)": vals[2024],
                         "2025 stock (thousands)": vals[2025], "2022-2025 change (thousands)": vals[2025] - vals[2022]})
    return pd.DataFrame(rows)


def retrospective(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["IS_AYRIL_YIL"].notna()].copy()
    d = d[(d["REFERANS_YIL"] - d["IS_AYRIL_YIL"]).between(0, 7)]
    d["sure"] = d["REFERANS_YIL"] - d["IS_AYRIL_YIL"]
    d["yas_ayrilma"] = d["YAS"] - d["sure"]
    d = d[d["yas_ayrilma"].between(30, 54) & d["male"].eq(1)].copy()
    d["olgun"] = d["yas_ayrilma"].between(43, 54).astype(int)
    d["emeklilik_nedeni"] = d["IS_AYRIL_NEDEN"].eq(9).astype(float) * 100
    comparisons = [(1, 2021, 2023), (1, 2021, 2024), (2, 2022, 2023),
                   (2, 2020, 2022), (3, 2019, 2021)]
    rows = []
    for sure, y0, y1 in comparisons:
        z = d[d["sure"].eq(sure) & d["IS_AYRIL_YIL"].isin([y0, y1])].copy()
        z["post_cohort"] = z["IS_AYRIL_YIL"].eq(y1).astype(int)
        est = fit_model("emeklilik_nedeni ~ olgun * post_cohort + secondary_education + tertiary_education + C(IBBS_1)", z, "olgun:post_cohort")
        rows.append({"Recall interval": sure, "Comparison": f"{y0} vs {y1}",
                     "Reform comparison": y1 >= 2023 and y0 < 2023, **est})
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def write_report(cells, main_res, sex, rd, mech, stock, retro):
    """Export a compact technical report; the manuscript is maintained separately."""
    lines = ["# Fixed-age analysis: technical output", "",
             "Men aged 43-49 are compared with men aged 40-42. Estimates are percentage points.",
             "Age groups proxy reform exposure; actual eligibility and individual transitions are unobserved.", ""]
    selected = main_res[(main_res["Sample"] == "Men") &
                        (main_res["Specification"] == "Unadjusted 2x2 DiD")]
    lines.extend(["| Period | Outcome | Coefficient | Household-clustered SE | p-value |",
                  "|---|---|---:|---:|---:|"])
    for _, row in selected.iterrows():
        lines.append(f"| {row['Period']} | {row['Outcome']} | {row['coef']:.3f} | {row['se_household']:.3f} | {row['p_household']:.4f} |")
    (PROJECT / "reports").mkdir(exist_ok=True)
    (PROJECT / "reports" / "age_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    missing = [str(p) for p in RAW.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing raw data: " + ", ".join(missing))
    df = add_outcomes(pd.concat([load_year(RAW[y]) for y in YEARS], ignore_index=True))
    cells, main_res = main_did(df)
    sex = sex_difference(df)
    rd = rdid(df)
    placebo = placebo_cutoffs(df)
    mech = mechanism_components(df)
    stock = stock_accounting(df)
    retro = retrospective(df)
    identity = df["formal"] + df["informal"] + df["retired_inactive"] + df["other_status"]
    if not np.allclose(identity, 100.0):
        raise AssertionError("Labor force status shares do not sum to 100.")
    for period in ("2024 vs 2022", "2025 vs 2022"):
        z = main_res[(main_res["Period"].eq(period)) & (main_res["Sample"].eq("Men")) &
                     (main_res["Specification"].eq("Unadjusted 2x2 DiD"))]
        total = float(z.loc[z["Outcome"].eq("Formal employment"), "coef"].iloc[0]) + float(z.loc[z["Outcome"].eq("Informal employment"), "coef"].iloc[0])
        employment = float(z.loc[z["Outcome"].eq("Total employment"), "coef"].iloc[0])
        if not np.isclose(total, employment, atol=1e-8):
            raise AssertionError(f"Employment accounting identity failed: {period}")
    outputs = {
        "table01_age_group_means.csv": cells,
        "table02_age_did.csv": main_res,
        "table03_gender_ddd.csv": sex,
        "table04_local_age_discontinuity.csv": rd,
        "table05_alternative_age_cutoffs.csv": placebo,
        "table06_informality_components.csv": mech,
        "table07_stock_accounting.csv": stock,
        "table08_retrospective_men.csv": retro,
    }
    for name, table in outputs.items():
        table.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    write_report(cells, main_res, sex, rd, mech, stock, retro)
    print(f"Completed: {len(df):,} individual records; {len(outputs)} tables and technical report generated.")


if __name__ == "__main__":
    main()
