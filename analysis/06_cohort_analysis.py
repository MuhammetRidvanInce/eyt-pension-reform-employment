"""Main fixed-birth-cohort model, pretrend diagnostics and cohort-window sensitivity. Cohorts proxy exposure, not observed pension eligibility."""

from pathlib import Path
import itertools
import os

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = Path(os.environ.get("EYT_DATA_ROOT", PROJECT.parent)).expanduser().resolve()
OUT = PROJECT / "tables"


def load(years):
    cols_new = ["REFERANS_YIL", "BIRIMNO", "CINSIYET", "DOGUM_YIL",
                "YAS", "AGIRLIK_KATSAYISI", "DURUM", "KAYITLILIK",
                "OKUL_BITEN_K", "IBBS_1", "IDO_NEDEN"]
    parts = []
    for year in years:
        path = WORKSPACE / str(year) / "HHIA_Isgucu" / f"mikro_veri_{year}.csv"
        available = pd.read_csv(path, sep=";", encoding="iso-8859-9", nrows=0).columns
        use = [c for c in cols_new if c in available]
        d = pd.read_csv(path, sep=";", encoding="iso-8859-9", usecols=use, low_memory=False)
        for col in use:
            if col != "IBBS_1":
                d[col] = pd.to_numeric(d[col], errors="coerce")
        if "DOGUM_YIL" not in d:
            d["DOGUM_YIL"] = year - d["YAS"]
        d = d[d["CINSIYET"].eq(1) & d["DOGUM_YIL"].between(1972, 1985)].copy()
        d["exposed_cohort"] = d["DOGUM_YIL"].le(1981).astype(int)
        d["total_employment"] = d["DURUM"].eq(1).astype(float) * 100
        d["formal"] = (d["DURUM"].eq(1) & d["KAYITLILIK"].eq(1)).astype(float) * 100
        d["informal"] = (d["DURUM"].eq(1) & d["KAYITLILIK"].eq(2)).astype(float) * 100
        d["retired_inactive"] = (d["DURUM"].eq(3) & d["IDO_NEDEN"].eq(36)).astype(float) * 100
        d["other_status"] = 100 - d["formal"] - d["informal"] - d["retired_inactive"]
        d["secondary_education"] = d["OKUL_BITEN_K"].isin([41, 42]).astype(int)
        d["tertiary_education"] = d["OKUL_BITEN_K"].isin([511, 512, 52]).astype(int)
        d["household_year_id"] = str(year) + "_" + d["BIRIMNO"].astype(str)
        d["cohort_year_id"] = d["DOGUM_YIL"].astype(int).astype(str) + "_" + str(year)
        d["trend"] = year - min(years)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def wild_p(formula_u, formula_r, data, term, cluster, draws=1999):
    yu, xu = patsy.dmatrices(formula_u, data, return_type="dataframe")
    yr, xr = patsy.dmatrices(formula_r, data, return_type="dataframe")
    idx = xu.index.intersection(xr.index)
    xu, xr = xu.loc[idx], xr.loc[idx]
    y = yu.loc[idx].iloc[:, 0].to_numpy()
    weights = data.loc[idx, "AGIRLIK_KATSAYISI"].to_numpy(float)
    sqrt_w = np.sqrt(weights)
    x_u, x_r = xu.to_numpy(), xr.to_numpy()
    beta_r = np.linalg.lstsq(x_r * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
    fitted = x_r @ beta_r
    resid = y - fitted
    term_index = xu.columns.get_loc(term)
    labels = cluster.loc[idx].astype(str).to_numpy()
    unique = np.unique(labels)
    if len(unique) <= 12:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(unique))))
    else:
        signs = np.random.default_rng(2403).choice((-1.0, 1.0), size=(draws, len(unique)))
    positions = {label: pos for pos, label in enumerate(unique)}
    cluster_index = np.asarray([positions[label] for label in labels])

    xtwx_inv = np.linalg.pinv(x_u.T @ (weights[:, None] * x_u))
    influence = (xtwx_inv @ (x_u.T * weights))[term_index]
    observed = abs(influence @ y)
    fitted_part = influence @ fitted
    cluster_scores = np.asarray([
        np.sum(influence[cluster_index == pos] * resid[cluster_index == pos])
        for pos in range(len(unique))
    ])
    boot = np.abs(fitted_part + signs @ cluster_scores)
    return float((np.sum(boot >= observed) + 1) / (len(boot) + 1))


def main():
    new_all = load([2022, 2024, 2025])
    new = new_all[new_all["DOGUM_YIL"].between(1975, 1984)].copy()
    labels = {"total_employment": "Total employment", "formal": "Formal employment",
              "informal": "Informal employment", "retired_inactive": "Retired and outside the labor force",
              "other_status": "Unemployed or otherwise inactive"}
    rows = []
    for post_year in (2024, 2025):
        data = new[new["REFERANS_YIL"].isin([2022, post_year])].copy()
        data["post"] = data["REFERANS_YIL"].eq(post_year).astype(int)
        for outcome, label in labels.items():
            specifications = [
                ("Cohort and year FE",
                 f"{outcome} ~ C(DOGUM_YIL) + C(REFERANS_YIL) + exposed_cohort:post",
                 f"{outcome} ~ C(DOGUM_YIL) + C(REFERANS_YIL)"),
                ("Cohort/year FE + education + region",
                 f"{outcome} ~ C(DOGUM_YIL) + C(REFERANS_YIL) + exposed_cohort:post + secondary_education + tertiary_education + C(IBBS_1)",
                 None),
            ]
            for specification, formula_u, formula_r in specifications:
                model = smf.wls(formula_u, data=data, weights=data["AGIRLIK_KATSAYISI"]).fit(
                    cov_type="cluster", cov_kwds={"groups": data["household_year_id"]})
                term = "exposed_cohort:post"
                ci = model.conf_int().loc[term]
                use_wild = outcome == "informal" and formula_r is not None
                p_birth = wild_p(formula_u, formula_r, data, term, data["DOGUM_YIL"]) if use_wild else np.nan
                p_birth_year = wild_p(formula_u, formula_r, data, term, data["cohort_year_id"]) if use_wild else np.nan
                rows.append({"Period": f"{post_year} vs 2022", "Specification": specification,
                             "Outcome": label, "coef": model.params[term],
                             "se_household": model.bse[term], "p_household": model.pvalues[term],
                             "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "n": int(model.nobs),
                             "p_wild_birthyear": p_birth,
                             "p_wild_birthyear_year": p_birth_year})

    old_all = load([2018, 2019, 2020])
    old = old_all[old_all["DOGUM_YIL"].between(1975, 1984)].copy()
    pre_rows = []
    for outcome, label in labels.items():
        formula = f"{outcome} ~ C(DOGUM_YIL) + C(REFERANS_YIL) + exposed_cohort:trend"
        model = smf.wls(formula, data=old, weights=old["AGIRLIK_KATSAYISI"]).fit(
            cov_type="cluster", cov_kwds={"groups": old["household_year_id"]})
        term = "exposed_cohort:trend"
        ci = model.conf_int().loc[term]
        pre_rows.append({"Test": "2018-2020 linear differential cohort trend", "Outcome": label,
                         "coef": model.params[term], "se_household": model.bse[term],
                         "p_household": model.pvalues[term], "ci_low": ci.iloc[0],
                         "ci_high": ci.iloc[1], "n": int(model.nobs)})

    sensitivity = []
    for lower, cutoff, upper in [(1972, 1980, 1983), (1972, 1981, 1984),
                                  (1972, 1982, 1985), (1975, 1981, 1984),
                                  (1978, 1981, 1984)]:
        sample = new_all[new_all["DOGUM_YIL"].between(lower, upper)].copy()
        sample["tedavi_alt"] = sample["DOGUM_YIL"].le(cutoff).astype(int)
        for post_year in (2024, 2025):
            data = sample[sample["REFERANS_YIL"].isin([2022, post_year])].copy()
            data["post"] = data["REFERANS_YIL"].eq(post_year).astype(int)
            model = smf.wls("informal ~ C(DOGUM_YIL) + C(REFERANS_YIL) + tedavi_alt:post",
                            data=data, weights=data["AGIRLIK_KATSAYISI"]).fit(
                                cov_type="cluster", cov_kwds={"groups": data["household_year_id"]})
            term = "tedavi_alt:post"
            ci = model.conf_int().loc[term]
            sensitivity.append({"Earliest birth year": lower, "Latest higher-exposure birth year": cutoff,
                                "Latest comparison birth year": upper, "Period": f"{post_year} vs 2022",
                                "coef": model.params[term], "se_household": model.bse[term],
                                "p_household": model.pvalues[term], "ci_low": ci.iloc[0],
                                "ci_high": ci.iloc[1], "n": int(model.nobs)})

    cells = []
    for (year, treated), group in new.groupby(["REFERANS_YIL", "exposed_cohort"]):
        row = {"Year": int(year), "Cohort": "1975-1981" if treated else "1982-1984",
               "N": len(group)}
        for outcome, label in labels.items():
            row[label] = np.average(group[outcome], weights=group["AGIRLIK_KATSAYISI"])
        cells.append(row)

    pd.DataFrame(rows).to_csv(OUT / "table15_cohort_did.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pre_rows).to_csv(OUT / "table16_cohort_pretrends.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(OUT / "table17_cohort_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cells).to_csv(OUT / "table18_cohort_means.csv", index=False, encoding="utf-8-sig")
    print("Fixed-cohort analysis completed.")


if __name__ == "__main__":
    main()
