# Aggregate result files

All CSV files contain aggregate estimates or diagnostics. They do not contain person-level records.

| Computational output | Contents |
|---|---|
| `table01_age_group_means.csv` | Weighted age-group outcome shares |
| `table02_age_did.csv` | Fixed-age DiD estimates |
| `table03_gender_ddd.csv` | Male-minus-female triple differences |
| `table04_local_age_discontinuity.csv` | Local age diagnostics |
| `table05_alternative_age_cutoffs.csv` | Alternative candidate age cutoffs |
| `table06_informality_components.csv` | Firm-size, sector, and employment-status components |
| `table07_stock_accounting.csv` | Repeated-cross-section stock accounting |
| `table08_retrospective_men.csv` | Retrospective job-exit comparisons |
| `table09_long_run_age_means.csv` | Age-group means across survey regimes |
| `table10_age_event_study.csv` | Discontinuous age-profile diagnostics |
| `table11_pretrends_and_placebos.csv` | Earlier-regime pretrend and placebo results |
| `table15_cohort_did.csv` | Main fixed-cohort estimates and wild p-values |
| `table16_cohort_pretrends.csv` | Earlier-regime fixed-cohort trends |
| `table17_cohort_sensitivity.csv` | Alternative fixed-cohort definitions |
| `table18_cohort_means.csv` | Weighted fixed-cohort outcome shares |
| `extension_composition.csv` | Education and NUTS-1 composition |
| `extension_education_effects.csv` | Education-group estimates and direct contrasts |
| `extension_standardization.csv` | Descriptive education–region standardization |
| `extension_support.csv` | Support-cell diagnostics |
| `extension_year_contrasts.csv` | Direct 2025-minus-2024 coefficient tests |
| `extension_education_codes.csv` | Education-code audit |
| `extension_audit.json` | Reproduction metadata and source hashes |

Computational numbering differs from the sequential table numbering in the article because source outputs were preserved for auditability. Coefficients are generally expressed in percentage points. Column names indicate household-clustered standard errors and p-values or the relevant wild-inference grouping level. Empty wild-p-value cells are intentional when that supplementary procedure was not run.

Current-job-tenure tables 12–14 are excluded because those analyses are not part of the final article and current-job start year does not measure first social insurance registration.
