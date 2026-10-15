# Validation on 15 October 2026

The portable release was run against all selected licensed input files. Both `28_cohort_informality.py` and `30_cohort_history.py` completed successfully. No raw files were changed. The eight aggregate CSV files matched the archived reference outputs using relative tolerance 1e-7 and absolute tolerance 1e-8. The largest difference in main coefficients was 4.44e-16. The main sample contained 217,716 observations.

The estimation script also cross-checked coefficients and cluster covariance against statsmodels, compared the baseline individual model to weighted cohort-year cells, checked bootstrap computations against direct refits, and checked that employment-status components add to total informality. The historical models were estimated separately on each side of the 2021 measurement break.

The English figures were regenerated with the portable release. All three PNGs were pixel-identical to the manuscript figures. This is computational validation, not verification of causal identification assumptions.

Input paths and individual or household records are not included in this repository. Source-manifest files generated locally are Git-ignored.
