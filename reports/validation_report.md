# Release validation

The public replication package was assembled from the English Project VII analysis files. It excludes restricted TurkStat microdata, current-job-tenure experiments, document-build utilities, copyrighted reference PDFs, author-only files, and temporary compilation artifacts.

The retained English outputs were previously matched numerically to the Turkish Project VI outputs. The added education, composition, and temporal-contrast results were independently generated in both projects and the six CSV files matched exactly. Weighted least-squares coefficients and household-clustered covariance for the extension were also cross-checked against `statsmodels`.

Run:

```bash
python analysis/validate_release.py
```

The validation checks the expected table structure, education accounting identities, the shared-baseline temporal contrasts, education–region support, figure presence, and absence of restricted-data filenames, absolute user paths, and common secret patterns. It does not independently validate TurkStat source data or replace substantive review of the identification assumptions.
