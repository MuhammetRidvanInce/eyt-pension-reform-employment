# Cohort informality release October 2026

Replication materials for “Türkiye's EYT Pension Reform and Informal Employment: Evidence from Birth Cohorts”, by Muhammet Rıdvan İnce.

## Scope

The main analysis uses the 2021–2025 TurkStat Household Labour Force Survey and compares men born in 1975–1981 with those born in 1982–1984 (reference year 2022; N = 217,716). The separately estimated 2014–2020 diagnostic and 2021–2025 comparison use approximate birth cohorts derived from survey year minus completed age. No model crosses the 2021 measurement break. Cohorts proxy for exposure, not verified eligibility or beneficiary status. The current manuscript focuses on informality; the repository's older fixed-age and other extensions are not this release's main analysis.

## Restricted data and configuration

This release contains code, aggregate estimates and figures only. No individual or household records are distributed. Researchers must obtain the relevant microdata independently from TurkStat under its access and reuse conditions: https://www.tuik.gov.tr/Kurumsal/Mikro_Veri . The author cannot redistribute the underlying microdata.

Copy `config/hlfs_paths.example.json` to a private location and replace each value with the licensed CSV file for that year. Paths can be absolute or relative to the configuration file. Set `EYT_HLFS_CONFIG` to the private configuration file. Do not commit that file or any microdata. The selected input hashes are in `config/input_checksums.json`; the CSV reader detects comma/semicolon delimiters and uses ISO-8859-9 encoding. Recent input columns are defined in `28_cohort_informality.py`; historical input columns are defined in `30_cohort_history.py`. The 2024 and 2025 inputs are the revised files used in the October 2026 analysis.

## Reproduction

From this release directory, install the recorded dependencies and run:

```text
python -m pip install -r requirements.txt
python analysis/codes/28_cohort_informality.py
python analysis/codes/30_cohort_history.py
python analysis/codes/35_english_figures.py
python validate_release.py
```

PowerShell configuration example:

```powershell
$env:EYT_HLFS_CONFIG = 'D:\private\hlfs_paths.json'
```

Without access to the microdata, figures can still be regenerated from the supplied aggregate results using `python analysis/codes/35_english_figures.py`; `python validate_release.py` also works without microdata.

## Output map

All current outputs are in `analysis/outputs/cohort_informality_2026-10-15/`. `sample_audit.csv` supplies Table 1; `group_means.csv` supplies Table 2; `dynamic_results.csv` supplies Table 3 and Figure 1; `sensitivity_results.csv` supplies Table 4; `informality_partitions.csv` supplies Figure 2; `historical_results.csv` supplies Figure 3. `time_contrasts.csv` contains coefficient differences. `reference_results/` preserves the original aggregate outputs for numerical comparison.

Confidence intervals use birth-cohort clustered CR1 covariance and t critical values. Bootstrap p-values are null-restricted and studentized, using exhaustive Rademacher weights and 99,999 Webb draws (seed 20261015); p-values are unadjusted. The portable release changes input resolution and removes local manuscript-backup/legacy-output dependencies; estimation equations, samples, seeds and inference calculations are retained.

Running estimation overwrites aggregate output files and creates local validation and source-manifest files. Generated source manifests contain local input paths and are ignored by Git. Input files are read only. The public validation script compares numeric columns with the reference tables; it does not assert that microdata are publicly available.

## Author declarations

The author reports no competing interests and no specific research funding. GPT Astra was used for language, translation and grammar support and to assist with testing analysis code. The author is responsible for the analysis, interpretation and final materials. No new license is granted by this update; the repository's existing reuse terms remain applicable.
