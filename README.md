# EYT Pension Reform and Employment

Replication materials for the study **“Formal and Informal Employment after Türkiye’s EYT Pension Reform: Reduced-Form Evidence from Birth-Cohort and Age-Group Comparisons.”** The study uses repeated cross-sections from the Turkish Statistical Institute Household Labour Force Survey (HLFS).

## What this repository contains

- `analysis/`: estimation, inference, validation, and figure code;
- `tables/`: machine-readable aggregate estimates and diagnostics;
- `figures/`: publication and supplementary figures in raster and vector formats;
- `reports/`: computational environment and validation notes;
- `docs/`: data layout and the scope of causal interpretation.

Restricted individual-level TurkStat microdata are **not** distributed. The repository contains no raw or de-identified person-level records. Current-job-tenure experiments, document-build utilities, author-only submission files, and copyrighted reference PDFs are excluded because they are not part of the final empirical package.

## Empirical designs

The main fixed-cohort design compares men born in 1975–1981, who had higher expected exposure to the EYT reform, with men born in 1982–1984. The complementary fixed-age design compares men aged 43–49 with men aged 40–42. Outcomes are population shares in total employment, formal employment, informal employment, retirement outside the labor force, and the residual labor-market status.

The package additionally reports pre-reform diagnostics within the earlier measurement regime, alternative cohort and age windows, male–female triple differences, local age diagnostics, work-arrangement decompositions, education heterogeneity, education–region composition standardization, and direct differences between the 2024 and 2025 coefficients.

Actual legal eligibility and individual transitions are unobserved. The estimates are reduced-form differences associated with expected reform exposure under the assumptions stated in the article; they are not treatment effects among verified beneficiaries. See [reproducibility scope](docs/reproducibility_scope.md).

## Data requirements

Obtain the 2018, 2019, 2020, 2022, 2024, and 2025 HLFS individual microdata under TurkStat's applicable access and reuse conditions. Arrange the files as described in [data layout](docs/data_layout.md), then set the `EYT_DATA_ROOT` environment variable to their parent directory.

PowerShell:

```powershell
$env:EYT_DATA_ROOT = "D:\path\to\data"
```

Bash:

```bash
export EYT_DATA_ROOT="/path/to/data"
```

If `EYT_DATA_ROOT` is omitted, the scripts look for the year directories beside the cloned repository.

## Reproduction

Python 3.14.3 was used for the archived results. Create an environment and install the recorded dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python analysis/00_run_all.py
python analysis/validate_release.py
```

The runner estimates the fixed-age and fixed-cohort models, performs the pretrend and supplementary analyses, builds the education/composition extension, and regenerates the figures. Existing files in `tables/`, `figures/`, and the generated technical report may be overwritten. Randomized sign-flip routines use fixed seeds where enumeration is not feasible.

## Code map

| File | Purpose |
|---|---|
| `01_age_analysis.py` | Fixed-age DiD, gender DDD, local age tests, decompositions, and accounting checks |
| `03_pretrend_analysis.py` | Earlier-regime trends, placebo comparisons, and age-profile diagnostics |
| `06_cohort_analysis.py` | Main fixed-birth-cohort estimates, wild inference, pretrends, and cohort windows |
| `09_education_composition_time.py` | Education heterogeneity, composition standardization, and temporal contrasts |
| `plotting.py` | Main and supplementary figure definitions |
| `02_age_figures.py`, `07_cohort_figures.py` | Figure entry points |
| `validate_release.py` | Structural, accounting, and disclosure-safety checks |

## Output map

The machine-readable tables retain their computational numbering. `table15_cohort_did.csv` contains the main fixed-cohort estimates; `table18_cohort_means.csv` contains their descriptive cells; `table02_age_did.csv` contains the fixed-age results; and `table06_informality_components.csv` contains the work-arrangement decomposition. Files beginning with `extension_` contain education, composition, and temporal-contrast results. The complete map is in [tables/README.md](tables/README.md).

## Reproducibility and disclosure

The archived English outputs were matched to the corresponding Turkish-project outputs, and the additional WLS coefficients and household-clustered covariance were cross-checked against `statsmodels`. Input hashes for the main extension are recorded in `tables/extension_audit.json`; they identify local source files without distributing them.

ChatGPT assisted with translation, manuscript development, and preparation of analysis and visualization code; QuillBot assisted with language editing. The author remains responsible for the study design, validation, interpretation, and final materials.

## Citation

Citation metadata are provided in `CITATION.cff`. The article's journal citation and DOI should replace the working title metadata after publication.

No open-source license is asserted in this initial deposit. Contact the author before reusing code beyond GitHub's default viewing and forking permissions.
