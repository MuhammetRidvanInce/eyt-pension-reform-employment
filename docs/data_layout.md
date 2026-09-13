# Data layout and access

The code expects the TurkStat Household Labour Force Survey (HLFS) individual microdata for 2018, 2019, 2020, 2022, 2024, and 2025. These files are not part of this repository. Access and reuse are governed by TurkStat's applicable conditions.

Set `EYT_DATA_ROOT` to a directory with this structure:

```text
EYT_DATA_ROOT/
├── 2018/HHIA_Isgucu/mikro_veri_2018.csv
├── 2019/HHIA_Isgucu/mikro_veri_2019.csv
├── 2020/HHIA_Isgucu/mikro_veri_2020.csv
├── 2022/HHIA_Isgucu/mikro_veri_2022.csv
├── 2024/HHIA_Isgucu/mikro_veri_2024.csv
└── 2025/HHIA_Isgucu/mikro_veri_2025.csv
```

The scripts retain official Turkish variable names to make the transformation from source files auditable. They read semicolon-delimited CSV files using the encodings specified in each module. No individual-level data are written by the analysis. Public outputs are aggregate regression results, weighted cell summaries, diagnostics, and figures.

The 2022, 2024, and 2025 waves form the main comparison under the newer HLFS measurement regime. The 2018–2020 waves are used only for diagnostics within the earlier regime. The methodological change prevents treating these periods as a continuous event study.
