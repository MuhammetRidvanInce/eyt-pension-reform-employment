from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis/outputs/cohort_informality_2026-10-15'
report={}
for reference in sorted((ROOT/'reference_results').glob('*.csv')):
    a=pd.read_csv(reference); b=pd.read_csv(OUT/reference.name)
    assert list(a.columns)==list(b.columns) and a.shape==b.shape, reference.name
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]):
            assert np.allclose(a[c],b[c],rtol=1e-7,atol=1e-8,equal_nan=True),(reference.name,c)
        else:
            assert a[c].fillna('').equals(b[c].fillna('')),(reference.name,c)
    report[reference.name]='matches reference'
assert pd.read_csv(OUT/'sample_audit.csv').valid_n.sum()==217716
for stem in ['figure1_cohort_employment','figure2_informal_employment_status','figure3_separate_period_cohort_check']:
    for ext in ['pdf','png','svg']: assert (OUT/'figures_en'/f'{stem}.{ext}').is_file()
print(json.dumps(report,indent=2))
