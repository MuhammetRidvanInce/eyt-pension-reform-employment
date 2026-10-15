"""Render English counterparts of the current Turkish figures from saved estimates.

Only labels and output destinations are localized. The original plotting
implementations supply geometry, estimates, confidence intervals, and styling.
"""
from pathlib import Path
import sys

CODES=Path(__file__).resolve().parent
ROOT=CODES.parents[1]
OUT=ROOT/'analysis/outputs/cohort_informality_2026-10-15'
(OUT/'figures_en').mkdir(exist_ok=True)
sys.path.insert(0,str(CODES))

TRANSLATIONS={
    'Kayıt dışı istihdam farkı (yüzde puan)':'Informal employment difference (percentage points)',
    'Emeklilik nedeniyle işgücü dışında':'Inactive due to retirement',
    'Kayıtlı istihdam':'Formal employment',
    'Kayıt dışı istihdam':'Informal employment',
    'Toplam istihdam':'Total employment',
    'Ücretli ve yevmiyeli':'Wage and casual workers',
    'Kendi hesabına':'Own-account workers',
    'İşveren':'Employers',
    'Ücretsiz aile işçisi':'Unpaid family workers',
    'Anket yılı':'Survey year',
    'Yüzde puan':'Percentage points',
    'Referans:':'Reference:',
    'sekil1_kohort_istihdam':'figure1_cohort_employment',
    'sekil2_kayitdisi_calisma_bicimleri':'figure2_informal_employment_status',
    'sekil3_ayri_donem_kohort_kontrolu':'figure3_separate_period_cohort_check',
    "OUT/'figures'":"OUT/'figures_en'",
    'figure_manifest.json':'figure_manifest_en.json',
}

for filename in ['29_cohort_figures.py','30_cohort_history.py']:
    path=CODES/filename
    source=path.read_text(encoding='utf-8')
    for original,translated in TRANSLATIONS.items():
        source=source.replace(original,translated)
    namespace={'__file__':str(path),'__name__':'english_figure_render'}
    exec(compile(source,str(path), 'exec'),namespace)
    if filename.startswith('30'):
        namespace['make_figure'](namespace['pd'].read_csv(OUT/'historical_results.csv'))
print('Three English figures saved as PNG, SVG, and PDF; Turkish figures retained.')
