"""Publication figures from English result tables; no model is re-estimated here."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TABLES, FIGURES = PROJECT / 'tables', PROJECT / 'figures'
ORANGE, BLUE, GRAY = '#C65D21', '#21618C', '#66717D'
PERIODS = [('2024 vs 2022', ORANGE, 'o', .13), ('2025 vs 2022', BLUE, 's', -.13)]
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'axes.titleweight': 'semibold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#A0A7AF', 'axes.linewidth': .7,
    'xtick.color': '#3B4652', 'ytick.color': '#3B4652',
    'legend.frameon': False, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
ORDER = ['Total employment', 'Formal employment', 'Informal employment',
         'Retired and outside the labor force', 'Unemployed or otherwise inactive']
SHORT = {'Retired and outside the labor force': 'Retired, outside the\nlabor force',
         'Unemployed or otherwise inactive': 'Unemployed or\notherwise inactive'}

def read(stem):
    return pd.read_csv(TABLES / (stem + '.csv'))

def save(fig, stem, note):
    FIGURES.mkdir(exist_ok=True)
    fig.text(.01, .015, textwrap.fill(note, 145), va='bottom', ha='left',
             color=GRAY, fontsize=8)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(FIGURES / f'{stem}.{ext}', dpi=400, bbox_inches='tight', pad_inches=.14)
    plt.close(fig)

def finish_axis(ax, horizontal=True):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', color='#E3E7EB', linewidth=.6)
    ax.tick_params(length=3, width=.6)
    if horizontal:
        ax.axvline(0, color=GRAY, lw=.9, zorder=1)
        ax.xaxis.set_major_locator(MaxNLocator(6))

def draw_forest(ax, data, categories, key='Outcome', display=None):
    y = np.arange(len(categories))[::-1]
    for period, color, marker, offset in PERIODS:
        z = data[data['Period'].eq(period)].set_index(key).loc[categories]
        ax.errorbar(z['coef'], y + offset,
                    xerr=np.vstack([z['coef']-z['ci_low'], z['ci_high']-z['coef']]),
                    fmt=marker, color=color, ms=6, lw=1.5, capsize=3,
                    label=period.replace(' vs ', '–'), zorder=3)
    ax.set_yticks(y, display or [SHORT.get(x, x) for x in categories])
    ax.set_ylim(-.6, len(categories)-.4)
    ax.set_xlabel('DiD estimate (percentage points)')
    finish_axis(ax)

def cohort_figures():
    d = read('table18_cohort_means')
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    for ax, outcome, letter in zip(axes, ORDER[1:3]+ORDER[:1], 'ABC'):
        for cohort, color, marker in [('1975-1981', ORANGE, 'o'), ('1982-1984', BLUE, 's')]:
            z = d[d.Cohort.eq(cohort)].sort_values('Year')
            ax.plot(z.Year, z[outcome], color=color, marker=marker, lw=1.8, ms=6,
                    label=f'{cohort} birth cohorts')
        ax.axvline(2023, color=GRAY, ls='--', lw=.8)
        ax.set_title(f'{letter}. {outcome}', loc='left')
        ax.set_xticks([2022, 2024, 2025]); ax.set_xlabel('Survey year')
        ax.set_ylabel('Share of male cohort population (%)')
        finish_axis(ax, False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.5, .09), ncol=2)
    fig.suptitle('Labor market status by fixed birth cohort', x=.07, ha='left', weight='bold', fontsize=14)
    fig.subplots_adjust(left=.07, right=.99, bottom=.29, top=.81, wspace=.34)
    save(fig, 'figure1_cohort_means', 'Source: TurkStat HLFS, individual weights. Birth cohorts are held constant. The dashed line marks the 2023 reform; no 2023 observation is interpolated.')
    d = read('table15_cohort_did')
    d = d[d.Specification.eq('Cohort and year FE')]
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    draw_forest(ax, d, ORDER)
    ax.set_title('Fixed-cohort difference-in-differences estimates', loc='left', pad=18)
    ax.legend(loc='lower right')
    fig.subplots_adjust(left=.28, right=.97, top=.86, bottom=.20)
    save(fig, 'figure2_cohort_did', 'Men born in 1975–1981 versus 1982–1984. Cohort and year fixed effects. Bars: 95% household-clustered confidence intervals; birth-year wild p-values are reported in the text.')

def age_figures():
    d = read('table02_age_did')
    d = d[d.Sample.eq('Men') & d.Specification.eq('Unadjusted 2x2 DiD')]
    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    draw_forest(ax, d, ORDER[:4])
    ax.set_title('Fixed-age difference-in-differences estimates', loc='left', pad=18)
    ax.legend(loc='lower right')
    fig.subplots_adjust(left=.28, right=.97, top=.85, bottom=.23)
    save(fig, 'figure3_age_did', 'Men aged 43–49 versus men aged 40–42. Unadjusted DiD. Bars: 95% household-clustered confidence intervals; age-level wild p-values are reported in Table 5.')
    d = read('table06_informality_components'); d = d[d.Sample.eq('Men')]
    panels = [
        ('A. Firm size', ['Firm: 1-9', 'Firm: 10-49', 'Firm: 50+ (known size)', 'Firm: 10+ (size unknown)'],
         ['1–9 workers', '10–49 workers', '50+ workers\n(known size)', '10+ workers\n(size unknown)']),
        ('B. Employment status', ['Status: wage/casual employee', 'Status: employer', 'Status: own-account worker', 'Status: unpaid family worker'],
         ['Wage/casual employee', 'Employer', 'Own-account worker', 'Unpaid family worker'])]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for ax, (title, categories, labels) in zip(axes, panels):
        draw_forest(ax, d, categories, 'Component', labels); ax.set_title(title, loc='left', pad=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .09), ncol=2)
    fig.suptitle('Components of the increase in informal employment', x=.04, ha='left', weight='bold', fontsize=14)
    fig.subplots_adjust(left=.13, right=.98, top=.80, bottom=.28, wspace=.68)
    save(fig, 'figure4_informality_components', 'Male age-group DiD, 2022 baseline. Bars: 95% household-clustered confidence intervals. Firm-size and employment-status classifications overlap and must not be added across panels.')
    supplementary_age_figures()

def supplementary_age_figures():
    d = read('table09_long_run_age_means'); d = d[d.Sample.eq('Men')].drop_duplicates(['Year','Group'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, outcome, letter in zip(axes, ORDER[1:4], 'ABC'):
        for group, color, marker in [('43-49', ORANGE, 'o'), ('40-42', BLUE, 's')]:
            for old in (True, False):
                z = d[d.Group.eq(group) & (d.Year.le(2020) if old else d.Year.ge(2022))].sort_values('Year')
                ax.plot(z.Year, z[outcome], color=color, marker=marker, lw=1.6, ms=5,
                        label=f'Ages {group}' if old else None)
        ax.axvline(2021, color=GRAY, ls='--', lw=.8)
        ax.set_title(f'{letter}. '+SHORT.get(outcome,outcome).replace('\n',' '), loc='left', fontsize=10)
        ax.set_xticks([2018,2020,2022,2024,2025]); ax.tick_params(axis='x', labelrotation=35)
        ax.set_xlabel('Survey year'); ax.set_ylabel('Population share (%)'); finish_axis(ax,False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.08), ncol=2)
    fig.suptitle('Supplementary Figure S1. Weighted age-group means', x=.07, ha='left', weight='bold')
    fig.subplots_adjust(left=.07,right=.99,top=.79,bottom=.30,wspace=.38)
    save(fig,'figureS1_long_run_age_means','Lines are deliberately broken at the 2021 HLFS methodological change. Old- and new-regime levels are not a continuous comparable series.')
    d = read('table05_alternative_age_cutoffs')
    fig, axes = plt.subplots(1,2,figsize=(11.7,4.8))
    for ax, outcome, color in zip(axes,['Informal','Formal'],[ORANGE,BLUE]):
        z=d[d.Outcome.eq(outcome)].sort_values('Cutoff')
        ax.errorbar(z.Cutoff,z.coef,yerr=[z.coef-z.ci_low,z.ci_high-z.coef],fmt='o',color=color,ecolor='#BFC7CF',capsize=3,ms=5)
        at=z[z.Cutoff.eq(43)].iloc[0]
        ax.scatter(43,at.coef,s=80,facecolor='white',edgecolor=color,lw=2,zorder=5)
        ax.axvline(43,color=GRAY,ls='--',lw=.8); ax.axhline(0,color=GRAY,lw=.8)
        ax.set_title(outcome+' employment',loc='left'); ax.set_xlabel('Candidate age cutoff')
        ax.set_ylabel('Local estimate (percentage points)'); ax.set_xticks(range(36,49,2)); finish_axis(ax,False)
    fig.suptitle('Supplementary Figure S2. Local estimates at alternative age cutoffs',x=.07,ha='left',weight='bold')
    fig.subplots_adjust(left=.08,right=.99,top=.81,bottom=.25,wspace=.28)
    save(fig,'figureS2_alternative_age_cutoffs','Men, 2024–2022. Bars: 95% household-clustered confidence intervals. Age 43 is a candidate exposure cutoff, not a statutory EYT eligibility threshold.')
    d=read('table07_stock_accounting'); d=d[d['Cohort/window'].eq('Common cohort 1970-1982')].set_index('Status')
    categories=['Long current-job tenure, formal','Short current-job tenure, formal','Total informal']
    fig,ax=plt.subplots(figsize=(9.5,4.8)); x=np.arange(3)
    for i,(year,color) in enumerate([(2022,'#8796A5'),(2024,ORANGE),(2025,BLUE)]):
        ax.bar(x+(i-1)*.23,d.loc[categories,f'{year} stock (thousands)'],.23,color=color,label=str(year))
    ax.set_xticks(x,['Long current-job tenure,\nformal employment','Short current-job tenure,\nformal employment','Informal employment'])
    ax.set_ylabel('Weighted stock (thousands)'); ax.legend(ncol=3); finish_axis(ax,False)
    ax.set_title('Supplementary Figure S3. Employment stocks: 1970–1982 cohorts',loc='left')
    fig.subplots_adjust(left=.10,right=.98,top=.84,bottom=.25)
    save(fig,'figureS3_stock_accounting','Repeated cross-sectional stocks, not individual transitions. Current-job tenure does not identify first social insurance registration.')
    d=read('table10_age_event_study')
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.8))
    for ax,outcome,color in zip(axes,['Informal employment','Formal employment'],[ORANGE,BLUE]):
        for old in (True,False):
            z=d[d.Outcome.eq(outcome)&(d.Year.le(2020) if old else d.Year.ge(2022))].sort_values('Year')
            ax.errorbar(z.Year,z.coef,yerr=[z.coef-z.ci_low,z.ci_high-z.coef],fmt='o-',color=color,lw=1.5,capsize=3)
        ax.axhline(0,color=GRAY,lw=.8);ax.axvline(2021,color=GRAY,ls='--',lw=.8)
        ax.set_title(outcome,loc='left');ax.set_xticks([2018,2020,2022,2024,2025]);ax.set_xlabel('Survey year')
        ax.set_ylabel('Age-group contrast relative to 2020 (pp)');finish_axis(ax,False)
    fig.suptitle('Supplementary Figure S4. Age-group event-study diagnostics',x=.07,ha='left',weight='bold')
    fig.subplots_adjust(left=.09,right=.99,top=.82,bottom=.25,wspace=.30)
    save(fig,'figureS4_age_event_study','Diagnostic only: the 2021 survey break prevents a continuous causal event-study interpretation. Lines are broken across regimes; bars are 95% household-clustered confidence intervals.')

def tenure_figures():
    d=read('table12_current_job_tenure_did');d=d[d.Specification.eq('Age FE + education + region')]
    fig,ax=plt.subplots(figsize=(9.5,4.2)); y=np.arange(len(d))[::-1]
    ax.errorbar(d.coef,y,xerr=[d.coef-d.ci_low,d.ci_high-d.coef],fmt='o',color=BLUE,capsize=3,lw=1.5)
    ax.set_yticks(y,d.Period.str.replace(' vs ','–'));ax.set_ylim(-.6,1.6)
    ax.set_xlabel('Current-job tenure DiD estimate (percentage points)');finish_axis(ax)
    ax.set_title('Supplementary Figure S5. Current-job tenure and informality',loc='left')
    fig.subplots_adjust(left=.15,right=.98,top=.83,bottom=.28)
    save(fig,'figureS5_current_job_tenure','Employed men aged 40–55; long tenure means starting the current job in 1998 or earlier. Adjusted model. Bars: 95% household-clustered confidence intervals.')
    d=read('table13_age_tenure_interaction');fig,ax=plt.subplots(figsize=(10.5,4.5))
    draw_forest(ax,d,['Within-age-group tenure DiD','Net age-tenure DDD'],'Model',
                ['Tenure DiD\nwithin ages 43–49','Net age × tenure × post\ntriple interaction'])
    ax.set_title('Supplementary Figure S6. Age and current-job tenure interactions',loc='left');ax.legend(loc='lower right')
    fig.subplots_adjust(left=.27,right=.98,top=.82,bottom=.27)
    save(fig,'figureS6_age_tenure_interaction','Conditional on employment. Current-job start year is not first insurance registration. Bars: 95% household-clustered confidence intervals. The net triple interaction is not statistically significant.')
