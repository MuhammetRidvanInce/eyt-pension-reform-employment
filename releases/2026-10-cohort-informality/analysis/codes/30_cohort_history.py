"""Earlier-data feasibility and separate-regime cohort diagnostic, 2014-2025.

Historical cohort proxies use survey year minus completed age, consistently in
both regimes. They do not replace recorded birth years in the primary model.
No common pre/post coefficient is estimated across the 2021 survey break.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import extension_common as ec
from extension_common import WeightedFit

spec=importlib.util.spec_from_file_location('focus',Path(__file__).with_name('28_cohort_informality.py'))
focus=importlib.util.module_from_spec(spec);spec.loader.exec_module(focus)
ROOT=ec.ROOT;OUT=focus.OUT;FIG=OUT/'figures'
HIGHLIGHT_COLOR='#c45b19'

def make_figure(result):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
       'axes.edgecolor':'#8a99a6','axes.linewidth':.7,'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,1,figsize=(10.8,7.2),layout='constrained')
    for ax,(o,title) in zip(axs,[('formal','Kayıtlı istihdam'),('informal','Kayıt dışı istihdam')]):
        color='#202020'
        ax.axvspan(2022.5,2025.4,color='#edf2f6',zorder=0)
        ax.axvspan(2020.35,2020.65,color='#e4e7e9',zorder=0)
        ax.axhline(0,color='#7f8991',lw=.85,zorder=1)
        for regime,years,ref in [('2014-2020',range(2014,2021),2020),('2021-2025',range(2021,2026),2022)]:
            q=result[(result.regime==regime)&(result.outcome==o)]
            values={int(r.contrast):(r.coef,r.ci_cluster_low,r.ci_cluster_high) for r in q.itertuples()};values[ref]=(0,0,0)
            x=np.array(list(years));v=np.array([values[y] for y in x]);mask=x!=ref
            ax.plot(x,v[:,0],color=color,lw=1,ls='--',alpha=.8,zorder=2)
            for highlighted in [False,True]:
                mask=(x!=ref)&(np.isin(x,[2024,2025])==highlighted)
                if not mask.any():continue
                ax.errorbar(x[mask],v[mask,0],yerr=np.vstack([v[mask,0]-v[mask,1],v[mask,2]-v[mask,0]]),
                    fmt='s' if highlighted else 'o',ms=4.8 if highlighted else 4.3,
                    elinewidth=2.5 if highlighted else 1.7,capthick=2.1 if highlighted else 1.5,
                    capsize=3.5 if highlighted else 3,color=HIGHLIGHT_COLOR if highlighted else color)
            ax.plot(ref,0,'o',mfc='white',mec=color,ms=5)
        ax.set_title(title,loc='left',pad=27)
        ax.text(2017,.99,'2014–2020  |  Referans: 2020',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,color='#53616b')
        ax.text(2023,.99,'2021–2025  |  Referans: 2022',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,color='#53616b')
        ax.set(xlabel='Anket yılı',ylabel='Yüzde puan',xticks=np.arange(2014,2026),xlim=(2013.6,2025.4))
        ax.margins(y=.15)
        ax.grid(axis='y',color='#e1e7ec',lw=.6);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg','pdf']:fig.savefig(FIG/('sekil3_ayri_donem_kohort_kontrolu.'+ext),dpi=300,facecolor='white')
    plt.close(fig)

def main():
    parts=[];audit=[];sources=[]
    for year in range(2014,2026):
        p=ec.source(year)
        cols=['REFERANS_YIL','BIRIMNO','CINSIYET','YAS','AGIRLIK_KATSAYISI','DURUM','KAYITLILIK']
        z=ec.read_csv(p,cols)
        for c in cols:z[c]=pd.to_numeric(z[c],errors='coerce')
        z=z.rename(columns={'REFERANS_YIL':'year','YAS':'age','AGIRLIK_KATSAYISI':'weight'})
        z['birth']=z.year-z.age
        z=z[z.CINSIYET.eq(1)&z.birth.between(1975,1984)].copy();raw=len(z)
        z=z[z.weight.gt(0)&z.DURUM.isin([1,2,3])&(~z.DURUM.eq(1)|z.KAYITLILIK.isin([1,2]))].copy()
        z['household']=str(year)+'_'+z.BIRIMNO.astype(str)
        z['treated']=z.birth.le(1981).astype(int)
        z['formal']=100.*(z.DURUM.eq(1)&z.KAYITLILIK.eq(1))
        z['informal']=100.*(z.DURUM.eq(1)&z.KAYITLILIK.eq(2))
        audit.append(dict(year=year,raw_n=raw,n=len(z),exposed_n=int(z.treated.sum()),control_n=int(z.treated.eq(0).sum())))
        sources.append(dict(year=year,path=str(p),sha256=ec.sha(p)))
        parts.append(z);print('HISTORY LOAD',year,len(z),flush=True)
    all_data=pd.concat(parts,ignore_index=True)
    rows=[]
    for regime,years,ref in [('2014-2020',range(2014,2021),2020),('2021-2025',range(2021,2026),2022)]:
        q=all_data[all_data.year.isin(years)].copy()
        active=[y for y in years if y!=ref]
        for y in active:q[f'e{y}']=q.treated*q.year.eq(y)
        f=WeightedFit(q,'C(birth)+C(year)+'+'+'.join(f'e{y}' for y in active),['formal','informal'])
        tests=[(o,{f'e{y}':1},str(y)) for o in f.outcomes for y in active]
        rr=focus.inference(f,tests,'history_'+regime,True)
        rows.extend([dict(regime=regime,reference_year=ref,**r) for r in rr])
    focus.csv('historical_results.csv',rows);focus.csv('historical_sample.csv',audit)
    focus.dump('historical_source_manifest.json',sources)
    result=pd.DataFrame(rows)
    make_figure(result)
    focus.dump('historical_validation.json',{'years':list(range(2014,2026)),'older_n':sum(x['n'] for x in audit if x['year']<2021),
       'recent_proxy_n':sum(x['n'] for x in audit if x['year']>=2021),'cohort_definition':'survey year - completed age; proxy 1975-1981 vs 1982-1984',
       'no_model_crosses_2021_break':True,'bootstrap_validation':focus.CHECK,
       'scope_decision':'Earlier microdata available back to 2000; diagnostic starts in 2014 to avoid the earlier 2014 survey break. Main model retains actual birth years and 2021-2025.'})
    print('HISTORY COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--figures-only',action='store_true',help='Redraw from saved estimates without rerunning analysis.')
    args=parser.parse_args()
    if args.figures_only:
        make_figure(pd.read_csv(OUT/'historical_results.csv'))
        print('Historical figure redrawn: PNG, SVG, PDF.')
    else:main()
