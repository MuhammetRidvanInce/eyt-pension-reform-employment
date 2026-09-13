"""Additional cross-sectional analyses; no parallel-trend sensitivity extrapolation.

Run in VI first. The identical file can be run in VII/analysis; only output paths
and chart language change. Raw microdata never leave the workspace.
"""
from pathlib import Path
import importlib.util
import itertools
import json
import hashlib
import os
import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
EN = True
OUT = PROJECT / ('tables' if EN else 'cikti_tablolari')
FIG = PROJECT / ('figures' if EN else 'grafikler')
DATA_ROOT = Path(os.environ.get('EYT_DATA_ROOT', PROJECT.parent)).expanduser().resolve()
OUTCOMES = ['total', 'formal', 'informal', 'retired']


def load():
    name = '06_cohort_analysis.py' if EN else '06_sabit_kohort_dogrulama.py'
    spec = importlib.util.spec_from_file_location('cohort_base', Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    d = module.load([2022, 2024, 2025])
    rename = {'REFERANS_YIL':'year','DOGUM_YIL':'birth','YAS':'age',
              'AGIRLIK_KATSAYISI':'weight','IBBS_1':'region','OKUL_BITEN_K':'school'}
    rename.update(dict(zip(['total_employment','formal','informal','retired_inactive','household_year_id']
                          if EN else ['toplam_istihdam','kayitli','kayitdisi','emekli_igd','HANE_KOD'],
                          OUTCOMES+['household'])))
    d = d.rename(columns=rename)
    assert d[['school','region','weight']].notna().all().all()
    assert set(d.school.unique()) <= {1,2,3,41,42,511,512,52}
    assert (d.weight >= 0).all()
    d['education'] = np.select([d.school.isin([1,2,3]),d.school.isin([41,42])],[0,1],default=2)
    d['high'] = (d.education >= 1).astype(int)
    d['treated'] = (d.birth <= 1981).astype(int)
    return d


def fit(x, y, w, groups):
    """WLS with household-cluster CR1 covariance, matching statsmodels WLS."""
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w,float)
    inv = np.linalg.pinv(x.T @ (w[:,None]*x))
    beta = inv @ (x.T @ (w*y)); residual = y-x@beta
    codes, labels = pd.factorize(groups)
    scores = np.zeros((len(labels),x.shape[1]))
    np.add.at(scores,codes,x*(w*residual)[:,None])
    rank = np.linalg.matrix_rank(x.T @ (w[:,None]*x))
    assert rank == x.shape[1], (rank,x.shape[1])
    cov = inv @ (scores.T@scores) @ inv * len(labels)/(len(labels)-1)*(len(y)-1)/(len(y)-rank)
    return beta,cov,inv


def estimate(beta,cov,l):
    coef = float(l@beta); se = float(np.sqrt(l@cov@l))
    return dict(coef=coef,se=se,p=float(2*norm.sf(abs(coef/se))),
                ci_low=coef-1.95996398454*se,ci_high=coef+1.95996398454*se)


def wild(x,y,w,beta,inv,l,groups):
    """Restricted unstudentized sign-flip diagnostic, as in the main analysis."""
    x=np.asarray(x,float); y=np.asarray(y,float); w=np.asarray(w,float)
    restricted=beta-(inv@l)*(l@beta)/(l@inv@l)
    fitted=x@restricted; resid=y-fitted
    influence=(l@inv@x.T)*w
    codes,labels=pd.factorize(groups)
    scores=np.bincount(codes,weights=influence*resid)
    signs=np.asarray(list(itertools.product([-1.,1.],repeat=len(labels))))
    boot=np.abs(influence@fitted+signs@scores)
    return float((np.sum(boot>=abs(l@beta))+1)/(len(boot)+1))


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    all_data=load(); d=all_data[all_data.birth.between(1975,1984)].copy()
    assert len(d)==128783
    school_audit=d.groupby(['year','school']).size().rename('n').reset_index()
    school_audit.to_csv(OUT/'extension_education_codes.csv',index=False)
    rows=[]; composition=[]
    categories=[('education',str(k),lambda z,k=k:z.education.eq(k)) for k in range(3)]
    categories += [('region',k,lambda z,k=k:z.region.eq(k)) for k in sorted(d.region.unique())]
    for year,z in d.groupby('year'):
        for variable,level,fn in categories:
            shares=[]
            for t in [1,0]:
                q=z[z.treated.eq(t)]; p=np.average(fn(q),weights=q.weight); shares.append(p)
                composition.append(dict(year=int(year),treated=t,variable=variable,level=level,share=100*p,n=len(q)))
            denom=np.sqrt(sum(p*(1-p) for p in shares)/2)
            rows.append(dict(year=int(year),variable=variable,level=level,
                             exposed=100*shares[0],comparison=100*shares[1],
                             smd=(shares[0]-shares[1])/denom if denom else 0))
    pd.DataFrame(rows).to_csv(OUT/'extension_composition.csv',index=False)
    # Joint education x region support, with each cohort fixed to its OWN 2022
    # distribution. This removes observed temporal composition changes, not
    # baseline cohort differences; it is descriptive standardization, not an ATT.
    d['cell']=d.education.astype(str)+'_'+d.region.astype(str)
    support=d.groupby(['year','treated','cell']).agg(n=('weight','size'),mass=('weight','sum')).reset_index()
    assert len(support)==3*2*3*12
    support['share']=support.mass/support.groupby(['year','treated']).mass.transform('sum')
    target=support[support.year.eq(2022)].set_index(['treated','cell'])['share']
    support['factor']=[target.loc[(r.treated,r.cell)]/r.share for r in support.itertuples()]
    support.to_csv(OUT/'extension_support.csv',index=False)
    d=d.merge(support[['year','treated','cell','factor']],on=['year','treated','cell'],validate='many_to_one')
    d['standard_weight']=d.weight*d.factor
    standardized=[]; means={}
    for (year,t),z in d.groupby(['year','treated']):
        for outcome in OUTCOMES:
            for kind,weight in [('raw','weight'),('standardized','standard_weight')]:
                means[(year,t,outcome,kind)]=np.average(z[outcome],weights=z[weight])
    for year in [2024,2025]:
        for outcome in OUTCOMES:
            for kind in ['raw','standardized']:
                delta=means[year,1,outcome,kind]-means[2022,1,outcome,kind]-means[year,0,outcome,kind]+means[2022,0,outcome,kind]
                standardized.append(dict(year=year,outcome=outcome,kind=kind,coef=delta))
    pd.DataFrame(standardized).to_csv(OUT/'extension_standardization.csv',index=False)
    hetero=[]
    for year in [2024,2025]:
        z=d[d.year.isin([2022,year])].copy(); z['post']=z.year.eq(year).astype(int)
        x=patsy.dmatrix('C(birth)*C(high)+C(year)*C(high)+C(region)*C(high)+treated:post+treated:post:high',z,return_type='dataframe')
        for outcome in OUTCOMES:
            b,v,inv=fit(x,z[outcome],z.weight,z.household)
            for group in ['low','high','difference']:
                l=np.zeros(x.shape[1])
                if group!='difference':l[x.columns.get_loc('treated:post')]=1
                if group!='low':l[x.columns.get_loc('treated:post:high')]=1
                r=dict(year=year,outcome=outcome,group=group,n=len(z),**estimate(b,v,l))
                if outcome=='informal':r['p_wild_birth']=wild(x,z[outcome],z.weight,b,inv,l,z.birth)
                hetero.append(r)
            print('Education',year,outcome,flush=True)
    h=pd.DataFrame(hetero); mask=h.group.eq('difference')
    h.loc[mask,'p_holm']=multipletests(h.loc[mask,'p'],method='holm')[1]
    h.to_csv(OUT/'extension_education_effects.csv',index=False)
    # Stack the original two pairwise specifications. A duplicated baseline
    # keeps its household identifier, preserving shared-baseline covariance.
    time=[]
    for design in ['cohort','age']:
        z=d.copy() if design=='cohort' else all_data[all_data.age.between(40,49)].copy()
        # Loader includes births 1972-1985, covering ages 40-49 in all three waves.
        z['treated']=z.birth.le(1981).astype(int) if design=='cohort' else z.age.ge(43).astype(int)
        fe='birth' if design=='cohort' else 'age'
        n_original=len(z)
        blocks=[]; frames=[]
        for year in [2024,2025]:
            q=z[z.year.isin([2022,year])].copy(); q['post']=q.year.eq(year).astype(int)
            formula=f'C({fe})+C(year)+treated:post' if design=='cohort' else 'treated*post'
            blocks.append(patsy.dmatrix(formula,q,return_type='dataframe'))
            frames.append(q)
        k=blocks[0].shape[1]; assert k==blocks[1].shape[1]
        x=np.zeros((sum(len(q) for q in frames),2*k)); n=len(frames[0])
        x[:n,:k]=blocks[0];x[n:,k:]=blocks[1]
        z=pd.concat(frames,ignore_index=True)
        l=np.zeros(2*k);l[blocks[0].columns.get_loc('treated:post')]=-1;l[k+blocks[1].columns.get_loc('treated:post')]=1
        for outcome in OUTCOMES:
            b,v,inv=fit(x,z[outcome],z.weight,z.household)
            r=dict(design=design,outcome=outcome,n_unique=n_original,
                   coefficient_2024=b[blocks[0].columns.get_loc('treated:post')],
                   coefficient_2025=b[k+blocks[1].columns.get_loc('treated:post')],**estimate(b,v,l))
            if outcome=='informal':r['p_wild']=wild(x,z[outcome],z.weight,b,inv,l,z[fe])
            time.append(r); print('Year contrast',design,outcome,flush=True)
    time=pd.DataFrame(time);time['p_holm']=multipletests(time.p,method='holm')[1]
    time.to_csv(OUT/'extension_year_contrasts.csv',index=False)
    # Reproducible inputs and support diagnostics; no person-level records saved.
    audit={'sample_n':len(d),'education_missing':int(d.school.isna().sum()),
           'zero_weight_observations':int(d.weight.eq(0).sum()),
           'support_min_cell_n':int(support.n.min()),'standardization_factor_min':float(support.factor.min()),
           'standardization_factor_max':float(support.factor.max()),
           'group_n':d.groupby(['year','treated','high']).size().to_dict().__str__(),
           'source_sha256':{str(y):hashlib.file_digest(open(DATA_ROOT/str(y)/'HHIA_Isgucu'/f'mikro_veri_{y}.csv','rb'),'sha256').hexdigest() for y in [2022,2024,2025]},
           'interpretation':'Exploratory subgroup contrasts; observed education is not pension eligibility. Standardization is descriptive with no inference. No pretrend extrapolation.'}
    (OUT/'extension_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    labels=['Total employment','Formal employment','Informal employment','Retired, outside\nlabor force'] if EN else ['Toplam istihdam','Kayıtlı istihdam','Kayıt dışı istihdam','İşgücü dışı\nemeklilik']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(174/25.4,88/25.4),sharey=True,sharex=True)
    for ax,year in zip(axes,[2024,2025]):
        for group,color,marker,offset,label in [('low','#C65D21','o',.12,'Below upper secondary' if EN else 'Lise altı'),('high','#21618C','s',-.12,'Upper secondary or higher' if EN else 'Lise ve üzeri')]:
            q=h[h.year.eq(year)&h.group.eq(group)].set_index('outcome').loc[OUTCOMES]
            ax.errorbar(q.coef,np.arange(4)[::-1]+offset,xerr=[q.coef-q.ci_low,q.ci_high-q.coef],fmt=marker,color=color,capsize=3,label=label)
        ax.set_title(f'{"a" if year==2024 else "b"}  {year}–2022');ax.axvline(0,color='#666666',lw=.8)
        ax.grid(axis='x',color='#E1E5E9',linewidth=.6);ax.set_axisbelow(True)
        ax.set_xlabel('DiD estimate (percentage points)' if EN else 'DiD katsayısı (yüzde puan)')
        ax.set_yticks(np.arange(4)[::-1],labels)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.10,1,1))
    for ext in ['png','pdf','eps']:fig.savefig(FIG/f'extension_education.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__': main()
