"""2021-2025 cohort study focused on informality; unadjusted inference only.

Rebuilds the analysis from immutable raw HLFS sources. Does not run old-period
models, multiplicity procedures, or edit either manuscript. Primary population:
men born 1975-1984; higher exposure 1975-1981, comparison 1982-1984.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import itertools, json, platform, shutil
import numpy as np
import pandas as pd
import scipy, statsmodels, statsmodels.api as sm
import extension_common as ec
from extension_common import WeightedFit, events, event_rhs

ROOT=ec.ROOT
OUT=ROOT/'analysis/outputs/cohort_informality_2026-10-15'
OUT.mkdir(parents=True,exist_ok=True)
B=99999
SEED=20261015
CHECK={}
OUTCOMES=['informal','formal','total','retired']
COLS=['REFERANS_YIL','BIRIMNO','FERTNO','CINSIYET','DOGUM_YIL','YAS',
      'AGIRLIK_KATSAYISI','DURUM','KAYITLILIK','IDO_NEDEN','IBBS_1','IBBS_2',
      'OKUL_BITEN_K','ISTEKI_DURUM_K','CALISAN_SAYI_HH']

def csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def weighted(d,c):return float(np.average(d[c],weights=d.weight))

def load():
    parts=[];audit=[];sources=[]
    for year in ec.YEARS:
        path=ec.source(year);z=ec.read_csv(path,COLS)
        raw_full=len(z)
        for c in COLS:
            if c not in ['IBBS_1','IBBS_2']:z[c]=pd.to_numeric(z[c],errors='coerce')
        z=z[z.CINSIYET.eq(1)&z.DOGUM_YIL.between(1972,1984)].copy()
        z=z.rename(columns={'REFERANS_YIL':'year','DOGUM_YIL':'birth','YAS':'age',
            'AGIRLIK_KATSAYISI':'weight','IBBS_1':'region','IBBS_2':'region2','OKUL_BITEN_K':'school'})
        assert z.year.eq(year).all()
        z['household']=str(year)+'_'+z.BIRIMNO.astype(str)
        for c in ['region','region2']:z[c]=z[c].astype(str).str.strip()
        z['education']=np.select([z.school.isin([1,2,3]),z.school.isin([41,42]),z.school.isin([511,512,52])],[0,1,2],default=-1)
        z['treated']=z.birth.le(1981).astype(int)
        z['valid']=z.weight.gt(0)&z.DURUM.isin([1,2,3])&z.education.ge(0)&(~z.DURUM.eq(1)|z.KAYITLILIK.isin([1,2]))&(~z.DURUM.eq(3)|z.IDO_NEDEN.notna())
        for name,mask in {'formal':z.DURUM.eq(1)&z.KAYITLILIK.eq(1),
                          'informal':z.DURUM.eq(1)&z.KAYITLILIK.eq(2),
                          'total':z.DURUM.eq(1),
                          'retired':z.DURUM.eq(3)&z.IDO_NEDEN.eq(36)}.items():z[name]=100.*mask
        q=z[z.birth.between(1975,1984)]
        audit.append(dict(year=year,raw_file_n=raw_full,raw_target_n=len(q),valid_n=int(q.valid.sum()),
             excluded_n=int((~q.valid).sum()),zero_weight_n=int(q.weight.eq(0).sum()),
             missing_status_n=int(q.DURUM.isna().sum()),treated_n=int((q.valid&q.treated.eq(1)).sum()),
             control_n=int((q.valid&q.treated.eq(0)).sum())))
        sources.append(dict(year=year,path=str(path),sha256=ec.sha(path)))
        parts.append(z[z.valid].copy())
        print('LOAD',year,'target',int(q.valid.sum()),flush=True)
    csv('sample_audit.csv',audit);dump('source_manifest.json',sources)
    return pd.concat(parts,ignore_index=True)

def inference(fit,tests,tag,validate=False):
    """Null-restricted WCR-t, exhaustive Rademacher and MC Webb weights."""
    fit.signs=np.asarray(list(itertools.product([-1.,1.],repeat=fit.g)))
    values=np.array([-np.sqrt(1.5),-1.,-np.sqrt(.5),np.sqrt(.5),1.,np.sqrt(1.5)])
    weights=np.random.default_rng(SEED).choice(values,size=(B,fit.g))
    rows=[];errors=[]
    for ti,(outcome,terms,label) in enumerate(tests):
        r=fit.contrast(terms,outcome)
        # WeightedFit uses +1 convention above G=12; here signs are exhaustive.
        if fit.g>12:r['p_wild_cluster_t']=(r['p_wild_cluster_t']*(len(fit.signs)+1)-1)/len(fit.signs)
        j=fit.outcomes.index(outcome);l=np.zeros(fit.p)
        for term,value in terms.items():l[fit.names.index(term)]=value
        u=fit.inv@l;shift=u*r['coef']/(l@u)
        score=fit.raw_c[j]+np.einsum('gij,j->gi',fit.grams,shift)
        threshold=abs(r['coef']/r['se_cluster']);hits=0
        for start in range(0,B,8192):
            v=weights[start:start+8192];delta=(v@score)@fit.inv
            rs=v*(score@u)[None,:]-delta@np.einsum('gij,j->gi',fit.grams,u).T
            variance=fit.correction(fit.g)*np.sum(rs**2,axis=1)
            assert (variance>1e-24).all()
            ts=np.abs(delta@l)/np.sqrt(variance);hits+=int((ts>=threshold-1e-10).sum())
            if validate and ti==0 and start==0:
                for ix in [0,23]:
                    b0=fit.beta[:,j]-shift
                    ys=fit.x@b0+(fit.y[:,j]-fit.x@b0)*v[ix,fit.codes]
                    bs=fit.inv@(fit.x.T@(fit.w*ys));res=ys-fit.x@bs
                    scores=np.zeros((fit.g,fit.p));np.add.at(scores,fit.codes,fit.x*(fit.w*res)[:,None])
                    cov=fit.inv@(scores.T@scores)@fit.inv*fit.correction(fit.g)
                    direct=abs(l@bs)/np.sqrt(l@cov@l);errors.append(float(abs(direct-ts[ix])))
        p=(hits+1)/(B+1)
        rows.append(dict(outcome=outcome,contrast=label,**r,p_webb=p,webb_draws=B,seed=SEED,mc_se=float(np.sqrt(p*(1-p)/(B+1)))))
    if errors:
        CHECK[tag+'_direct_bootstrap_error']=max(errors);assert max(errors)<1e-7
    print('FIT',tag,'N',fit.n,'G',fit.g,'contrasts',len(tests),flush=True)
    return rows

def main():
    all_data=load();d=events(all_data[all_data.birth.between(1975,1984)].copy())
    assert len(d)==217716 and set(d.year)==set(range(2021,2026))
    assert np.allclose(d.formal+d.informal,d.total)
    assert not d.duplicated(['year','BIRIMNO','FERTNO']).any()
    CHECK['sample_n']=len(d);CHECK['years']=sorted(map(int,d.year.unique()))
    rows=[]
    for (year,g),q in d.groupby(['year','treated']):
        rows.append(dict(year=int(year),treated=int(g),n=len(q),weight_sum=q.weight.sum(),
            **{c:weighted(q,c) for c in OUTCOMES},mean_age=weighted(q,'age'),
            conditional_informal=100*weighted(q,'informal')/weighted(q,'total')))
    csv('group_means.csv',rows)
    results=[];contrasts=[];fits={}
    for spec,adjust in [('base',False),('adjusted',True)]:
        fit=WeightedFit(d,event_rhs(adjust=adjust),OUTCOMES);fits[spec]=fit
        tests=[(o,{f'e{y}':1},str(y)) for o in OUTCOMES for y in ec.EVENTS]
        results += [dict(specification=spec,**r) for r in inference(fit,tests,spec,True)]
        contrasts += [dict(specification=spec,**r) for r in inference(fit,[('informal',{'e2025':1,'e2023':-1},'2025-2023'),('informal',{'e2025':1,'e2021':-1},'2025 versus 2021')],spec+'_contrasts')]
    csv('dynamic_results.csv',results);csv('time_contrasts.csv',contrasts)
    # Independently validate CR1 against statsmodels, and base cell equivalence.
    f=fits['base'];m=sm.WLS(f.y[:,0],f.x,weights=f.w).fit(cov_type='cluster',cov_kwds={'groups':d.birth})
    CHECK['statsmodels_coef_max_error']=float(np.max(np.abs(m.params-f.beta[:,0])))
    CHECK['statsmodels_cov_max_error']=float(np.max(np.abs(m.cov_params()-f.v_c[0])))
    assert CHECK['statsmodels_coef_max_error']<1e-7 and CHECK['statsmodels_cov_max_error']<1e-7
    cells=[]
    for (birth,year),q in d.groupby(['birth','year']):
        cells.append(dict(birth=birth,year=year,treated=int(birth<=1981),household=f'{birth}_{year}',weight=q.weight.sum(),**{o:weighted(q,o) for o in OUTCOMES}))
    cf=WeightedFit(events(pd.DataFrame(cells)),event_rhs(),OUTCOMES)
    CHECK['cell_coefficient_max_error']=float(np.max(np.abs(cf.beta-f.beta)))
    assert CHECK['cell_coefficient_max_error']<1e-7
    # Exhaustive work-status partition; the denominator always remains all men.
    inf=d.informal.eq(100);status=d.ISTEKI_DURUM_K
    masks={'informal_wage':inf&status.isin([11,12]),'informal_employer':inf&status.eq(2),
           'informal_own':inf&status.eq(3),'informal_family':inf&status.eq(4)}
    for o,mask in masks.items():d[o]=100.*mask
    assert np.allclose(d[list(masks)].sum(axis=1),d.informal)
    pf=WeightedFit(d,event_rhs(adjust=True),list(masks))
    rr=inference(pf,[(o,{f'e{y}':1},str(y)) for o in masks for y in ec.EVENTS],'partitions',True)
    for y in ec.EVENTS:
        maincoef=fits['adjusted'].contrast({f'e{y}':1},'informal',False)['coef']
        assert abs(sum(z['coef'] for z in rr if z['contrast']==str(y))-maincoef)<1e-8
    csv('informality_partitions.csv',rr)
    CHECK['partitions_additive']=True
    # Sensitivity definitions fixed by scope; no selection based on significance.
    scenarios=[('region_year',d,event_rhs(adjust=True,region_year=True)),
      ('exclude5',d[~d.region2.isin(['TR62','TR63','TRB1','TRC1','TRC2'])],event_rhs(adjust=True)),
      ('window1972',events(all_data[all_data.birth.between(1972,1984)]),event_rhs(adjust=True)),
      ('window1978',events(all_data[all_data.birth.between(1978,1984)]),event_rhs(adjust=True))]
    for b in [1982,1983,1984]:scenarios.append((f'omit{b}',d[d.birth.ne(b)],event_rhs(adjust=True)))
    robust=[r for r in results if r['outcome']=='informal']
    for name,q,rhs in scenarios:
        fit=WeightedFit(q,rhs,['informal'])
        robust += [dict(specification=name,**r) for r in inference(fit,[('informal',{f'e{y}':1},str(y)) for y in ec.EVENTS],name)]
    csv('sensitivity_results.csv',robust)
    # Match the archived results for this release, without manuscript dependencies.
    previous=pd.read_csv(ROOT/'reference_results/dynamic_results.csv')
    errors=[]
    for r in results:
        old=previous[(previous.specification==r['specification'])&(previous.outcome==r['outcome'])&(previous.contrast.astype(str)==r['contrast'])].iloc[0]
        errors.append(abs(old.coef-r['coef']))
    CHECK['reference_coefficient_max_error']=max(errors)
    assert max(errors)<1e-8
    CHECK['raw_sources_unchanged']=all(x['sha256']==ec.sha(Path(x['path'])) for x in json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8')))
    assert CHECK['raw_sources_unchanged']
    CHECK['versions']={k:v for k,v in [('python',platform.python_version()),('numpy',np.__version__),('pandas',pd.__version__),('scipy',scipy.__version__),('statsmodels',statsmodels.__version__)]}
    CHECK['inference']={'bootstrap':'null-restricted studentized cluster wild','rademacher':'all 2^G signs','webb_draws':B,'seed':SEED,'ci':'CR1 birth-cluster covariance and t(G-1), pointwise','p_values':'unadjusted, two-sided'}
    dump('analysis_validation.json',CHECK)
    print('COMPLETE',json.dumps(CHECK),flush=True)

if __name__=='__main__':main()
