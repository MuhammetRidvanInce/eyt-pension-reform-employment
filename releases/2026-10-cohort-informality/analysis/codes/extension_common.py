"""Shared read-only data resolution and transparent weighted least squares."""
from pathlib import Path
import hashlib, itertools, json, platform, os
import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm, t

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/outputs/cohort_informality_2026-10-15'
OUT.mkdir(parents=True,exist_ok=True)
YEARS=[2021,2022,2023,2024,2025]
EVENTS=[2021,2023,2024,2025]


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def source(year,family='hia',original=False):
    if family != 'hia' or original:
        raise ValueError('This release supports the selected HLFS input files only.')
    setting=os.environ.get('EYT_HLFS_CONFIG')
    if not setting:
        raise RuntimeError('Set EYT_HLFS_CONFIG to your private JSON map of year to licensed CSV file.')
    config=Path(setting).expanduser().resolve()
    mapping=json.loads(config.read_text(encoding='utf-8-sig'))
    p=Path(mapping[str(year)]).expanduser()
    if not p.is_absolute(): p=config.parent/p
    if not p.is_file(): raise FileNotFoundError(p)
    return p.resolve()


def read_csv(p,cols=None):
    with Path(p).open(encoding='iso-8859-9') as f:header=f.readline()
    sep=';' if header.count(';')>header.count(',') else ','
    return pd.read_csv(p,sep=sep,encoding='iso-8859-9',usecols=cols,low_memory=False)


def save_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save_csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')


class WeightedFit:
    """CR1 WLS; few-cluster restricted, studentized wild bootstrap diagnostic.

    Cohort clustering and sign flips assume independent cohort disturbances;
    this is a sensitivity calculation, not exact design-based randomization.
    """
    def __init__(self,d,rhs,outcomes,weight='weight',household='household',cluster='birth'):
        x=patsy.dmatrix(rhs,d,return_type='dataframe',NA_action='raise')
        self.names=list(x.columns); self.x=np.asarray(x,float)
        self.y=d[outcomes].to_numpy(float); self.outcomes=outcomes
        assert np.isfinite(self.x).all() and np.isfinite(self.y).all()
        self.w=d[weight].to_numpy(float); assert (self.w>=0).all()
        self.w=self.w/self.w.mean(); self.n,self.p=self.x.shape
        gram=self.x.T@(self.w[:,None]*self.x)
        assert np.linalg.matrix_rank(gram)==self.p,(self.p,np.linalg.matrix_rank(gram))
        self.inv=np.linalg.inv(gram)
        self.beta=self.inv@(self.x.T@(self.w[:,None]*self.y))
        self.residual=self.y-self.x@self.beta
        self.hh_codes,self.hh_labels=pd.factorize(d[household],sort=True)
        self.codes,self.labels=pd.factorize(d[cluster],sort=True)
        self.g=len(self.labels)
        self.v_h=[];self.v_c=[];self.raw_c=[]
        for j in range(len(outcomes)):
            individual=self.x*(self.w*self.residual[:,j])[:,None]
            hs=np.zeros((len(self.hh_labels),self.p)); np.add.at(hs,self.hh_codes,individual)
            cs=np.zeros((self.g,self.p));np.add.at(cs,self.codes,individual)
            self.raw_c.append(cs)
            self.v_h.append(self.inv@(hs.T@hs)@self.inv*self.correction(len(self.hh_labels)))
            self.v_c.append(self.inv@(cs.T@cs)@self.inv*self.correction(self.g))
        self.grams=np.stack([self.x[self.codes==g].T@(self.w[self.codes==g,None]*self.x[self.codes==g]) for g in range(self.g)])
        self.signs=np.asarray(list(itertools.product([-1.,1.],repeat=self.g))) if self.g<=12 else np.random.default_rng(7438).choice([-1.,1.],size=(9999,self.g))

    def correction(self,g):return g/(g-1)*(self.n-1)/(self.n-self.p)

    def contrast(self,terms,outcome,wild=True):
        j=self.outcomes.index(outcome);l=np.zeros(self.p)
        for k,v in terms.items():l[self.names.index(k)]=v
        b=float(l@self.beta[:,j]);sh=float(np.sqrt(max(0,l@self.v_h[j]@l)));sc=float(np.sqrt(max(0,l@self.v_c[j]@l)))
        row=dict(coef=b,se_household=sh,p_household=float(2*norm.sf(abs(b/sh))) if sh else None,
                 ci_low=b-norm.ppf(.975)*sh,ci_high=b+norm.ppf(.975)*sh,
                 se_cluster=sc,p_cluster_t=float(2*t.sf(abs(b/sc),self.g-1)) if sc else None,
                 ci_cluster_low=b-t.ppf(.975,self.g-1)*sc,ci_cluster_high=b+t.ppf(.975,self.g-1)*sc,
                 n=self.n,clusters=self.g,households=len(self.hh_labels))
        if wild:
            u=self.inv@l
            shift=u*b/float(l@u)
            # Null-restricted scores: Xg'W e_restricted.
            scores=self.raw_c[j]+np.einsum('gij,j->gi',self.grams,shift)
            delta=(self.signs@scores)@self.inv
            influence=scores@u
            residual_scores=self.signs*influence[None,:]-delta@np.einsum('gij,j->gi',self.grams,u).T
            var=self.correction(self.g)*np.sum(residual_scores**2,axis=1)
            valid=var>1e-24
            boot=np.abs((delta@l)[valid])/np.sqrt(var[valid])
            if sc and valid.all():
                hits=np.sum(boot>=abs(b/sc)-1e-10)
                row['p_wild_cluster_t']=float(hits/len(boot)) if self.g<=12 else float((hits+1)/(len(boot)+1))
            else:row['p_wild_cluster_t']=None
            row['wild_draws']=int(valid.sum())
        return row


def events(d,exposed='treated'):
    d=d.copy()
    for y in EVENTS:d[f'e{y}']=d[exposed]*d.year.eq(y)
    return d


def event_rhs(unit='birth',adjust=False,region_year=False):
    rhs=f'C({unit})+C(year)+'+'+'.join(f'e{y}' for y in EVENTS)
    if adjust:rhs+=' + C(education) + C(region)'
    if region_year:rhs+=' + C(region):C(year)'
    return rhs
