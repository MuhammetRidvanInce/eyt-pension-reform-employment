"""Publication figures for the Turkish, informality-focused manuscript."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/outputs/cohort_informality_2026-10-15'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'axes.edgecolor':'#8a99a6',
 'axes.linewidth':.7,'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})
COLORS=['#202020']*4
HIGHLIGHT_COLOR='#c45b19'

def panel(ax,q,label,color):
    q=q.assign(year=q.contrast.astype(int)).sort_values('year')
    vals={int(r.year):(r.coef,r.ci_cluster_low,r.ci_cluster_high) for r in q.itertuples()}
    vals[2022]=(0,0,0)
    years=np.arange(2021,2026);v=np.array([vals[y] for y in years])
    ax.axvspan(2022.5,2025.5,color='#edf2f6',zorder=0)
    ax.axhline(0,color='#7f8991',lw=.85,zorder=1)
    ax.plot(years,v[:,0],color=color,lw=1,ls='--',alpha=.8,zorder=2)
    for highlighted in [False,True]:
        mask=(years!=2022)&(np.isin(years,[2024,2025])==highlighted)
        ax.errorbar(years[mask],v[mask,0],yerr=np.vstack([v[mask,0]-v[mask,1],v[mask,2]-v[mask,0]]),
            fmt='s' if highlighted else 'o',ms=4.8 if highlighted else 4.3,
            elinewidth=2.5 if highlighted else 1.7,capthick=2.1 if highlighted else 1.5,
            capsize=3.5 if highlighted else 3,color=HIGHLIGHT_COLOR if highlighted else color,zorder=3)
    ax.plot(2022,0,'o',mfc='white',mec=color,ms=5,zorder=4)
    ax.set(title=label,xlabel='Anket yılı',ylabel='Yüzde puan',xlim=(2020.65,2025.35),xticks=years)
    ax.grid(axis='y',color='#e1e7ec',lw=.6,alpha=.8);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

def make(data,items,name):
    fig,axs=plt.subplots(2,2,figsize=(10.8,7.2),layout='constrained')
    for ax,(o,title),c in zip(axs.flat,items,COLORS):panel(ax,data[data.outcome==o],title,c)
    for ext in ['png','svg','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=300)
    plt.close(fig)

def make_partitions(data):
    items=[('informal_wage','Ücretli ve yevmiyeli'),('informal_own','Kendi hesabına'),
           ('informal_employer','İşveren'),('informal_family','Ücretsiz aile işçisi')]
    fig,ax=plt.subplots(figsize=(10.8,7.2),layout='constrained')
    centers=np.arange(len(items))
    for i in centers:
        if i%2==0:ax.axhspan(i-.43,i+.43,color='#f2f5f7',zorder=0)
    ax.axvline(0,color='#7f8991',lw=1,zorder=1)
    for year,offset,marker in [(2023,-.23,'o'),(2024,0,'^'),(2025,.23,'s')]:
        q=data[data.contrast==year].set_index('outcome').loc[[o for o,_ in items]]
        v=q[['coef','ci_cluster_low','ci_cluster_high']].to_numpy()
        highlighted=year>=2024
        color=HIGHLIGHT_COLOR if highlighted else '#202020'
        ax.errorbar(v[:,0],centers+offset,xerr=np.vstack([v[:,0]-v[:,1],v[:,2]-v[:,0]]),
            fmt=marker,ms=6 if highlighted else 5,color=color,
            mfc='white' if year==2024 else color,
            elinewidth=2.5 if highlighted else 1.7,capthick=2.1 if highlighted else 1.5,
            capsize=3.5 if highlighted else 3,label=str(year),zorder=3)
    ax.set(yticks=centers,yticklabels=[title for _,title in items],
           xlabel='Kayıt dışı istihdam farkı (yüzde puan)',ylim=(3.55,-.55))
    ax.tick_params(axis='y',length=0,pad=12,labelsize=11)
    ax.grid(axis='x',color='#e1e7ec',lw=.6);ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False)
    ax.legend(loc='lower right',ncol=3,frameon=False)
    for ext in ['png','svg','pdf']:
        fig.savefig(FIG/('sekil2_kayitdisi_calisma_bicimleri.'+ext),dpi=300)
    plt.close(fig)

main=pd.read_csv(OUT/'dynamic_results.csv')
make(main[main.specification=='base'],[
 ('formal','Kayıtlı istihdam'),('informal','Kayıt dışı istihdam'),
 ('total','Toplam istihdam'),('retired','Emeklilik nedeniyle işgücü dışında')],'sekil1_kohort_istihdam')
part=pd.read_csv(OUT/'informality_partitions.csv')
make_partitions(part)
(OUT/'figure_manifest.json').write_text(json.dumps({'data':['dynamic_results.csv','informality_partitions.csv'],
 'figures':['sekil1_kohort_istihdam','sekil2_kayitdisi_calisma_bicimleri'],
 'intervals':'birth-cluster CR1 and t(G-1), pointwise 95%, not bootstrap-inverted',
 'reference_year':2022,'ci_linewidth':1.7,'highlight_years':[2024,2025],
 'base_color':'#202020','highlight_color':HIGHLIGHT_COLOR,'connecting_line':'dashed',
 'highlight_ci_linewidth':2.5,'layouts':{'figure1':'four-panel event study','figure2':'horizontal grouped coefficients, 2023-2025'},
 'figure2_markers':{'2023':'circle','2024':'open triangle','2025':'filled square'},
 'no_embedded_suptitle_or_notes':True,'dpi':300},indent=2),encoding='utf-8')
print('Two figures created: PNG, SVG, PDF.')
