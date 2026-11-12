"""Evaluate post-hoc intercept updates and transfer to other US cycles.

No person-level data are written; the updating cycle is not validation.
"""
from pathlib import Path
import json
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'work/pydeps'),str(ROOT/'work')]
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as old
from korea2019_independent_survey import predict
from review_intercept_recalibration import citl
from cross_country_calibration_and_matching import weighted_calibration

OUT=ROOT/'outputs'

def sample_old(year):
    d=old.load_cycle(year);d['outcome'],d['weight']=d.outcome_100,d.WTSAU2YR
    return d[d.age.between(40,69)&d.outcome.notna()&d.self_hearing_difficulty.notna()&d.weight.gt(0)].copy()

def sample_recent(cycle):
    d=nh.load(cycle);d['outcome']=d.outcome_100
    return d[d.age.between(40,69)&d.outcome.notna()&d.self_hearing_difficulty.notna()&d.weight.gt(0)].copy()

def main():
    card=json.loads((OUT/'增量模型_系数与插补规则_研究用.json').read_text(encoding='utf-8'))
    samples={'US_2001_2002':sample_old(2001),'US_2003_2004':sample_old(2003),
             'US_2011_2012':sample_recent('2011_2012'),'US_2015_2016':sample_recent('2015_2016')}
    update=samples['US_2015_2016'];rows=[]; curves=[]
    dev=pd.concat([c.korean_year(y) for y in (2010,2011,2012)],ignore_index=True)
    dev=dev[dev.age.ge(40)&dev.outcome.notna()&dev.self_hearing_difficulty.notna()&dev.weight.gt(0)]
    assert len(dev)==11679 and len(update)==2564
    dev_w=dev.weight/3
    dev_prev=float(np.average(dev.outcome,weights=dev_w));target_prev=float(np.average(update.outcome,weights=update.weight))
    for model in ('base','expanded'):
        prob=predict(update,card['models'][model]);shift=citl(update.outcome,prob,update.weight)
        for name,d in samples.items():
            y=d.outcome.to_numpy(int);w=d.weight.to_numpy(float)
            before=predict(d,card['models'][model]);after=expit(logit(before)+shift)
            i0,s0=weighted_calibration(y,before,w);i1,s1=weighted_calibration(y,after,w)
            a0=roc_auc_score(y,before,sample_weight=w);a1=roc_auc_score(y,after,sample_weight=w)
            assert np.isclose(i1,i0-shift,atol=1e-8) and np.isclose(a0,a1,atol=1e-12)
            rows.append({'model':model,'cohort':name,'n':len(d),'events':int(y.sum()),
                         'development_unweighted_prevalence':dev.outcome.mean(),
                         'development_weighted_prevalence':dev_prev,'update_weighted_prevalence':target_prev,
                         'update_cycle_prevalence_log_odds_difference_vs_development':logit(target_prev)-logit(dev_prev),
                         'weighted_prevalence':np.average(y,weights=w),'mean_pred_before':np.average(before,weights=w),
                         'mean_pred_after':np.average(after,weights=w),'intercept_update':shift,
                         'citl_before':i0,'citl_after':i1,'slope_before':s0,'slope_after':s1,
                         'auc_before':a0,'auc_after':a1,'brier_before':np.average((y-before)**2,weights=w),
                         'brier_after':np.average((y-after)**2,weights=w),
                         'status':'apparent same-sample update' if name=='US_2015_2016' else 'post-hoc transfer; not blinded'})
            if name=='US_2015_2016':
                # Fixed groups based on the original score keep the same people in each pair.
                groups=pd.qcut(pd.Series(before).rank(method='first'),q=10,labels=False).to_numpy()
                for group in range(10):
                    mask=groups==group
                    for label,p in [('Before',before),('After apparent update',after)]:
                        curves.append({'model':model,'bin':group+1,'stage':label,'n':int(mask.sum()),
                                       'mean_predicted':np.average(p[mask],weights=w[mask]),
                                       'observed':np.average(y[mask],weights=w[mask])})
    results=pd.DataFrame(rows); results.to_csv(OUT/'增量模型_美国2015同样本截距更新.csv',index=False)
    bins=pd.DataFrame(curves);bins.to_csv(OUT/'增量模型_同样本校准图分组汇总.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,5.1),dpi=180,sharex=True,sharey=True)
    for ax,model in zip(axes,('base','expanded')):
        x=bins[bins.model.eq(model)]
        ax.plot([0,1],[0,1],color='#687675',ls='--',lw=1,label='Ideal')
        for stage,color in [('Before','#2c6864'),('After apparent update','#b75b3d')]:
            z=x[x.stage.eq(stage)];ax.plot(z.mean_predicted,z.observed,marker='o',ms=3.5,lw=1.5,color=color,label=stage)
        ax.set_title(model.capitalize()+' model',loc='left',fontsize=11)
        ax.set(xlim=(0,.75),ylim=(0,.75),xlabel='Mean predicted probability')
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Observed outcome proportion')
    axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    fig.suptitle('US 2015-2016: intercept update on the same sample',fontsize=13,ha='left',x=.08)
    fig.text(.08,.01,'Apparent recalibration, not independent validation; ten fixed groups by original prediction. Weighted.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(OUT/'增量模型_美国2015同样本校准图.png',dpi=220,bbox_inches='tight');plt.close(fig)
    print(results[['model','cohort','intercept_update','citl_before','citl_after','brier_before','brier_after','auc_before']].to_string(index=False))
    print('development unweighted/weighted prevalence',dev.outcome.mean(),dev_prev,'US weighted',target_prev,'log odds diff',logit(target_prev)-logit(dev_prev))

if __name__=='__main__':main()
