"""Post-hoc intercept updating on one US cycle, evaluated on other US cycles."""
from pathlib import Path
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps'));sys.path.insert(0,str(ROOT/'work'))
warnings.filterwarnings('ignore',category=FutureWarning);warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as older

def citl(y,p,w):
    w=np.asarray(w,float);w=w/w.mean()
    x=logit(np.clip(p,1e-7,1-1e-7))
    return float(np.asarray(sm.GLM(np.asarray(y),np.ones((len(y),1)),offset=x,
                                   family=sm.families.Binomial(),freq_weights=w).fit().params)[0])

def sample(year):
    d=older.load_cycle(year);d['outcome']=d.outcome_100;d['weight']=d.WTSAU2YR
    return d[d.age.between(40,69)&d.outcome.notna()&d.weight.gt(0)].copy()

def main():
    dev=pd.concat([c.korean_year(y) for y in (2010,2011,2012)],ignore_index=True)
    dev=dev[dev.age.ge(40)&dev.outcome.notna()]
    model=c.logistic();model.fit(dev[c.FEATURES],dev.outcome.astype(int),groups=dev.group)
    update=sample(2001); p=model.predict_proba(update[c.FEATURES])[:,1]
    delta=citl(update.outcome,p,update.weight)
    targets=[('US_2001_2002_apparent_update',update),('US_2003_2004',sample(2003))]
    for cycle in ('2011_2012','2015_2016'):
        d=nh.load(cycle);d['outcome']=d.outcome_100
        targets.append((f'US_{cycle}',d[d.age.between(40,69)&d.outcome.notna()&d.weight.gt(0)].copy()))
    rows=[]
    for name,d in targets:
        y=d.outcome.to_numpy(int);w=d.weight.to_numpy(float)
        before=model.predict_proba(d[c.FEATURES])[:,1];after=expit(logit(np.clip(before,1e-7,1-1e-7))+delta)
        rows.append({'cohort':name,'n':len(d),'events':int(y.sum()),'update_intercept':delta,
                     'citl_before':citl(y,before,w),'citl_after':citl(y,after,w),
                     'brier_before':np.average((y-before)**2,weights=w),'brier_after':np.average((y-after)**2,weights=w),
                     'auc_before':roc_auc_score(y,before,sample_weight=w),'auc_after':roc_auc_score(y,after,sample_weight=w),
                     'status':'post-hoc; same-cycle apparent update' if 'apparent' in name else 'post-hoc transfer to separate cycle; not blinded'})
    pd.DataFrame(rows).to_csv(ROOT/'outputs/复核后_截距再校准_探索性.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
