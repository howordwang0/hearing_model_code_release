"""Secondary >=40 dB better-ear PTA outcome and 888 audit."""
from pathlib import Path
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps')); sys.path.insert(0,str(ROOT/'work'))
warnings.filterwarnings('ignore',category=FutureWarning); warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as old

def main():
    years={y:c.korean_year(y) for y in (2010,2011,2012,2019,2020,2021,2022)}
    dev=pd.concat([years[y] for y in (2010,2011,2012)],ignore_index=True)
    dev=dev[dev.age.ge(40)&dev.outcome.notna()]
    model=c.logistic(); model.fit(dev[c.FEATURES],dev.outcome.astype(int),groups=dev.group)
    threshold=json.loads((ROOT/'outputs/cross_country_hearing_exploratory_report.json').read_text(encoding='utf-8'))['development']['risk_threshold_for_80pct_sensitivity_from_development_oof']
    def compare(z,p,y):
        mask=z.self_hearing_difficulty.notna().to_numpy()
        zz=z.loc[mask]; yy=np.asarray(y)[mask]; pp=p[mask]
        if len(np.unique(yy))<2: return {'comparison_n':len(zz)}
        model_scores=c.screen(yy,pp>=threshold,zz.weight)
        self_scores=c.screen(yy,zz.self_hearing_difficulty.eq(1),zz.weight)
        return {'comparison_n':len(zz),'model_sensitivity':model_scores['sensitivity'],
                'model_referrals_per_1000':model_scores['referrals_per_1000'],
                'self_report_sensitivity':self_scores['sensitivity'],
                'self_report_referrals_per_1000':self_scores['referrals_per_1000']}
    rows=[]
    for year in (2019,2020,2021,2022):
        d=years[year]; z=d[d.age.between(40,69)&d.outcome_40.notna()&d.quality_pass&d.weight.gt(0)].copy()
        p=model.predict_proba(z[c.FEATURES])[:,1]
        rows.append({'cohort':f'Korea_{year}','outcome':'better_ear_pta_ge40','n':len(z),'events':int(z.outcome_40.sum()),
                     'prevalence_weighted':np.average(z.outcome_40,weights=z.weight),
                     'auc_weighted':roc_auc_score(z.outcome_40,p,sample_weight=z.weight),
                     **compare(z,p,z.outcome_40)})
    for year in (2001,2003):
        d=old.load_cycle(year); d['weight']=d.WTSAU2YR
        z=d[d.age.between(40,69)&d.outcome_100_40.notna()&d.weight.gt(0)].copy()
        p=model.predict_proba(z[c.FEATURES])[:,1]
        rows.append({'cohort':f'US_{year}_{year+1}','outcome':'better_ear_pta_ge40','n':len(z),'events':int(z.outcome_100_40.sum()),
                     'prevalence_weighted':np.average(z.outcome_100_40,weights=z.weight),
                     'auc_weighted':roc_auc_score(z.outcome_100_40,p,sample_weight=z.weight),
                     'n_666_any':int(z.no_response_any.sum()),'n_888_any':int(z.n_could_not_obtain_888.gt(0).sum()),
                     **compare(z,p,z.outcome_100_40)})
    for cycle in ('2011_2012','2015_2016'):
        d=nh.load(cycle); z=d[d.age.between(40,69)&d.outcome_100_40.notna()&d.weight.gt(0)].copy()
        p=model.predict_proba(z[c.FEATURES])[:,1]
        rows.append({'cohort':f'US_{cycle}','outcome':'better_ear_pta_ge40','n':len(z),'events':int(z.outcome_100_40.sum()),
                     'prevalence_weighted':np.average(z.outcome_100_40,weights=z.weight),
                     'auc_weighted':roc_auc_score(z.outcome_100_40,p,sample_weight=z.weight),
                     'n_666_any':int(z.n_no_response_666.gt(0).sum()),'n_888_any':int(z.n_could_not_obtain_888.gt(0).sum()),
                     **compare(z,p,z.outcome_100_40)})
    out=ROOT/'outputs'; pd.DataFrame(rows).to_csv(out/'hearing_outcome_ge40_sensitivity.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
