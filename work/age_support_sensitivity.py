"""Explore whether restricting Korean training ages changes transfer estimates.

All evaluation periods have already been inspected. This is not a new locked
external validation or a replacement for the prespecified primary analysis.
"""
from pathlib import Path
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work/pydeps'))
sys.path.insert(0, str(ROOT/'work'))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c
import incremental_self_report_models as inc
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as old


def main():
    years = {year:c.korean_year(year) for year in (2010,2011,2012,2019,2020,2021,2022)}
    dev = pd.concat([years[y] for y in (2010,2011,2012)], ignore_index=True)
    dev = dev[dev.age.ge(40)&dev.outcome.notna()&dev.self_hearing_difficulty.notna()&dev.weight.gt(0)].copy()
    restricted = dev[dev.age.between(40,69)].copy()
    models = {label:(inc.fit(sample,inc.BASE),inc.fit(sample,inc.EXT))
              for label,sample in [('age_40plus',dev),('age_40_69',restricted)]}
    sources=[]
    for year in (2019,2020,2021,2022):
        d=years[year]
        sources.append((f'Korea_{year}',d[d.age.between(40,69)&d.outcome.notna()&d.quality_pass].copy()))
    for year in (2001,2003):
        d=old.load_cycle(year)
        d['outcome'],d['weight']=d.outcome_100,d.WTSAU2YR
        sources.append((f'US_{year}_{year+1}',d[d.age.between(40,69)&d.outcome.notna()].copy()))
    for cycle in ('2011_2012','2015_2016'):
        d=nh.load(cycle)
        d['outcome']=d.outcome_100
        sources.append((f'US_{cycle}',d[d.age.between(40,69)&d.outcome.notna()].copy()))
    rows=[]
    for name,d in sources:
        d=d[d.self_hearing_difficulty.notna()&d.weight.gt(0)].copy()
        y=d.outcome.to_numpy(dtype=int)
        for label,(base,expanded) in models.items():
            p0=base.predict_proba(d[inc.BASE])[:,1]
            p1=expanded.predict_proba(d[inc.EXT])[:,1]
            a=roc_auc_score(y,p0,sample_weight=d.weight)
            b=roc_auc_score(y,p1,sample_weight=d.weight)
            rows.append({'cohort':name,'training_age':label,'n':len(d),'events':int(y.sum()),
                         'base_auc_weighted':a,'expanded_auc_weighted':b,'delta_auc_weighted':b-a,
                         'base_brier_weighted':np.average((y-p0)**2,weights=d.weight),
                         'expanded_brier_weighted':np.average((y-p1)**2,weights=d.weight)})
        print(name,'age-restricted delta AUC',round(rows[-1]['delta_auc_weighted'],4),flush=True)
    out=ROOT/'outputs'
    pd.DataFrame(rows).to_csv(out/'incremental_training_age_sensitivity.csv',index=False)
    summary={'status':'Post hoc sensitivity analysis after evaluation cohorts were inspected.',
             'full_training_n':len(dev),'full_training_events':int(dev.outcome.sum()),
             'age_restricted_training_n':len(restricted),'age_restricted_training_events':int(restricted.outcome.sum()),
             'model_C':{label:{'base':base.best_params_['model__C'],'expanded':exp.best_params_['model__C']}
                        for label,(base,exp) in models.items()},
             'limitations':['Not independently validated: age restriction was chosen after external data inspection.',
                            'Training and tuning are unweighted; sampling uncertainty not computed.',
                            'Same validation participants used for both training-age models.'],
             'results':rows}
    (out/'incremental_training_age_sensitivity.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
    main()
