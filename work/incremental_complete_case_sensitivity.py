"""Exploratory complete-case retraining and same-participant comparisons."""
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
import validate_nhanes_2003_2004 as older


def main():
    years = {year:c.korean_year(year) for year in (2010,2011,2012,2019,2020,2021,2022)}
    dev = pd.concat([years[y] for y in (2010,2011,2012)], ignore_index=True)
    dev = dev[dev.age.ge(40) & dev.outcome.notna() & dev.self_hearing_difficulty.notna() & dev.weight.gt(0)].copy()
    complete_dev = dev[dev[inc.EXT].notna().all(axis=1)].copy()
    models = {}
    for name, training in [('original_imputation', dev), ('complete_case_retrained', complete_dev)]:
        models[name] = (inc.fit(training, inc.BASE), inc.fit(training, inc.EXT))
    sources = []
    for year in (2019,2020,2021,2022):
        d=years[year]
        sources.append((f'Korea_{year}',d[d.age.between(40,69)&d.outcome.notna()&d.quality_pass].copy()))
    for year in (2001,2003):
        d=older.load_cycle(year)
        d['outcome'],d['weight']=d.outcome_100,d.WTSAU2YR
        sources.append((f'US_{year}_{year+1}',d[d.age.between(40,69)&d.outcome.notna()].copy()))
    for cycle in ('2011_2012','2015_2016'):
        d=nh.load(cycle)
        d['outcome']=d.outcome_100
        sources.append((f'US_{cycle}',d[d.age.between(40,69)&d.outcome.notna()].copy()))
    rows=[]
    for name,d in sources:
        d=d[d[inc.EXT].notna().all(axis=1)&d.weight.gt(0)].copy()
        y=d.outcome.to_numpy(dtype=int)
        for scenario,(base,expanded) in models.items():
            p0=base.predict_proba(d[inc.BASE])[:,1]
            p1=expanded.predict_proba(d[inc.EXT])[:,1]
            a=roc_auc_score(y,p0,sample_weight=d.weight)
            b=roc_auc_score(y,p1,sample_weight=d.weight)
            rows.append({'cohort':name,'scenario':scenario,'n':len(d),'events':int(y.sum()),
                         'base_auc_weighted':a,'expanded_auc_weighted':b,'delta_auc_weighted':b-a,
                         'base_brier_weighted':np.average((y-p0)**2,weights=d.weight),
                         'expanded_brier_weighted':np.average((y-p1)**2,weights=d.weight)})
        print(name,'complete n',len(d),'delta',round(rows[-1]['delta_auc_weighted'],4),flush=True)
    out=ROOT/'outputs'
    pd.DataFrame(rows).to_csv(out/'incremental_complete_case_sensitivity.csv',index=False)
    summary={'status':'Post hoc descriptive sensitivity; not independent of prior inspection.',
             'development_original_n':len(dev),'development_complete_n':len(complete_dev),
             'development_original_events':int(dev.outcome.sum()),
             'development_complete_events':int(complete_dev.outcome.sum()),
             'models':{name:{'base_C':base.best_params_['model__C'],'expanded_C':exp.best_params_['model__C']}
                       for name,(base,exp) in models.items()},
             'limitations':['Complete-case selection can introduce bias and changes the target sample.',
                            'Tuning and fitting remain unweighted; results have no design-based uncertainty.',
                            'Observed self-report and complete outcome remain required in both strategies.'],
             'results':rows}
    (out/'incremental_complete_case_sensitivity.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
    main()
