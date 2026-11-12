"""Exploratory added value of routine predictors beyond age/sex/self-report.

All datasets here have been inspected previously. This analysis does not use
the reserved NHANES 2001-2002 cycle or establish independent validation.
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
from sklearn.model_selection import GroupKFold
import cross_country_hearing_pilot as c
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as older

BASE = ['age', 'female', 'self_hearing_difficulty']
EXT = c.FEATURES + ['self_hearing_difficulty']


def metric(y, p, weight):
    return {'auc_weighted': float(roc_auc_score(y, p, sample_weight=weight)),
            'brier_weighted': float(np.average((y-p)**2, weights=weight))}


def fit(data, features):
    model = c.logistic()
    model.fit(data[features], data.outcome.astype(int), groups=data.group)
    return model


def add_result(rows, name, y, p_base, p_ext, weight):
    a, b = metric(y, p_base, weight), metric(y, p_ext, weight)
    rows.append({'cohort': name, 'n': len(y), 'events': int(y.sum()),
                 'base_auc_weighted': a['auc_weighted'], 'expanded_auc_weighted': b['auc_weighted'],
                 'delta_auc_weighted': b['auc_weighted']-a['auc_weighted'],
                 'base_brier_weighted': a['brier_weighted'], 'expanded_brier_weighted': b['brier_weighted'],
                 'delta_brier_weighted': b['brier_weighted']-a['brier_weighted']})
    print(name, 'AUC', round(a['auc_weighted'], 3), round(b['auc_weighted'], 3), flush=True)


def main():
    years = {y: c.korean_year(y) for y in (2010, 2011, 2012, 2019, 2020, 2021, 2022)}
    dev = pd.concat([years[y] for y in (2010, 2011, 2012)], ignore_index=True)
    dev = dev[dev.age.ge(40) & dev.outcome.notna() & dev.self_hearing_difficulty.notna() & dev.weight.gt(0)].copy()
    oof_base, oof_ext = np.zeros(len(dev)), np.zeros(len(dev))
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(dev, dev.outcome, dev.group), 1):
        train, test = dev.iloc[tr], dev.iloc[te]
        for features, output in [(BASE, oof_base), (EXT, oof_ext)]:
            model = fit(train, features)
            output[te] = model.predict_proba(test[features])[:, 1]
        print('grouped nested fold', fold, 'complete', flush=True)
    y = dev.outcome.astype(int).to_numpy()
    rows = []
    add_result(rows, 'Korea_2010_2012_nested_cv_40plus', y, oof_base, oof_ext, dev.weight / 3)
    base, expanded = fit(dev, BASE), fit(dev, EXT)
    cycles = []
    for year in (2019, 2020, 2021, 2022):
        d = years[year]
        cycles.append((f'Korea_{year}', d[d.age.between(40, 69) & d.outcome.notna() & d.quality_pass]))
    d = older.load_cycle()
    d['outcome'], d['weight'] = d.outcome_100, d.WTSAU2YR
    cycles.append(('US_2003_2004', d[d.age.between(40, 69) & d.outcome.notna()]))
    for cycle in ('2011_2012', '2015_2016'):
        d = nh.load(cycle)
        d['outcome'] = d.outcome_100
        cycles.append((f'US_{cycle}', d[d.age.between(40, 69) & d.outcome.notna()]))
    for name, d in cycles:
        d = d[d.self_hearing_difficulty.notna() & d.weight.gt(0)].copy()
        y = d.outcome.astype(int).to_numpy()
        add_result(rows, name, y, base.predict_proba(d[BASE])[:, 1],
                   expanded.predict_proba(d[EXT])[:, 1], d.weight)
    result = {'status': 'Exploratory after prior inspection of all these evaluation samples; no independent validation.',
              'comparison': {'base': BASE, 'expanded': EXT},
              'base_selected_C': base.best_params_['model__C'],
              'expanded_selected_C': expanded.best_params_['model__C'],
              'limitations': ['Models and tuning are unweighted; weighted point estimates only.',
                              'Self-report questions and response categories vary across surveys.',
                              'All comparisons restricted to observed self-report; selection may matter.',
                              'AUC and Brier differences lack design-based uncertainty.',
                              'Prior data inspection influenced research question and model design.'],
              'results': rows}
    (ROOT/'outputs/incremental_self_report_exploratory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(ROOT/'outputs/incremental_self_report_exploratory.csv', index=False)


if __name__ == '__main__':
    main()
