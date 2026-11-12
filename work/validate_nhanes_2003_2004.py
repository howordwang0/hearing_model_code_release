"""Evaluate the previously specified Korean model on NHANES 2003-2004.

The analysis plan was written before loading this cycle. Outputs are aggregate.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work/pydeps'))
sys.path.insert(0, str(ROOT / 'work'))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c
from cross_country_calibration_and_matching import weighted_calibration

OUT = ROOT / 'outputs'
DATA = ROOT / 'work/data/nhanes/2003_2004'


def load_cycle(year=2003):
    if year not in (2001, 2003):
        raise ValueError('Supported cycles: 2001 and 2003')
    suffix = 'b' if year == 2001 else 'c'
    data = ROOT / f'work/data/nhanes/{year}_{year+1}'
    creatinine = 'LBDSCR' if year == 2001 else 'LBXSCR'
    specs = {
        f'demo_{suffix}': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'SDMVPSU', 'SDMVSTRA'],
        f'aux_{suffix}': ['SEQN', 'WTSAU2YR', 'AUXU500R', 'AUXU1K1R', 'AUXU2KR', 'AUXU4KR',
                  'AUXU500L', 'AUXU1K1L', 'AUXU2KL', 'AUXU4KL'],
        f'bmx_{suffix}': ['SEQN', 'BMXBMI'], f'l40_{suffix}': ['SEQN', creatinine],
        f'l25_{suffix}': ['SEQN', 'LBXHGB'], f'diq_{suffix}': ['SEQN', 'DIQ010'],
        f'smq_{suffix}': ['SEQN', 'SMQ020', 'SMQ040'], f'auq_{suffix}': ['SEQN', 'AUQ130'],
    }
    d = None
    for file, columns in specs.items():
        frame, _ = pyreadstat.read_xport(data / (file + '.xpt'), usecols=columns)
        if not frame.SEQN.is_unique or frame.SEQN.isna().any():
            raise ValueError(f'Invalid or repeated participant identifier in {file}')
        d = frame if d is None else d.merge(frame, on='SEQN', how='left', validate='one_to_one')
    d['age'] = d.RIDAGEYR
    d['female'] = d.RIAGENDR.map({1.: 0., 2.: 1.})
    d['bmi'], d['creatinine'], d['hemoglobin'] = d.BMXBMI, d[creatinine], d.LBXHGB
    d['diabetes'] = d.DIQ010.map({1.: 1., 2.: 0.})
    d['ever_smoked'] = d.SMQ020.map({1.: 1., 2.: 0.})
    d['current_smoker'] = np.nan
    d.loc[d.SMQ020.eq(2), 'current_smoker'] = 0
    d.loc[d.SMQ020.eq(1) & d.SMQ040.eq(3), 'current_smoker'] = 0
    d.loc[d.SMQ020.eq(1) & d.SMQ040.isin([1, 2]), 'current_smoker'] = 1
    d['self_hearing_difficulty'] = d.AUQ130.map({1.: 0., 2.: 1., 3.: 1., 4.: 1.})
    freq = specs[f'aux_{suffix}'][2:]
    d['no_response_any'] = d[freq].eq(666).any(axis=1)
    d['n_could_not_obtain_888'] = d[freq].eq(888).sum(axis=1)
    for value in (100, 120):
        vals = d[freq].replace(666, value)
        vals = vals.where(vals.ge(-10) & vals.le(120))
        complete = vals.notna().all(axis=1)
        pta = np.minimum(vals.iloc[:, :4].mean(axis=1), vals.iloc[:, 4:].mean(axis=1))
        d[f'outcome_{value}'] = np.where(complete, (pta > 25).astype(float), np.nan)
        d[f'outcome_{value}_40'] = np.where(complete, (pta >= 40).astype(float), np.nan)
    return d


def main():
    report = json.loads((OUT / 'cross_country_hearing_exploratory_report.json').read_text(encoding='utf-8'))
    threshold = report['development']['risk_threshold_for_80pct_sensitivity_from_development_oof']
    dev = pd.concat([c.korean_year(y) for y in (2010, 2011, 2012)], ignore_index=True)
    dev = dev[(dev.age >= 40) & dev.outcome.notna()]
    assert len(dev) == report['development']['n'] and int(dev.outcome.sum()) == report['development']['events']
    model = c.logistic()
    model.fit(dev[c.FEATURES], dev.outcome.astype(int), groups=dev.group)
    age = c.logistic(False)
    age.fit(dev[['age']], dev.outcome.astype(int))
    assert model.best_params_['model__C'] == report['development']['selected_C']

    full = load_cycle()
    age_mask = full.age.between(40, 69)
    eligible = full[age_mask & full.outcome_100.notna() & full.WTSAU2YR.gt(0)].copy()
    assert eligible.outcome_100.equals(eligible.outcome_120)
    y = eligible.outcome_100.astype(int).to_numpy()
    w = eligible.WTSAU2YR.to_numpy()
    p = model.predict_proba(eligible[c.FEATURES])[:, 1]
    pa = age.predict_proba(eligible[['age']])[:, 1]
    intercept, slope = weighted_calibration(y, p, w)
    common = eligible.self_hearing_difficulty.notna().to_numpy()
    assert common.any() and len(np.unique(y)) == 2
    comparisons = []
    for name, positive in [('model_fixed', p[common] >= threshold),
                           ('age_60_plus', eligible.age.to_numpy()[common] >= 60),
                           ('self_report', eligible.self_hearing_difficulty.to_numpy()[common] == 1)]:
        comparisons.append({'strategy': name, **c.screen(y[common], positive, w[common])})

    result = {
        'status': 'Additional exploratory external sample; plan written before analysis; earlier US cycles influenced model choices.',
        'cycle': '2003-2004', 'age': '40-69',
        'source_population_age_40_69': int(age_mask.sum()),
        'n_outcome_and_positive_audiometry_weight': len(eligible), 'events': int(y.sum()),
        'positive_no_response_666_any': int(eligible.no_response_any.sum()),
        'binary_outcome_changes_100_vs_120': int((eligible.outcome_100 != eligible.outcome_120).sum()),
        'predictor_missing_n': {feature: int(eligible[feature].isna().sum()) for feature in c.FEATURES},
        'self_report_n': int(common.sum()), 'model_C': model.best_params_['model__C'],
        'fixed_threshold': threshold, 'weight': 'WTSAU2YR',
        'auc_core_weighted': float(roc_auc_score(y, p, sample_weight=w)),
        'auc_age_only_weighted': float(roc_auc_score(y, pa, sample_weight=w)),
        'brier_weighted': float(np.average((y-p)**2, weights=w)),
        'prevalence_weighted': float(np.average(y, weights=w)),
        'mean_prediction_weighted': float(np.average(p, weights=w)),
        'calibration_in_large_weighted': intercept, 'calibration_slope_weighted': slope,
        'comparisons': comparisons,
        'plan_sha256': hashlib.sha256((OUT/'NHANES_2003_2004_验证方案_分析前记录.md').read_bytes()).hexdigest(),
        'limitations': ['Weights used for point estimates only; no design-based inference.',
                        'Comparative self-report definition differs from later cycles.',
                        'Complete outcome selection may affect generalizability.',
                        'Prior US data informed model choice before this cycle was selected.']}
    (OUT / 'nhanes_2003_2004_additional_validation.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(comparisons).to_csv(OUT / 'nhanes_2003_2004_screening_comparison.csv', index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
