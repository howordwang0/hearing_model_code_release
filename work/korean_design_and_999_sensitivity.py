"""Scenario checks for 2019 singleton PSUs and 2010 undocumented 999 values.

All generated person-level intermediates are temporary. Scenarios are not
assertions of code meaning or recommendations for an official survey design.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'work/pydeps'), str(ROOT/'work')]
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c
import incremental_self_report_models as inc
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as old
from survey_paired_auc import auc_and_influence
from cross_country_calibration_and_matching import weighted_calibration

R = r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'
OUT = ROOT/'outputs'


def fitted(train):
    return inc.fit(train, inc.BASE), inc.fit(train, inc.EXT)


def metric(z, base, expanded):
    y, w = z.outcome.astype(int), z.weight
    p0 = base.predict_proba(z[inc.BASE])[:, 1]
    p1 = expanded.predict_proba(z[inc.EXT])[:, 1]
    a = roc_auc_score(y, p0, sample_weight=w)
    b = roc_auc_score(y, p1, sample_weight=w)
    citl0, slope0 = weighted_calibration(y, p0, w)
    citl1, slope1 = weighted_calibration(y, p1, w)
    return {'base_auc': a, 'expanded_auc': b, 'auc_difference': b-a,
            'base_brier': np.average((y-p0)**2, weights=w),
            'expanded_brier': np.average((y-p1)**2, weights=w),
            'base_citl': citl0, 'expanded_citl': citl1,
            'base_calibration_slope': slope0, 'expanded_calibration_slope': slope1}


def main():
    years = {y: c.korean_year(y) for y in (2010, 2011, 2012, 2019, 2020, 2021, 2022)}
    early = pd.concat([years[y] for y in (2010, 2011, 2012)], ignore_index=True)
    values = early[c.FREQ].apply(pd.to_numeric, errors='coerce')
    any999 = values.eq(999).any(axis=1)
    all999 = values.eq(999).all(axis=1)
    pos = early.age.ge(40) & early.weight.gt(0) & early.self_hearing_difficulty.notna()
    original = early[pos & early.outcome.notna()].copy()
    assert len(original) == 11679 and int(original.outcome.sum()) == 3105
    special = early[pos & early.year.eq(2010) & any999].copy()
    patterns = []
    for label, mask in [('all_40plus_2010', early.age.ge(40) & early.year.eq(2010)),
                        ('eligible_except_outcome_2010', pos & early.year.eq(2010))]:
        v = values.loc[mask]
        valid = v.ge(-10) & v.le(120)
        patterns.append({'population': label, 'n': len(v), 'any_999': int(v.eq(999).any(axis=1).sum()),
                         'all_8_999': int(v.eq(999).all(axis=1).sum()),
                         'mixed_999_and_valid': int((v.eq(999).any(axis=1) & valid.any(axis=1)).sum()),
                         'both_ears_any_valid_and_999': int((v.eq(999).any(axis=1) & valid.iloc[:,:4].any(axis=1) & valid.iloc[:,4:].any(axis=1)).sum())})
    pd.DataFrame(patterns).to_csv(OUT/'korean_2010_999_pattern_audit.csv', index=False)
    assert special.outcome.isna().all()

    evaluation = []
    for year in (2019, 2020, 2021, 2022):
        d = years[year]
        mask = d.age.between(40,69) & d.outcome.notna() & d.quality_pass & d.self_hearing_difficulty.notna() & d.weight.gt(0)
        evaluation.append((f'Korea_{year}', d.loc[mask].copy()))
    for year in (2001, 2003):
        d = old.load_cycle(year); d['outcome'], d['weight'] = d.outcome_100, d.WTSAU2YR
        mask = d.age.between(40,69) & d.outcome.notna() & d.self_hearing_difficulty.notna() & d.weight.gt(0)
        evaluation.append((f'US_{year}_{year+1}', d.loc[mask].copy()))
    for cycle in ('2011_2012', '2015_2016'):
        d = nh.load(cycle); d['outcome'] = d.outcome_100
        mask = d.age.between(40,69) & d.outcome.notna() & d.self_hearing_difficulty.notna() & d.weight.gt(0)
        evaluation.append((f'US_{cycle}', d.loc[mask].copy()))

    # Identification bounds on labels only: these are deliberately extreme
    # model-development scenarios, not claims that the values are thresholds.
    scenarios = [('exclude_999', original)]
    for value, label in [(0., 'all_999_negative'), (1., 'all_999_positive')]:
        x = special.copy(); x['outcome'] = value
        scenarios.append((label, pd.concat([original, x], ignore_index=True)))
    rows = []
    baseline_models = None
    for label, train in scenarios:
        models = fitted(train)
        if label == 'exclude_999': baseline_models = models
        for name, z in evaluation:
            rows.append({'scenario': label, 'cohort': name, 'training_n': len(train),
                         'training_events': int(train.outcome.sum()), 'added_999_n': len(train)-len(original),
                         'eval_n': len(z), **metric(z, *models),
                         'base_C': models[0].best_params_['model__C'], 'expanded_C': models[1].best_params_['model__C']})
        print(label, 'training', len(train), 'events', int(train.outcome.sum()), flush=True)
    pd.DataFrame(rows).to_csv(OUT/'korean_2010_999_extreme_label_sensitivity.csv', index=False)

    d = years[2019]
    ok = np.isfinite(d.weight) & d.weight.gt(0)
    domain = ok & d.age.between(40,69) & d.outcome.notna() & d.quality_pass & d.self_hearing_difficulty.notna()
    z = d.loc[domain]
    base, expanded = baseline_models
    p0, p1 = base.predict_proba(z[inc.BASE])[:,1], expanded.predict_proba(z[inc.EXT])[:,1]
    a0, i0 = auc_and_influence(z.outcome, p0, z.weight)
    a1, i1 = auc_and_influence(z.outcome, p1, z.weight)
    design = d.loc[ok, ['KSTRATA', 'PSU', 'weight']].copy()
    design['domain'] = domain.loc[design.index].astype(int)
    design['outcome'] = 0.; design.loc[z.index, 'outcome'] = z.outcome
    design['influence_per_weight'] = 0.; design.loc[z.index, 'influence_per_weight'] = (i1-i0)/z.weight.to_numpy()
    keys = design[['KSTRATA','PSU']].drop_duplicates()
    assert int(keys.groupby('KSTRATA').PSU.nunique().eq(1).sum()) == 5
    with tempfile.TemporaryDirectory(prefix='survey_sensitivity_',dir=ROOT/'work/data') as temp:
        src, dst = Path(temp)/'design.csv', Path(temp)/'result.csv'
        design.rename(columns={'KSTRATA':'stratum','PSU':'psu'}).to_csv(src,index=False)
        run = subprocess.run([R,str(ROOT/'work/korea2019_lonely_sensitivity.R'),str(src),str(dst)],
                             capture_output=True,text=True,encoding='utf-8',errors='replace')
        if run.returncode: raise RuntimeError(run.stderr)
        result = pd.read_csv(dst)
        result['delta_auc'] = a1-a0
        result['delta_low_t'] = result.delta_auc - t.ppf(.975, result.domain_df)*result.delta_se
        result['delta_high_t'] = result.delta_auc + t.ppf(.975, result.domain_df)*result.delta_se
        result['full_component_psus'], result['full_component_strata'], result['full_singleton_strata'] = len(keys), keys.KSTRATA.nunique(), 5
        result.to_csv(OUT/'korea_2019_singleton_assumptions_sensitivity.csv',index=False)
        print('2019 scenarios',result.to_string(index=False),flush=True)
        if run.stderr: print(run.stderr,flush=True)
    print('2010 code patterns',patterns,flush=True)


if __name__ == '__main__':
    main()
