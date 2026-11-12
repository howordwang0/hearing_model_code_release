"""Aggregate cohort flow and baseline summaries for the exploratory paper."""
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work/pydeps'))
sys.path.insert(0, str(ROOT/'work'))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
import cross_country_hearing_pilot as c
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as older


def weighted_mean(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else None


def main():
    years = {y: c.korean_year(y) for y in (2010, 2011, 2012, 2019, 2020, 2021, 2022)}
    samples = []
    dev = pd.concat([years[y] for y in (2010, 2011, 2012)], ignore_index=True)
    dev['weight'] = dev.weight / 3
    samples.append(('Korea_2010_2012_development', '40+', dev, dev.age.ge(40), dev.outcome.notna(),
                    pd.Series(True, index=dev.index)))
    for year in (2019, 2020, 2021, 2022):
        d = years[year]
        samples.append((f'Korea_{year}', '40-69', d, d.age.between(40, 69), d.outcome.notna(), d.quality_pass))
    d = older.load_cycle(2001)
    d['outcome'], d['weight'] = d.outcome_100, d.WTSAU2YR
    samples.append(('US_2001_2002_incremental_only', '40-69', d, d.age.between(40, 69), d.outcome.notna(),
                    d.self_hearing_difficulty.notna()))
    d = older.load_cycle()
    d['outcome'], d['weight'] = d.outcome_100, d.WTSAU2YR
    samples.append(('US_2003_2004', '40-69', d, d.age.between(40, 69), d.outcome.notna(),
                    pd.Series(True, index=d.index)))
    for cycle in ('2011_2012', '2015_2016'):
        d = nh.load(cycle)
        d['outcome'] = d.outcome_100
        samples.append((f'US_{cycle}', '40-69', d, d.age.between(40, 69), d.outcome.notna(),
                        pd.Series(True, index=d.index)))
    flow, baseline, missing = [], [], []
    for name, age_group, d, age, outcome, quality in samples:
        good_weight = np.isfinite(d.weight) & d.weight.gt(0)
        eligible = age & outcome & quality & good_weight
        z = d.loc[eligible]
        w = z.weight
        flow.append({'cohort': name, 'age_group': age_group, 'source_records': len(d),
                     'age_eligible': int(age.sum()), 'age_and_outcome': int((age & outcome).sum()),
                     'additional_selection': ('observed_self_report' if name == 'US_2001_2002_incremental_only'
                                              else 'booth_noise_flags' if name.startswith('Korea_20') and name != 'Korea_2010_2012_development'
                                              else 'none'),
                     'age_outcome_quality': int((age & outcome & quality).sum()),
                     'analysis_positive_weight': len(z), 'events': int(z.outcome.sum()),
                     'self_report_available': int(z.self_hearing_difficulty.notna().sum())})
        baseline.append({'cohort': name, 'age_group': age_group, 'n_unweighted': len(z),
                         'events_unweighted': int(z.outcome.sum()),
                         'age_mean_weighted': weighted_mean(z.age, w),
                         'female_fraction_weighted': weighted_mean(z.female, w),
                         'bmi_mean_weighted_among_nonmissing': weighted_mean(z.bmi, w),
                         'diabetes_fraction_weighted_among_nonmissing': weighted_mean(z.diabetes, w),
                         'outcome_fraction_weighted': weighted_mean(z.outcome, w),
                         'any_predictor_missing_n': int(z[c.FEATURES].isna().any(axis=1).sum())})
        missing.extend({'cohort': name, 'feature': f, 'missing_n': int(z[f].isna().sum()),
                        'eligible_n': len(z), 'missing_fraction_unweighted': float(z[f].isna().mean())}
                       for f in c.FEATURES)
    pd.DataFrame(flow).to_csv(ROOT/'outputs/cross_country_exploratory_cohort_flow.csv', index=False)
    pd.DataFrame(baseline).to_csv(ROOT/'outputs/cross_country_exploratory_baseline.csv', index=False)
    pd.DataFrame(missing).to_csv(ROOT/'outputs/cross_country_exploratory_predictor_missingness.csv', index=False)
    print(pd.DataFrame(flow).to_string(index=False))


if __name__ == '__main__':
    main()
