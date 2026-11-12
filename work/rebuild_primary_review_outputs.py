"""Rebuild incremental-model flow and check fixed-score influence variance in R.

Temporary person-level intermediates remain under work/data and are deleted on
exit. Aggregate outputs do not establish independent design or clinical review.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'work/pydeps'), str(ROOT / 'work')]
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
import cross_country_hearing_pilot as c
import incremental_self_report_models as inc
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as older
from cross_country_cohort_baseline import weighted_mean
from survey_paired_auc import auc_and_influence, paired_delta

R = Path(r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe')
OUT = ROOT / 'outputs'


def main():
    years = {year: c.korean_year(year) for year in (2010, 2011, 2012, 2019, 2020, 2021, 2022)}
    dev = pd.concat([years[y] for y in (2010, 2011, 2012)], ignore_index=True)
    dev['weight'] = dev.weight / 3
    dsel = dev.age.ge(40) & dev.outcome.notna() & dev.self_hearing_difficulty.notna() & dev.weight.gt(0)
    base = inc.fit(dev.loc[dsel], inc.BASE)
    expanded = inc.fit(dev.loc[dsel], inc.EXT)
    assert len(dev.loc[dsel]) == 11679 and int(dev.loc[dsel, 'outcome'].sum()) == 3105
    sources = [('Korea 2010-2012', dev, dev.age.ge(40), pd.Series(True, index=dev.index), 'survey_strata', 'survey_psu')]
    for year in (2019, 2020, 2021, 2022):
        d = years[year]
        sources.append((f'Korea {year}', d, d.age.between(40, 69), d.quality_pass, 'KSTRATA', 'PSU'))
    for year in (2001, 2003):
        d = older.load_cycle(year)
        d['outcome'], d['weight'] = d.outcome_100, d.WTSAU2YR
        sources.append((f'US {year}-{year+1}', d, d.age.between(40, 69), pd.Series(True, index=d.index), 'SDMVSTRA', 'SDMVPSU'))
    for cycle in ('2011_2012', '2015_2016'):
        d = nh.load(cycle)
        d['outcome'] = d.outcome_100
        sources.append((f'US {cycle.replace("_", "-")}', d, d.age.between(40, 69), pd.Series(True, index=d.index), 'SDMVSTRA', 'SDMVPSU'))
    flow, baseline, checks = [], [], []
    with tempfile.TemporaryDirectory(prefix='survey_check_', dir=ROOT / 'work/data') as tmp:
        tmp = Path(tmp)
        for name, d, age, quality, strata, psu in sources:
            outcome = d.outcome.notna()
            self_report = d.self_hearing_difficulty.notna()
            pos = np.isfinite(d.weight) & d.weight.gt(0)
            steps = [age, age & outcome, age & outcome & quality,
                     age & outcome & quality & self_report,
                     age & outcome & quality & self_report & pos]
            z = d.loc[steps[-1]].copy()
            flow.append({'cohort': name, 'source_records': len(d), 'age_eligible': int(steps[0].sum()),
                         'valid_outcome': int(steps[1].sum()), 'quality_pass': int(steps[2].sum()),
                         'self_report_observed': int(steps[3].sum()), 'positive_weight': len(z),
                         'events': int(z.outcome.sum())})
            baseline.append({'cohort': name, 'n': len(z), 'events': int(z.outcome.sum()),
                             'weighted_mean_age': weighted_mean(z.age, z.weight),
                             'weighted_female': weighted_mean(z.female, z.weight),
                             'weighted_mean_bmi_nonmissing': weighted_mean(z.bmi, z.weight),
                             'weighted_prevalence': weighted_mean(z.outcome, z.weight),
                             'expanded_predictors_missing_n': int(z[inc.EXT].isna().any(axis=1).sum())})
            if name == 'Korea 2010-2012':
                continue  # out-of-fold development predictions differ from final-model scores
            design = d.loc[pos, [strata, psu, 'weight']].copy()
            if design[[strata, psu]].isna().any(axis=None):
                raise ValueError(f'{name}: missing design identifiers')
            design['domain'] = steps[-1].loc[design.index].astype(int)
            design['influence_per_weight'] = 0.0
            p0 = base.predict_proba(z[inc.BASE])[:, 1]
            p1 = expanded.predict_proba(z[inc.EXT])[:, 1]
            y, w = z.outcome.to_numpy(int), z.weight.to_numpy(float)
            a0, i0 = auc_and_influence(y, p0, w)
            a1, i1 = auc_and_influence(y, p1, w)
            design.loc[z.index, 'influence_per_weight'] = (i1 - i0) / w
            keys = design[[strata, psu]].drop_duplicates()
            singletons = keys.groupby(strata)[psu].nunique().eq(1).sum()
            row = {'cohort': name, 'n': len(z), 'auc_difference': a1-a0, 'singleton_strata': int(singletons)}
            if singletons:
                row['status'] = 'not estimated: full positive-weight component has singleton strata'
                checks.append(row)
                continue
            py = paired_delta(y, p0, p1, w, z[strata], z[psu], list(map(tuple, keys.to_numpy())))
            datafile, outfile = tmp / 'design.csv', tmp / 'r_result.csv'
            design.rename(columns={strata: 'stratum', psu: 'psu'}).to_csv(datafile, index=False)
            run = subprocess.run([str(R), str(ROOT/'work/verify_survey_auc.R'), str(datafile), str(outfile)],
                                 capture_output=True, text=True, encoding='utf-8', errors='replace')
            if run.returncode:
                raise RuntimeError(f'{name}: R survey failed: {run.stderr}')
            rr = pd.read_csv(outfile).iloc[0]
            row.update(python_se=py['linearized_standard_error'], r_survey_se=float(rr['se']),
                       r_domain_df=int(rr['df']), python_domain_df=py['domain_degrees_of_freedom'],
                       abs_se_difference=abs(py['linearized_standard_error']-float(rr['se'])),
                       status='same influence, independently aggregated with R survey')
            if row['abs_se_difference'] > 1e-9:
                raise AssertionError(f'{name}: variance mismatch {row}')
            checks.append(row)
            print(name, row, flush=True)
    expected = [11679, 1341, 2698, 2618, 2443, 1040, 977, 2236, 2564]
    assert [r['positive_weight'] for r in flow] == expected
    pd.DataFrame(flow).to_csv(OUT/'incremental_primary_flow_verified.csv', index=False)
    pd.DataFrame(baseline).to_csv(OUT/'incremental_primary_baseline_verified.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT/'incremental_auc_r_survey_crosscheck.csv', index=False)
    print('Aggregate flow, baseline, and R survey comparison written.', flush=True)


if __name__ == '__main__':
    main()
