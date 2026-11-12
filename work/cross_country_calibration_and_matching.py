"""Exploratory weighted calibration and retrospective matched operating points.

Only aggregate results are written. Matched thresholds use the evaluation sample
and are not independently validated operating points.
"""
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work/pydeps'))
sys.path.insert(0, str(ROOT / 'work'))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import cross_country_hearing_pilot as c
import nhanes_corrected_pilot as nh

OUT = ROOT / 'outputs'


def weighted_calibration(y, p, weight):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, int)
    weight = np.asarray(weight, float)
    weight = weight / weight.mean()
    logit = np.log(p / (1 - p))
    intercept = sm.GLM(y, np.ones((len(y), 1)), offset=logit,
                       family=sm.families.Binomial(), freq_weights=weight).fit()
    slope = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial(),
                   freq_weights=weight).fit()
    return float(intercept.params[0]), float(slope.params[1])


def risk_bins(y, p, weight):
    result = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (p >= low) & ((p < high) if index < 9 else (p <= high))
        if mask.sum() == 0:
            continue
        result.append({'risk_bin': f'{low:.1f}-{high:.1f}', 'n': int(mask.sum()),
                       'events': int(y[mask].sum()),
                       'mean_predicted_weighted': np.average(p[mask], weights=weight[mask]),
                       'observed_weighted': np.average(y[mask], weights=weight[mask])})
    return result


def matching_threshold(y, p, weight, target, match):
    """Nearest achievable weighted rate, using entire tied prediction groups."""
    order = np.argsort(-p, kind='stable')
    scores, yy, ww = p[order], y[order], weight[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    positive_w = np.cumsum(ww * yy)[ends]
    referred_w = np.cumsum(ww)[ends]
    rates = referred_w / ww.sum() if match == 'referral' else positive_w / np.sum(ww * yy)
    # Include a zero-referral option; break equal-distance ties toward fewer referrals.
    rates = np.r_[0., rates]
    choice = int(np.argmin(np.abs(rates - target)))
    return float(np.nextafter(scores[0], np.inf) if choice == 0 else scores[ends[choice - 1]])


def main():
    years = {year: c.korean_year(year) for year in [2010, 2011, 2012, 2019, 2020, 2021, 2022]}
    dev = pd.concat([years[y][(years[y].age >= 40) & years[y].outcome.notna()]
                     for y in (2010, 2011, 2012)], ignore_index=True)
    model = c.logistic()
    model.fit(dev[c.FEATURES], dev.outcome.astype(int), groups=dev.group)
    datasets = []
    for year in (2019, 2020, 2021, 2022):
        d = years[year]
        datasets.append((f'Korea_{year}', d[d.age.between(40, 69) & d.outcome.notna() & d.quality_pass].copy()))
    for cycle in ('2011_2012', '2015_2016'):
        d = nh.load(cycle)
        d['outcome'] = d.outcome_100
        datasets.append((f'US_{cycle}', d[d.age.between(40, 69) & d.outcome.notna()].copy()))

    calibrations, bins, comparisons = [], [], []
    for name, d in datasets:
        d = d[np.isfinite(d.weight) & (d.weight > 0)].copy()
        y = d.outcome.to_numpy(dtype=int)
        p = model.predict_proba(d[c.FEATURES])[:, 1]
        w = d.weight.to_numpy(dtype=float)
        intercept, slope = weighted_calibration(y, p, w)
        calibrations.append({'cohort': name, 'n': len(d), 'events': int(y.sum()),
                             'mean_predicted_weighted': np.average(p, weights=w),
                             'observed_weighted': np.average(y, weights=w),
                             'calibration_in_large_weighted': intercept,
                             'calibration_slope_weighted': slope})
        bins.extend({'cohort': name, **row} for row in risk_bins(y, p, w))
        mask = d.self_hearing_difficulty.notna().to_numpy()
        z = d.loc[mask]
        yy, pp, ww = y[mask], p[mask], w[mask]
        self_pos = z.self_hearing_difficulty.eq(1).to_numpy()
        self_metrics = c.screen(yy, self_pos, ww)
        comparisons.append({'cohort': name, 'strategy': 'self_report', 'threshold': np.nan,
                            'target': np.nan, **self_metrics})
        for match, target in [('referral', self_metrics['referrals_per_1000'] / 1000),
                              ('sensitivity', self_metrics['sensitivity'])]:
            threshold = matching_threshold(yy, pp, ww, target, match)
            comparisons.append({'cohort': name, 'strategy': f'model_matched_{match}',
                                'threshold': threshold, 'target': target,
                                **c.screen(yy, pp >= threshold, ww)})
        print(name, 'weighted calibration', round(intercept, 3), round(slope, 3), flush=True)

    calibration = pd.DataFrame(calibrations)
    binned = pd.DataFrame(bins)
    matched = pd.DataFrame(comparisons)
    calibration.to_csv(OUT / 'cross_country_exploratory_weighted_calibration.csv', index=False)
    binned.to_csv(OUT / 'cross_country_exploratory_calibration_bins.csv', index=False)
    matched.to_csv(OUT / 'cross_country_exploratory_matched_screening.csv', index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), dpi=180, sharex=True, sharey=True)
    for ax, (name, _) in zip(axes.flat, datasets):
        row = calibration[calibration.cohort == name].iloc[0]
        part = binned[(binned.cohort == name) & (binned.n >= 30)]
        ax.plot([0, 1], [0, 1], color='#888888', lw=1, ls='--')
        ax.plot(part.mean_predicted_weighted, part.observed_weighted,
                color='#007f78', marker='o', lw=1.4, ms=4)
        ax.set(title=name.replace('_', ' '), xlim=(0, 1), ylim=(0, 1))
        ax.text(.04, .94, f'n = {int(row.n):,}', transform=ax.transAxes, va='top', fontsize=8)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Mean predicted risk (weighted)', y=.065)
    fig.supylabel('Observed outcome proportion (weighted)')
    fig.suptitle('Exploratory calibration, ages 40–69', fontsize=14)
    fig.text(.5, .015, 'Fixed 0.1 risk bins; bins with fewer than 30 participants omitted. No uncertainty shown.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(.02, .10, 1, .96))
    fig.savefig(OUT / 'cross_country_exploratory_calibration.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
