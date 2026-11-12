"""Weighted retrospective decision curves for exploratory US comparisons."""
from pathlib import Path
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work/pydeps'))
sys.path.insert(0, str(ROOT/'work'))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cross_country_hearing_pilot as c
import incremental_self_report_models as inc
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as older


def net_benefit(y, positive, weight, threshold):
    y = np.asarray(y, int)
    positive = np.asarray(positive, bool)
    weight = np.asarray(weight, float)
    if not 0 < threshold < 1 or not np.isfinite(weight).all() or not (weight > 0).all():
        raise ValueError('Threshold or weights invalid')
    tp = np.sum(weight * positive * (y == 1))
    fp = np.sum(weight * positive * (y == 0))
    return float((tp - fp*threshold/(1-threshold))/weight.sum())


def main():
    dev = pd.concat([c.korean_year(y) for y in (2010, 2011, 2012)], ignore_index=True)
    dev = dev[dev.age.ge(40) & dev.outcome.notna() & dev.self_hearing_difficulty.notna() & dev.weight.gt(0)]
    base, expanded = inc.fit(dev, inc.BASE), inc.fit(dev, inc.EXT)
    known = json.loads((ROOT/'outputs/incremental_self_report_exploratory.json').read_text(encoding='utf-8'))
    assert base.best_params_['model__C'] == known['base_selected_C']
    assert expanded.best_params_['model__C'] == known['expanded_selected_C']
    sources = []
    for year in (2001, 2003):
        d = older.load_cycle(year)
        d['outcome'], d['weight'] = d.outcome_100, d.WTSAU2YR
        sources.append((f'US {year}–{year+1}', d))
    for cycle in ('2011_2012', '2015_2016'):
        d = nh.load(cycle)
        d['outcome'] = d.outcome_100
        sources.append((f'US {cycle.replace("_", "–")}', d))
    rows = []
    for name, full in sources:
        d = full[full.age.between(40,69) & full.outcome.notna() &
                 full.self_hearing_difficulty.notna() & full.weight.gt(0)].copy()
        y = d.outcome.to_numpy(dtype=int)
        w = d.weight.to_numpy(dtype=float)
        p_base = base.predict_proba(d[inc.BASE])[:,1]
        p_exp = expanded.predict_proba(d[inc.EXT])[:,1]
        for threshold in np.arange(.05, .401, .005):
            for strategy, positive in [('base', p_base >= threshold), ('expanded', p_exp >= threshold),
                                       ('self_report', d.self_hearing_difficulty.eq(1).to_numpy()),
                                       ('refer_all', np.ones(len(d), dtype=bool)),
                                       ('refer_none', np.zeros(len(d), dtype=bool))]:
                rows.append({'cohort': name, 'n':len(d), 'threshold':round(float(threshold), 3),
                             'strategy':strategy, 'net_benefit_weighted':net_benefit(y, positive, w, threshold)})
        print(name, 'n', len(d), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT/'outputs/incremental_us_decision_curve_exploratory.csv', index=False)
    styles = {'base':('#007f78','-','Age, sex, self-report'),
              'expanded':('#b95535','--','Plus routine variables'),
              'self_report':('#3d6082',':','Self-report alone'),
              'refer_all':('#777777','-.','Refer all'), 'refer_none':('#222222','-','Refer none')}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180, sharex=True, sharey=True)
    for ax, (name, _) in zip(axes.flat, sources):
        sub = result[result.cohort.eq(name)]
        for strategy, (color, linestyle, title) in styles.items():
            s = sub[sub.strategy.eq(strategy)]
            ax.plot(s.threshold, s.net_benefit_weighted, color=color, ls=linestyle, lw=1.4, label=title)
        ax.set_title(name, loc='left', fontsize=11)
        ax.set_ylim(-.05, .16)
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Hypothetical referral threshold probability', y=.075)
    fig.supylabel('Survey-weighted net benefit')
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', ncol=3,
               bbox_to_anchor=(.5, .018), frameon=False, fontsize=8)
    fig.suptitle('Exploratory decision curves in US analytic samples', fontsize=13)
    fig.text(.5, .94, 'Display range is limited to net benefit −0.05 to 0.16.', ha='center', fontsize=8)
    fig.tight_layout(rect=(.02,.13,1,.96))
    fig.savefig(ROOT/'outputs/incremental_us_decision_curve_exploratory.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
