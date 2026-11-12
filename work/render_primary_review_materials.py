"""Render aggregate primary-model review tables and sample-flow figure."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work/pydeps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

OUT = ROOT / 'outputs'


def main():
    f = pd.read_csv(OUT/'incremental_primary_flow_verified.csv')
    b = pd.read_csv(OUT/'incremental_primary_baseline_verified.csv')
    assert f.cohort.tolist() == b.cohort.tolist()
    assert (f.positive_weight == b.n).all() and (f.events == b.events).all()
    lines = ['# Primary Incremental-Model Tables', '',
             '**Author-review version. Survey design and hearing data rules remain under review.**', '',
             '## Table 1. Selection for the incremental comparison', '',
             '| Survey | Source records | Age eligible | Valid outcome | After noise rule | Self-report observed | Positive weight | Events |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in f.itertuples(index=False):
        nums = [x.source_records, x.age_eligible, x.valid_outcome, x.quality_pass,
                x.self_report_observed, x.positive_weight, x.events]
        lines.append('| ' + x.cohort + ' | ' + ' | '.join(f'{int(n):,}' for n in nums) + ' |')
    lines += ['', 'All counts are unweighted and sequential. Development eligibility was age 40 years or older; all evaluation cohorts were aged 40–69. '
              'Noise flags were required only in Korea 2019–2022. Source records include participants outside the relevant component; '
              'especially, Korea 2019 ENT testing was restricted to July–December and the source count is not a component-eligible denominator. '
              'US 2001–2004 used audiometry subsamples. Valid outcome means the study-specific complete better-ear PTA calculation; '
              'US no-response values were handled as described in Methods.', '',
              '## Table 2. Selected participant characteristics', '',
              '| Survey | n | Events | Mean age | Female | Mean BMI | Outcome | Any expanded predictor missing, n |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in b.itertuples(index=False):
        lines.append(f'| {x.cohort} | {x.n:,} | {x.events:,} | {x.weighted_mean_age:.1f} | '
                     f'{100*x.weighted_female:.1f}% | {x.weighted_mean_bmi_nonmissing:.1f} | '
                     f'{100*x.weighted_prevalence:.1f}% | {x.expanded_predictors_missing_n:,} |')
    lines += ['', 'Counts are unweighted; means and percentages are weighted point estimates within each selected analytic sample. '
              'BMI is summarized among those with observed BMI. The development row is age 40+ and is not age-matched to the evaluation rows. '
              'Complete outcome and observed self-report restrictions may affect representativeness. Estimates require design review.', '',
              '## Table 3. Final exploratory model coefficients', '']
    card = json.loads((OUT/'增量模型_系数与插补规则_研究用.json').read_text(encoding='utf-8'))
    models = card['models']
    coef = {name: {c['term']: c['coefficient_on_unscaled_imputed_input'] for c in m['coefficients']}
            for name, m in models.items()}
    names = list(dict.fromkeys([c['term'] for model in models.values() for c in model['coefficients']]))
    lines += ['| Term | Base coefficient | Expanded coefficient | Expanded training median |',
              '|---|---:|---:|---:|',
              f"| Intercept | {models['base']['log_odds_intercept']:.8f} | {models['expanded']['log_odds_intercept']:.8f} | - |"]
    for term in names:
        values = [f'{coef[k][term]:.8f}' if term in coef[k] else '-' for k in ('base', 'expanded')]
        median = models['expanded']['training_medians'].get(term)
        lines.append(f'| {term} | {values[0]} | {values[1]} | {median:.8f} |' if median is not None
                     else f'| {term} | {values[0]} | {values[1]} | - |')
    lines += ['', 'Coefficients are on the unscaled, median-imputed input scale; missing__ terms are indicators for missing inputs. '
              'Probability = 1 / (1 + exp(-linear predictor)). Medians and coding must follow the complete model card. '
              'These penalized coefficients are not causal associations or a validated clinical tool.', '']
    (OUT/'增量主分析_流程基线与系数表.md').write_text('\n'.join(lines), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(15, 7.6), dpi=300)
    ax.axis('off')
    ax.text(.02, .96, 'Study selection for the incremental-model comparison', fontsize=15, weight='bold', transform=ax.transAxes)
    columns = ['Stage', 'Korea\n2010-2012', 'Korea\n2019', 'Korea\n2020', 'Korea\n2021', 'Korea\n2022',
               'US\n2001-2002', 'US\n2003-2004', 'US\n2011-2012', 'US\n2015-2016']
    rows = [('Source records', 'source_records'), ('Age eligible', 'age_eligible'),
            ('Valid outcome', 'valid_outcome'), ('Noise rule met', 'quality_pass'),
            ('Self-report observed', 'self_report_observed'), ('Positive weight', 'positive_weight'), ('Events', 'events')]
    cells = [[label] + [f'{n:,}' for n in f[col]] for label, col in rows]
    table = ax.table(cellText=cells, colLabels=columns, colLoc='center', cellLoc='center',
                     colWidths=[.17]+[.092]*9, bbox=[.015, .18, .97, .71])
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d6dfe0'); cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor('#244a48'); cell.get_text().set_color('white'); cell.get_text().set_weight('bold')
        elif row == 6:
            cell.set_facecolor('#e4f1eb'); cell.get_text().set_weight('bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f4f7f6')
        if col == 0 and row > 0:
            cell.get_text().set_ha('left')
    ax.text(.02, .105, 'Counts are sequential and unweighted. Korea development: age 40+; evaluation cohorts: age 40-69.',
            fontsize=9, transform=ax.transAxes)
    ax.text(.02, .065, 'Noise flags applied only in Korea 2019-2022. Korea 2019 source records include people outside the part-year ENT component.',
            fontsize=9, transform=ax.transAxes)
    fig.savefig(OUT/'增量主分析_样本流程图.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Wrote primary-model tables and flow figure.')


if __name__ == '__main__':
    main()
