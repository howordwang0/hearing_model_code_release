"""Generate reviewer-ready aggregate tables/figures from audited results."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=ROOT/'outputs'


def main():
    result=pd.read_csv(OUT/'复核后_配对抽样_探索性结果.csv')
    korea19=pd.read_csv(OUT/'korea_2019_R独立计算_探索性.csv')
    korea19=korea19[korea19.domain_adjust.eq(True)]
    order=['Korea_2020','Korea_2021','Korea_2022',
           'US_2001_2002','US_2003_2004','US_2011_2012','US_2015_2016']
    stats={}
    for cohort in order:
        sub=result[result.cohort.eq(cohort)].set_index('metric')
        assert sub.index.is_unique
        stats[cohort]=sub
    counts=pd.read_csv(OUT/'incremental_self_report_exploratory.csv').set_index('cohort')
    counts2001=pd.read_csv(OUT/'incremental_paired_auc_exploratory_linearization.csv').set_index('cohort')
    lines=['# Incremental Model Results','',
           '**Working table: fixed-model intervals are exploratory and require survey-design review.**','',
           '| Survey, age 40–69 | n | Events | Base AUC | Expanded AUC | Difference | Exploratory interval |',
           '|---|---:|---:|---:|---:|---:|---:|']
    fig,ax=plt.subplots(figsize=(9.2,5.4),dpi=300)
    for pos,cohort in enumerate(order):
        sub=stats[cohort];a=sub.loc['auc_base'];b=sub.loc['auc_expanded'];d=sub.loc['auc_difference']
        s=counts2001.loc[cohort] if cohort=='US_2001_2002' else counts.loc[cohort]
        assert np.isclose(b.point-a.point,d.point,atol=1e-12)
        low,high=d.low,d.high
        interval='Not estimated' if pd.isna(low) else f'{low:+.4f} to {high:+.4f} (bootstrap)'
        label=cohort.replace('_',' ')
        lines.append(f'| {label} | {int(s.n):,} | {int(s.events):,} | {a.point:.3f} | {b.point:.3f} | {d.point:+.4f} | {interval} |')
        color='#007f78' if cohort.startswith('Korea') else '#b55236'
        if pd.notna(low):
            ax.plot([low,high],[pos,pos],color=color,lw=1.4)
        ax.scatter([d.point],[pos],color=color,s=34,zorder=3)
    r=korea19[korea19.method.eq('adjust')].set_index('metric').loc['auc_difference']
    r_average=korea19[korea19.method.eq('average')].set_index('metric').loc['auc_difference']
    sub=result[result.cohort.eq('Korea_2019')].set_index('metric')
    assert np.isclose(r.point,sub.loc['auc_difference','point'],atol=1e-12)
    lines+=['','Base model: age, recorded sex and self-reported hearing difficulty. Expanded model: base predictors plus body mass index, diabetes, smoking, creatinine and hemoglobin. Positive values favor the expanded model. Counts and events are unweighted; AUCs are weighted point estimates. US 2001–2002 was inspected after a local cycle-specific plan was written, but the research question was selected using other examined data. These intervals are descriptive fixed-model percentiles from 500 paired within-stratum PSU resamples; they omit development uncertainty and require design review. The comparison is not an equivalence test.','',
            '## Supplement: Korea 2019 under two singleton-PSU assumptions','',
            '| n | Events | Base AUC | Expanded AUC | Difference | Singleton adjustment | Exploratory t interval |',
            '|---:|---:|---:|---:|---:|---|---:|',
            f"| 1,341 | 187 | {sub.loc['auc_base','point']:.3f} | {sub.loc['auc_expanded','point']:.3f} | {r.point:+.4f} | adjust | {r.low:+.4f} to {r.high:+.4f} |",
            f"| 1,341 | 187 | {sub.loc['auc_base','point']:.3f} | {sub.loc['auc_expanded','point']:.3f} | {r_average.point:+.4f} | average | {r_average.low:+.4f} to {r_average.high:+.4f} |",'',
            'The 2019 intervals use fixed-model linearization in R survey, applying the stated singleton adjustment at the component and domain levels. The average interval is wider for this data set; this does not establish the appropriate design or singleton treatment. The assumed component design and both adjustments remain unverified. These sensitivity analyses are not pooled estimates or validated design-based confidence intervals. No multiplicity adjustment was applied across the eight cohort comparisons.','']
    (OUT/'增量模型_复核后结果表.md').write_text('\n'.join(lines),encoding='utf-8')
    ax.axvline(0,color='#666666',ls='--',lw=1)
    ax.set_yticks(range(len(order)),[x.replace('_',' ') for x in order]);ax.invert_yaxis()
    ax.set_xlabel('Weighted AUC difference (expanded minus base)')
    ax.set_title('Incremental discrimination in adults aged 40–69',loc='left',fontsize=12)
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.text(.12,.01,'500 paired PSU resamples; fixed models. Korea 2019 is reported separately. Design review needed.',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/'增量模型_配对差值_探索性.png',dpi=300,bbox_inches='tight');plt.close(fig)
    print('Wrote revised incremental table and exploratory paired-resampling figure.')


if __name__=='__main__':main()
