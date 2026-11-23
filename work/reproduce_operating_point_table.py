"""Create an aggregate operating-point table for the routine-variable model."""
from pathlib import Path
import json
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work'))
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)
import pandas as pd
import cross_country_hearing_pilot as c
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as old

OUT=ROOT/'outputs'

def main():
    years={y:c.korean_year(y) for y in (2010,2011,2012)}
    dev=pd.concat([years[y] for y in (2010,2011,2012)],ignore_index=True)
    dev=dev[dev.age.ge(40)&dev.outcome.notna()&dev.weight.gt(0)].copy()
    model=c.logistic(); model.fit(dev[c.FEATURES],dev.outcome.astype(int),groups=dev.group)
    report=json.loads((OUT/'cross_country_hearing_exploratory_report.json').read_text(encoding='utf-8'))
    threshold=report['development']['risk_threshold_for_80pct_sensitivity_from_development_oof']
    cycles=[]
    for year in (2001,2003):
        d=old.load_cycle(year);d['outcome'],d['weight']=d.outcome_100,d.WTSAU2YR
        cycles.append((f'US_{year}_{year+1}',d))
    for cycle in ('2011_2012','2015_2016'):
        d=nh.load(cycle);d['outcome']=d.outcome_100;cycles.append((f'US_{cycle}',d))
    rows=[]
    for name,d in cycles:
        z=d[d.age.between(40,69)&d.outcome.notna()&d.self_hearing_difficulty.notna()&d.weight.gt(0)].copy()
        p=model.predict_proba(z[c.FEATURES])[:,1]
        model_stats=c.screen(z.outcome,(p>=threshold),z.weight)
        self_stats=c.screen(z.outcome,z.self_hearing_difficulty.eq(1),z.weight)
        rows.append({'cohort':name,'n':len(z),'events':int(z.outcome.sum()),
                     'development_target_sensitivity':0.8,'fixed_threshold':threshold,
                     'model_sensitivity':model_stats['sensitivity'],'model_referral_per_1000':model_stats['referrals_per_1000'],
                     'model_missed_per_1000':model_stats['missed_per_1000'],
                     'self_report_sensitivity':self_stats['sensitivity'],'self_report_referral_per_1000':self_stats['referrals_per_1000'],
                     'self_report_missed_per_1000':self_stats['missed_per_1000']})
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'operating_point_transportability_supplement.csv',index=False)
    lines=['# Supplementary Table. Operating-point transportability in US cycles','',
           'The routine-variable model threshold was selected in Korean development data to target approximately 80% unweighted sensitivity. It was then held fixed in each US cycle. Self-report is a separate comparator; referral burdens were not matched. All estimates are exploratory and survey-weighted point estimates.','',
           '| Cohort | n | Events | Fixed threshold | Model sensitivity | Model referrals/1,000 | Model missed/1,000 | Self-report sensitivity | Self-report referrals/1,000 | Self-report missed/1,000 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cohort'].replace('_',' ')} | {r['n']:,} | {r['events']:,} | {r['fixed_threshold']:.4f} | {100*r['model_sensitivity']:.1f}% | {r['model_referral_per_1000']:.1f} | {r['model_missed_per_1000']:.1f} | {100*r['self_report_sensitivity']:.1f}% | {r['self_report_referral_per_1000']:.1f} | {r['self_report_missed_per_1000']:.1f} |")
    (OUT/'operating_point_transportability_supplement.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(out.to_string(index=False))

if __name__=='__main__':main()
