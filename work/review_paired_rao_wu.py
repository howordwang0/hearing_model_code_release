"""Exploratory paired PSU resampling on complete positive-weight designs.

Models, threshold and questionnaire coding are fixed before resampling. This
does not include model development uncertainty. Singleton strata fail closed.
"""
from pathlib import Path
import json
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps')); sys.path.insert(0,str(ROOT/'work'))
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c
import incremental_self_report_models as inc
import nhanes_corrected_pilot as nh
import validate_nhanes_2003_2004 as older


def compute(y,w,p0,p1,pr,self_report,threshold):
    good=w>0
    y,w,p0,p1,pr,self_report=[np.asarray(x)[good] for x in (y,w,p0,p1,pr,self_report)]
    if len(np.unique(y))<2:
        raise ValueError('one outcome class')
    w=w/w.mean()
    a0=roc_auc_score(y,p0,sample_weight=w); a1=roc_auc_score(y,p1,sample_weight=w)
    def calibration(p):
        p=np.clip(p,1e-7,1-1e-7); logits=np.log(p/(1-p))
        citl=sm.GLM(y,np.ones((len(y),1)),offset=logits,family=sm.families.Binomial(),freq_weights=w).fit()
        slope=sm.GLM(y,sm.add_constant(logits),family=sm.families.Binomial(),freq_weights=w).fit()
        return float(citl.params[0]),float(slope.params[1])
    i0,s0=calibration(p0); i1,s1=calibration(p1); ir,srout=calibration(pr)
    model=pr>=threshold; self_pos=self_report==1
    sens=lambda x:np.sum(w*x*(y==1))/np.sum(w*(y==1))
    refer=lambda x:np.sum(w*x)/np.sum(w)
    return {'auc_base':a0,'auc_expanded':a1,'auc_difference':a1-a0,
            'routine_auc':roc_auc_score(y,pr,sample_weight=w),
            'base_calibration_intercept':i0,'expanded_calibration_intercept':i1,
            'calibration_intercept_difference':i1-i0,
            'base_calibration_slope':s0,'expanded_calibration_slope':s1,'calibration_slope_difference':s1-s0,
            'routine_calibration_intercept':ir,'routine_calibration_slope':srout,
            'model_sensitivity':sens(model),'model_referral':refer(model),
            'self_sensitivity':sens(self_pos),'self_referral':refer(self_pos),
            'sensitivity_difference_model_minus_self':sens(model)-sens(self_pos),
            'referral_difference_model_minus_self':refer(model)-refer(self_pos)}


def main():
    reps=500
    years={y:c.korean_year(y) for y in (2010,2011,2012,2019,2020,2021,2022)}
    dev=pd.concat([years[y] for y in (2010,2011,2012)],ignore_index=True)
    dev=dev[dev.age.ge(40)&dev.outcome.notna()&dev.weight.gt(0)]
    incr=dev[dev.self_hearing_difficulty.notna()]
    base=inc.fit(incr,inc.BASE); expanded=inc.fit(incr,inc.EXT)
    routine=c.logistic(); routine.fit(dev[c.FEATURES],dev.outcome.astype(int),groups=dev.group)
    threshold=json.loads((ROOT/'outputs/cross_country_hearing_exploratory_report.json').read_text(encoding='utf-8'))['development']['risk_threshold_for_80pct_sensitivity_from_development_oof']
    sources=[]
    for year in (2019,2020,2021,2022):
        f=years[year]; domain=f.age.between(40,69)&f.outcome.notna()&f.quality_pass&f.self_hearing_difficulty.notna()
        sources.append((f'Korea_{year}',f,domain,'weight','KSTRATA','PSU'))
    for year in (2001,2003):
        f=older.load_cycle(year);f['outcome']=f.outcome_100
        domain=f.age.between(40,69)&f.outcome.notna()&f.self_hearing_difficulty.notna()
        sources.append((f'US_{year}_{year+1}',f,domain,'WTSAU2YR','SDMVSTRA','SDMVPSU'))
    for cycle in ('2011_2012','2015_2016'):
        f=nh.load(cycle);f['outcome']=f.outcome_100
        domain=f.age.between(40,69)&f.outcome.notna()&f.self_hearing_difficulty.notna()
        sources.append((f'US_{cycle}',f,domain,'weight','SDMVSTRA','SDMVPSU'))
    rows=[]; metadata=[]
    for number,(name,full,mask,wcol,stratum,psu) in enumerate(sources):
        weightok=np.isfinite(full[wcol])&full[wcol].gt(0)
        design=full.loc[weightok,[stratum,psu]].dropna().drop_duplicates()
        groupsizes=design.groupby(stratum)[psu].nunique()
        d=full[mask&weightok].copy()
        if d[[stratum,psu]].isna().any(axis=None):raise ValueError(f'missing design in {name}')
        y=d.outcome.to_numpy(int); w=d[wcol].to_numpy(float)
        p0=base.predict_proba(d[inc.BASE])[:,1];p1=expanded.predict_proba(d[inc.EXT])[:,1]
        pr=routine.predict_proba(d[c.FEATURES])[:,1];sr=d.self_hearing_difficulty.to_numpy(int)
        point=compute(y,w,p0,p1,pr,sr,threshold)
        singleton=int(groupsizes.eq(1).sum())
        meta={'cohort':name,'n':len(d),'events':int(y.sum()),'design_psus':len(design),
              'design_strata':len(groupsizes),'singleton_strata':singleton,'requested_replicates':reps}
        if singleton:
            meta.update(status='not estimated: singleton component strata; method requires expert review',valid_replicates=0)
            rows.extend({'cohort':name,'metric':k,'point':float(v),'low':None,'high':None,
                         'valid_replicates':0,'status':meta['status']} for k,v in point.items())
            metadata.append(meta);print(name,meta['status'],flush=True);continue
        keys=list(map(tuple,design.to_numpy())); keymap={k:i for i,k in enumerate(keys)}
        groups=[np.array([keymap[tuple(row)] for row in block.to_numpy()],dtype=int)
                for _,block in design.groupby(stratum,sort=True)]
        idx=np.array([keymap[tuple(row)] for row in d[[stratum,psu]].to_numpy()])
        rng=np.random.default_rng(66100+number); out=[];failed=0
        for _ in range(reps):
            mult=np.zeros(len(keys))
            for psus in groups:
                n=len(psus); draw=rng.choice(psus,size=n-1,replace=True)
                mult[psus]=np.bincount(draw,minlength=len(keys))[psus]*n/(n-1)
            try:
                val=compute(y,w*mult[idx],p0,p1,pr,sr,threshold)
                if not all(np.isfinite(v) for v in val.values()):raise ValueError('nonfinite')
                out.append(val)
            except (ValueError,ZeroDivisionError,np.linalg.LinAlgError,sm.tools.sm_exceptions.PerfectSeparationError):
                failed+=1
        frame=pd.DataFrame(out)
        meta.update(status='exploratory fixed-model PSU resampling; design review required',
                    valid_replicates=len(frame),failed_replicates=failed)
        for metric,value in point.items():
            low,high=(frame[metric].quantile([.025,.975]) if len(frame)>=475 else (np.nan,np.nan))
            rows.append({'cohort':name,'metric':metric,'point':float(value),'low':float(low),'high':float(high),
                         'valid_replicates':len(frame),'status':meta['status']})
        metadata.append(meta);print(name,'complete',len(frame),'failures',failed,flush=True)
    out=ROOT/'outputs'
    pd.DataFrame(rows).to_csv(out/'复核后_配对抽样_探索性结果.csv',index=False)
    (out/'复核后_配对抽样_方法.json').write_text(json.dumps({'method':'Within each positive-weight design stratum of n PSUs, sample n-1 PSUs with replacement and multiply PSU weights by n/(n-1); evaluate all strategies using identical replicate weights.',
        'limitation':'Fixed models and threshold; no training uncertainty; no FPC; missingness is not corrected. Singleton-stratum policy unverified. Not final survey inference.',
        'comparison_domain':'40-69, outcome and self-report observed, positive cycle-specific component weight',
        'threshold':threshold,'cohorts':metadata},indent=2),encoding='utf-8')

if __name__=='__main__':main()
