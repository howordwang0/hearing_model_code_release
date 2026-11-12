from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work' / 'pydeps'))
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'work/data/nhanes'
OUT = ROOT / 'outputs'
SPECS = {
    '2011_2012': ('demo_g','aux_g','bmx_g','biopro_g','cbc_g','diq_g','smq_g','auq_g'),
    '2015_2016': ('demo_i','aux_i','bmx_i','biopro_i','cbc_i','diq_i','smq_i','auq_i'),
    '2017_2020': ('p_demo','p_aux','p_bmx','p_biopro','p_cbc','p_diq','p_smq','p_auq'),
}
CORE = ['age','female','bmi','diabetes','ever_smoked','current_smoker','creatinine','hemoglobin']
EXT = CORE + ['tinnitus','occupational_noise']

def load(cycle):
    names = SPECS[cycle]
    frames = [pyreadstat.read_xport(DATA / cycle / (n+'.xpt'))[0] for n in names]
    d = frames[0][['SEQN','RIDAGEYR','RIAGENDR','SDMVPSU','SDMVSTRA', 'WTMECPRP' if cycle == '2017_2020' else 'WTMEC2YR']].copy()
    fields = [
        ['AUXU500R','AUXU1K1R','AUXU2KR','AUXU4KR','AUXU500L','AUXU1K1L','AUXU2KL','AUXU4KL'],
        ['BMXBMI'], ['LBXSCR'], ['LBXHGB'], ['DIQ010'], ['SMQ020','SMQ040'],
        ['AUQ054','AUQ191','AUQ331' if cycle == '2015_2016' else 'AUQ330']
    ]
    for f, cols in zip(frames[1:], fields):
        d = d.merge(f[['SEQN']+cols], on='SEQN', how='left', validate='one_to_one')
    d['cycle'] = cycle
    d['weight'] = d['WTMECPRP' if cycle == '2017_2020' else 'WTMEC2YR']
    d['age'] = d.RIDAGEYR
    d['female'] = d.RIAGENDR.map({1.:0.,2.:1.})
    d['bmi'], d['creatinine'], d['hemoglobin'] = d.BMXBMI, d.LBXSCR, d.LBXHGB
    for new, col in [('diabetes','DIQ010'),('ever_smoked','SMQ020'),('tinnitus','AUQ191')]:
        d[new] = d[col].map({1.:1., 2.:0.})
    d['current_smoker'] = np.nan
    d.loc[d.SMQ020.eq(2), 'current_smoker'] = 0
    d.loc[d.SMQ020.eq(1) & d.SMQ040.eq(3), 'current_smoker'] = 0
    d.loc[d.SMQ020.eq(1) & d.SMQ040.isin([1,2]), 'current_smoker'] = 1
    col = 'AUQ331' if cycle == '2015_2016' else 'AUQ330'
    d['occupational_noise'] = d[col].map({1.:1.,2.:0.,3.:0.})
    d['self_hearing_difficulty'] = d.AUQ054.map({1.:0.,2.:0.,3.:1.,4.:1.,5.:1.,6.:1.})
    freq = fields[0]
    d['n_no_response_666'] = d[freq].eq(666).sum(axis=1)
    d['n_could_not_obtain_888'] = d[freq].eq(888).sum(axis=1)
    d['n_other_invalid_audiometry'] = (~d[freq].isin(list(range(-10,121))+[666,888])).sum(axis=1)
    for replacement in (100,120):
        vals = d[freq].copy().replace(666,replacement)
        vals = vals.where((vals >= -10) & (vals <= 120))
        complete = vals.notna().all(axis=1)
        right, left = vals.iloc[:,:4].mean(axis=1), vals.iloc[:,4:].mean(axis=1)
        better = np.minimum(right,left)
        d[f'outcome_{replacement}'] = np.where(complete, better>25, np.nan)
        d[f'outcome_{replacement}_40'] = np.where(complete, better>=40, np.nan)
    return d

def pipeline(features, tune):
    base = Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),
                     ('logit',LogisticRegression(solver='liblinear',penalty='l1',max_iter=5000,random_state=2026))])
    if tune:
        return GridSearchCV(base, {'logit__C':[0.03,0.1,0.3,1.,3.,10.]},
                            cv=StratifiedKFold(4,shuffle=True,random_state=44),scoring='neg_log_loss',n_jobs=1)
    return base

def scores(y, p, w):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p),1e-6,1-1e-6); w=np.asarray(w)
    logits=np.log(p/(1-p))
    slope_fit=sm.GLM(y,sm.add_constant(logits),family=sm.families.Binomial()).fit()
    citl=sm.GLM(y,np.ones((len(y),1)),offset=logits,family=sm.families.Binomial()).fit()
    good=np.isfinite(w)&(w>0)
    return {'n':len(y),'events':int(y.sum()),'auc_unweighted':roc_auc_score(y,p),
            'brier_unweighted':brier_score_loss(y,p),'calibration_slope_unweighted':float(slope_fit.params[1]),
            'calibration_in_large_unweighted':float(citl.params[0]),
            'auc_weighted_point_estimate':roc_auc_score(y[good],p[good],sample_weight=w[good]),
            'brier_weighted_point_estimate':brier_score_loss(y[good],p[good],sample_weight=w[good]),
            'weighted_prevalence':float(np.average(y[good],weights=w[good]))}

def bootstrap_delta(y, pa, pb, seed=2026, reps=1000):
    rng=np.random.default_rng(seed); diffs=[]
    y=np.asarray(y,dtype=int); pa=np.asarray(pa); pb=np.asarray(pb)
    for _ in range(reps):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))==2:
            diffs.append(roc_auc_score(y[idx],pa[idx])-roc_auc_score(y[idx],pb[idx]))
    return {'difference':roc_auc_score(y,pa)-roc_auc_score(y,pb),
            'ci_low':float(np.quantile(diffs,.025)),'ci_high':float(np.quantile(diffs,.975)),'replicates':len(diffs)}

def threshold(y, positive, w):
    y=np.asarray(y,int); positive=np.asarray(positive,bool); w=np.asarray(w,float)
    tp=np.sum(w*(positive & (y==1))); fn=np.sum(w*(~positive & (y==1)))
    fp=np.sum(w*(positive & (y==0))); tn=np.sum(w*(~positive & (y==0)))
    return {'n':len(y),'sensitivity':float(tp/(tp+fn)), 'specificity':float(tn/(tn+fp)),
            'referral_fraction':float((tp+fp)/(tp+fp+tn+fn)), 'missed_cases_per_1000':float(1000*fn/w.sum()),
            'detected_cases_per_1000':float(1000*tp/w.sum()), 'referrals_per_detected_case':float((tp+fp)/tp) if tp else None}

def main():
    all_data=pd.concat([load(c) for c in SPECS],ignore_index=True)
    cohorts={
        'development_2011_2012_40_69':all_data[(all_data.cycle=='2011_2012') & all_data.age.between(40,69) & all_data.outcome_100.notna()].copy(),
        'validation_2015_2016_40_69':all_data[(all_data.cycle=='2015_2016') & all_data.age.between(40,69) & all_data.outcome_100.notna()].copy(),
        'supplement_2017_2020_70plus':all_data[(all_data.cycle=='2017_2020') & (all_data.age>=70) & all_data.outcome_100.notna()].copy()}
    dev=cohorts['development_2011_2012_40_69']; val=cohorts['validation_2015_2016_40_69']
    rows=[]; predictions={}; models={}
    for name,features in [('core',CORE),('extended',EXT),('age_only',['age'])]:
        tune=name!='age_only'
        model=pipeline(features,tune)
        if not tune: model.set_params(logit__C=100000)
        model.fit(dev[features],dev.outcome_100.astype(int)); models[name]=model
        cv=StratifiedKFold(5,shuffle=True,random_state=2026)
        oof=np.zeros(len(dev))
        for tr,te in cv.split(dev,dev.outcome_100):
            fold=pipeline(features,tune)
            if not tune: fold.set_params(logit__C=100000)
            fold.fit(dev.iloc[tr][features],dev.iloc[tr].outcome_100.astype(int))
            oof[te]=fold.predict_proba(dev.iloc[te][features])[:,1]
        rows.append({'model':name,'cohort':'development_nested_cv',**scores(dev.outcome_100,oof,dev.weight)})
        for cname,cohort in cohorts.items():
            p=model.predict_proba(cohort[features])[:,1]
            predictions[(name,cname)]=p
            if cname!='development_2011_2012_40_69':
                rows.append({'model':name,'cohort':cname,**scores(cohort.outcome_100,p,cohort.weight)})
    perf=pd.DataFrame(rows)
    perf.to_csv(OUT/'hearing_screening_corrected_pilot_performance.csv',index=False)
    delta=bootstrap_delta(val.outcome_100,predictions[('core','validation_2015_2016_40_69')],predictions[('age_only','validation_2015_2016_40_69')])
    valid=val.self_hearing_difficulty.notna() & val.weight.notna() & (val.weight>0)
    z=val[valid]; core=predictions[('core','validation_2015_2016_40_69')][valid.to_numpy()]
    comparisons={
        'age_60_plus':threshold(z.outcome_100,z.age>=60,z.weight),
        'self_report_little_trouble_or_worse':threshold(z.outcome_100,z.self_hearing_difficulty==1,z.weight),
        'core_risk_10_percent_exploratory':threshold(z.outcome_100,core>=.1,z.weight)}
    audit={}
    for name,d in cohorts.items():
        audit[name]={'n':len(d),'events':int(d.outcome_100.sum()), 'no_response_666_any':int((d.n_no_response_666>0).sum()),
                     'outcome_changes_when_666_set_to_120':int((d.outcome_100!=d.outcome_120).sum()),
                     'predictor_missing':{x:int(d[x].isna().sum()) for x in EXT},
                     'self_report_available':int(d.self_hearing_difficulty.notna().sum())}
    results={'audit':audit,'performance':rows,'validation_core_minus_age_auc_unweighted_individual_bootstrap':delta,
             'validation_screening_comparison_weighted_point_estimates':comparisons,
             'limitations':['Exploratory NHANES-only analysis. Outcome uses 100 dB for 666, 120 dB sensitivity.',
                            'MEC weights applied only to point estimates, not to fitting, tuning or variance.',
                            'Individual bootstrap intervals do not reflect survey PSU or strata.',
                            'The 70+ group is out-of-age-range extrapolation, not a same-age validation.',
                            'No geographic validation, no prospective impact study, and no established undiagnosed case definition.',
                            'Data-driven model choices and threshold 10% are exploratory.']}
    (OUT/'hearing_screening_corrected_pilot_report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
