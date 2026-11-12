"""Exploratory cross-survey pilot. Outputs only aggregate statistics, no patient records."""
from pathlib import Path
import json
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps'))
sys.path.insert(0,str(ROOT/'work'))
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import nhanes_corrected_pilot as nh

OUT=ROOT/'outputs'
DATA=ROOT/'work/data/knhanes'
FEATURES=['age','female','bmi','diabetes','ever_smoked','current_smoker','creatinine','hemoglobin']
FREQ=[f'T_HR{freq}_{side}' for side in ('RT','LT') for freq in (500,1000,2000,4000)]
PASS=[f'T_HR{freq}_PASS' for freq in (500,1000,2000,4000)]

def read_selected(path, variables):
    _,meta=pyreadstat.read_sas7bdat(path,metadataonly=True,encoding='euc-kr')
    names={n.upper():n for n in meta.column_names}
    cols=[names[c] for c in variables if c in names]
    df,_=pyreadstat.read_sas7bdat(path,encoding='euc-kr',usecols=cols)
    df.columns=[c.upper() for c in df.columns]
    return df

def korean_year(year):
    base=DATA/'extracted'/f'hn{year%100:02d}_all.sas7bdat'
    common=['ID','AGE','SEX','PSU','KSTRATA','WT_ITVEX','WT_EX1','HE_BMI','DE1_DG',
            'BS1_1','BS3_1','HE_CREA','HE_HB']
    if year>=2019: common += FREQ+PASS+['T_Q_HR']
    d=read_selected(base,common)
    assert d.ID.notna().all() and d.ID.is_unique
    if year<=2012:
        ent=read_selected(DATA/f'{year}_ent.sas7bdat',['ID']+FREQ+['T_Q_HR'])
        assert ent.ID.notna().all() and ent.ID.is_unique
        d=d.merge(ent,on='ID',how='left',validate='one_to_one')
    assert set(FREQ).issubset(d.columns)
    values=d[FREQ].apply(pd.to_numeric,errors='coerce')
    # In later KNHANES files some no-responses are encoded as missing. We do not
    # infer that every missing value is a no-response; this pilot is complete-case.
    upper=110 if year>=2019 else 120
    valid=values.notna().all(axis=1)&values.ge(-10).all(axis=1)&values.le(upper).all(axis=1)
    right=values.iloc[:,:4].mean(axis=1); left=values.iloc[:,4:].mean(axis=1)
    d['better_ear_pta']=np.minimum(left,right)
    d['outcome']=np.where(valid,(d.better_ear_pta>25).astype(float),np.nan)
    d['outcome_40']=np.where(valid,(d.better_ear_pta>=40).astype(float),np.nan)
    d['quality_pass']=True if year<=2012 else d[PASS].eq(0).all(axis=1)
    d['age']=d.AGE; d['female']=d.SEX.map({1.:0.,2.:1.})
    d['bmi']=d.HE_BMI; d['creatinine']=d.HE_CREA; d['hemoglobin']=d.HE_HB
    diabetes={0.:0.,1.:1.}
    if year<=2012: diabetes[8.]=0.
    d['diabetes']=d.DE1_DG.map(diabetes)
    d['ever_smoked']=d.BS1_1.map({1.:0.,2.:1.,3.:0.})
    d['current_smoker']=np.nan
    d.loc[d.BS1_1.eq(3),'current_smoker']=0
    d.loc[d.BS1_1.isin([1,2]) & d.BS3_1.isin([1,2,3]),'current_smoker']=(
        d.BS1_1.eq(2) & d.BS3_1.isin([1,2])).astype(float)
    d['self_hearing_difficulty']=d.T_Q_HR.map({1.:0.,2.:1.,3.:1.,4.:1.})
    d['weight']=d.WT_EX1 if year==2019 else d.WT_ITVEX
    d['year']=year
    # Survey strata and PSU identifiers must be unique when annual files are pooled.
    d['survey_strata']=str(year)+'_'+d.KSTRATA.astype(str)
    d['survey_psu']=str(year)+'_'+d.PSU.astype(str)
    d['group']=d['survey_psu']
    return d

def logistic(tuned=True):
    pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                   ('scale',StandardScaler()),
                   ('model',LogisticRegression(solver='liblinear',penalty='l1',max_iter=5000,random_state=42))])
    if tuned:
        return GridSearchCV(pipe,{'model__C':[0.01,0.03,0.1,0.3,1,3]},scoring='neg_log_loss',
                            cv=GroupKFold(n_splits=4),n_jobs=1)
    return pipe.set_params(model__C=100000)

def metric(y,p,w):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p),1e-7,1-1e-7); w=np.asarray(w,float)
    logits=np.log(p/(1-p)); valid=np.isfinite(w)&(w>0)
    fit=sm.GLM(y,sm.add_constant(logits),family=sm.families.Binomial()).fit()
    citl=sm.GLM(y,np.ones((len(y),1)),offset=logits,family=sm.families.Binomial()).fit()
    return {'n':len(y),'events':int(y.sum()),'prevalence_unweighted':float(y.mean()),
            'auc_unweighted':float(roc_auc_score(y,p)),'pr_auc_unweighted':float(average_precision_score(y,p)),
            'brier_unweighted':float(brier_score_loss(y,p)),
            'calibration_slope_unweighted':float(fit.params[1]),'calibration_in_large_unweighted':float(citl.params[0]),
            'n_valid_weights':int(valid.sum()),
            'auc_weighted_point':float(roc_auc_score(y[valid],p[valid],sample_weight=w[valid])),
            'brier_weighted_point':float(brier_score_loss(y[valid],p[valid],sample_weight=w[valid])),
            'prevalence_weighted_point':float(np.average(y[valid],weights=w[valid]))}

def screen(y,pos,w):
    y=np.asarray(y,int); pos=np.asarray(pos,bool); w=np.asarray(w,float)
    tp=np.sum(w*pos*(y==1)); fn=np.sum(w*(~pos)*(y==1))
    fp=np.sum(w*pos*(y==0)); tn=np.sum(w*(~pos)*(y==0)); total=w.sum()
    return {'n':len(y),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),
            'referrals_per_1000':float(1000*(tp+fp)/total),'detected_per_1000':float(1000*tp/total),
            'missed_per_1000':float(1000*fn/total),'referrals_per_detected':float((tp+fp)/tp) if tp else None}

def main():
    kd=[korean_year(y) for y in [2010,2011,2012,2019,2020,2021,2022]]
    flow=[]
    for df in kd:
        year=int(df.year.iloc[0]); age=df.age.ge(40); out=df.outcome.notna(); quality=df.quality_pass
        flow.append({'year':year,'basic_n':len(df),'age_40plus':int(age.sum()),
                     'age_40plus_complete_outcome':int((age&out).sum()),
                     'age_40plus_complete_quality':int((age&out&quality).sum()),
                     'missing_predictors_among_eligible':{c:int(df.loc[age&out&quality,c].isna().sum()) for c in FEATURES}})
    dev=pd.concat([d[(d.age>=40)&d.outcome.notna()] for d in kd if d.year.iloc[0]<=2012],ignore_index=True)
    later=pd.concat([d[(d.age>=40)&d.outcome.notna()&d.quality_pass] for d in kd if d.year.iloc[0]>=2019],ignore_index=True)
    assert dev.group.notna().all() and dev.outcome.notna().all()
    dev['model_year']=dev.year.astype(str)
    model=logistic(); model.fit(dev[FEATURES],dev.outcome.astype(int),groups=dev.group)
    age_model=logistic(False); age_model.fit(dev[['age']],dev.outcome.astype(int))
    folds=GroupKFold(n_splits=5); oof=np.zeros(len(dev)); oof_age=np.zeros(len(dev))
    for fold,(tr,te) in enumerate(folds.split(dev,dev.outcome,dev.group),1):
        m=logistic(); m.fit(dev.iloc[tr][FEATURES],dev.iloc[tr].outcome.astype(int),groups=dev.iloc[tr].group)
        a=logistic(False); a.fit(dev.iloc[tr][['age']],dev.iloc[tr].outcome.astype(int))
        oof[te]=m.predict_proba(dev.iloc[te][FEATURES])[:,1]
        oof_age[te]=a.predict_proba(dev.iloc[te][['age']])[:,1]
        print('Nested grouped CV fold',fold,'complete',flush=True)
    ydev=dev.outcome.astype(int)
    positives=oof[ydev.eq(1)]
    threshold_80=float(np.quantile(positives,0.2))
    performance=[{'dataset':'Korea_2010_2012_development_grouped_nested_cv','age':'40+',
                  'model':'core',**metric(ydev,oof,dev.weight/3)},
                 {'dataset':'Korea_2010_2012_development_grouped_nested_cv','age':'40+',
                  'model':'age_only',**metric(ydev,oof_age,dev.weight/3)}]
    comparisons=[]
    def evaluate(df,label,age_label):
        if len(df)==0 or df.outcome.nunique()<2: return
        core=model.predict_proba(df[FEATURES])[:,1]
        age=age_model.predict_proba(df[['age']])[:,1]
        for name,p in [('core',core),('age_only',age)]:
            performance.append({'dataset':label,'age':age_label,'model':name,**metric(df.outcome,p,df.weight)})
        mask=df.self_hearing_difficulty.notna()&df.weight.notna()&(df.weight>0)
        z=df.loc[mask]; ps=core[mask.to_numpy()]
        if len(z) and z.outcome.nunique()==2:
            for name,pred in [('age_60_plus',z.age>=60),('self_report_difficulty',z.self_hearing_difficulty==1),
                              ('model_threshold_development_80pct_sensitivity',ps>=threshold_80)]:
                comparisons.append({'dataset':label,'age':age_label,'strategy':name,
                                    **screen(z.outcome,pred,z.weight)})
    for year in [2019,2020,2021,2022]:
        for tag,mask in [('40-69',later.age.between(40,69)),('70+',later.age>=70)]:
            evaluate(later[(later.year==year)&mask],f'Korea_{year}',tag)
    for cyc,label,age_min,age_max in [('2011_2012','US_2011_2012',40,69),('2015_2016','US_2015_2016',40,69),
                                     ('2017_2020','US_2017_2020_pre_pandemic',70,200)]:
        x=nh.load(cyc); x=x[x.age.between(age_min,age_max)&x.outcome_100.notna()].copy()
        x['outcome']=x.outcome_100; evaluate(x,label,'70+' if age_min==70 else '40-69')
    perf=pd.DataFrame(performance); perf.to_csv(OUT/'cross_country_hearing_exploratory_performance.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'cross_country_hearing_exploratory_screening.csv',index=False)
    report={'status':'Exploratory pilot, not submission-ready','cohort_flow':flow,
            'development':{'n':len(dev),'events':int(ydev.sum()),'features':FEATURES,
                           'selected_C':model.best_params_['model__C'],
                           'risk_threshold_for_80pct_sensitivity_from_development_oof':threshold_80},
            'performance':performance,'screening_comparisons':comparisons,
            'limitations':['All analyses are exploratory after prior data inspection; no final protocol locked.',
                           'Complete numeric audiograms only for Korea; missing may include no-response and excluded severe cases.',
                           'Noise flags are only available and applied to 2019-2022; early period has unverified quality criteria.',
                           'Survey weights used only in point estimates; fitting and CV selection are unweighted.',
                           'No complex survey design confidence intervals; uncertainty and recalibration not yet assessed.',
                           'Self-report questions differ between countries; comparison on complete responses only.',
                           'Calibration and operating points may not transport across surveys; 80% threshold is exploratory.',
                           'A clinical hearing loss diagnosis or undiagnosed status is not established by audiometry alone.']}
    (OUT/'cross_country_hearing_exploratory_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'development':report['development'],'performance':performance,'screening_comparisons':comparisons},indent=2),flush=True)

if __name__=='__main__': main()
