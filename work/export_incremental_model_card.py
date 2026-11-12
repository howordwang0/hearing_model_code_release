"""Export aggregate coefficients and preprocessing, with reconstruction checks."""
from pathlib import Path
import json
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps'));sys.path.insert(0,str(ROOT/'work'))
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np
import pandas as pd
from scipy.special import expit
import cross_country_hearing_pilot as c
import incremental_self_report_models as inc
import validate_nhanes_2003_2004 as old


def record(model,features,validation):
    best=model.best_estimator_; imputer=best.named_steps['impute'];scaler=best.named_steps['scale'];clf=best.named_steps['model']
    assert len(imputer.statistics_)==len(features)
    marker=[features[i] for i in imputer.indicator_.features_]
    names=features+[f'missing__{name}' for name in marker]
    coef=clf.coef_[0]/scaler.scale_
    intercept=float(clf.intercept_[0]-np.dot(coef,scaler.mean_))
    assert len(names)==len(coef)
    records=[{'term':name,'coefficient_on_unscaled_imputed_input':float(value)} for name,value in zip(names,coef)]
    prep={'input_features':features,'training_medians':{k:float(v) for k,v in zip(features,imputer.statistics_)},
          'training_missing_indicator_features':marker,'log_odds_intercept':intercept,
          'coefficients':records,'selected_C':float(model.best_params_['model__C'])}
    x=validation[features]
    imputed=x.fillna(pd.Series(prep['training_medians']))
    design=np.column_stack([imputed.to_numpy(float)]+[x[[name]].isna().to_numpy(float) for name in marker])
    reconstructed=expit(intercept+design@coef)
    expected=model.predict_proba(x)[:,1]
    assert np.allclose(reconstructed,expected,atol=1e-12,rtol=1e-10)
    return prep,float(np.max(np.abs(reconstructed-expected)))


def main():
    full=pd.concat([c.korean_year(y) for y in (2010,2011,2012)],ignore_index=True)
    dev=full[full.age.ge(40)&full.outcome.notna()&full.self_hearing_difficulty.notna()&full.weight.gt(0)]
    val=old.load_cycle(2001)
    val=val[val.age.between(40,69)&val.outcome_100.notna()&val.self_hearing_difficulty.notna()&val.WTSAU2YR.gt(0)]
    out={'status':'Exploratory research models only; not validated for clinical deployment.',
         'development_n':len(dev),'events':int(dev.outcome.sum()),
         'model_development':'Unweighted L1 logistic, GroupKFold penalty selection. Indicator terms added only for predictors missing in training.',
         'models':{}}
    for name,features in [('base',inc.BASE),('expanded',inc.EXT)]:
        model=inc.fit(dev,features);data,error=record(model,features,val)
        data['verification_max_absolute_probability_error']=error
        out['models'][name]=data
        print(name,'C',data['selected_C'],'reconstruction error',error,flush=True)
    path=ROOT/'outputs'/'增量模型_系数与插补规则_研究用.json'
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
