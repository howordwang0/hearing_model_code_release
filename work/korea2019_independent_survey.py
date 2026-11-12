"""Independently evaluate the 2019 component using the exported model card.

Only aggregate results are retained; the temporary survey-level file is deleted.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'work/pydeps'),str(ROOT/'work')]
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import cross_country_hearing_pilot as c

OUT=ROOT/'outputs'

def predict(data, model):
    p=np.full(len(data),model['log_odds_intercept'],dtype=float)
    for term in model['coefficients']:
        name=term['term']; coef=term['coefficient_on_unscaled_imputed_input']
        if name.startswith('missing__'):
            val=data[name[len('missing__'):]].isna().to_numpy(dtype=float)
        else:
            val=data[name].fillna(model['training_medians'][name]).to_numpy(dtype=float)
        p+=coef*val
    return expit(p)

def main():
    card=json.loads((OUT/'增量模型_系数与插补规则_研究用.json').read_text(encoding='utf-8'))
    data=c.korean_year(2019)
    component=data.weight.gt(0)&np.isfinite(data.weight)
    domain=component&data.age.between(40,69)&data.outcome.notna()&data.self_hearing_difficulty.notna()&data.quality_pass
    z=data.loc[domain]
    design=data.loc[component,['KSTRATA','PSU','weight']].rename(columns={'KSTRATA':'stratum','PSU':'psu'}).copy()
    design['domain']=domain.loc[design.index].astype(int)
    design['outcome']=0.;design['p0']=.5;design['p1']=.5
    design.loc[z.index,'outcome']=z.outcome
    for name,column in [('base','p0'),('expanded','p1')]:
        design.loc[z.index,column]=predict(z,card['models'][name])
    keys=design[['stratum','psu']].drop_duplicates()
    assert len(design)==2127 and len(keys)==96 and int(keys.groupby('stratum').psu.nunique().eq(1).sum())==5
    domainkeys=design[design.domain.eq(1)][['stratum','psu']].drop_duplicates()
    component_counts=keys.groupby('stratum').psu.nunique()
    domain_counts=domainkeys.groupby('stratum').psu.nunique()
    audit={'positive_weight_records':len(design),'component_strata':len(component_counts),
           'component_psus':len(keys),'component_singleton_strata':int(component_counts.eq(1).sum()),
           'domain_records':int(design.domain.sum()),'domain_strata':len(domain_counts),
           'domain_psus':len(domainkeys),'component_psus_without_domain':len(keys)-len(domainkeys),
           'domain_singleton_strata':int(domain_counts.eq(1).sum()),
           'domain_singleton_from_multipsu_component':int((domain_counts.eq(1)&component_counts.loc[domain_counts.index].gt(1)).sum())}
    pd.DataFrame([audit]).to_csv(OUT/'korea_2019_抽样设计与分析域审计.csv',index=False)
    with tempfile.TemporaryDirectory(prefix='survey_independent_',dir=ROOT/'work/data') as tmp:
        src,dst=Path(tmp)/'design.csv',Path(tmp)/'results.csv'
        design.to_csv(src,index=False)
        run=subprocess.run([r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe',
                            str(ROOT/'work/korea2019_independent_survey.R'),str(src),str(dst)],
                           capture_output=True,text=True,encoding='utf-8',errors='replace')
        if run.returncode: raise RuntimeError(run.stderr)
        result=pd.read_csv(dst)
    for method in ('adjust','average','certainty'):
        q=result[result.method.eq(method)&result.domain_adjust.eq(True)].set_index('metric')
        for metric,col in [('auc_base','p0'),('auc_expanded','p1')]:
            p=roc_auc_score(design.loc[z.index,'outcome'],design.loc[z.index,col],sample_weight=design.loc[z.index,'weight'])
            assert np.isclose(q.loc[metric,'point'],p,atol=1e-12)
        prior=pd.read_csv(OUT/'korea_2019_singleton_assumptions_sensitivity.csv').set_index('method')
        assert np.isclose(q.loc['auc_difference','se'],prior.loc[method,'delta_se'],atol=1e-11)
    assert len(result)==60 and result.domain_df.eq(60).all()
    result.to_csv(OUT/'korea_2019_R独立计算_探索性.csv',index=False)
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
