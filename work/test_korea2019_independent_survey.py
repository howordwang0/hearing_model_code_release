"""Check independent 2019 aggregate results against previous point estimates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps'))
import numpy as np
import pandas as pd

out=ROOT/'outputs'
d=pd.read_csv(out/'korea_2019_R独立计算_探索性.csv')
audit=pd.read_csv(out/'korea_2019_抽样设计与分析域审计.csv').iloc[0]
assert len(d)==60 and d.domain_df.eq(60).all()
assert set(d.method)=={'adjust','average','certainty'} and set(d.domain_adjust)=={True,False}
assert (audit.component_strata,audit.component_psus,audit.component_singleton_strata)==(26,96,5)
assert (audit.domain_records,audit.domain_strata,audit.domain_psus)==(1341,26,86)
assert audit.component_psus_without_domain==10 and audit.domain_singleton_strata==6
assert audit.domain_singleton_from_multipsu_component==1
reference=pd.read_csv(out/'复核后_配对抽样_探索性结果.csv')
reference=reference[reference.cohort.eq('Korea_2019')].set_index('metric')
mapping={'auc_base':'auc_base','auc_expanded':'auc_expanded','auc_difference':'auc_difference',
         'citl_base':'base_calibration_intercept','citl_expanded':'expanded_calibration_intercept',
         'calibration_slope_base':'base_calibration_slope','calibration_slope_expanded':'expanded_calibration_slope'}
for method in ('adjust','average','certainty'):
    for adjustment in (True,False):
        sub=d[d.method.eq(method)&d.domain_adjust.eq(adjustment)].set_index('metric')
        assert len(sub)==10 and (sub.se>0).all()
        assert (sub.low<sub.point).all() and (sub.high>sub.point).all()
        for metric,original in mapping.items():
            assert np.isclose(sub.loc[metric,'point'],reference.loc[original,'point'],atol=1e-10)
assert d.scope.str.contains('exploratory').all()
print('2019 design/domain counts, independently computed AUC and calibration estimates verified.')
