"""Check intercept-update identities and primary cohort consistency."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/pydeps'))
import numpy as np
import pandas as pd

d=pd.read_csv(ROOT/'outputs/增量模型_美国2015同样本截距更新.csv')
assert len(d)==8 and set(d.model)=={'base','expanded'}
assert np.allclose(d.citl_after,d.citl_before-d.intercept_update,atol=1e-8)
assert np.allclose(d.auc_before,d.auc_after,atol=1e-12)
assert np.allclose(d.slope_before,d.slope_after,atol=1e-8)
apparent=d[d.cohort.eq('US_2015_2016')].set_index('model')
assert apparent.status.eq('apparent same-sample update').all()
assert np.allclose(apparent.citl_after,0,atol=1e-8)
assert np.isclose(apparent.loc['base','citl_before'],-.723711,atol=1e-6)
assert np.isclose(apparent.loc['base','brier_before'],.091648,atol=1e-6)
assert np.isclose(apparent.loc['base','brier_after'],.083707,atol=1e-6)
assert (apparent.brier_after < apparent.brier_before).all()
assert np.isclose(apparent.loc['base','development_unweighted_prevalence'],3105/11679)
assert np.isclose(apparent.loc['base','development_weighted_prevalence'],.21768517949910238)
transferred=d[(d.cohort=='US_2001_2002')&(d.model=='base')].iloc[0]
assert transferred.citl_after>0 and transferred.brier_after>transferred.brier_before
bins=pd.read_csv(ROOT/'outputs/增量模型_同样本校准图分组汇总.csv')
assert len(bins)==40 and bins.groupby(['model','stage']).n.sum().eq(2564).all()
print('Apparent intercept-update identities, prevalences, transfer and calibration groups verified.')
