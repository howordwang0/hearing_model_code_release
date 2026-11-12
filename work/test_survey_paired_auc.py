"""Check weighted ties and influence derivatives against brute force."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work/pydeps'))
import numpy as np
from sklearn.metrics import roc_auc_score
from survey_paired_auc import auc_and_influence, paired_delta

y = np.array([0, 1, 0, 1, 0, 1])
p = np.array([.2, .3, .3, .3, .8, .9])
w = np.array([1., 2., 3., 1., 2., 4.])
auc, influence = auc_and_influence(y, p, w)
assert np.isclose(auc, roc_auc_score(y, p, sample_weight=w))
for i in range(len(y)):
    step = 1e-6
    perturbed = w.copy()
    perturbed[i] *= 1+step
    numerical = (auc_and_influence(y, p, perturbed)[0]-auc)/step
    assert np.isclose(influence[i], numerical, atol=1e-6)
strata = np.array([1, 1, 1, 2, 2, 2])
psus = np.array([1, 2, 2, 1, 2, 2])
result = paired_delta(y, p, p, w, strata, psus, [(1,1), (1,2), (2,1), (2,2)])
assert result['difference'] == 0 and result['linearized_standard_error'] == 0
print('Weighted AUC ties, influence derivatives, and paired equality passed.')
