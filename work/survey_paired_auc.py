"""Survey-linearized paired AUC difference, conditional on fitted scores.

The first-order derivative treats survey weights as observation weights. Stratum
variance uses all PSUs represented in the positive-weight survey sample, with
zero contribution from PSUs outside the analytic domain. No FPC or model
training uncertainty is included. Independent specialist review is required.
"""
import numpy as np
from scipy.stats import t


def auc_and_influence(y, scores, weights):
    y, scores, weights = np.asarray(y, int), np.asarray(scores, float), np.asarray(weights, float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all() or not np.all(weights > 0):
        raise ValueError('Invalid outcome, score, or weight')
    cases, controls = y == 1, y == 0
    if not cases.any() or not controls.any():
        raise ValueError('Both outcome classes are required')

    def weighted_cdf(query, ref_score, ref_weight):
        order = np.argsort(ref_score, kind='stable')
        ordered, rw = ref_score[order], ref_weight[order]
        cum = np.r_[0., np.cumsum(rw)]
        left = np.searchsorted(ordered, query, side='left')
        right = np.searchsorted(ordered, query, side='right')
        return (cum[left] + .5 * (cum[right] - cum[left])) / cum[-1]

    case_concordance = weighted_cdf(scores[cases], scores[controls], weights[controls])
    control_concordance = 1 - weighted_cdf(scores[controls], scores[cases], weights[cases])
    auc = np.average(case_concordance, weights=weights[cases])
    influence = np.zeros(len(y))
    influence[cases] = weights[cases]/weights[cases].sum() * (case_concordance-auc)
    influence[controls] = weights[controls]/weights[controls].sum() * (control_concordance-auc)
    return float(auc), influence


def paired_delta(y, p_base, p_expanded, weights, strata, psu, full_design):
    a, ia = auc_and_influence(y, p_base, weights)
    b, ib = auc_and_influence(y, p_expanded, weights)
    influence = ib - ia
    totals = {tuple(key): 0. for key in full_design}
    for key, value in zip(zip(strata, psu), influence):
        if key not in totals:
            raise ValueError('Domain PSU missing from full survey design')
        totals[key] += value
    grouped = {}
    for (stratum, cluster), total in totals.items():
        grouped.setdefault(stratum, []).append(total)
    variance = 0.
    for values in grouped.values():
        n = len(values)
        if n < 2:
            raise ValueError('Singleton PSU stratum: policy requires independent verification')
        variance += n/(n-1) * np.sum((np.asarray(values)-np.mean(values))**2)
    domain_keys = set(zip(strata, psu))
    df = len(domain_keys) - len(set(strata))
    if df <= 0:
        raise ValueError('Insufficient subgroup degrees of freedom')
    se = float(np.sqrt(variance))
    delta = b-a
    bound = float(t.ppf(.975, df)*se)
    return {'base_auc': a, 'expanded_auc': b, 'difference': delta,
            'linearized_standard_error': se, 'domain_degrees_of_freedom': df,
            'full_design_psus': len(totals), 'full_design_strata': len(grouped),
            'conditional_interval_low': delta-bound, 'conditional_interval_high': delta+bound}
