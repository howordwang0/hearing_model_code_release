"""Sanity-check aggregate sensitivity outputs without exposing records."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work/pydeps'))
import numpy as np
import pandas as pd

out = ROOT/'outputs'
patterns = pd.read_csv(out/'korean_2010_999_pattern_audit.csv').set_index('population')
assert patterns.loc['eligible_except_outcome_2010', 'any_999'] == 280
assert patterns.loc['eligible_except_outcome_2010', 'all_8_999'] == 279
assert patterns.loc['eligible_except_outcome_2010', 'mixed_999_and_valid'] == 1
assert patterns.loc['eligible_except_outcome_2010', 'both_ears_any_valid_and_999'] == 0
results = pd.read_csv(out/'korean_2010_999_extreme_label_sensitivity.csv')
assert len(results) == 24
for scenario, n, events in [('exclude_999', 11679, 3105),
                             ('all_999_negative', 11959, 3105),
                             ('all_999_positive', 11959, 3385)]:
    d = results[results.scenario.eq(scenario)]
    assert len(d) == 8 and d.cohort.nunique() == 8
    assert d.training_n.eq(n).all() and d.training_events.eq(events).all()
    assert np.allclose(d.auc_difference, d.expanded_auc-d.base_auc)
    assert np.isfinite(d[['base_citl','expanded_citl','base_brier','expanded_brier']]).all().all()
lonely = pd.read_csv(out/'korea_2019_singleton_assumptions_sensitivity.csv')
assert set(lonely.method) == {'adjust', 'average', 'certainty'}
assert lonely.full_singleton_strata.eq(5).all() and lonely.domain_df.eq(60).all()
assert (lonely.delta_low_t < lonely.delta_auc).all() and (lonely.delta_high_t > lonely.delta_auc).all()
assert lonely.status.str.contains('exploratory').all()
formats = pd.read_csv(out/'knhanes_2010_sas_spss听阈编码核对.csv')
assert len(formats) == 8 and formats.mismatched_values.eq(0).all()
assert formats.sas_999.eq(formats.spss_999).all()
assert not formats.spss_value_labels_present.any()
bounds = pd.read_csv(out/'knhanes_2010_2012听阈上限敏感性审计.csv')
assert set(bounds.year) == {2010, 2011, 2012}
assert bounds.any_111_to_120.eq(0).all()
assert bounds.changed_complete_outcome_status.eq(0).all()
print('2010 code patterns, source-format match, bounds, calibration, and 2019 assumptions passed.')
