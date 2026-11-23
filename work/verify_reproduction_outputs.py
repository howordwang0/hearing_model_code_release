"""Check high-priority manuscript aggregates after a full reproduction run."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
checks = []

def ok(condition, message):
    if not condition:
        raise AssertionError(message)
    checks.append(message)

dev = pd.read_csv(OUT / "incremental_self_report_exploratory.csv")
r = dev[dev.cohort.eq("Korea_2010_2012_nested_cv_40plus")].iloc[0]
ok(int(r.n) == 11679 and int(r.events) == 3105, "development n/events 11,679/3,105")
ok(abs(r.base_auc_weighted - 0.8711713438418063) < 1e-10, "base development weighted OOF AUC")
ok(abs(r.expanded_auc_weighted - 0.871603898780278) < 1e-10, "expanded development weighted OOF AUC")

paired = pd.read_csv(OUT / "复核后_配对抽样_探索性结果.csv")
paired = paired[paired.metric.eq("auc_difference")]
expected = {
    "Korea_2020": (0.0024, -0.0004, 0.0054),
    "Korea_2021": (-0.0007, -0.0036, 0.0021),
    "Korea_2022": (0.0024, -0.0003, 0.0050),
    "US_2001_2002": (0.0011, -0.0060, 0.0106),
    "US_2003_2004": (0.0049, -0.0046, 0.0135),
    "US_2011_2012": (0.0015, -0.0037, 0.0069),
    "US_2015_2016": (0.0012, -0.0040, 0.0067),
}
for cohort, wanted in expected.items():
    row = paired[paired.cohort.eq(cohort)].iloc[0]
    got = tuple(round(float(row[k]), 4) for k in ("point", "low", "high"))
    ok(int(row.valid_replicates) == 500 and got == wanted, f"500-replicate paired AUC interval: {cohort}")
row19 = paired[paired.cohort.eq("Korea_2019")].iloc[0]
ok(int(row19.valid_replicates) == 0, "Korea 2019 singleton strata remain separately unbootstrapped")

ops = pd.read_csv(OUT / "operating_point_transportability_supplement.csv")
expected_ops = {
    "US_2001_2002": (40.3, 151.4, 83.0),
    "US_2003_2004": (40.1, 166.8, 74.7),
    "US_2011_2012": (42.1, 192.2, 69.9),
    "US_2015_2016": (51.7, 233.3, 57.8),
}
for cohort, wanted in expected_ops.items():
    row = ops[ops.cohort.eq(cohort)].iloc[0]
    got = (round(100 * float(row.model_sensitivity), 1), round(float(row.model_referral_per_1000), 1), round(float(row.model_missed_per_1000), 1))
    ok(got == wanted, f"Table 4 operating point: {cohort}")

cc = json.loads((OUT / "incremental_complete_case_sensitivity.json").read_text(encoding="utf-8"))
ok(cc["development_complete_n"] == 10826 and cc["development_complete_events"] == 2699, "complete-case development count")
ge40 = pd.read_csv(OUT / "hearing_outcome_ge40_sensitivity.csv")
r = ge40[ge40.cohort.eq("US_2015_2016")].iloc[0]
ok(round(100 * float(r.model_sensitivity), 1) == 67.2 and round(100 * float(r.self_report_sensitivity), 1) == 93.1, "PTA >=40 dB US sensitivity")

lonely = pd.read_csv(OUT / "korea_2019_singleton_assumptions_sensitivity.csv")
for method, expected_ci in (("adjust", (0.0012, 0.0092)), ("average", (0.0007, 0.0097))):
    row = lonely[lonely.method.eq(method)].iloc[0]
    got = (round(float(row.delta_low_t), 4), round(float(row.delta_high_t), 4))
    ok(got == expected_ci, f"Korea 2019 singleton sensitivity: {method}")

print("PASS: " + str(len(checks)) + " manuscript aggregate checks")
for check in checks:
    print("- " + check)
