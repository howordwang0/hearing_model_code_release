"""Run the exploratory manuscript analyses in reproducible order.

Acquire all required survey files from KDCA/CDC and place them under work/data
as described in README.md. Individual-level source data are never copied to
outputs or committed by this script.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    *(f"work/data/knhanes/extracted/hn{year}_all.sas7bdat" for year in (10, 11, 12, 19, 20, 21, 22)),
    *(f"work/data/knhanes/{year}_ent.sas7bdat" for year in (2010, 2011, 2012)),
]
NHANES_FILES = {
    "2001_2002": ("demo_b", "aux_b", "bmx_b", "l40_b", "l25_b", "diq_b", "smq_b", "auq_b"),
    "2003_2004": ("demo_c", "aux_c", "bmx_c", "l40_c", "l25_c", "diq_c", "smq_c", "auq_c"),
    "2011_2012": ("demo_g", "aux_g", "bmx_g", "biopro_g", "cbc_g", "diq_g", "smq_g", "auq_g"),
    "2015_2016": ("demo_i", "aux_i", "bmx_i", "biopro_i", "cbc_i", "diq_i", "smq_i", "auq_i"),
    # Used by the earlier exploratory pilot output, not by the final manuscript's primary estimates.
    "2017_2020": ("p_demo", "p_aux", "p_bmx", "p_biopro", "p_cbc", "p_diq", "p_smq", "p_auq"),
}
for cycle, stems in NHANES_FILES.items():
    REQUIRED_FILES.extend(f"work/data/nhanes/{cycle}/{stem}.xpt" for stem in stems)

SCRIPTS = [
    "cross_country_hearing_pilot.py",
    "incremental_self_report_models.py",
    "review_paired_rao_wu.py",
    "cross_country_cohort_baseline.py",
    "export_incremental_model_card.py",
    "incremental_complete_case_sensitivity.py",
    "hearing_outcome_threshold_sensitivity.py",
    "age_support_sensitivity.py",
    "incremental_same_cycle_recalibration.py",
    "incremental_exploratory_dca.py",
    "reproduce_operating_point_table.py",
    "korean_design_and_999_sensitivity.py",
    "verify_reproduction_outputs.py",
]


def main():
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit("Required survey files are missing; obtain them from KDCA/CDC:\n" + "\n".join(missing))
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if not rscript or not Path(rscript).is_file():
        raise SystemExit("Rscript is required for the Korea 2019 and 2010-code sensitivity checks. Set RSCRIPT or add it to PATH.")
    for name in SCRIPTS:
        path = ROOT / "work" / name
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
