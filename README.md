# Hearing-loss model analysis code

This repository contains the analysis code and aggregate outputs accompanying the exploratory manuscript “Incremental Value of Routine Examination Variables Beyond Self-reported Hearing for Audiometric Hearing Loss.” No individual-level KNHANES or NHANES records are included. Obtain source files from the official Korea Disease Control and Prevention Agency (KDCA) and US Centers for Disease Control and Prevention (CDC) portals under their terms of use. Do not commit or redistribute survey microdata.

The analysis is exploratory and was performed after prior inspection of the survey samples. Re-running these scripts is computational reproduction, not a new blinded validation. The KNHANES 2019 component design and singleton-PSU variance treatment remain unresolved and are presented as sensitivity analyses only.

## Reproduce the analysis

Tested environment: Windows x64, Python 3.12.14, NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, pyreadstat 1.3.7, statsmodels 0.15.0, and matplotlib 3.11.2. Install the pinned environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

The 2019 KNHANES and 2010 code-sensitivity checks use R 4.5.2 and the R package `survey` 4.4-8. Add `Rscript` to `PATH`, or set `RSCRIPT` to the full executable path (for example, `$env:RSCRIPT = 'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'`).

Download and arrange the source data as follows. `.xpt` files should use the lower-case names shown below; original data formats must be preserved for KNHANES.

```text
work/data/knhanes/extracted/hn10_all.sas7bdat
work/data/knhanes/extracted/hn11_all.sas7bdat
work/data/knhanes/extracted/hn12_all.sas7bdat
work/data/knhanes/extracted/hn19_all.sas7bdat
work/data/knhanes/extracted/hn20_all.sas7bdat
work/data/knhanes/extracted/hn21_all.sas7bdat
work/data/knhanes/extracted/hn22_all.sas7bdat
work/data/knhanes/2010_ent.sas7bdat
work/data/knhanes/2011_ent.sas7bdat
work/data/knhanes/2012_ent.sas7bdat
work/data/nhanes/2001_2002/{demo_b,aux_b,bmx_b,l40_b,l25_b,diq_b,smq_b,auq_b}.xpt
work/data/nhanes/2003_2004/{demo_c,aux_c,bmx_c,l40_c,l25_c,diq_c,smq_c,auq_c}.xpt
work/data/nhanes/2011_2012/{demo_g,aux_g,bmx_g,biopro_g,cbc_g,diq_g,smq_g,auq_g}.xpt
work/data/nhanes/2015_2016/{demo_i,aux_i,bmx_i,biopro_i,cbc_i,diq_i,smq_i,auq_i}.xpt
work/data/nhanes/2017_2020/{p_demo,p_aux,p_bmx,p_biopro,p_cbc,p_diq,p_smq,p_auq}.xpt
```

The 2017–2020 NHANES files support an earlier exploratory 70+ pilot included in `cross_country_hearing_pilot.py`; that result is outside the final manuscript’s primary evaluation samples but is run by the full pipeline.

From the repository root, run the complete sequence and aggregate-output checks:

```powershell
python work/reproduce_manuscript_outputs.py
```

The runner checks that inputs are present, runs the model, paired resampling, cohort summaries, calibration, decision-curve and sensitivity scripts in dependency order, creates the Table 4 operating-point summary, and finishes with `work/verify_reproduction_outputs.py`. Intermediate R survey files are written to temporary directories under `work/data` and removed on normal completion. If a run is interrupted, inspect and remove only temporary files created by that interrupted run before restarting.

Key entry points are `work/incremental_self_report_models.py` (primary incremental models), `work/review_paired_rao_wu.py` (500 paired PSU resamples), `work/reproduce_operating_point_table.py` (US operating-point table), and `work/korean_design_and_999_sensitivity.py` (Korea 2019 singleton and 2010 code sensitivity). The model development and fixed-model evaluation remain exploratory; point estimates use survey weights while model fitting is unweighted.

All generated files in `outputs/` are aggregate summaries, model metadata or figures. Check the versioned `manifest.json` for file sizes and SHA-256 hashes. The release does not redistribute source survey files.

## Software version and archive

This GitHub source update is version 0.1.2 (`v0.1.2`). The persistent Zenodo concept DOI is [10.5281/zenodo.24149658](https://doi.org/10.5281/zenodo.24149658); it may resolve to an earlier archived snapshot until the updated source package is deposited there. Check the DOI record’s version before treating the Zenodo archive as identical to the current GitHub tag.

## License

Code and aggregate outputs are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Survey data remain subject to the source agencies’ terms. See `LICENSE`.
