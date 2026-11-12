# Incremental hearing-loss model analysis code

This source-only release accompanies the exploratory manuscript “Incremental Value of Routine Examination Variables Beyond Self-reported Hearing for Audiometric Hearing Loss.” It contains analysis scripts, aggregate outputs, and model metadata. It contains no individual-level KNHANES or NHANES records.

The source surveys must be obtained from the official Korea Disease Control and Prevention Agency and US Centers for Disease Control and Prevention portals under their data-use terms. Do not commit downloaded survey files. The current analysis is exploratory; Korean 2019 singleton-PSU handling remains design-sensitive, and the code does not establish clinical deployment.

## Reproducibility

The included scripts were run in the local environment documented in `README.md` at the archive root. Some entry points require R and local paths that must be adapted. The included aggregate-output check reproduces the seven-cohort results table without individual-level records.

## Status

Public repository: https://github.com/howordwang0/hearing_model_code_release (version tag v0.1.0).
Archived copy and persistent DOI: https://doi.org/10.5281/zenodo.24149659 (Zenodo record 24149659, version 0.1.0, licence CC BY 4.0).

## License

Code and aggregate outputs in this release are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See `LICENSE`. The underlying KNHANES and NHANES source data are not covered by this license and remain subject to their own data-use terms.
