# Incremental hearing-loss model analysis code

This source-only release accompanies the exploratory manuscript “Incremental Value of Routine Examination Variables Beyond Self-reported Hearing for Audiometric Hearing Loss.” It contains analysis scripts, aggregate outputs, and model metadata; it does not contain individual-level KNHANES or NHANES records.

Obtain survey files from the official Korea Disease Control and Prevention Agency and US Centers for Disease Control and Prevention portals under their data-use terms. Never commit or redistribute survey microdata. See `README.md` for the exact input layout, pinned Python environment, R requirements, and one-command reproduction workflow.

The full workflow includes the operating-point Table 4 generator and an aggregate-result verifier. It was rerun from a clean Python 3.12.14 environment in November 2026. Korean 2019 singleton-PSU intervals remain design-sensitive, and the analysis is exploratory rather than a clinical decision tool or a new blinded validation.

## Current source and archive

GitHub source tag: `v0.1.2`. Persistent Zenodo concept DOI: https://doi.org/10.5281/zenodo.24149658. The Zenodo concept record may still resolve to archived version 0.1.1 until this updated source snapshot is deposited; verify the version shown on the DOI landing page.

## License

Code and aggregate outputs are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). The underlying KNHANES and NHANES source data are not covered by this license and remain subject to agency terms. See `LICENSE`.
