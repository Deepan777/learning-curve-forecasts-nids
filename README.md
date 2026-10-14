# Learning-curve forecasts of labelling needs under distribution shift (network intrusion detection)

Code, protocols, per-fit records and result files for the article

> Joshitha N, P. Deepanramkumar and A. Helen Sharmila.
> **Learning-curve forecasts underestimate the labelling needs of network intrusion detectors facing unseen attack variants**.
> Manuscript submitted to *Scientific Reports* (2026).

The study asks whether the number of labelled flows that an intrusion detector needs can be forecast from a small
labelled pilot, and when such a forecast can be trusted. It pairs pilot learning-curve forecasts with the
requirements realised on target traffic in 99 source-target scenarios (in-distribution, later traffic, changed
benign traffic and held-out attack variants) built from four public benchmarks, five classifiers and 24 attack
families, and evaluates scores computed without target labels as abstention signals. It does not propose a new
detector.

## Repository layout

```
04_Code/
  run.py                    pre-specified in-distribution analysis (H1-H6): tasks | curves | geometry | analyze | figures
  run_shift.py              follow-up study (F1-F6): scenarios | curves | floors | analyze | robustness | figures | tables
  configs/default.json      pre-specified settings, with dated amendments
  configs/shift.json        follow-up protocol (written before the follow-up experiments), with its dated amendment
  lcurve/                   data loading, task construction, learning curves, curve fitting, first analysis
  shiftfc/                  scenarios, paired curves, label-free scores, forecasts, analysis, figures, tables;
                            audit_temporal_composition.py = post hoc descriptive check (Supplementary Table S15)
  launch_shift_workers.ps1  starts parallel follow-up workers (Windows)
  requirements.txt          exact package versions used
05_Data/                    data audit and NSL-KDD verification scripts and outputs (datasets not included)
06_Experiments/raw/         per-fit records of the first analysis
06_Experiments/shift/       follow-up per-fit records (curves/), scores (signals/), residual-error controls (floors/)
06_Experiments/audit/       outputs of the audit of the first analysis and of the post hoc composition check
09_Statistical_Analysis/    intermediate statistics (forecast table, LODO thresholds, bootstrap outputs)
10_Robustness/              sensitivity analyses of the first analysis
13_Results/                 Master_Results.csv (first analysis), Shift_Results.csv and Shift_Robustness.csv
                            (follow-up): every number reported in the article is a row of these files
11_Figures/, 12_Tables/     generated figures (PDF/SVG/PNG) and LaTeX tables
figure1_tikz/               editable TikZ source of Figure 1 (compile Figure_1_Framework.tex with any LaTeX engine)
```

## Installation

Python 3.11 (CPU only):

```
pip install -r 04_Code/requirements.txt
```

## Data

The public datasets are not redistributed. Download them from the sources in [`05_Data/README.md`](05_Data/README.md)
and either place them under `data/raw/` or set `LCF_UNSW_DIR`, `LCF_CICIDS2017_DIR` and `LCF_CICIOT2023_DIR`.

## Reproducing the results

From `04_Code/`:

```
python run.py tasks                 # task arrays, de-duplication, label conflicts
python run.py curves --jobs 6       # learning curves of the first analysis
python run.py geometry
python ../05_Data/data_audit.py
python run.py analyze               # -> 13_Results/Master_Results.csv
python run_shift.py scenarios       # follow-up scenarios
python run_shift.py curves --worker 0 --nworkers 1    # or launch_shift_workers.ps1 for parallel workers
python run_shift.py floors
python run_shift.py analyze         # -> 13_Results/Shift_Results.csv
python run_shift.py robustness      # -> 13_Results/Shift_Robustness.csv
python run_shift.py figures
python run_shift.py tables
python -m shiftfc.audit_temporal_composition && python -m shiftfc.table_temporal_composition
```

Every stage skips outputs that already exist. The follow-up study comprises 77,000 model fits (about 9 CPU hours).
All random choices are seeded (global seed 2026; per-cell seeds derived from the cell name), so reruns reproduce the
per-fit records. Because the per-fit records are included, `analyze`, `figures` and `tables` can be run without the
raw data.

## Notes on the analyses

* The first analysis (H1-H6) was pre-specified in `configs/default.json`. The follow-up hypotheses (F1-F6) were
  written in `configs/shift.json` after the first results were known and before the follow-up experiments; its one
  amendment (removal of three CIC-IoT2023 capture-artefact features) is dated in the file.
* The composition check of the temporal scenarios (`audit_temporal_composition.py`) was added after all results
  were known and is reported as post hoc.
* Parts of the code were developed with the assistance of Claude (Anthropic); all code and results were checked by
  the authors.

## Licence and citation

Code: MIT licence (see `LICENSE`). The public datasets remain subject to their own licences. If you use this
repository, please cite the article above (see `CITATION.cff`).
