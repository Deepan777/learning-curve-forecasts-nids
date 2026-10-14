# lcurve — sample complexity of network intrusion detection

Code for the study *"How many labelled flows does an intrusion detector need? Learning curves, irreducible error and time across four benchmarks"* (working title).

The code measures, for every attack family of four public benchmarks:
- how fast standard learners learn to separate the family from benign traffic;
- what limits the error that remains.

It does not propose a detector.

## Layout

```
04_Code/
  run.py                  stage driver (tasks | curves | geometry | analyze | figures | smoke)
  configs/default.json    every pre-specified setting (sizes, repeats, learners, eps, seeds), with dated amendments
  lcurve/
    config.py             paths (raw data read in place; caches in 05_Data/cache)
    data.py               loaders that keep time, host and attack family; cleaning rules
    tasks.py              one task per attack family vs benign; de-duplication; exact-conflict floor
    curves.py             learning-curve protocol (nested training sets, 6 learners, 10 repeats;
                          random, chronological and random-control orders)
    geometry.py           intrinsic dimension (TwoNN, MLE) and cross-class nearest-neighbour overlap
    fitting.py            curve models (pure / saturating power law, exponential), AICc, n*(eps), forecasts
    analysis.py           pre-specified hypotheses H1-H6, sensitivity analyses, Master_Results.csv
    figures.py, tables.py publication figures and LaTeX tables
  ../05_Data/data_audit.py  data-integrity audit (Supplementary Table S1)
```

All code under `04_Code/` was written for this study. It has no dependency on other research code.

## Environment
- Python 3.11.5, CPU only.
- Install with `pip install -r requirements.txt`.

## Data
The raw public datasets are read in place from the paths in `lcurve/config.py`:
- NSL-KDD KDDTrain+;
- UNSW-NB15 raw `UNSW-NB15_1..4.csv` with `NUSW-NB15_features.csv`;
- CICIDS2017 MachineLearningCVE per-day CSVs;
- CIC-IoT2023 `train.csv`.

On first use, the loaders build compact caches in `05_Data/cache`. `run.py tasks` then writes the per-task arrays to `05_Data/tasks`.

## Reproducing every result
Run from `04_Code/`, in this order:
```
python run.py tasks                     # task arrays, de-duplication, conflict counts (~2 min)
python run.py curves --jobs 6           # all learning curves (~1.5 h on 6 cores)
python run.py geometry                  # intrinsic dimension per task (~1 min)
python ../05_Data/data_audit.py         # data-integrity audit
python run.py analyze                   # fits + bootstrap + hypotheses -> 13_Results/Master_Results.csv
python run.py figures                   # 11_Figures, 12_Tables, 15_Supplementary tables
cd ../14_Manuscript
python count_words.py && python trace_numbers.py && python overlap_check.py
tectonic -o _build main.tex && python make_docx.py
```

Every stage skips outputs that already exist. Delete the corresponding files to recompute.

**Determinism.** Every random choice uses a NumPy generator seeded from `(seed, repeat, CRC32 of the cell name)`, so Python's salted `hash` is never used. Learners are seeded per repeat.

## Where each manuscript number comes from
Every number in the manuscript is a row of `13_Results/Master_Results.csv`. Its columns are analysis, dataset, family, learner, metric, value, ci_low, ci_high, n, note and source.

The `source` column names the intermediate CSV in `09_Statistical_Analysis/` or `10_Robustness/`. Those CSVs are built from the raw per-fit records in `06_Experiments/raw/curves/`.
