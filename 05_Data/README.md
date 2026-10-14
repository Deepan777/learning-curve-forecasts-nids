# 05_Data

The four public datasets are **not** redistributed here. `04_Code/lcurve/config.py` reads them in place from the
folders given by the environment variables below (defaults: `data/raw/...` in this repository).

| Dataset | Source | Environment variable | Expected content |
|---|---|---|---|
| NSL-KDD (KDDTrain+, 41 features) | https://github.com/jmnwong/NSL-KDD-Dataset (copy of the archived UNB release) | - | built into `05_Data/source/KDDTrain_plus_full.parquet` by `verify_nslkdd_download.py` |
| UNSW-NB15 (raw release) | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `LCF_UNSW_DIR` | `UNSW-NB15_1..4.csv`, `NUSW-NB15_features.csv` |
| CICIDS2017 (MachineLearningCVE) | https://www.unb.ca/cic/datasets/ids-2017.html | `LCF_CICIDS2017_DIR` | the eight per-day CSV files |
| CIC-IoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html | `LCF_CICIOT2023_DIR` | `train.csv` |

`data_audit.py` produces the data-integrity audit (`data_audit_results.json`, Supplementary Table S1);
`verify_nslkdd_download.py` documents the row-by-row check of the NSL-KDD copy (`nslkdd_download_verification.json`).
`ciciot_fine_verification.json` records that the fine-grained CIC-IoT2023 labels were recovered exactly.
On first use the loaders write caches to `05_Data/cache` and task arrays to `05_Data/tasks` (not tracked).
