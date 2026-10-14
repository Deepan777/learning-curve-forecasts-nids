"""Paths and global constants.

Raw datasets are read in place (read-only). Compact caches are written to
05_Data/cache. Paths to the raw public releases are read from environment variables (see README).
"""

from __future__ import annotations

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(PROJECT_ROOT, "04_Code")
DATA_CACHE = os.path.join(PROJECT_ROOT, "05_Data", "cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "13_Results")
RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, "06_Experiments", "raw")
FIG_DIR = os.path.join(PROJECT_ROOT, "11_Figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "12_Tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "18_Logs")

# Raw data sources (read in place, never modified). Set the environment variables to the folders that hold the
# public releases, or place them under data/raw/ in this repository (see README.md, "Data").
_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
UNSW_DIR = os.environ.get("LCF_UNSW_DIR", os.path.join(_RAW, "UNSW_NB15"))
CICIDS_DIR = os.environ.get("LCF_CICIDS2017_DIR", os.path.join(_RAW, "CIC-IDS2017"))
CICIOT_DIR = os.environ.get("LCF_CICIOT2023_DIR", os.path.join(_RAW, "ciciot2023"))
# NSL-KDD KDDTrain+ with all 41 features, label and difficulty level. Built by 05_Data/verify_nslkdd_download.py
# from KDDTrain+.txt (github.com/jmnwong/NSL-KDD-Dataset, a copy of the archived UNB release) after verifying it
# row by row against the earlier 36-feature copy.
NSLKDD_PARQUET = os.path.join(PROJECT_ROOT, "05_Data", "source", "KDDTrain_plus_full.parquet")

GLOBAL_SEED = 2026

for _d in (DATA_CACHE, RESULTS_DIR, RAW_RESULTS_DIR, FIG_DIR, TABLE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)
