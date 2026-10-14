"""Verify the downloaded NSL-KDD KDDTrain+.txt against the 36-feature copy used so far,
then write the complete 41-feature copy as 05_Data/source/KDDTrain_plus_full.parquet.

Usage (run with -I):  python -I verify_nslkdd_download.py <path to KDDTrain+.txt>
Checks: row count 125,973; 43 fields per row; the 36 shared columns, label and difficulty
identical row by row to 05_Data/source/KDDTrain_plus.parquet (string-encoded categories
compared as strings, numbers compared exactly)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "class", "classnum"]


def main(path):
    new = pd.read_csv(path, header=None, names=COLS)
    old = pd.read_parquet(os.path.join(HERE, "source", "KDDTrain_plus.parquet"))
    rep = {"rows_new": len(new), "rows_old": len(old), "fields_new": new.shape[1]}
    shared = [c for c in old.columns if c in new.columns]
    rep["shared_columns"] = len(shared)
    rep["missing_in_old"] = [c for c in COLS if c not in old.columns]
    mism = {}
    for c in shared:
        a, b = new[c], old[c]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            # the earlier copy stores rates as float32: compare at float32 precision
            bad = int((a.to_numpy(np.float32) != b.to_numpy(np.float32)).sum())
        else:
            bad = int((a.astype(str).str.strip() != b.astype(str).str.strip()).sum())
        if bad:
            mism[c] = bad
    rep["mismatching_cells_by_column"] = mism
    rep["identical_on_shared_columns"] = (len(new) == len(old)) and not mism
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(HERE, "nslkdd_download_verification.json"), "w"), indent=1)
    if rep["identical_on_shared_columns"] and len(new) == 125973 and new.shape[1] == 43:
        new.to_parquet(os.path.join(HERE, "source", "KDDTrain_plus_full.parquet"))
        print("verified; wrote source/KDDTrain_plus_full.parquet")
    else:
        print("NOT verified; nothing written")


if __name__ == "__main__":
    main(sys.argv[1])
