"""Dataset loaders that keep capture time, protected host and attack family.

Every loader returns a `FlowData` object with
  X        float32 feature matrix (identifiers and timestamps removed)
  y        int8 binary label (0 benign, 1 attack)
  family   attack-family string ("Benign" for benign flows)
  order    integer giving the temporal order of records (or file order)
  period   coarse time period (capture session / day) where available
  host     protected-host identifier where available (UNSW-NB15 only)
  features list of feature names

Caches are compact parquet files in 05_Data/cache. Raw files are read-only.
Sub-sampling rules are fixed, documented, seeded, and stated in
05_Data/Data_Integrity_Audit.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# ----------------------------------------------------------------- container
@dataclass
class FlowData:
    name: str
    X: np.ndarray
    y: np.ndarray
    family: np.ndarray
    order: np.ndarray
    period: np.ndarray
    host: np.ndarray
    features: list
    src_flag: np.ndarray = None   # UNSW-NB15: source in attacker subnet

    def subset(self, idx):
        idx = np.asarray(idx)
        sf = None if self.src_flag is None else self.src_flag[idx]
        return FlowData(self.name, self.X[idx], self.y[idx], self.family[idx],
                        self.order[idx], self.period[idx], self.host[idx],
                        self.features, sf)

    def drop_features(self, names):
        keep = [i for i, f in enumerate(self.features) if f not in set(names)]
        return FlowData(self.name, self.X[:, keep], self.y, self.family, self.order,
                        self.period, self.host, [self.features[i] for i in keep],
                        self.src_flag)

    def __len__(self):
        return len(self.y)


def _from_frame(name, df, feat_cols):
    return FlowData(
        name=name,
        X=df[feat_cols].to_numpy(dtype=np.float32),
        y=df["label"].to_numpy(dtype=np.int8),
        family=df["family"].astype(str).to_numpy(),
        order=df["order"].to_numpy(dtype=np.int64),
        period=df["period"].astype(str).to_numpy(),
        host=df["host"].astype(str).to_numpy(),
        features=list(feat_cols),
    )


def _label_encode_inplace(df, cols):
    for c in cols:
        df[c] = pd.factorize(df[c].astype(str), sort=True)[0].astype(np.float32)


# ----------------------------------------------------------------- UNSW-NB15
UNSW_ID_COLS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]
UNSW_CAT_COLS = ["proto", "state", "service"]


def build_unsw_raw_cache():
    """Read the four raw UNSW-NB15 CSVs (2,540,047 flows) and keep time,
    destination host and attack category. Identifiers (IPs, ports) and
    timestamps are retained as metadata but never used as features."""
    names = pd.read_csv(os.path.join(config.UNSW_DIR, "NUSW-NB15_features.csv"),
                        encoding="latin1")["Name"].str.strip().str.lower()
    names = [n.replace(" ", "") for n in names]
    parts = []
    for i in range(1, 5):
        p = pd.read_csv(os.path.join(config.UNSW_DIR, f"UNSW-NB15_{i}.csv"),
                        header=None, names=names, low_memory=False,
                        encoding="latin1")
        p["file_part"] = i
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df["attack_cat"] = (df["attack_cat"].fillna("Normal").astype(str).str.strip()
                        .replace({"Backdoors": "Backdoor", "": "Normal"}))
    df["family"] = df["attack_cat"].replace({"Normal": "Benign"})
    df["label"] = df["label"].astype(np.int8)
    df["stime"] = pd.to_numeric(df["stime"], errors="coerce").astype(np.int64)
    # two capture sessions (22-Jan-2015 and 18-Feb-2015)
    day = pd.to_datetime(df["stime"], unit="s").dt.date.astype(str)
    df["period"] = np.where(day >= "2015-02-01", "S2", "S1")
    df["host"] = df["dstip"].astype(str)
    df["src_attacker"] = df["srcip"].astype(str).str.startswith("175.45.176.")
    df["order"] = np.lexsort((np.arange(len(df)), df["stime"].to_numpy()))
    # `order` must be a rank, not an argsort permutation
    rank = np.empty(len(df), dtype=np.int64)
    rank[df["order"].to_numpy()] = np.arange(len(df))
    df["order"] = rank
    for c in ["ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]:
        df[c] = pd.to_numeric(df[c].replace(" ", np.nan), errors="coerce").fillna(0)
    _label_encode_inplace(df, UNSW_CAT_COLS)
    feat = [c for c in names if c not in UNSW_ID_COLS + ["attack_cat", "label"]]
    out = df[feat + ["label", "family", "period", "host", "order", "stime",
                     "src_attacker"]].copy()
    out[feat] = out[feat].astype(np.float32)
    out["family"] = out["family"].astype("category")
    out["host"] = out["host"].astype("category")
    out["period"] = out["period"].astype("category")
    out.to_parquet(os.path.join(config.DATA_CACHE, "unsw_raw.parquet"),
                   compression="zstd")
    return out


UNSW_TTL_FEATURES = ["sttl", "dttl", "ct_state_ttl"]   # encode attacker subnet (testbed artefact)
UNSW_PROTECTED_HOSTS = [f"149.171.126.{i}" for i in range(20)]


def load_unsw_raw(clean=True, drop_ttl=True):
    """clean=True applies the two pre-declared integrity rules documented in
    Data_Integrity_Audit.md: (i) keep flows to the 20 protected servers
    149.171.126.0-19; (ii) drop benign-labelled flows whose source is in the
    attacker subnet 175.45.176.0/24 (40,413 flows; every attack flow originates
    there, so these labels are unreliable). drop_ttl=True removes the TTL-derived
    features that act as an attacker-subnet shortcut."""
    import pyarrow.parquet as pq
    path = os.path.join(config.DATA_CACHE, "unsw_raw.parquet")
    if not os.path.exists(path):
        build_unsw_raw_cache()
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    if "src_attacker" not in names:
        build_unsw_raw_cache(); pf = pq.ParquetFile(path); names = pf.schema_arrow.names
    meta_cols = ["label", "family", "period", "host", "order", "stime", "src_attacker"]
    meta = pd.read_parquet(path, columns=meta_cols)
    keep = np.ones(len(meta), dtype=bool)
    if clean:
        keep = meta["host"].astype(str).isin(UNSW_PROTECTED_HOSTS).to_numpy().copy()
        keep &= ~((meta["label"] == 0) & meta["src_attacker"]).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    feat = [c for c in names if c not in meta_cols and not c.startswith("__")]
    if drop_ttl:
        feat = [c for c in feat if c not in UNSW_TTL_FEATURES]
    # column-wise read keeps peak memory near the size of the final matrix
    X = np.empty((int(keep.sum()), len(feat)), dtype=np.float32)
    for j, c in enumerate(feat):
        X[:, j] = pf.read(columns=[c]).column(0).to_numpy()[keep]
    d = FlowData(name="UNSW-NB15", X=X, y=meta["label"].to_numpy(dtype=np.int8),
                 family=meta["family"].astype(str).to_numpy(),
                 order=meta["order"].to_numpy(dtype=np.int64),
                 period=meta["period"].astype(str).to_numpy(),
                 host=meta["host"].astype(str).to_numpy(), features=feat,
                 src_flag=meta["src_attacker"].to_numpy())
    return d


# ----------------------------------------------------------------- CICIDS2017
CICIDS_FILES = [  # chronological order of the capture week
    ("Mon", "Monday-WorkingHours.pcap_ISCX.csv"),
    ("Tue", "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("Wed", "Wednesday-workingHours.pcap_ISCX.csv"),
    ("Thu-AM", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("Thu-PM", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("Fri-AM", "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("Fri-PM1", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("Fri-PM2", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
]
CICIDS_FAMILY = {
    "BENIGN": "Benign",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS", "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS", "Heartbleed": "Heartbleed",
    "Web Attack Brute Force": "WebAttack", "Web Attack XSS": "WebAttack",
    "Web Attack Sql Injection": "WebAttack",
    "Infiltration": "Infiltration", "Bot": "Botnet", "PortScan": "PortScan",
    "DDoS": "DDoS",
}
CICIDS_KEEP_FRAC = 0.20       # uniform order-preserving thinning of large classes
CICIDS_SMALL_CLASS = 10_000   # classes below this size are kept completely
SUBSAMPLE_SEED = 42


def build_cicids_cache():
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    parts = []
    offset = 0
    for day, fname in CICIDS_FILES:
        p = pd.read_csv(os.path.join(config.CICIDS_DIR, fname), encoding="latin1",
                        low_memory=False)
        p.columns = [c.strip() for c in p.columns]
        lab = (p["Label"].astype(str).str.strip()
               .str.replace("�", "-", regex=False)
               .str.replace("Web Attack - ", "Web Attack ", regex=False)
               .str.replace("Web Attack \x96 ", "Web Attack ", regex=False))
        lab = lab.str.replace(r"Web Attack\W+", "Web Attack ", regex=True)
        p = p.drop(columns=["Label"])
        p = p.apply(pd.to_numeric, errors="coerce").astype(np.float64)
        p["raw_label"] = lab.values
        p["file_row"] = np.arange(len(p)) + offset
        offset += len(p)
        p = p.replace([np.inf, -np.inf], np.nan).dropna()
        sizes = p["raw_label"].value_counts()
        keep = np.zeros(len(p), dtype=bool)
        for c, n in sizes.items():
            m = (p["raw_label"] == c).to_numpy()
            if n < CICIDS_SMALL_CLASS:
                keep |= m
            else:
                keep |= m & (rng.random(len(p)) < CICIDS_KEEP_FRAC)
        p = p.loc[keep]
        p["period"] = day
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    # duplicated column "Fwd Header Length.1" exists in the MachineLearningCVE export
    df = df.loc[:, ~df.columns.duplicated()]
    df["family"] = df["raw_label"].map(CICIDS_FAMILY).fillna("Other")
    df["label"] = (df["family"] != "Benign").astype(np.int8)
    df["host"] = "NA"
    df["order"] = df["file_row"].astype(np.int64)
    feat = [c for c in df.columns if c not in
            ("raw_label", "file_row", "period", "family", "label", "host", "order")]
    # constant columns carry no information; drop them deterministically
    const = [c for c in feat if df[c].nunique() <= 1]
    feat = [c for c in feat if c not in const]
    out = df[feat + ["label", "family", "raw_label", "period", "host", "order"]].copy()
    out[feat] = out[feat].astype(np.float32)
    for c in ("family", "raw_label", "period", "host"):
        out[c] = out[c].astype("category")
    out.to_parquet(os.path.join(config.DATA_CACHE, "cicids2017_days.parquet"),
                   compression="zstd")
    return out


def load_cicids():
    path = os.path.join(config.DATA_CACHE, "cicids2017_days.parquet")
    df = pd.read_parquet(path) if os.path.exists(path) else build_cicids_cache()
    feat = [c for c in df.columns if c not in
            ("label", "family", "raw_label", "period", "host", "order")]
    return _from_frame("CICIDS2017", df, feat)


# ----------------------------------------------------------------- CIC-IoT2023
def ciciot_family(lab: str) -> str:
    if lab == "BenignTraffic":
        return "Benign"
    for pre, fam in (("DDoS", "DDoS"), ("DoS", "DoS"), ("Mirai", "Mirai"),
                     ("Recon", "Recon"), ("VulnerabilityScan", "Recon"),
                     ("DNS_Spoofing", "Spoofing"), ("MITM", "Spoofing"),
                     ("DictionaryBruteForce", "BruteForce")):
        if lab.startswith(pre):
            return fam
    return "Web"  # SqlInjection, CommandInjection, XSS, Uploading, Backdoor, BrowserHijacking


CICIOT_BENIGN_N = 100_000
CICIOT_FAMILY_CAP = 25_000


def build_ciciot_cache():
    """Benign flows are rare (about 2.4 %) in CIC-IoT2023, so benign records are
    over-represented relative to the native ratio to allow benign calibration;
    each attack family is capped at CICIOT_FAMILY_CAP records. Rule fixed in
    advance and documented in the data audit."""
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    path = os.path.join(config.CICIOT_DIR, "train.csv")
    pool = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunk["family"] = chunk["label"].astype(str).map(ciciot_family)
        # reservoir-style thinning per family keeps memory bounded
        keep = np.zeros(len(chunk), dtype=bool)
        for fam, g in chunk.groupby("family").groups.items():
            g = np.asarray(list(g)) - chunk.index[0]
            frac = {"Benign": 1.0, "DDoS": 0.02, "DoS": 0.05, "Mirai": 0.10}.get(fam, 1.0)
            keep[g[rng.random(len(g)) < frac]] = True
        pool.append(chunk.loc[keep])
    df = pd.concat(pool, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    parts = []
    for fam, g in df.groupby("family"):
        cap = CICIOT_BENIGN_N if fam == "Benign" else CICIOT_FAMILY_CAP
        parts.append(g.sample(min(cap, len(g)), random_state=SUBSAMPLE_SEED))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=SUBSAMPLE_SEED).reset_index(drop=True)
    df["label"] = (df["family"] != "Benign").astype(np.int8)
    df["raw_label"] = df["label"].astype(str)
    df["period"] = "NA"
    df["host"] = "NA"
    df["order"] = np.arange(len(df))
    feat = [c for c in df.columns if c not in
            ("label", "family", "raw_label", "period", "host", "order")]
    out = df[feat + ["label", "family", "period", "host", "order"]].copy()
    out[feat] = out[feat].astype(np.float32)
    for c in ("family", "period", "host"):
        out[c] = out[c].astype("category")
    out.to_parquet(os.path.join(config.DATA_CACHE, "ciciot2023_families.parquet"),
                   compression="zstd")
    return out


def load_ciciot():
    path = os.path.join(config.DATA_CACHE, "ciciot2023_families.parquet")
    df = pd.read_parquet(path) if os.path.exists(path) else build_ciciot_cache()
    feat = [c for c in df.columns if c not in
            ("label", "family", "period", "host", "order")]
    return _from_frame("CIC-IoT2023", df, feat)


# ----------------------------------------------------------------- NSL-KDD
NSL_FAMILY = {
    "normal": "Benign",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
}


def load_nslkdd():
    df = pd.read_parquet(config.NSLKDD_PARQUET)
    df["family"] = df["class"].astype(str).map(NSL_FAMILY).fillna("Other")
    df["label"] = (df["family"] != "Benign").astype(np.int8)
    df = df.drop(columns=["class", "classnum"])  # classnum = difficulty metadata (leaks)
    _label_encode_inplace(df, ["protocol_type", "service", "flag"])
    df["period"] = "NA"
    df["host"] = "NA"
    df["order"] = np.arange(len(df))
    feat = [c for c in df.columns if c not in
            ("label", "family", "period", "host", "order")]
    df[feat] = df[feat].astype(np.float32)
    return _from_frame("NSL-KDD", df, feat)


LOADERS = {"unsw": load_unsw_raw, "cicids": load_cicids,
           "ciciot": load_ciciot, "nslkdd": load_nslkdd}
