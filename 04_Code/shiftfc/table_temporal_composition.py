"""Supplementary Table S15 (post hoc, added 2026-10-13): attack sub-type composition of the source and target periods
of the CICIDS2017 temporal scenarios, with the unsafe share of their verifiable forecasts.
Input: 06_Experiments/audit/temporal_composition.csv (audit_temporal_composition.py).
Output: 12_Tables/supp_temporal_composition.tex and a copy in 15_Supplementary/."""
import os
import shutil

import pandas as pd

from lcurve import config

SRC = os.path.join(config.PROJECT_ROOT, "06_Experiments", "audit", "temporal_composition.csv")
OUT = os.path.join(config.PROJECT_ROOT, "12_Tables", "supp_temporal_composition.tex")
FAM = {"Botnet": "Botnet", "BruteForce": "Brute force", "DDoS": "DDoS", "DoS": "DoS", "PortScan": "Port scan",
       "WebAttack": "Web attack"}


def esc(s):
    return str(s).replace("%", "\\%").replace("_", "\\_").replace("Web Attack ", "").replace("Sql", "SQL")


def main():
    d = pd.read_csv(SRC)
    c = d[d.dataset == "cicids"].sort_values("family")
    rows = []
    for r in c.itertuples():
        new = esc(r.subtypes_only_in_target) if isinstance(r.subtypes_only_in_target, str) and r.subtypes_only_in_target else "none"
        rows.append(f"{FAM[r.family]} & {esc(r.subtypes_src)} & {esc(r.subtypes_tgt)} & {new} & "
                    f"{r.tv_distance:.2f} & {int(r.unsafe)}/{int(r.verifiable)} \\\\")
    u = d[d.dataset == "unsw"]
    rows.append("\\midrule")
    rows.append(f"UNSW-NB15 (8 families) & \\multicolumn{{4}}{{l}}{{no sub-type labels}} & "
                f"{int(u.unsafe.sum())}/{int(u.verifiable.sum())} \\\\")
    L = ["\\begin{tabular}{l>{\\raggedright}p{3.2cm}>{\\raggedright}p{4.0cm}>{\\raggedright}p{2.6cm}rr}", "\\toprule",
         "Family & Source period (first 60\\%) & Target period (last 40\\%) & Sub-types only in target & TV & Unsafe \\\\",
         "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    shutil.copy(OUT, os.path.join(config.PROJECT_ROOT, "15_Supplementary", os.path.basename(OUT)))
    print(open(OUT, encoding="utf-8").read())


if __name__ == "__main__":
    main()
