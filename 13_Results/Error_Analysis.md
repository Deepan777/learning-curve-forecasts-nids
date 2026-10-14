# Error analysis — where and why learning fails

All values are from `13_Results/Master_Results.csv` and from `09_Statistical_Analysis/H2–H5` CSVs.

## 1. Class overlap (irreducible in this feature space)
**Affected:** the CIC-IoT2023 reconnaissance, spoofing, web and brute-force families.

- **Floors.** The best learner's fitted floors are 5.0%, 5.6%, 1.9% and 2.7%. Balanced error at the largest available size is 5.8%, 6.2%, 4.6% and 5.5%.
- **Cross-class neighbours.** 18–33% of attack flows have a benign nearest neighbour, against a median of 0.13% overall.
- **Intrinsic dimension.** The attack-class dimension (4.9–5.5) is close to that of benign IoT traffic (5.6–5.8). The attack traffic fills the same high-dimensional region as benign traffic.
- **Not label conflicts.** The exact-conflict floor is 0.
- **Not lack of data.** Recon and spoofing curves are flat from about 4,096 flows per class.
- **Implication.** Richer features or context (sequences, hosts) are needed. More labels or other learners will not help.

## 2. Time / variant shift (reducible only by covering the variants)
**Affected:** CICIDS2017 families whose day contains several attack tools in sequence (the within-day split is first 60% for training, last 40% for testing). Random-forest balanced error, random split vs past→future:

| Family | Random split | Past → future | What changes between the two pools |
|---|---|---|---|
| Brute force | 0.05% | 43.8% | Training is FTP-Patator (+ a few SSH-Patator); testing is SSH-Patator only |
| Web attack | 0.43% | 1.8% | Training is web brute force only; testing adds XSS and SQL injection |
| DoS | 0.20% | 3.7% | Testing includes GoldenEye, which is absent from training |
| Botnet | 0.81% | 32.3% | Single label; behaviour drifts within the morning |
| DDoS, PortScan | 0.08–0.10% | 0.10–0.20% | Homogeneous attacks; little change |

**UNSW-NB15 sessions 1 → 2** (same traffic generator, four weeks apart) show no consistent effect.

## 3. Too little data
**NSL-KDD R2L** has only 995 unique attack flows. It stood at 1.2% at 512 flows per class and was still decreasing, so it is censored rather than plateaued.

## 4. Learner-specific limits
- **Logistic regression** shows floors in 88% of curves. Its linear boundary cannot fit the nonlinear separation; for example, NSL-KDD Probe never reaches 1%.
- **5-nearest-neighbours** is the slowest learner (median n\* 394). Increasing k to 10 slows it further (+0.52 log₂ units). Larger neighbourhoods blur the narrow class regions at small n.
- **Tree ensembles** are the fastest learners, with medians of 79 (random forest) and 131 (gradient boosting) flows per class.

## 5. Artefacts that make learning look easier
- **UNSW-NB15 TTL features.** A depth-2 tree on the three TTL features alone has balanced error 0.0. Restoring them lowers n\* by 0.98 log₂ units (random forest) and 2.47 (gradient boosting).
- **Exact duplicates.** They are 75.6% of UNSW-NB15 attack rows (Generic 91.3%). Keeping them lowers the random forest's n\* by a median of 0.16 log₂ units (P = 0.036) and leaves gradient boosting unchanged. The share of tasks reaching 1% with ≤ 1,024 flows is unchanged (79%).

## 6. Where the pilot-based forecast fails
- The saturating-power-law forecast missed the factor-of-2 band for 13 of 144 curves (`09_Statistical_Analysis/H6_forecast.csv`).
- 12 of the 13 are overestimates: the curve steepened after the pilot range (n ≤ 128). For example, CICIDS2017 DoS with the random forest needed 660 flows, but the forecast was 2,018.
- Only one is an underestimate (CICIDS2017 botnet, 1-NN).
- Six of the 13 involve 5-nearest-neighbours, the learner whose curves bend most.
- The plateauing CIC-IoT2023 overlap families are forecast correctly as not reaching 1%. In practice, the pilot forecast is conservative: when it errs, it asks for too many labels rather than too few.
