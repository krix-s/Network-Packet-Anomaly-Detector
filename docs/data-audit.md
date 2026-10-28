# Initial data audit

Scanned the local Parquet file in Arrow batches of at most 200,000 rows on 2026-10-28. The scan read all rows but did not construct a full pandas DataFrame.

## Verified facts

- 17,129,715 rows, 43 columns, one row group; file size 641,603,980 bytes.
- No null values in any column and no non-finite values in numeric features.
- `FTP_COMMAND_RET_CODE` is the only constant numeric feature, with value zero.
- `Label` agrees with `Attack` for every row: `Benign` maps to 0 and every other category maps to 1.
- Class frequencies:

| Attack | Rows |
|---|---:|
| Benign | 15,101,685 |
| DDOS attack-HOIC | 1,066,881 |
| DoS attacks-Hulk | 432,648 |
| DDoS attacks-LOIC-HTTP | 207,078 |
| Infilteration | 115,513 |
| SSH-Bruteforce | 94,979 |
| FTP-BruteForce | 25,933 |
| Bot | 28,033 |
| DoS attacks-GoldenEye | 27,723 |
| DoS attacks-SlowHTTPTest | 14,116 |
| DoS attacks-Slowloris | 9,512 |
| Brute Force -Web | 2,143 |
| DDOS attack-LOIC-UDP | 2,112 |
| Brute Force -XSS | 927 |
| SQL Injection | 432 |

- `SRC_TO_DST_SECOND_BYTES` ranges from 0 to about `6.588e304`; `DST_TO_SRC_SECOND_BYTES` ranges from 0 to about `3.122e73`. These are finite in the source data but may overflow or dominate scale-sensitive models. Their formula/meaning is not established by this scan. Preserve them in the source; compare a documented exclusion/robust transformation when modeling.
- `FLOW_DURATION_MILLISECONDS` ranges from 0 to 4,294,967. This range alone does not explain the distribution or justify a transformation.
- Duration zero counts are 14,448,674 of 15,101,685 benign rows (95.68%) and 258,534 of 2,028,030 attack rows (12.75%). Exactly 210 benign rows and no attack rows equal the maximum value 4,294,967. This confirms a strong label association in this file; it does not establish why the zeros or upper value occur.
- 19,234 rows exceed `1e9` in at least one of the two directional bytes-per-second columns. Of these, 1,703 are benign and 17,531 are attacks. Their source formula and validity remain unknown; no cause is inferred.

## Consequences for experiments

- Accuracy is misleading for both tasks; binary evaluation must emphasize recall, precision, F1, PR-AUC, and false-positive rate.
- Multiclass evaluation needs per-class support and macro metrics. With only 432 SQL Injection and 927 Brute Force -XSS examples, any held-out per-class estimate will have substantial uncertainty; do not merge or silently omit those classes.
- Fit all learned preprocessing on the training partition. Remove the all-zero feature as non-informative; the suspicious rate columns require an explicit model comparison rather than an assumed semantic explanation.
- This scan verifies structural quality only. It does not rule out duplicate flows across partitions, temporal leakage, source-level grouping, or distribution shift; there are no obvious time/session/id columns in the 43-column schema.
- A prior notebook contains code for a stratified 70/15/15 split with seed 42, but no persisted index files were found. The new bounded-sample experiments use their own deterministic stratified splits. Exact duplicates and capture-level dependence have not been independently assessed on the complete table.
