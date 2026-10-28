# Research basis

## Source checked

The repository did not contain a paper or bibliography file. The associated NetFlow v2 paper was identified from the University of Queensland (UQ) dataset page and read from its public preprint:

Sarhan, Mohanad; Layeghy, Siamak; Portmann, Marius. “Towards a Standard Feature Set for Network Intrusion Detection System Datasets.” *Mobile Networks and Applications* 27 (2022), 357–370. [DOI](https://doi.org/10.1007/s11036-021-01843-0) · [arXiv preprint](https://arxiv.org/abs/2101.11315).

UQ describes NetFlow v2 as 43 extended features and links the paper as the methodological reference. The local feature dictionary is `data/raw/NetFlow v2 Features.csv`.

## What the paper did

- It proposed and evaluated common 12-feature and extended 43-feature NetFlow sets across multiple NIDS datasets. The stated goal was comparability across datasets and network environments.
- It generated the NetFlow records from public PCAPs using nProbe and matched flow identifiers against published ground-truth events for labeling. The paper describes the features as flow/header-derived and not payload-dependent.
- Its model comparison used Extra Trees. It reports binary and multiclass results, including accuracy, AUC, F1, detection rate, false-alarm rate, and prediction time.
- The methods section reports dropping flow identifiers (IDs, endpoint IPs/ports, timestamps, start/end times), applying min-max normalization, and a 70/30 train/test setup with five cross-validation splits. TTL features were additionally removed for UNSW-NB15 variants due to extreme correlation with labels.
- For the paper's NF-CSE-CIC-IDS2018-v2 dataset, it reports 18,893,708 flows and seven categories (Benign plus six aggregated attack categories). Its Extra Trees results are 99.35% binary accuracy, 0.9829 AUC, 0.97 F1, 96.89% detection rate, and 0.31% false-alarm rate; multiclass accuracy is 96.90% with 0.98 weighted F1. The paper also identifies weak classes, including web brute-force categories, infiltration, and SQL Injection.

These are published-paper results, not results from this implementation.

## What this project reproduces

- The same flow-level problem framing and 43-field NetFlow v2 family, with separate binary `Label` and multiclass `Attack` tasks.
- The paper's use of a tree ensemble as a relevant comparison point and reporting of intrusion-sensitive metrics.
- A train/validation/test workflow that keeps the test partition out of model selection.

## What this project changes or extends

- The current local Parquet file has 17,129,715 rows and 15 attack categories. This differs from the UQ/paper reference distribution (18,893,708 rows, seven categories), so exact reproduction of the paper's reported experiment is not possible with the checked-in local data variant.
- This dataset's 43 columns comprise 41 numeric flow features plus `Label` and `Attack`; it does not include the endpoint IP columns discussed in the paper. The supplied CSV dictionary describes the broader feature set, so its row names are reconciled to the actual Parquet schema before use.
- The first experiment uses bounded, reproducible per-class samples to fit on a laptop, while streaming the source Parquet file. Results must be labeled sample-based, not full-data fitting.
- It follows the paper's removal of endpoint ports as identifiers to reduce endpoint-specific bias; the local table lacks endpoint IPs and timestamps.
- Logistic regression provides a simple learned baseline, and histogram gradient boosting is the stronger tabular candidate. Preprocessing is fitted through the training pipeline. Class-frequency weights are used for evaluation to correct the deliberate capped sampling; training uses balanced class weights.
- The local audit found finite but extreme rate features and a highly suspicious duration distribution. Such fields require explicit ablation; they are not explained by the paper's methodology.

## What cannot be verified from the source

- The paper does not give a complete estimator parameter configuration or random seed in the accessible methods text.
- The exact split indices, capture grouping, deduplication policy, and whether the five cross-validation folds were group/time-aware are not specified.
- It does not describe a class-resampling or class-weighting method in the cited evaluation section.
- The current local Parquet file's provenance relative to the UQ download is not represented by a checksum or version manifest. Its materially different counts and labels are verified facts; the cause of the difference is unknown.
- Feature formulas beyond the supplied descriptions (especially unusual rate fields and duration edge cases) are not established by the local dictionary. Treat formulas and interpretations as unverified unless independently documented by the exporter.

## Dataset and feature references

- [UQ NetFlow dataset page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) describes the v2 feature count, release counts, and category definitions.
- [UQ Cyber Research Centre overview](https://www.cyber.uq.edu.au/node/824) identifies the NetFlow v2 methodology paper and feature dictionary.
