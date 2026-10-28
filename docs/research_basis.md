# Research basis

## Reference source

The user supplied the IEEE conference paper *Optimization of Algorithm for Network Traffic Anomaly Detection Using Convolutional Neural Networks (CNN)* by Teng Li, presented at IACIS 2024 (DOI: [10.1109/IACIS61494.2024.10721912](https://doi.org/10.1109/IACIS61494.2024.10721912)). The local source PDF is the paper cited here. The project dataset is the local 43-column table described below; it is not one of the datasets used in the paper.

## What the paper actually did and reported

- **Objective:** design a CNN-based method for network traffic anomaly detection and discuss multi-class attack recognition, noise sensitivity, class imbalance, and cross-dataset generalization.
- **Datasets:** it names KDD Cup 99 and NSL-KDD. The reported cross-dataset table trains on KDD Cup 99 and tests on NSL-KDD. It does not use this project's local 43-column dataset.
- **Preprocessing:** it recommends standardizing features and discusses augmentation with random noise and time-series extension. It gives no exact scaler, augmentation algorithm, or fit protocol.
- **Model:** three convolution layers, 3×3 kernels, 2×2 max pooling, Adam, learning rate 0.001, batch size 64. It gives no layer widths, activations, padding, input tensor construction, regularization values, or code. Cross entropy is described for binary classification.
- **Comparators and environment:** CNN versus SVM and decision tree; Python 3.7 and TensorFlow 2.3 on a high-performance server.
- **Imbalance:** the paper describes an imbalance experiment and reports accuracy falling from 97.5% to 80.0%, but its listed class-count sequence and experimental protocol are ambiguous. It does not clearly specify class weights, resampling, or another reproducible imbalance treatment.
- **Metrics/results:** Table I reports CNN accuracy 98.5%, precision 97.8%, recall 99.0%, F1 98.4%, training time 3.5 hours, test time 0.002 seconds/sample, and memory 3000 MB. It reports 95.2% accuracy for SVM and 93.1% for decision tree. The abstract separately says 99.0% accuracy without noise, so the paper contains an unexplained discrepancy. Its KDD-to-NSL table reports accuracy 98.5% to 96.2%, precision 97.8% to 95.1%, recall 99.0% to 97.0%, F1 98.4% to 96.0%, false alarm rate 1.5% to 3.8%, and underreporting rate 1.0% to 3.0%.
- **Limitations:** the paper acknowledges lower performance on some attack types, sensitivity to increased noise, imbalance-related degradation, cross-dataset performance drops, and the need for real-world testing.

These are paper-reported claims, not results obtained by this project. The paper's missing architecture and preprocessing details prevent exact reproduction.

## What this project will reproduce or use from the reference

- Keep the goal as flow-level network anomaly detection, and report attack-sensitive metrics rather than accuracy alone.
- Treat class imbalance, rare-category performance, robustness, and cross-domain generalization as limitations that need explicit evaluation.
- Use the paper's CNN as research context, not as a claim that its specific model was reproduced.

## Project-specific reproduction and changes

- **Dataset:** use the local Parquet file `data/raw/NF-CSE-CIC-IDS2018-V2.parquet`. Its audited shape is 17,129,715 rows and 43 columns: 41 numeric predictors plus `Label` and `Attack`. It has 15 observed attack categories. These counts describe the local file only.
- **Tasks:** separate binary classification using `Label` and multiclass classification using `Attack`; neither target is an input feature.
- **Models:** the implemented baseline and selected tabular candidate are majority prediction, logistic regression, and histogram gradient boosting. No CNN is currently implemented: the paper does not specify how its 2D CNN consumes tabular columns, and imposing an arbitrary 2D arrangement would not be a faithful or defensible reproduction. The local runtime also has no TensorFlow or PyTorch installed. The tabular models provide a reproducible baseline for a later, explicitly justified neural-network comparison.
- **Scale and evaluation:** streaming per-class samples bound memory on the laptop; splits are stratified 70/15/15; model selection uses validation data; selected-model test metrics are reported with source-frequency weights. Results are sample-based and are not full-dataset training results.
- **Feature handling:** remove the audited constant feature; omit endpoint ports as a conservative leakage-control choice because label construction can be related to endpoint tuples, not because the supplied paper instructs it. The extreme rate fields exceed float32 range raw; a validation-only `log1p` variant included them but did not change binary F1. A separate validation-only ablation also found unchanged binary F1 after dropping `FLOW_DURATION_MILLISECONDS`.

## What cannot be verified

- The local dataset's exact provenance/version manifest and the reason its row/category counts differ from other releases carrying similar names.
- Whether the paper's CNN input was 2D tabular reshaping or another representation; the feature order and tensor construction are not specified.
- The paper's exact train/test partitions, random seed, CNN layer widths/activations, augmentation and standardization formulas, detailed class frequencies, and imbalance-handling procedure.
- Whether port exclusion improves transferability for this local dataset. It is a precautionary choice, not a verified finding.
- The calculation formulas and causes of the extreme directional rate values and duration values. The supplied local dictionary gives names/descriptions, but does not resolve those edge cases.

## Local data sources

- `data/raw/NF-CSE-CIC-IDS2018-V2.parquet`: local records used by the experiments.
- `data/raw/NetFlow v2 Features.csv`: local feature dictionary. It describes a wider set than the 41 predictors present in the Parquet schema; only matching columns are used.
