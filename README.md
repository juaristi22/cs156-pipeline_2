# cs156-pipeline_2

Deep learning imputation benchmarking pipeline for CS156: Finding Patterns in Data with Machine Learning.

The main pipeline is in `code.ipynb`, which benchmarks RealNVP, MDN, and TabSyn against a Quantile Random Forest baseline for imputing net worth from the Survey of Consumer Finances onto the Current Population Survey.

The `tabsyn/` directory contains an adaptation of the TabSyn model (Zhang et al., 2024) for full variable imputation across datasets. The `data/` folder stores preprocessed SCF and CPS data along with cross-validation folds, while `impute/` contains the imputation outputs. The `report/` directory contains the LaTeX source for the accompanying paper explaining the methodology and results.
