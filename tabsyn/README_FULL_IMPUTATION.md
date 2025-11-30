# TabSyn Full Variable Imputation Pipeline

## Overview

This is an improved implementation of TabSyn designed specifically for **full variable imputation** - imputing an entire variable (net worth) from a donor dataset (SCF) to a receiver dataset (CPS) that completely lacks that variable.

### Key Improvements

1. **Cross-dataset normalization**: Uses SCF statistics to normalize CPS data, ensuring consistent scaling
2. **Full dataset imputation**: Processes entire CPS dataset, not just a test split
3. **Notebook integration**: Easy-to-use wrapper functions callable from Jupyter notebooks
4. **Batch processing**: Handles large datasets efficiently
5. **Proper inverse transformation**: Correctly reverses normalization and transformations

## Architecture

```
SCF Data (Donor)          CPS Data (Receiver)
     |                           |
     v                           v
  Fit Preprocessor -----> Apply SCF Statistics
     |                           |
     v                           v
  Train VAE                 Use as "Test Set"
     |                           |
     v                           v
  Train Diffusion               |
     |                           |
     v                           v
  Save Models ---------> Apply Models
                                |
                                v
                          Imputed Values
                                |
                                v
                        Inverse Transform
                                |
                                v
                         Final Net Worth
```

## Files Description

### Core Components

- **`tabsyn_custom_preprocess.py`**: Custom preprocessing that fits on donor and transforms receiver
- **`impute_full.py`**: Modified imputation script for full dataset imputation
- **`tabsyn_wrapper.py`**: High-level wrapper for notebook integration
- **`configs/scf_cps_config.json`**: Configuration file with all parameters

### Example Usage

- **`tabsyn_example_usage.py`**: Complete examples for using the pipeline

## Installation

```bash
# Ensure you have the required dependencies
pip install torch numpy pandas scikit-learn tqdm

# Clone TabSyn if not already done
git clone https://github.com/amazon-science/tabsyn.git
```

## Quick Start

### Method 1: Simple One-Function Approach

```python
from tabsyn.tabsyn_wrapper import run_tabsyn_imputation

# Run imputation
imputed_values = run_tabsyn_imputation(
    scf_data=scf_data,           # Donor DataFrame
    cps_data=cps_data,           # Receiver DataFrame
    predictors=predictors,        # List of predictor columns
    target_variable='networth',   # Variable to impute
    weights='wgt',                # Weight column
    num_trials=50,                # Number of trials
    python_path='python3',        # Python executable
    verbose=True
)

# If using transformed target (e.g., arcsinh)
final_values = np.sinh(imputed_values)
```

### Method 2: Pipeline Class (More Control)

```python
from tabsyn.tabsyn_wrapper import TabSynPipeline

# Initialize pipeline
pipeline = TabSynPipeline(
    working_dir='tabsyn',
    python_path='python3',
    use_gpu=True,
    verbose=True
)

# Run full pipeline
results = pipeline.full_pipeline(
    scf_data=scf_data,
    cps_data=cps_data,
    predictors=predictors,
    target_variable='transformed_networth',
    weights='wgt',
    vae_epochs=100,
    diffusion_epochs=100,
    num_trials=50
)

# Get imputed values
imputed_networth = results['imputed_values']
```

### Method 3: Step-by-Step Control

```python
# 1. Preprocess data
preprocess_results = pipeline.preprocess_data(
    scf_data=scf_data,
    cps_data=cps_data,
    predictors=predictors,
    target_variable='transformed_networth'
)

# 2. Train VAE
pipeline.train_vae(epochs=100)

# 3. Train Diffusion
pipeline.train_diffusion(epochs=100)

# 4. Impute
imputed_df = pipeline.impute(num_trials=50)

# 5. Get final values
final_values = pipeline.get_imputed_values(
    target_column='transformed_networth',
    inverse_transform=True
)
```

## Key Features Explained

### 1. Cross-Dataset Normalization

The preprocessor fits normalization parameters (e.g., quantiles) on SCF data and applies them to CPS:

```python
preprocessor = CrossDatasetPreprocessor(
    num_col_idx=[...],
    cat_col_idx=[...],
    target_col_idx=[...],
    normalization='quantile'
)

# Fit on donor
preprocessor.fit_donor(scf_data)

# Transform receiver using donor statistics
X_num_cps, X_cat_cps, y_cps = preprocessor.transform_receiver(cps_data)
```

### 2. Full Dataset Processing

Unlike the original impute.py which only processes test splits, impute_full.py processes the entire receiver dataset:

```python
# Process all CPS observations
for batch_start in range(0, n_samples, batch_size):
    batch = get_batch(batch_start, batch_end)
    imputed_batch = process_batch(batch)
    results.append(imputed_batch)
```

### 3. Batch Processing for Memory Efficiency

Large datasets are processed in batches to avoid memory issues:

```python
imputed_df = pipeline.impute(
    num_trials=50,
    batch_size=2048  # Process 2048 samples at a time
)
```

### 4. Multiple Trial Averaging

The pipeline runs multiple imputation trials and averages results:

```python
# Runs 50 independent imputations
# Automatically averages results
# Saves individual trials for analysis
```

## Configuration

Edit `configs/scf_cps_config.json` to customize:

- Preprocessing parameters (normalization type, quantiles)
- Model architecture (layers, dimensions)
- Training parameters (epochs, batch size, learning rate)
- Imputation settings (trials, batch size)

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch_size in imputation
2. **Training failures**: Check data has no NaNs after preprocessing
3. **Import errors**: Ensure tabsyn directory is in Python path
4. **Model not found**: Run training steps before imputation

### Validation

Compare imputed distributions:

```python
from scipy.stats import wasserstein_distance

# Compare distributions
distance = wasserstein_distance(
    scf_data['networth'],
    imputed_values,
    scf_weights,
    cps_weights
)
```

## Statistical Considerations

### Normalization Approach

We use SCF statistics for normalizing CPS because:
1. The target variable only exists in SCF
2. Ensures consistent scale for imputation
3. Preserves SCF's distribution characteristics

### Alternative Approaches

If you prefer different normalization:

```python
# Option 1: Hybrid approach
# Use CPS stats for predictors, SCF for target

# Option 2: No normalization
# Set normalization=None in preprocessor

# Option 3: Domain adaptation
# Apply additional transformation to account for dataset shift
```

## Performance Tips

1. **Use GPU**: Set `use_gpu=True` for faster training
2. **Adjust batch size**: Larger batches = faster but more memory
3. **Reduce trials**: Start with 10 trials for testing
4. **Cache models**: Reuse trained models for multiple imputations

## Citations

If you use this implementation, please cite:

1. The original TabSyn paper: [https://arxiv.org/pdf/2310.09656](https://arxiv.org/pdf/2310.09656)
2. Your SCF-CPS imputation work

## Support

For issues or questions:
1. Check this documentation
2. Review example code in `tabsyn_example_usage.py`
3. Examine the source code comments
4. Test with small data subsets first

## License

This implementation extends TabSyn for full variable imputation. See original TabSyn repository for licensing terms.