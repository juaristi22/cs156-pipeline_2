"""
Wrapper functions for TabSyn that can be easily called from Jupyter notebooks.
This provides a high-level interface for training and imputation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')
# Specifically ignore the pkg_resources deprecation warning from zero package
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

# Add TabSyn directory to path if not already there
tabsyn_dir = Path(__file__).parent
if str(tabsyn_dir) not in sys.path:
    sys.path.insert(0, str(tabsyn_dir))

from tabsyn_custom_preprocess import CrossDatasetPreprocessor, preprocess_for_full_imputation
from impute_full import impute_full_dataset


class TabSynPipeline:
    """
    High-level wrapper for TabSyn training and imputation pipeline.
    Designed for easy use from Jupyter notebooks.
    """

    def __init__(self,
                 working_dir: str = 'tabsyn',
                 python_path: str = 'python3',
                 use_gpu: bool = True,
                 verbose: bool = True):
        """
        Initialize the TabSyn pipeline.

        Args:
            working_dir: Base directory for TabSyn operations
            python_path: Path to Python executable (e.g., 'venv3.10/bin/python3')
            use_gpu: Whether to use GPU if available
            verbose: Whether to print detailed progress
        """
        self.working_dir = Path(working_dir)
        self.python_path = python_path
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.device = 'cuda:0' if use_gpu else 'cpu'

        # Ensure working directory exists
        self.working_dir.mkdir(exist_ok=True)

        # Store state
        self.donor_name = None
        self.receiver_name = None
        self.preprocessor = None
        self.trained_models = {}

    def preprocess_data(self,
                       scf_data: pd.DataFrame,
                       cps_data: pd.DataFrame,
                       predictors: List[str],
                       target_variable: str,
                       weights: Optional[str] = None,
                       numerical_cols: Optional[List[str]] = None,
                       categorical_cols: Optional[List[str]] = None,
                       donor_name: str = 'scf_2022',
                       receiver_name: str = 'cps_2023') -> Dict:
        """
        Preprocess SCF and CPS data using SCF statistics for normalization.

        Args:
            scf_data: DataFrame with SCF data (donor)
            cps_data: DataFrame with CPS data (receiver)
            predictors: List of predictor column names
            target_variable: Name of target variable to impute
            weights: Name of weight column (optional)
            numerical_cols: List of numerical column names (if None, auto-detect)
            categorical_cols: List of categorical column names (if None, auto-detect)
            donor_name: Name for donor dataset
            receiver_name: Name for receiver dataset

        Returns:
            Dictionary with preprocessing results
        """
        if self.verbose:
            print("Preprocessing data...")

        self.donor_name = donor_name
        self.receiver_name = receiver_name

        # Prepare column indices
        all_cols = predictors + [target_variable]
        if weights:
            all_cols.append(weights)

        # Reorder dataframes
        scf_ordered = scf_data[all_cols].copy()
        cps_ordered = cps_data[predictors].copy()

        # Add placeholder target to CPS if not present
        if target_variable not in cps_ordered.columns:
            cps_ordered[target_variable] = np.nan
        if weights and weights not in cps_ordered.columns:
            cps_ordered[weights] = 1.0

        # Ensure column order matches
        cps_ordered = cps_ordered[all_cols]

        # Determine column types
        num_col_idx = []
        cat_col_idx = []
        target_col_idx = []

        # Use explicit specifications if provided
        if numerical_cols is not None and categorical_cols is not None:
            for i, col in enumerate(all_cols):
                if col == target_variable:
                    target_col_idx.append(i)
                elif col == weights:
                    continue  # Skip weight column
                elif col in numerical_cols:
                    num_col_idx.append(i)
                elif col in categorical_cols:
                    cat_col_idx.append(i)
        else:
            # Auto-detect if not explicitly specified
            for i, col in enumerate(all_cols):
                if col == target_variable:
                    target_col_idx.append(i)
                elif col == weights:
                    continue  # Skip weight column
                elif scf_ordered[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Check if it's actually categorical
                    n_unique = scf_ordered[col].nunique()
                    if n_unique <= 10 and col != target_variable:
                        cat_col_idx.append(i)
                    else:
                        num_col_idx.append(i)
                else:
                    cat_col_idx.append(i)

        # Determine task type
        if scf_ordered[target_variable].dtype in ['float64', 'float32']:
            task_type = 'regression'
        else:
            task_type = 'bin_class'

        # Create info dictionary
        info = {
            'name': donor_name,
            'task_type': task_type,
            'num_col_idx': num_col_idx,
            'cat_col_idx': cat_col_idx,
            'target_col_idx': target_col_idx,
            'column_names': all_cols,
            'data_path': f'{self.working_dir}/data/{donor_name}.csv',
            'file_type': 'csv',
            'header': 'infer'
        }

        # Save raw data
        data_dir = self.working_dir / 'data'
        data_dir.mkdir(exist_ok=True)

        scf_ordered.to_csv(data_dir / f'{donor_name}.csv', index=False)
        cps_ordered.to_csv(data_dir / f'{receiver_name}.csv', index=False)

        if self.verbose:
            print(f"SCF shape: {scf_ordered.shape}")
            print(f"CPS shape: {cps_ordered.shape}")
            print(f"Predictors: {predictors}")
            print(f"Target: {target_variable}")
            print(f"Task type: {task_type}")
            print(f"Numerical columns: {[all_cols[i] for i in num_col_idx]}")
            print(f"Categorical columns: {[all_cols[i] for i in cat_col_idx]}")

        # Run custom preprocessing
        preprocess_results = preprocess_for_full_imputation(
            donor_dataname=donor_name,
            receiver_dataname=receiver_name,
            donor_df=scf_ordered,
            receiver_df=cps_ordered,
            info=info,
            save_dir='data'
        )

        # Store preprocessor
        self.preprocessor = CrossDatasetPreprocessor.load(
            preprocess_results['preprocessor_path']
        )

        if self.verbose:
            print(f"Saved to: data/{donor_name} and data/{receiver_name}")

        return preprocess_results

    def train_vae(self, epochs: int = 100, force_retrain: bool = False) -> bool:
        """
        Train the VAE model on donor data.

        Args:
            epochs: Number of training epochs
            force_retrain: If True, retrain even if model exists

        Returns:
            True if training successful
        """
        # Check if VAE model already exists
        vae_model_path = self.working_dir / 'tabsyn' / 'vae' / 'ckpt' / self.donor_name / 'model.pt'
        if vae_model_path.exists() and not force_retrain:
            if self.verbose:
                print(f"VAE model already exists at {vae_model_path}, skipping training.")
                print("Set force_retrain=True to retrain the model.")
            return True

        if self.verbose:
            print("Training VAE...")

        cmd = [
            self.python_path,
            f'{self.working_dir}/main.py',
            '--dataname', self.donor_name,
            '--method', 'vae',
            '--mode', 'train',
            '--epochs', str(epochs)
        ]

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        # Set environment to suppress warnings in subprocess
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::UserWarning'

        result = subprocess.run(
            cmd,
            capture_output=not self.verbose,
            text=True,
            env=env
        )

        if result.returncode != 0:
            print(f"VAE training failed with return code {result.returncode}")
            if not self.verbose:
                print(f"Error: {result.stderr}")
            return False

        self.trained_models['vae'] = True

        if self.verbose:
            print("VAE training complete")

        return True

    def train_diffusion(self, epochs: int = 100, force_retrain: bool = False) -> bool:
        """
        Train the diffusion model on donor data.

        Args:
            epochs: Number of training epochs
            force_retrain: If True, retrain even if model exists

        Returns:
            True if training successful
        """
        # Check if diffusion model already exists
        diffusion_model_path = self.working_dir / 'tabsyn' / 'ckpt' / self.donor_name / 'model.pt'
        if diffusion_model_path.exists() and not force_retrain:
            if self.verbose:
                print(f"Diffusion model already exists at {diffusion_model_path}, skipping training.")
                print("Set force_retrain=True to retrain the model.")
            return True

        if self.verbose:
            print("Training diffusion model...")

        cmd = [
            self.python_path,
            f'{self.working_dir}/main.py',
            '--dataname', self.donor_name,
            '--method', 'tabsyn',
            '--mode', 'train',
            '--total_epochs_both', str(epochs)
        ]

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        # Set environment to suppress warnings in subprocess
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::UserWarning'

        result = subprocess.run(
            cmd,
            capture_output=not self.verbose,
            text=True,
            env=env
        )

        if result.returncode != 0:
            print(f"Diffusion training failed with return code {result.returncode}")
            if not self.verbose:
                print(f"Error: {result.stderr}")
            return False

        self.trained_models['diffusion'] = True

        if self.verbose:
            print("Diffusion model training complete")

        return True

    def setup_model_links(self) -> None:
        """
        Set up symbolic links so receiver data can use donor-trained models.
        """
        if self.verbose:
            print("Setting up model links...")

        # VAE model links
        vae_source = Path(f'{self.working_dir}/tabsyn/vae/ckpt/{self.donor_name}')
        vae_target = Path(f'{self.working_dir}/tabsyn/vae/ckpt/{self.receiver_name}')

        if vae_target.exists() or vae_target.is_symlink():
            vae_target.unlink()

        # Ensure parent directory exists
        vae_target.parent.mkdir(parents=True, exist_ok=True)
        vae_target.symlink_to(vae_source.absolute())

        # Diffusion model links
        diff_source = Path(f'{self.working_dir}/tabsyn/ckpt/{self.donor_name}')
        diff_target = Path(f'{self.working_dir}/tabsyn/ckpt/{self.receiver_name}')

        if diff_target.exists() or diff_target.is_symlink():
            diff_target.unlink()

        # Ensure parent directory exists
        diff_target.parent.mkdir(parents=True, exist_ok=True)
        diff_target.symlink_to(diff_source.absolute())

        if self.verbose:
            print(f"Created model links: {self.receiver_name} -> {self.donor_name}")

    def impute(self,
               num_trials: int = 50,
               batch_size: int = 2048,
               use_custom_script: bool = True) -> pd.DataFrame:
        """
        Perform imputation from donor to receiver.

        Args:
            num_trials: Number of imputation trials
            batch_size: Batch size for processing
            use_custom_script: Whether to use custom full imputation script

        Returns:
            DataFrame with imputed values
        """
        if self.verbose:
            print("Performing imputation...")
            print(f"Trials: {num_trials}")
            print(f"Batch size: {batch_size}")

        # Setup model links
        self.setup_model_links()

        if use_custom_script:
            # Use custom full imputation script
            import torch
            device = 'cuda:0' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'

            results = impute_full_dataset(
                donor_name=self.donor_name,
                receiver_name=self.receiver_name,
                model_name=self.donor_name,
                device=device,
                num_trials=num_trials,
                batch_size=batch_size,
                use_random_init=True,  # Use weighted random init
                save_individual_trials=True
            )

            return results['averaged_result']

        else:
            # Use original impute.py (for backward compatibility)
            cmd = [
                self.python_path,
                f'{self.working_dir}/impute.py',
                '--dataname', self.receiver_name,
                '--gpu', '0' if self.use_gpu else '-1'
            ]

            if self.verbose:
                print(f"Running: {' '.join(cmd)}")

            # Set environment to suppress warnings in subprocess
            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore::UserWarning'

            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                env=env
            )

            if result.returncode != 0:
                print(f"Imputation failed with return code {result.returncode}")
                if not self.verbose:
                    print(f"Error: {result.stderr}")
                return None

            # Load and average results
            impute_dir = Path(f'impute/{self.receiver_name}')
            all_trials = []

            for trial in range(num_trials):
                trial_path = impute_dir / f'{trial}.csv'
                if trial_path.exists():
                    df = pd.read_csv(trial_path)
                    all_trials.append(df)

            if all_trials:
                averaged = pd.concat(all_trials).groupby(level=0).mean()
                return averaged
            else:
                print("No imputation results found!")
                return None

    def get_imputed_values(self,
                          target_column: str = 'networth',
                          inverse_transform: bool = True) -> np.ndarray:
        """
        Get the imputed target values.

        Args:
            target_column: Name of target column
            inverse_transform: Whether to apply inverse transformation

        Returns:
            Array of imputed values
        """
        # Load averaged imputation results
        impute_path = Path(f'impute/{self.receiver_name}/averaged_imputation.csv')

        if not impute_path.exists():
            print(f"No averaged imputation found at {impute_path}")
            return None

        df = pd.read_csv(impute_path)

        # The columns in the imputed file are named by their indices (0, 1, 2, ..., n)
        # We need to find which index corresponds to our target variable
        # Load info.json to get the target column index
        info_path = Path(f'data/{self.donor_name}/info.json')
        if not info_path.exists():
            print(f"Info file not found at {info_path}")
            # Try to use column name directly
            if target_column in df.columns:
                values = df[target_column].values
            else:
                print(f"Target column '{target_column}' not found in results")
                print(f"Available columns: {list(df.columns)}")
                return None
        else:
            with open(info_path) as f:
                info = json.load(f)

            # Get the target column index from info
            target_col_indices = info.get('target_col_idx', [])
            if not target_col_indices:
                print(f"No target column indices found in info.json")
                return None

            # Use the first target column index (typically there's only one for regression)
            target_col_idx = str(target_col_indices[0])

            # Check if column exists (as integer index)
            if target_col_idx not in df.columns:
                # Try the target column name directly as fallback
                if target_column not in df.columns:
                    print(f"Neither target column index {target_col_idx} nor name '{target_column}' found in results")
                    print(f"Available columns: {list(df.columns)}")
                    return None
                values = df[target_column].values
            else:
                values = df[target_col_idx].values

        if inverse_transform and self.preprocessor is not None:
            # Check if values need inverse transformation
            if target_column == 'transformed_networth':
                # Apply inverse sinh transformation
                values = np.sinh(values)
            elif hasattr(self.preprocessor, 'inverse_transform_target'):
                # Use preprocessor's inverse transform
                values = self.preprocessor.inverse_transform_target(
                    values.reshape(-1, 1)
                ).flatten()

        return values

    def fit(self, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray = None,
            numerical_cols: List[str] = None, categorical_cols: List[str] = None,
            target_col: str = 'transformed_networth', donor_name: str = 'cv_donor',
            vae_epochs: int = 1000, diffusion_epochs: int = 5000,
            force_retrain: bool = False) -> 'TabSynPipeline':
        """
        Unified fit interface for CV compatibility.
        Wraps preprocess_data + train_vae + train_diffusion.

        Args:
            X: DataFrame of predictors
            y: array of target values
            weights: survey weights (array)
            numerical_cols: list of numerical column names
            categorical_cols: list of categorical column names
            target_col: name for target column
            donor_name: name for this training dataset
            vae_epochs: VAE training epochs
            diffusion_epochs: diffusion training epochs
            force_retrain: if False, reuse existing models if available

        Returns:
            self for method chaining
        """
        # Build donor dataframe
        donor_df = X.copy()
        donor_df[target_col] = y
        if weights is not None:
            donor_df['wgt'] = weights
        else:
            donor_df['wgt'] = 1.0

        self.target_col = target_col
        self._fit_numerical_cols = numerical_cols
        self._fit_categorical_cols = categorical_cols
        self._fit_predictors = list(X.columns)
        self._fit_vae_epochs = vae_epochs
        self._fit_diffusion_epochs = diffusion_epochs

        # Use existing preprocess_data method
        self.preprocess_data(
            scf_data=donor_df,
            cps_data=donor_df,  # Placeholder for preprocessing
            predictors=self._fit_predictors,
            target_variable=target_col,
            weights='wgt',
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            donor_name=donor_name,
            receiver_name=f'{donor_name}_placeholder'
        )

        # Use existing training methods (respects force_retrain flag)
        self.train_vae(epochs=vae_epochs, force_retrain=force_retrain)
        self.train_diffusion(epochs=diffusion_epochs, force_retrain=force_retrain)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, weights: np.ndarray = None,
                receiver_name: str = 'cv_receiver', num_trials: int = 10) -> np.ndarray:
        """
        Unified predict interface for CV compatibility.
        Wraps preprocess + impute + get_imputed_values.

        Args:
            X: DataFrame of predictors to impute onto
            weights: survey weights for receiver
            receiver_name: name for receiver dataset
            num_trials: number of imputation trials

        Returns:
            array of imputed target values (not inverse transformed)
        """
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Store expected number of samples for validation
        expected_n_samples = len(X)

        # IMPORTANT: Reset index to ensure consistent row ordering
        # This prevents issues when X has non-sequential indices from iloc slicing
        X = X.reset_index(drop=True)

        # Build receiver dataframe with same structure as donor
        receiver_df = X.copy()
        receiver_df[self.target_col] = np.nan
        if weights is not None:
            receiver_df['wgt'] = weights
        else:
            receiver_df['wgt'] = 1.0

        # Load the donor data that was saved during fit()
        donor_path = self.working_dir / 'data' / f'{self.donor_name}.csv'
        donor_df = pd.read_csv(donor_path)

        # Reorder columns to match
        all_cols = self._fit_predictors + [self.target_col, 'wgt']
        receiver_df = receiver_df[all_cols]

        # Re-run preprocessing with new receiver data
        # This re-uses the donor data and creates new receiver preprocessing
        self.preprocess_data(
            scf_data=donor_df,
            cps_data=receiver_df,
            predictors=self._fit_predictors,
            target_variable=self.target_col,
            weights='wgt',
            numerical_cols=self._fit_numerical_cols,
            categorical_cols=self._fit_categorical_cols,
            donor_name=self.donor_name,
            receiver_name=receiver_name
        )

        # Setup model links for the new receiver
        self.setup_model_links()

        # Clear any existing imputation results for this receiver to force fresh imputation
        import shutil
        impute_dir = Path(f'impute/{receiver_name}')
        if impute_dir.exists():
            shutil.rmtree(impute_dir)

        # Use existing impute method
        self.impute(num_trials=num_trials)

        # Return imputed values (not inverse transformed for CV comparison)
        result = self.get_imputed_values(target_column=self.target_col, inverse_transform=False)

        # Validate output shape matches input
        if result is not None and len(result) != expected_n_samples:
            print(f"WARNING: Output shape mismatch! Expected {expected_n_samples}, got {len(result)}")

        return result

    def full_pipeline(self,
                     scf_data: pd.DataFrame,
                     cps_data: pd.DataFrame,
                     predictors: List[str],
                     target_variable: str,
                     weights: Optional[str] = None,
                     numerical_cols: Optional[List[str]] = None,
                     categorical_cols: Optional[List[str]] = None,
                     vae_epochs: int = 100,
                     diffusion_epochs: int = 100,
                     num_trials: int = 50,
                     force_retrain: bool = False) -> Dict:
        """
        Run the complete pipeline from preprocessing to imputation.

        Args:
            scf_data: SCF DataFrame
            cps_data: CPS DataFrame
            predictors: List of predictor columns
            target_variable: Target to impute
            weights: Weight column name
            numerical_cols: List of numerical column names (if None, auto-detect)
            categorical_cols: List of categorical column names (if None, auto-detect)
            vae_epochs: VAE training epochs
            diffusion_epochs: Diffusion training epochs
            num_trials: Number of imputation trials
            force_retrain: If True, retrain models even if they exist

        Returns:
            Dictionary with all results
        """
        if self.verbose:
            print("Running full TabSyn pipeline...")

        # Step 1: Preprocess
        preprocess_results = self.preprocess_data(
            scf_data=scf_data,
            cps_data=cps_data,
            predictors=predictors,
            target_variable=target_variable,
            weights=weights,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols
        )

        # Step 2: Train VAE
        if not self.train_vae(epochs=vae_epochs, force_retrain=force_retrain):
            raise RuntimeError("VAE training failed")

        # Step 3: Train Diffusion
        if not self.train_diffusion(epochs=diffusion_epochs, force_retrain=force_retrain):
            raise RuntimeError("Diffusion training failed")

        # Step 4: Impute
        imputed_df = self.impute(num_trials=num_trials)

        # Step 5: Get final values
        imputed_values = self.get_imputed_values(
            target_column=target_variable,
            inverse_transform=True
        )

        results = {
            'imputed_df': imputed_df,
            'imputed_values': imputed_values,
            'preprocessor': self.preprocessor,
            'preprocess_results': preprocess_results,
            'donor_name': self.donor_name,
            'receiver_name': self.receiver_name
        }

        if self.verbose:
            print("Imputation complete")
            print(f"Imputed {len(imputed_values)} values")
            print(f"Mean: {np.mean(imputed_values):,.2f}")
            print(f"Median: {np.median(imputed_values):,.2f}")
            print(f"Std: {np.std(imputed_values):,.2f}")

        return results


# Convenience functions for direct notebook usage

def run_tabsyn_imputation(
    scf_data: pd.DataFrame,
    cps_data: pd.DataFrame,
    predictors: List[str],
    target_variable: str,
    weights: Optional[str] = None,
    num_trials: int = 50,
    python_path: str = 'python3',
    verbose: bool = True
) -> np.ndarray:
    """
    Simple one-function interface for TabSyn imputation.

    Args:
        scf_data: SCF DataFrame (donor)
        cps_data: CPS DataFrame (receiver)
        predictors: List of predictor columns
        target_variable: Target to impute
        weights: Weight column
        num_trials: Number of imputation trials
        python_path: Python executable path
        verbose: Print progress

    Returns:
        Array of imputed values
    """
    pipeline = TabSynPipeline(
        python_path=python_path,
        verbose=verbose
    )

    results = pipeline.full_pipeline(
        scf_data=scf_data,
        cps_data=cps_data,
        predictors=predictors,
        target_variable=target_variable,
        weights=weights,
        num_trials=num_trials
    )

    return results['imputed_values']


if __name__ == "__main__":
    print("TabSyn wrapper module loaded.")
    print("Use TabSynPipeline class or run_tabsyn_imputation() function.")