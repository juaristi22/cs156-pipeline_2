"""
Custom preprocessing for TabSyn that uses SCF statistics to normalize CPS data.
This enables full variable imputation from SCF (donor) to CPS (receiver).
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CrossDatasetPreprocessor:
    """
    Preprocessor that fits on donor data (SCF) and transforms receiver data (CPS)
    using the donor's statistics for normalization.
    """

    def __init__(self,
                 num_col_idx,
                 cat_col_idx,
                 target_col_idx,
                 task_type='regression',
                 normalization='quantile',
                 n_quantiles=1000):
        """
        Initialize the preprocessor.

        Args:
            num_col_idx: List of numerical column indices
            cat_col_idx: List of categorical column indices
            target_col_idx: List of target column indices
            task_type: 'regression' or 'bin_class'
            normalization: Type of normalization ('quantile', 'standard', 'minmax')
            n_quantiles: Number of quantiles for QuantileTransformer
        """
        self.num_col_idx = num_col_idx
        self.cat_col_idx = cat_col_idx
        self.target_col_idx = target_col_idx
        self.task_type = task_type
        self.normalization = normalization
        self.n_quantiles = n_quantiles

        # Will store fitted transformers
        self.num_transformer = None
        self.cat_encoders = {}
        self.target_transformer = None
        self.column_info = {}

    def fit_donor(self, donor_df):
        """
        Fit the preprocessor on donor data (SCF).

        Args:
            donor_df: DataFrame containing donor data
        """
        print(f"Fitting preprocessor on donor data with shape: {donor_df.shape}")

        # Convert DataFrame columns to list for indexing
        columns = donor_df.columns.tolist()

        # Fit numerical columns
        if self.num_col_idx:
            num_cols = [columns[i] for i in self.num_col_idx]
            X_num = donor_df[num_cols].values.astype(np.float32)

            if self.normalization == 'quantile':
                self.num_transformer = QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=min(self.n_quantiles, len(X_num)),
                    subsample=int(1e9),
                    random_state=42
                )
            elif self.normalization == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.num_transformer = StandardScaler()
            else:  # minmax
                from sklearn.preprocessing import MinMaxScaler
                self.num_transformer = MinMaxScaler()

            self.num_transformer.fit(X_num)
            print(f"Fitted numerical transformer on {len(num_cols)} columns")

            # Store column info for numerical columns
            for idx, col in zip(self.num_col_idx, num_cols):
                self.column_info[idx] = {
                    'type': 'numerical',
                    'max': float(donor_df[col].max()),
                    'min': float(donor_df[col].min()),
                    'mean': float(donor_df[col].mean()),
                    'std': float(donor_df[col].std())
                }

        # Fit categorical columns
        if self.cat_col_idx:
            cat_cols = [columns[i] for i in self.cat_col_idx]
            for idx, col in zip(self.cat_col_idx, cat_cols):
                le = LabelEncoder()
                # Handle missing values
                col_data = donor_df[col].fillna('nan').astype(str)
                le.fit(col_data)
                self.cat_encoders[idx] = le

                # Store column info
                self.column_info[idx] = {
                    'type': 'categorical',
                    'categories': list(le.classes_),
                    'n_categories': len(le.classes_)
                }
            print(f"Fitted categorical encoders on {len(cat_cols)} columns")

        # Fit target column
        if self.target_col_idx:
            target_cols = [columns[i] for i in self.target_col_idx]

            if self.task_type == 'regression':
                X_target = donor_df[target_cols].values.astype(np.float32)

                if self.normalization == 'quantile':
                    self.target_transformer = QuantileTransformer(
                        output_distribution='normal',
                        n_quantiles=min(self.n_quantiles, len(X_target)),
                        subsample=int(1e9),
                        random_state=42
                    )
                elif self.normalization == 'standard':
                    from sklearn.preprocessing import StandardScaler
                    self.target_transformer = StandardScaler()
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    self.target_transformer = MinMaxScaler()

                self.target_transformer.fit(X_target)

                for idx, col in zip(self.target_col_idx, target_cols):
                    self.column_info[idx] = {
                        'type': 'numerical',
                        'max': float(donor_df[col].max()),
                        'min': float(donor_df[col].min()),
                        'mean': float(donor_df[col].mean()),
                        'std': float(donor_df[col].std())
                    }
            else:
                # For classification targets
                for idx, col in zip(self.target_col_idx, target_cols):
                    le = LabelEncoder()
                    col_data = donor_df[col].fillna('nan').astype(str)
                    le.fit(col_data)
                    self.cat_encoders[idx] = le

                    self.column_info[idx] = {
                        'type': 'categorical',
                        'categories': list(le.classes_),
                        'n_categories': len(le.classes_)
                    }

            print(f"Fitted target transformer on {len(target_cols)} columns")

    def transform_donor(self, donor_df):
        """
        Transform donor data using fitted transformers.

        Args:
            donor_df: DataFrame containing donor data

        Returns:
            Tuple of (X_num, X_cat, y) numpy arrays
        """
        columns = donor_df.columns.tolist()

        # Transform numerical columns
        X_num = None
        if self.num_col_idx:
            num_cols = [columns[i] for i in self.num_col_idx]
            X_num = donor_df[num_cols].values.astype(np.float32)
            X_num = self.num_transformer.transform(X_num).astype(np.float32)

        # Transform categorical columns
        X_cat = None
        if self.cat_col_idx:
            cat_cols = [columns[i] for i in self.cat_col_idx]
            X_cat_list = []
            for idx, col in zip(self.cat_col_idx, cat_cols):
                col_data = donor_df[col].fillna('nan').astype(str)
                encoded = self.cat_encoders[idx].transform(col_data)
                X_cat_list.append(encoded)
            X_cat = np.column_stack(X_cat_list) if X_cat_list else None

        # Transform target
        y = None
        if self.target_col_idx:
            target_cols = [columns[i] for i in self.target_col_idx]
            if self.task_type == 'regression':
                y = donor_df[target_cols].values.astype(np.float32)
                y = self.target_transformer.transform(y).astype(np.float32)
            else:
                y_list = []
                for idx, col in zip(self.target_col_idx, target_cols):
                    col_data = donor_df[col].fillna('nan').astype(str)
                    encoded = self.cat_encoders[idx].transform(col_data)
                    y_list.append(encoded)
                y = np.column_stack(y_list) if y_list else None

        return X_num, X_cat, y

    def transform_receiver(self, receiver_df, fill_target_with_mean=True):
        """
        Transform receiver data (CPS) using donor statistics.

        Args:
            receiver_df: DataFrame containing receiver data
            fill_target_with_mean: If True, fill missing target with donor mean

        Returns:
            Tuple of (X_num, X_cat, y) numpy arrays
        """
        columns = receiver_df.columns.tolist()

        # Transform numerical columns
        X_num = None
        if self.num_col_idx:
            num_cols = [columns[i] for i in self.num_col_idx]
            X_num = receiver_df[num_cols].values.astype(np.float32)
            # Use donor statistics for normalization
            X_num = self.num_transformer.transform(X_num).astype(np.float32)

        # Transform categorical columns
        X_cat = None
        if self.cat_col_idx:
            cat_cols = [columns[i] for i in self.cat_col_idx]
            X_cat_list = []
            for idx, col in zip(self.cat_col_idx, cat_cols):
                col_data = receiver_df[col].fillna('nan').astype(str)
                # Handle unseen categories by mapping to most frequent or 'nan'
                le = self.cat_encoders[idx]
                col_data_encoded = []
                for val in col_data:
                    if val in le.classes_:
                        col_data_encoded.append(le.transform([val])[0])
                    else:
                        # Map unseen to first category (could be 'nan' or most frequent)
                        col_data_encoded.append(0)
                X_cat_list.append(np.array(col_data_encoded))
            X_cat = np.column_stack(X_cat_list) if X_cat_list else None

        # Handle target (for imputation, this will be replaced)
        y = None
        if self.target_col_idx:
            target_cols = [columns[i] for i in self.target_col_idx]

            # Check if target exists in receiver
            has_target = all(col in columns for col in target_cols)

            if has_target and not receiver_df[target_cols].isna().all().all():
                # Target exists and has some values
                if self.task_type == 'regression':
                    y = receiver_df[target_cols].values.astype(np.float32)
                    y = self.target_transformer.transform(y).astype(np.float32)
                else:
                    y_list = []
                    for idx, col in zip(self.target_col_idx, target_cols):
                        col_data = receiver_df[col].fillna('nan').astype(str)
                        le = self.cat_encoders[idx]
                        col_data_encoded = []
                        for val in col_data:
                            if val in le.classes_:
                                col_data_encoded.append(le.transform([val])[0])
                            else:
                                col_data_encoded.append(0)
                        y_list.append(np.array(col_data_encoded))
                    y = np.column_stack(y_list) if y_list else None
            elif fill_target_with_mean:
                # Fill with mean/mode from donor
                n_samples = len(receiver_df)
                if self.task_type == 'regression':
                    # Use mean of normalized donor target (should be ~0 for quantile)
                    y = np.zeros((n_samples, len(self.target_col_idx)), dtype=np.float32)
                else:
                    # Use most frequent class (encoded as 0)
                    y = np.zeros((n_samples, len(self.target_col_idx)), dtype=int)

        return X_num, X_cat, y

    def inverse_transform_target(self, y_transformed):
        """
        Inverse transform the target variable back to original scale.

        Args:
            y_transformed: Transformed target values

        Returns:
            Original scale target values
        """
        if self.task_type == 'regression' and self.target_transformer is not None:
            return self.target_transformer.inverse_transform(y_transformed)
        elif self.task_type != 'regression':
            # For categorical, decode back to original labels
            y_decoded_list = []
            for i, idx in enumerate(self.target_col_idx):
                if idx in self.cat_encoders:
                    col_data = y_transformed[:, i].astype(int)
                    decoded = self.cat_encoders[idx].inverse_transform(col_data)
                    y_decoded_list.append(decoded)
            return np.column_stack(y_decoded_list) if y_decoded_list else y_transformed
        else:
            return y_transformed

    def save(self, save_path):
        """Save the fitted preprocessor."""
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {save_path}")

    @classmethod
    def load(cls, load_path):
        """Load a fitted preprocessor."""
        with open(load_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {load_path}")
        return preprocessor


def preprocess_for_full_imputation(
    donor_dataname,
    receiver_dataname,
    donor_df,
    receiver_df,
    info,
    save_dir='data'
):
    """
    Preprocess donor and receiver data for full variable imputation.

    Args:
        donor_dataname: Name of donor dataset (e.g., 'scf_2022')
        receiver_dataname: Name of receiver dataset (e.g., 'cps_2023')
        donor_df: DataFrame with donor data
        receiver_df: DataFrame with receiver data
        info: Dictionary with dataset information
        save_dir: Directory to save preprocessed data

    Returns:
        Dictionary with paths to saved preprocessed data
    """

    # Initialize preprocessor
    preprocessor = CrossDatasetPreprocessor(
        num_col_idx=info['num_col_idx'],
        cat_col_idx=info['cat_col_idx'],
        target_col_idx=info['target_col_idx'],
        task_type=info['task_type'],
        normalization='quantile'
    )

    # Fit on donor data
    preprocessor.fit_donor(donor_df)

    # Transform donor data
    X_num_donor, X_cat_donor, y_donor = preprocessor.transform_donor(donor_df)

    # Transform receiver data using donor statistics
    X_num_receiver, X_cat_receiver, y_receiver = preprocessor.transform_receiver(
        receiver_df,
        fill_target_with_mean=True
    )

    # Save donor data
    donor_save_dir = os.path.join(save_dir, donor_dataname)
    os.makedirs(donor_save_dir, exist_ok=True)

    # For TabSyn, we treat donor as "train" and save accordingly
    if X_num_donor is not None:
        np.save(os.path.join(donor_save_dir, 'X_num_train.npy'), X_num_donor)
    if X_cat_donor is not None:
        np.save(os.path.join(donor_save_dir, 'X_cat_train.npy'), X_cat_donor)
    if y_donor is not None:
        np.save(os.path.join(donor_save_dir, 'y_train.npy'), y_donor)

    # Save weights if they exist (for weighted statistics)
    if 'wgt' in donor_df.columns:
        weights_donor = donor_df['wgt'].values
        np.save(os.path.join(donor_save_dir, 'weights_train.npy'), weights_donor)
        print(f"Saved donor weights for weighted statistics computation")

    # Create minimal test set from donor (TabSyn requires it)
    test_size = min(100, len(donor_df) // 10)
    if X_num_donor is not None:
        np.save(os.path.join(donor_save_dir, 'X_num_test.npy'), X_num_donor[:test_size])
    if X_cat_donor is not None:
        np.save(os.path.join(donor_save_dir, 'X_cat_test.npy'), X_cat_donor[:test_size])
    if y_donor is not None:
        np.save(os.path.join(donor_save_dir, 'y_test.npy'), y_donor[:test_size])

    # Save receiver data for imputation
    receiver_save_dir = os.path.join(save_dir, receiver_dataname)
    os.makedirs(receiver_save_dir, exist_ok=True)

    # For imputation, receiver becomes "test" set
    if X_num_receiver is not None:
        np.save(os.path.join(receiver_save_dir, 'X_num_test.npy'), X_num_receiver)
    if X_cat_receiver is not None:
        np.save(os.path.join(receiver_save_dir, 'X_cat_test.npy'), X_cat_receiver)
    if y_receiver is not None:
        np.save(os.path.join(receiver_save_dir, 'y_test.npy'), y_receiver)

    # Also need minimal train set for receiver (use first 100 samples)
    train_size = min(100, len(receiver_df))
    if X_num_receiver is not None:
        np.save(os.path.join(receiver_save_dir, 'X_num_train.npy'), X_num_receiver[:train_size])
    if X_cat_receiver is not None:
        np.save(os.path.join(receiver_save_dir, 'X_cat_train.npy'), X_cat_receiver[:train_size])
    if y_receiver is not None:
        np.save(os.path.join(receiver_save_dir, 'y_train.npy'), y_receiver[:train_size])

    # Save preprocessor
    preprocessor_path = os.path.join(save_dir, f'{donor_dataname}_preprocessor.pkl')
    preprocessor.save(preprocessor_path)

    # Update info with preprocessing details
    info_copy = info.copy()
    info_copy['column_info'] = preprocessor.column_info
    info_copy['train_num'] = len(donor_df)
    info_copy['test_num'] = len(receiver_df)
    info_copy['preprocessor_path'] = preprocessor_path

    # Save updated info for both datasets
    with open(os.path.join(donor_save_dir, 'info.json'), 'w') as f:
        json.dump(info_copy, f, indent=4)

    with open(os.path.join(receiver_save_dir, 'info.json'), 'w') as f:
        json.dump(info_copy, f, indent=4)

    print(f"Preprocessing complete:")
    print(f"  Donor ({donor_dataname}): {len(donor_df)} samples")
    print(f"  Receiver ({receiver_dataname}): {len(receiver_df)} samples")
    print(f"  Preprocessor saved to: {preprocessor_path}")

    return {
        'donor_dir': donor_save_dir,
        'receiver_dir': receiver_save_dir,
        'preprocessor_path': preprocessor_path,
        'info': info_copy
    }


if __name__ == "__main__":
    # Example usage
    print("CrossDatasetPreprocessor module loaded.")
    print("Use preprocess_for_full_imputation() to preprocess donor and receiver data.")