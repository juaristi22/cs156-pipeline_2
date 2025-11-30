"""
Full dataset imputation script for TabSyn with correct hyperparameters.
Implements full variable imputation from donor (SCF) to receiver (CPS) datasets.

Key features:
- Uses exact hyperparameters from TabSyn paper
- Implements weighted random initialization using donor statistics
- Processes entire receiver dataset (not just test split)
- Handles batch processing for large datasets
"""

import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import warnings
import json
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import recover_data, split_num_cat_target
from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from tabsyn_custom_preprocess import CrossDatasetPreprocessor

warnings.filterwarnings('ignore')

# ============================================================================
# HYPERPARAMETERS FROM ORIGINAL TABSYN PAPER AND IMPLEMENTATION
# ============================================================================

# VAE Model Architecture (from tabsyn/vae/main.py)
VAE_PARAMS = {
    'num_layers': 2,      # Transformer layers
    'd_token': 4,         # Token dimension
    'n_head': 1,          # Attention heads
    'factor': 32,         # Hidden dimension factor
}

# Diffusion Model (from tabsyn/main.py)
DIFFUSION_HIDDEN_DIM = 1024  # NOT 512! This is crucial

# Diffusion Sampling Parameters (from impute.py)
SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1

def step(net, num_steps, i, t_cur, t_next, x_next):
    """One denoising step from t to t-1."""
    x_cur = x_next
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def impute_full_dataset(
    donor_name,
    receiver_name,
    model_name,
    device='cuda:0',
    num_trials=50,
    batch_size=2048,
    num_steps=50,
    N=20,
    use_random_init=True,  # NEW: Use random init instead of mean
    save_individual_trials=True
):
    """
    Perform full dataset imputation from donor to receiver using TabSyn.

    Uses exact hyperparameters from the TabSyn paper and implements
    weighted random initialization for better representation of the
    target distribution.
    """

    print(f"Starting full dataset imputation:")
    print(f"  Donor: {donor_name}")
    print(f"  Receiver: {receiver_name}")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Random init for target: {use_random_init}")
    print("\nUsing hyperparameters from TabSyn paper:")
    print(f"  VAE layers: {VAE_PARAMS['num_layers']}")
    print(f"  Diffusion hidden dim: {DIFFUSION_HIDDEN_DIM}")
    print(f"  Sampling steps: {num_steps}")

    # Load dataset info - need both model and receiver info
    model_info_path = f'data/{model_name}/info.json'
    receiver_info_path = f'data/{receiver_name}/info.json'

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    with open(receiver_info_path, 'r') as f:
        receiver_info = json.load(f)

    # Use model's structure for architecture
    num_col_idx = model_info['num_col_idx']
    cat_col_idx = model_info['cat_col_idx']
    target_col_idx = model_info['target_col_idx']
    task_type = model_info['task_type']

    # Use model info for categories to match training
    info = model_info

    # Create idx_mapping if not present
    # This maps global column indices to their position in the concatenated array
    # The concatenated array order is: [num_cols, cat_cols, target_cols]
    if 'idx_mapping' not in info:
        idx_mapping = {}
        position = 0

        # Map numerical columns
        for idx in num_col_idx:
            idx_mapping[str(idx)] = position
            position += 1

        # Map categorical columns
        for idx in cat_col_idx:
            idx_mapping[str(idx)] = position
            position += 1

        # Map target columns
        for idx in target_col_idx:
            idx_mapping[str(idx)] = position
            position += 1

        info['idx_mapping'] = idx_mapping

    # Load preprocessor if available
    preprocessor = None
    if 'preprocessor_path' in info and os.path.exists(info['preprocessor_path']):
        preprocessor = CrossDatasetPreprocessor.load(info['preprocessor_path'])
        print(f"Loaded preprocessor from {info['preprocessor_path']}")

    # Load preprocessed receiver data (actual data to impute)
    data_dir = f'data/{receiver_name}'
    X_num_test = np.load(f'{data_dir}/X_num_test.npy') if os.path.exists(
        f'{data_dir}/X_num_test.npy') else None
    X_cat_test = np.load(f'{data_dir}/X_cat_test.npy') if os.path.exists(
        f'{data_dir}/X_cat_test.npy') else None

    n_samples = len(X_num_test) if X_num_test is not None else len(X_cat_test)
    print(f"Processing {n_samples} samples from receiver dataset")

    # Get categories info
    categories = []
    if X_cat_test is not None:
        for col_idx in cat_col_idx:
            # column_info uses string keys, not integers
            col_str = str(col_idx)
            if col_str in info.get('column_info', {}):
                n_cats = info['column_info'][col_str].get('n_categories', 2)
            else:
                n_cats = len(np.unique(X_cat_test[:, cat_col_idx.index(col_idx)]))
            categories.append(n_cats)
    # Load target data (will be NaN for receiver)
    y_test = np.load(f'{data_dir}/y_test.npy') if os.path.exists(
        f'{data_dir}/y_test.npy') else None

    # Concatenate target with numerical features (as done during training)
    if X_num_test is not None and y_test is not None:
        X_num_with_target = np.concatenate([X_num_test, y_test], axis=1)
    else:
        X_num_with_target = X_num_test

    d_numerical = X_num_with_target.shape[1] if X_num_with_target is not None else 0

    # Convert to tensors (use the concatenated version for numerical)
    X_test_num = torch.tensor(X_num_with_target).float() if X_num_with_target is not None else None
    X_test_cat = torch.tensor(X_cat_test).long() if X_cat_test is not None else None

    # Model paths - note the double 'tabsyn' because training runs from tabsyn/ dir
    vae_ckpt_dir = f'tabsyn/tabsyn/vae/ckpt/{model_name}'
    vae_model_path = f'{vae_ckpt_dir}/model.pt'
    embedding_save_path = f'{vae_ckpt_dir}/train_z.npy'
    diffusion_ckpt_path = f'tabsyn/tabsyn/ckpt/{model_name}/model.pt'

    # Check if models exist
    if not os.path.exists(vae_model_path):
        raise FileNotFoundError(f"VAE model not found at {vae_model_path}")
    if not os.path.exists(diffusion_ckpt_path):
        raise FileNotFoundError(f"Diffusion model not found at {diffusion_ckpt_path}")

    # Initialize VAE with correct hyperparameters
    model = Model_VAE(
        num_layers=VAE_PARAMS['num_layers'],
        d_numerical=d_numerical,
        categories=categories,
        d_token=VAE_PARAMS['d_token'],
        n_head=VAE_PARAMS['n_head'],
        factor=VAE_PARAMS['factor'],
        bias=True
    ).to(device)
    model.load_state_dict(torch.load(vae_model_path, map_location=device))
    model.eval()

    pre_encoder = Encoder_model(
        num_layers=VAE_PARAMS['num_layers'],
        d_numerical=d_numerical,
        categories=categories,
        d_token=VAE_PARAMS['d_token'],
        n_head=VAE_PARAMS['n_head'],
        factor=VAE_PARAMS['factor']
    ).to(device)

    pre_decoder = Decoder_model(
        num_layers=VAE_PARAMS['num_layers'],
        d_numerical=d_numerical,
        categories=categories,
        d_token=VAE_PARAMS['d_token'],
        n_head=VAE_PARAMS['n_head'],
        factor=VAE_PARAMS['factor']
    ).to(device)

    pre_encoder.load_weights(model)
    pre_decoder.load_weights(model)

    # Load training embeddings for statistics
    train_z = torch.tensor(np.load(embedding_save_path)).float()
    train_z = train_z[:, 1:, :]  # Remove first token
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    mean, std = train_z.mean(0), train_z.std(0)

    # Initialize diffusion model with CORRECT hidden dimension
    denoise_fn = MLPDiffusion(in_dim, DIFFUSION_HIDDEN_DIM).to(device)
    diffusion_model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_ckpt_path, map_location=device))
    diffusion_model.eval()

    # Create save directory
    save_dir = f'impute/{receiver_name}'
    os.makedirs(save_dir, exist_ok=True)

    # Store all trial results
    all_results = []

    # If using random init, compute statistics from training data for random initialization
    if use_random_init and task_type == 'regression':
        # Load training target statistics for random init range
        train_data_path = f'data/{donor_name}/y_train.npy'
        if os.path.exists(train_data_path):
            train_y = np.load(train_data_path)

            # Try to load weights for weighted statistics
            weights_path = f'data/{donor_name}/weights_train.npy'
            if os.path.exists(weights_path):
                weights = np.load(weights_path)
                # Compute weighted mean and std
                weights_normalized = weights / weights.sum()
                target_mean = np.average(train_y, weights=weights_normalized, axis=0)
                # Weighted standard deviation
                variance = np.average((train_y - target_mean)**2, weights=weights_normalized, axis=0)
                target_std = np.sqrt(variance)
                if len(target_mean.shape) > 0:
                    print(f"Using WEIGHTED random init with mean={target_mean[0]:.3f}, std={target_std[0]:.3f}")
                else:
                    print(f"Using WEIGHTED random init with mean={target_mean:.3f}, std={target_std:.3f}")
            else:
                # Fall back to unweighted if weights not found
                target_mean = train_y.mean()
                target_std = train_y.std()
                print(f"Using unweighted random init with mean={target_mean:.3f}, std={target_std:.3f}")
                print("Note: Consider saving weights for better initialization")
        else:
            use_random_init = False
            print("Warning: Training target not found, falling back to zero init")

    # Check for existing trial results
    impute_dir = Path(f'impute/{receiver_name}')
    existing_trials = []

    # Check if we can reuse existing trials
    if impute_dir.exists():
        for trial_idx in range(num_trials):
            trial_path = impute_dir / f'{trial_idx}.csv'
            if trial_path.exists():
                existing_trials.append(trial_idx)

    # If all trials exist, just load them instead of regenerating
    if len(existing_trials) == num_trials:
        print(f"\nFound all {num_trials} existing trials in {impute_dir}, loading them...")
        # Clear all_results and load existing trials
        all_results = []
        for trial_idx in range(num_trials):
            trial_path = impute_dir / f'{trial_idx}.csv'
            trial_df = pd.read_csv(trial_path)
            all_results.append(trial_df)
            print(f"  Loaded trial {trial_idx + 1}/{num_trials}")
    else:
        # Run new imputation trials
        if existing_trials:
            print(f"\nFound {len(existing_trials)} existing trials but need {num_trials}. Running all trials fresh...")
        else:
            print(f"\nNo existing trials found. Running {num_trials} new imputation trials...")

        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials}")

            trial_results = []

            # Process in batches
            for batch_start in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
                batch_end = min(batch_start + batch_size, n_samples)

                # Get batch
                batch_num = X_test_num[batch_start:batch_end].to(device) if X_test_num is not None else None
                batch_cat = X_test_cat[batch_start:batch_end].to(device) if X_test_cat is not None else None

                # Initialize target column for encoding
                if task_type == 'regression' and batch_num is not None:
                        # Target is at the end of the numerical columns after concatenation
                        # batch_num contains: [features..., target]
                        # So target starts at index (total_cols - num_targets)
                        target_start_idx = batch_num.shape[1] - len(target_col_idx)

                        if use_random_init:
                            # Use random values from approximate target distribution
                            batch_size_actual = batch_num.shape[0]
                            random_target = torch.randn(batch_size_actual, len(target_col_idx)).to(device)
                            if 'target_std' in locals():
                                random_target = random_target * target_std + target_mean
                            # Place target values at the correct position (last columns)
                            for i in range(len(target_col_idx)):
                                batch_num[:, target_start_idx + i] = random_target[:, i]
                        else:
                            # Use mean (0 for normalized data)
                            for i in range(len(target_col_idx)):
                                batch_num[:, target_start_idx + i] = 0.0
                elif task_type != 'regression' and batch_cat is not None:
                    # For categorical, use random class
                    for idx in target_col_idx:
                        n_classes = categories[cat_col_idx.index(idx)]
                        batch_cat[:, idx] = torch.randint(0, n_classes, (batch_cat.shape[0],)).to(device)

                # Encode to latent space
                with torch.no_grad():
                    z_batch = pre_encoder(batch_num, batch_cat).detach()
                    z_batch = z_batch[:, 1:, :].contiguous()
                    z_batch = z_batch.view(z_batch.size(0), -1)
                    z_batch = (z_batch - mean.to(device)) / 2  # Normalize as in original

                    # Diffusion sampling
                    batch_size_actual = z_batch.shape[0]
                    x_t = torch.randn([batch_size_actual, in_dim], device=device)

                    # Setup diffusion timesteps
                    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
                    net = diffusion_model.denoise_fn_D

                    sigma_min = max(SIGMA_MIN, net.sigma_min)
                    sigma_max = min(SIGMA_MAX, net.sigma_max)

                    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
                                (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

                    # Create mask for target columns
                    mask_idx = target_col_idx if task_type == 'regression' else target_col_idx + [d_numerical]
                    mask_list = [list(range(i * token_dim, (i + 1) * token_dim)) for i in mask_idx]
                    mask = torch.zeros(num_tokens * token_dim, dtype=torch.bool, device=device)
                    mask[mask_list] = True
                    mask = mask.to(torch.int)

                    x_t = x_t.to(torch.float32) * t_steps[0]

                    # Diffusion sampling loop
                    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                        if i < num_steps - 1:
                            for j in range(N):
                                n_prev = torch.randn_like(z_batch).to(device) * t_next
                                x_known_t_prev = z_batch + n_prev
                                x_unknown_t_prev = step(net, num_steps, i, t_cur, t_next, x_t)

                                # Key line: mask ensures diffusion output replaces target
                                x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask

                                if j == N - 1:
                                    x_t = x_t_prev
                                else:
                                    n = torch.randn_like(z_batch) * (t_cur.pow(2) - t_next.pow(2)).sqrt()
                                    x_t = x_t_prev + n

                    # Denormalize
                    x_t = x_t * 2 + mean.to(device)

                    # Store batch results
                    trial_results.append(x_t.cpu().numpy())

            # Concatenate all batches (outside batch loop, inside trial loop)
            trial_data = np.concatenate(trial_results, axis=0)

            # Decode and recover data
            from utils_train import preprocess
            _, _, _, _, num_inverse, cat_inverse = preprocess(
                f'data/{model_name}',
                task_type=task_type,
                inverse=True
            )

            # Prepare info for recovery
            info['pre_decoder'] = pre_decoder
            info['token_dim'] = token_dim

            # Split and recover
            syn_num, syn_cat, syn_target = split_num_cat_target(
                trial_data, info, num_inverse, cat_inverse, device
            )

            # Create DataFrame
            syn_df = recover_data(syn_num, syn_cat, syn_target, info)

            # Apply inverse transformation if preprocessor available
            if preprocessor is not None and len(target_col_idx) > 0:
                # Columns are named by their indices, so use the index value itself
                target_col_name = str(target_col_idx[0])  # This will be 6 for the target column
                if target_col_name in syn_df.columns:
                    imputed_values = syn_df[target_col_name].values.reshape(-1, 1)
                    original_scale = preprocessor.inverse_transform_target(imputed_values)
                    syn_df[target_col_name] = original_scale.flatten()

            # Rename columns if mapping exists
            if 'idx_name_mapping' in info:
                idx_name_mapping = {int(k): v for k, v in info['idx_name_mapping'].items()}
                syn_df.rename(columns=idx_name_mapping, inplace=True)

            # Save trial result
            if save_individual_trials:
                trial_path = f'{save_dir}/{trial}.csv'
                syn_df.to_csv(trial_path, index=False)
                print(f"  Saved trial {trial} to {trial_path}")

            all_results.append(syn_df)

    # Calculate average across all trials
    print("\nAveraging results across all trials...")
    # Reset index on all DataFrames to ensure consistent indexing
    all_results_reset = [df.reset_index(drop=True) for df in all_results]
    # Stack along new axis and compute mean
    stacked = pd.concat(all_results_reset, keys=range(len(all_results_reset))).groupby(level=1).mean()
    stacked = stacked.reset_index(drop=True)
    avg_path = f'{save_dir}/averaged_imputation.csv'
    stacked.to_csv(avg_path, index=False)
    print(f"Averaged results saved to {avg_path}")
    print(f"Averaged result shape: {stacked.shape}")

    return {
        'individual_trials': all_results,
        'averaged_result': stacked,
        'save_dir': save_dir,
        'n_samples': n_samples,
        'n_trials': num_trials
    }

def main():
    parser = argparse.ArgumentParser(
        description='Full dataset imputation for TabSyn with correct hyperparameters'
    )
    parser.add_argument('--donor', type=str, required=True,
                        help='Name of donor dataset (SCF)')
    parser.add_argument('--receiver', type=str, required=True,
                        help='Name of receiver dataset (CPS)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (defaults to donor name)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (-1 for CPU)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of imputation trials')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for processing')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of diffusion steps')
    parser.add_argument('--random_init', action='store_true',
                        help='Use random initialization instead of mean for target')

    args = parser.parse_args()

    # Set device
    if args.gpu != -1 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    # Default model name to donor name
    model_name = args.model if args.model else args.donor

    # Run imputation
    results = impute_full_dataset(
        donor_name=args.donor,
        receiver_name=args.receiver,
        model_name=model_name,
        device=device,
        num_trials=args.trials,
        batch_size=args.batch_size,
        num_steps=args.steps,
        use_random_init=args.random_init
    )

    print(f"\nImputation complete!")
    print(f"Results saved in: {results['save_dir']}")
    print(f"Processed {results['n_samples']} samples over {results['n_trials']} trials")

if __name__ == "__main__":
    main()