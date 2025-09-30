import torch 
import os
from gan.utils import load_pickle
from alphagen_generic.features import *
from alphagen.data.expression import *
from typing import Tuple, List, Union
import json
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown
from gan.utils.builder import exprs2tensor


def remove_linearly_dependent_rows(x, y, to_pred, tol=1e-10):
    """
    Remove linearly dependent rows using efficient rank detection for speed.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        y: Target matrix (n_samples, n_targets)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        tol: Tolerance for linear independence
    
    Returns:
        x_filtered: Filtered training matrix
        y_filtered: Filtered target matrix  
        to_pred: Original prediction matrix (unchanged)
        selected_rows: List of selected row indices
    """
    if x.shape[0] <= x.shape[1]:
        # If we have fewer samples than features, keep all samples
        return x, y, to_pred, list(range(x.shape[0]))
    
    # For efficiency, only check for linear dependence if we have many more samples than features
    # This is the common case where linear dependence in rows matters
    sample_ratio = x.shape[0] / x.shape[1]
    
    if sample_ratio < 5:  # Not enough samples to worry about row dependence
        return x, y, to_pred, list(range(x.shape[0]))
    
    try:
        # Use SVD on transposed matrix to find rank efficiently
        U, S, Vh = torch.linalg.svd(x.T, full_matrices=False)
        
        # Effective rank based on singular values
        rank = torch.sum(S > tol * S[0]).item()
        
        if rank >= min(x.shape[0], x.shape[1]):
            # Matrix is full rank, no need to remove rows
            return x, y, to_pred, list(range(x.shape[0]))
        
        # If rank deficient, use QR decomposition to find independent rows
        Q, R = torch.linalg.qr(x.T, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol
        
        if not torch.any(pivot_mask):
            selected_rows = [0]
        else:
            selected_rows = torch.where(pivot_mask)[0].tolist()
            if len(selected_rows) == 0:
                selected_rows = [0]
                
    except:
        # If SVD/QR fails, keep all rows
        return x, y, to_pred, list(range(x.shape[0]))
    
    # Filter matrices
    x_filtered = x[selected_rows]
    y_filtered = y[selected_rows] if y is not None else None
    
    return x_filtered, y_filtered, to_pred, selected_rows


def remove_linearly_dependent_cols(x, to_pred, tol=1e-10):
    """
    Remove linearly dependent columns (factors) using QR decomposition with pivoting for speed.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        tol: Tolerance for linear independence
    
    Returns:
        x_filtered: Filtered training matrix
        to_pred_filtered: Filtered prediction matrix
        selected_factors: List of selected factor indices
    """
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    # Use SVD for more robust rank detection (faster than iterative QR)
    try:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        
        # Find columns corresponding to significant singular values
        rank = torch.sum(S > tol * S[0]).item()  # Relative tolerance
        
        if rank == 0:
            selected_factors = [0]
        else:
            # Use the first 'rank' columns as they correspond to largest singular values
            selected_factors = list(range(min(rank, x.shape[1])))
            
    except:
        # Fallback to QR decomposition if SVD fails
        Q, R = torch.linalg.qr(x, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol
        
        if not torch.any(pivot_mask):
            selected_factors = [0]
        else:
            selected_factors = torch.where(pivot_mask)[0].tolist()
            if len(selected_factors) == 0:
                selected_factors = [0]
    
    # Filter matrices
    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]
    
    return x_filtered, to_pred_filtered, selected_factors


def calculate_vif(x):
    """
    Calculate Variance Inflation Factor for each feature.
    VIF > 10 indicates multicollinearity issues.
    """
    n_features = x.shape[1]
    vif_scores = torch.zeros(n_features)
    
    for i in range(n_features):
        # Regression of feature i on all other features
        y_i = x[:, i]
        x_others = torch.cat([x[:, :i], x[:, i+1:]], dim=1)
        
        if x_others.shape[1] == 0:
            vif_scores[i] = 1.0
            continue
            
        try:
            # Add constant term
            ones = torch.ones(x_others.shape[0], 1, device=x.device)
            x_others_const = torch.cat([x_others, ones], dim=1)
            
            # Solve regression
            coef = torch.linalg.lstsq(x_others_const, y_i.unsqueeze(1), rcond=1e-15).solution
            y_pred = x_others_const @ coef
            
            # Calculate R-squared
            ss_res = torch.sum((y_i.unsqueeze(1) - y_pred) ** 2)
            ss_tot = torch.sum((y_i - torch.mean(y_i)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            
            # VIF = 1 / (1 - R^2)
            vif_scores[i] = 1.0 / (1.0 - torch.clamp(r_squared, max=0.999))
            
        except:
            vif_scores[i] = float('inf')
    
    return vif_scores


def remove_multicollinearity_vif(x, to_pred, vif_threshold=10.0):
    """
    Remove factors with high VIF to address multicollinearity.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        vif_threshold: VIF threshold above which factors are removed
    
    Returns:
        x_filtered: Filtered training matrix
        to_pred_filtered: Filtered prediction matrix
        selected_factors: List of selected factor indices
    """
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    selected_factors = list(range(x.shape[1]))
    
    while len(selected_factors) > 1:
        # Calculate VIF for current factors
        x_current = x[:, selected_factors]
        vif_scores = calculate_vif(x_current)
        
        # Find factor with highest VIF
        max_vif_idx = torch.argmax(vif_scores)
        max_vif = vif_scores[max_vif_idx]
        
        # If max VIF is below threshold, stop
        if max_vif <= vif_threshold:
            break
            
        # Remove the factor with highest VIF
        selected_factors.pop(max_vif_idx.item())
    
    # Filter matrices
    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]
    
    return x_filtered, to_pred_filtered, selected_factors



def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    exprs_raw = raw['exprs']
    weights = raw['weights']
    exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
    return exprs, weights

def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as f:
                raw = json.load(f)
                return load_alpha_pool(raw)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        exprs = df['exprs'].tolist()
        exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs if "Ensemble" not in expr_raw]
        try:
            weights = df['weight'].tolist()[:-1]
        except:
            weights = None
        return exprs, weights
    else:
        raise ValueError(f"Unsupported file extension: {path}")

def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list= []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i+chunk_size], y[i:i+chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list

def get_tensor_metrics(x, y, risk_free_rate=0.0):
    # Ensure tensors are 2D (days, stocks)
    if x.dim() > 2: x = x.squeeze(-1)
    if y.dim() > 2: y = y.squeeze(-1)

    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=args.chunk_size)
    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0.)
    ric_s = torch.nan_to_num(ric_s, nan=0.)
    ret_s = torch.nan_to_num(ret_s, nan=0.) / args.label_days
    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item() if ic_s.std().item() > 1e-6 else 1.0
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item() if ric_s.std().item() > 1e-6 else 1.0
    ret_s_mean = (ret_s).mean().item()
    ret_s_std = (ret_s).std().item() if (ret_s).std().item() > 1e-6 else 1.0

    # Calculate Sharpe Ratio and Maximum Drawdown for ret series
    ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item()
    ret_mdd = batch_max_drawdown(ret_s).item()

    result = dict(
        ic=ic_s_mean,
        ic_std=ic_s_std,
        icir=ic_s_mean / ic_s_std,
        ric=ric_s_mean,
        ric_std=ric_s_std,
        ricir=ric_s_mean / ric_s_std,
        ret=ret_s_mean * len(ret_s) / 3,
        ret_std=ret_s_std,
        retir=ret_s_mean / ret_s_std,
        ret_sharpe=ret_sharpe,
        ret_mdd=ret_mdd,
    )
    return result, ret_s


def run(args):
    """
    Main function to run adaptive factor combination and evaluation.
    """
    window = args.window
    if isinstance(window, str):
        assert window == 'inf'
        window = float('inf')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if args.instruments == 'sp500':
        QLIB_PATH = '/your_path/data/qlib_data/us_data_qlib'
    else:
        QLIB_PATH = '/your_path/data/qlib_data/cn_data_rolling'
    # 1. Define Target and Load Data
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -args.label_days) / close - 1

    train_end_time = f'{args.train_end_year}-12-31'
    valid_start_time = f'{args.train_end_year + 1}-01-01'
    valid_end_time = f'{args.train_end_year + 1}-12-31'
    test_start_time = f'{args.train_end_year + 2}-01-01'
    test_end_time = f'{args.train_end_year + 4}-12-31'

    data_all = StockData(instrument=args.instruments,
                         start_time='2010-01-01',
                         end_time=test_end_time,
                         qlib_path=QLIB_PATH)
    data_valid = StockData(instrument=args.instruments,
                           start_time=valid_start_time,
                           end_time=valid_end_time,
                           qlib_path=QLIB_PATH)
    data_test = StockData(instrument=args.instruments,
                          start_time=test_start_time,
                          end_time=test_end_time,
                          qlib_path=QLIB_PATH)

    # 2. Load expressions and convert to tensor
    print(f"Loading expressions from {args.expressions_file}...")
    expressions, weights = load_alpha_pool_by_path(args.expressions_file)
    print(f"Loaded {len(expressions)} expressions.")

    if args.use_weights:
        fct_tensor = exprs2tensor(expressions, data_test, normalize=True)
        weights = torch.tensor(weights).cuda()
        fct_tensor = fct_tensor @ weights
        tgt_tensor = exprs2tensor([target], data_test, normalize=False)
        test_results, ret_s = get_tensor_metrics(fct_tensor.cuda(), tgt_tensor.cuda())
        ret_s = ret_s.cpu().numpy()
        save_path = os.path.join(os.path.dirname(args.expressions_file), 'ret_s.npy')
        np.save(save_path, ret_s)
        # Format and print results
        results_df = pd.DataFrame([test_results], index=['Test'])
        print("\n--- Final Performance Metrics ---")
        
        # Print with full precision and no truncation
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.round(4))
        
        # Also print in a more parseable format
        print("\n--- Parseable Format ---")
        print(f"{'Dataset':<12} {'IC':>8} {'IC_STD':>8} {'ICIR':>8} {'RIC':>8} {'RIC_STD':>8} {'RICIR':>8} {'RET':>8} {'RET_STD':>8} {'RETIR':>8} {'RET_SR':>8} {'RET_MDD':>8}")
        for index, row in results_df.iterrows():
            print(f"{index:<12} {row['ic']:>8.4f} {row['ic_std']:>8.4f} {row['icir']:>8.4f} {row['ric']:>8.4f} {row['ric_std']:>8.4f} {row['ricir']:>8.4f} {row['ret']:>8.4f} {row['ret_std']:>8.4f} {row['retir']:>8.4f} {row['ret_sharpe']:>8.4f} {row['ret_mdd']:>8.4f}")
        
        print("="*50)
        
    else:
        fct_tensor = exprs2tensor(expressions, data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)

        # 3. Pre-calculate daily metrics for all factors
        ic_list, ric_list, ret_list = [], [], []
        print("Pre-calculating daily metrics for each factor...")
        for i in tqdm(range(fct_tensor.shape[-1])):
            factor_slice = fct_tensor[..., i]
            target_slice = tgt_tensor[..., 0]
            ic_s = batch_pearsonr(factor_slice, target_slice)
            ric_s = chunk_batch_spearmanr(factor_slice, target_slice, chunk_size=args.chunk_size)
            #ret_s = batch_ret(factor_slice, target_slice)
            ic_list.append(torch.nan_to_num(ic_s, nan=0.))
            ric_list.append(torch.nan_to_num(ric_s, nan=0.))
            #ret_list.append(torch.nan_to_num(ret_s, nan=0.))

        ic_s = torch.stack(ic_list, dim=-1)
        ric_s = torch.stack(ric_list, dim=-1)
        #ret_s = torch.stack(ret_list, dim=-1)
        torch.cuda.empty_cache()

        # 4. Main adaptive combination loop
        pred_list = []
        shift = args.label_days + 1  # To avoid lookahead bias
        
        valid_test_days = data_valid.n_days + data_test.n_days
        start_day = len(fct_tensor) - valid_test_days
        
        print("Starting adaptive combination process...")
        pbar = tqdm(range(start_day, len(fct_tensor)))
        for cur in pbar:
            # Define rolling window for evaluation
            begin = 0 if not np.isfinite(window) else max(0, cur - window - shift)
            
            # Slice metrics for the current window
            cur_ic = ic_s[begin:cur-shift]
            cur_ric = ric_s[begin:cur-shift]
            
            # Calculate performance metrics over the window
            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0)
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0)

            icir = ic_mean / ic_std
            ricir = ric_mean / ric_std
            
            # Filter and select best factors
            metrics_df = pd.DataFrame({
                'ric': ric_mean.cpu().numpy(),
                'ricir': ricir.cpu().numpy()
            })
            good_factors = metrics_df[(metrics_df['ric'].abs() > args.threshold_ric) & (metrics_df['ricir'].abs() > args.threshold_ricir)]
            if len(good_factors) < 1:
                good_factors = metrics_df.reindex(metrics_df.ricir.abs().sort_values(ascending=False).index).iloc[:1]
            
            good_idx = good_factors.iloc[:args.n_factors].index.to_list()
            
            # Prepare data for linear regression
            x = fct_tensor[begin:cur-shift, :, good_idx]
            y = tgt_tensor[begin:cur-shift, :, :]
            to_pred = fct_tensor[cur, :, good_idx]
            y = y.reshape(-1, y.shape[-1])
            x = x.reshape(-1, x.shape[-1])
            
            # Filter out NaNs
            valid_mask = torch.isfinite(y)[:, 0]
            y = y[valid_mask]
            x = x[valid_mask]
            
            to_pred = torch.nan_to_num(to_pred, nan=0.)
            
            # Remove linearly dependent columns (factors) for speed
            x, to_pred, selected_factors = remove_linearly_dependent_cols(x, to_pred, tol=args.linear_dep_tol)
            
            # Remove linearly dependent rows (samples) for speed  
            x, y, to_pred, selected_rows = remove_linearly_dependent_rows(x, y, to_pred, tol=args.linear_dep_tol)
            
            # Add constant for intercept
            ones = torch.ones_like(x[..., 0:1])
            x = torch.cat([x, ones], dim=-1)
            ones_pred = torch.ones_like(to_pred[..., 0:1])
            to_pred = torch.cat([to_pred, ones_pred], dim=-1)
            
            # Train regression and predict with improved stability
            try:
                # Check condition number before solving
                coef = torch.linalg.lstsq(x, y).solution
                
                pred = to_pred @ coef
                
            except Exception as e:
                print(f"Warning: Regression failed with error {e}, using zero prediction")
                # Handle singular matrix case
                pred = torch.zeros_like(to_pred[:, 0:1])

            pred_list.append(pred[:, 0])
            
            # Update progress bar description with running IC
            if len(pred_list) > 1:
                running_preds = torch.stack(pred_list, dim=0)
                running_targets = tgt_tensor[start_day:cur+1, :, 0]
                running_ic = batch_pearsonr(running_preds, running_targets).mean().item()
                pbar.set_description(f"Running IC: {running_ic:.4f}, Factors selected: {len(good_idx)}")


        # 5. Evaluate and display results
        print("\n" + "="*50)
        print("Adaptive combination finished. Calculating final metrics...")
        
        all_pred = torch.stack(pred_list, dim=0)
        
        # Slice predictions and targets for validation and test sets
        pred_valid = all_pred[:data_valid.n_days]
        pred_test = all_pred[data_valid.n_days:]
        
        tgt_valid = tgt_tensor[start_day : start_day + data_valid.n_days, :, 0]
        tgt_test = tgt_tensor[start_day + data_valid.n_days :, :, 0]
        
        # Calculate metrics
        valid_results, _ = get_tensor_metrics(pred_valid.cuda(), tgt_valid.cuda())
        test_results, ret_s = get_tensor_metrics(pred_test.cuda(), tgt_test.cuda())
        ret_s = ret_s.cpu().numpy()
        save_path = os.path.join(os.path.dirname(args.expressions_file), 'ret_s.npy')
        np.save(save_path, ret_s)
        # Format and print results
        results_df = pd.DataFrame([valid_results, test_results], index=['Validation', 'Test'])
        print("\n--- Final Performance Metrics ---")
        
        # Print with full precision and no truncation
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.round(4))
        
        # Also print in a more parseable format
        print("\n--- Parseable Format ---")
        print(f"{'Dataset':<12} {'IC':>8} {'IC_STD':>8} {'ICIR':>8} {'RIC':>8} {'RIC_STD':>8} {'RICIR':>8} {'RET':>8} {'RET_STD':>8} {'RETIR':>8} {'RET_SR':>8} {'RET_MDD':>8}")
        for index, row in results_df.iterrows():
            print(f"{index:<12} {row['ic']:>8.4f} {row['ic_std']:>8.4f} {row['icir']:>8.4f} {row['ric']:>8.4f} {row['ric_std']:>8.4f} {row['ricir']:>8.4f} {row['ret']:>8.4f} {row['ret_std']:>8.4f} {row['retir']:>8.4f} {row['ret_sharpe']:>8.4f} {row['ret_mdd']:>8.4f}")
        
        print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expressions_file', type=str, required=True,
                        help='Path to a JSON file containing a list of alpha expressions.')
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--train_end_year', type=int, default=2020)
    parser.add_argument('--threshold_ric', type=float, default=0.015)
    parser.add_argument('--threshold_ricir', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--n_factors', type=int, default=10,
                        help='Maximum number of factors to select at each step.')
    parser.add_argument('--chunk_size', type=int, default=400,
                        help='Chunk size for calculating Spearman correlation.')
    parser.add_argument('--window', type=str, default='inf',
                        help="Rolling window size for factor evaluation. 'inf' for expanding window.")
    parser.add_argument('--label_days', type=int, default=20,
                        help="Number of days to label the target.")
    parser.add_argument('--use_weights', type=bool, default=False,
                        help="Whether to use weights for the factors.")
    parser.add_argument('--corr_threshold', type=float, default=0.95,
                        help="Correlation threshold for multicollinearity detection.")
    parser.add_argument('--ridge_alpha', type=float, default=1e-6,
                        help="Ridge regression regularization parameter.")
    parser.add_argument('--use_vif', type=bool, default=False,
                        help="Whether to use VIF for multicollinearity detection.")
    parser.add_argument('--linear_dep_tol', type=float, default=1e-10,
                        help="Tolerance for linear dependence detection.")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)
