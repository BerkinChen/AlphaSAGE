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

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret
from gan.utils.builder import exprs2tensor


QLIB_PATH = '/DATA1/home/chenbq/AlphaStruct/data/qlib_data/cn_data_rolling'

def load_alpha_pool(raw) -> List[Expression]:
    exprs_raw = raw['exprs']
    exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
    return exprs

def load_alpha_pool_by_path(path: str) -> List[Expression]:
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as f:
                raw = json.load(f)
                return load_alpha_pool(raw)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        exprs = df['exprs'].tolist()
        exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs]
        return exprs
    else:
        raise ValueError(f"Unsupported file extension: {path}")

def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list= []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i+chunk_size], y[i:i+chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list

def get_tensor_metrics(x, y):
    # Ensure tensors are 2D (days, stocks)
    if x.dim() > 2: x = x.squeeze(-1)
    if y.dim() > 2: y = y.squeeze(-1)

    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)
    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0.)
    ric_s = torch.nan_to_num(ric_s, nan=0.)
    ret_s = torch.nan_to_num(ret_s, nan=0.)

    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item() if ic_s.std().item() > 1e-6 else 1.0
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item() if ric_s.std().item() > 1e-6 else 1.0
    ret_s_mean = ret_s.mean().item()
    ret_s_std = ret_s.std().item() if ret_s.std().item() > 1e-6 else 1.0

    result = dict(
        ic=ic_s_mean,
        ic_std=ic_s_std,
        icir=ic_s_mean / ic_s_std,
        ric=ric_s_mean,
        ric_std=ric_s_std,
        ricir=ric_s_mean / ric_s_std,
        ret=ret_s_mean,
        ret_std=ret_s_std,
        retir=ret_s_mean / ret_s_std,
    )
    return result


def run(args):
    """
    Main function to run adaptive factor combination and evaluation.
    """
    window = args.window
    if isinstance(window, str):
        assert window == 'inf'
        window = float('inf')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # 1. Define Target and Load Data
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

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
    expressions = load_alpha_pool_by_path(args.expressions_file)
    print(f"Loaded {len(expressions)} expressions.")

    fct_tensor = exprs2tensor(expressions, data_all, normalize=True)
    tgt_tensor = exprs2tensor([target], data_all, normalize=False)

    # 3. Pre-calculate daily metrics for all factors
    ic_list, ric_list, ret_list = [], [], []
    print("Pre-calculating daily metrics for each factor...")
    for i in tqdm(range(fct_tensor.shape[-1])):
        factor_slice = fct_tensor[..., i]
        target_slice = tgt_tensor[..., 0]
        ic_s = batch_pearsonr(factor_slice, target_slice)
        ric_s = chunk_batch_spearmanr(factor_slice, target_slice, chunk_size=400)
        ret_s = batch_ret(factor_slice, target_slice)
        ic_list.append(torch.nan_to_num(ic_s, nan=0.))
        ric_list.append(torch.nan_to_num(ric_s, nan=0.))
        ret_list.append(torch.nan_to_num(ret_s, nan=0.))

    ic_s = torch.stack(ic_list, dim=-1)
    ric_s = torch.stack(ric_list, dim=-1)
    ret_s = torch.stack(ret_list, dim=-1)
    torch.cuda.empty_cache()

    # 4. Main adaptive combination loop
    pred_list = []
    shift = 21  # To avoid lookahead bias
    
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
        
        good_factors = metrics_df[(metrics_df['ric'] > 0.02) & (metrics_df['ricir'] > 0.2)]
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
        
        # Add constant for intercept
        ones = torch.ones_like(x[..., 0:1])
        x = torch.cat([x, ones], dim=-1)
        ones_pred = torch.ones_like(to_pred[..., 0:1])
        to_pred = torch.cat([to_pred, ones_pred], dim=-1)
        
        # Train regression and predict
        try:
            coef = torch.linalg.lstsq(x, y).solution
            pred = to_pred @ coef
        except torch.linalg.LinAlgError:
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
    valid_results = get_tensor_metrics(pred_valid.cuda(), tgt_valid.cuda())
    test_results = get_tensor_metrics(pred_test.cuda(), tgt_test.cuda())

    # Format and print results
    results_df = pd.DataFrame([valid_results, test_results], index=['Validation', 'Test'])
    print("\n--- Final Performance Metrics ---")
    print(results_df.round(4))
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expressions_file', type=str, required=True,
                        help='Path to a JSON file containing a list of alpha expressions.')
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--train_end_year', type=int, default=2020)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--n_factors', type=int, default=10,
                        help='Maximum number of factors to select at each step.')
    parser.add_argument('--window', type=str, default='inf',
                        help="Rolling window size for factor evaluation. 'inf' for expanding window.")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)
