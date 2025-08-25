import torch
import random
import numpy as np
import argparse
import os
import json
from datetime import datetime
from torch.optim import Adam
from torch.distributions import Categorical
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from alphagen.rl.env.wrapper import action2token

from src.alpha_gfn.config import *
from src.alpha_gfn.env.core import GFNEnvCore
from src.alpha_gfn.modules import SequenceEncoder
from src.alpha_gfn.alpha_pool import AlphaPoolGFN
from src.alphagen.data.expression import *
from src.alphagen_qlib.stock_data import StockData
from src.alphagen.utils.correlation import batch_pearsonr

from gfn.samplers import Sampler
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.modules import NeuralNet

QLIB_PATH = '/DATA1/home/chenbq/AlphaStruct/data/qlib_data/cn_data_rolling'

def tokens_to_tensor(tokens: list, token_to_action):
    # The `+1` is for the padding value 0
    # The first token is BEG, skip it.
    token_indices = [token_to_action(token) + 1 for token in tokens[1:]]
    # Pad to max length
    padded_tokens = token_indices + [0] * (MAX_EXPR_LENGTH - len(token_indices))
    return torch.LongTensor(padded_tokens).unsqueeze(0)


class GFNLogger:
    def __init__(self, model: nn.Module, pool: AlphaPoolGFN, log_dir: str, test_data: StockData, target: Expression):
        self.model = model
        self.pool = pool
        self.log_dir = log_dir
        self.test_data = test_data
        self.target = target
        self.writer = SummaryWriter(log_dir)
        self.target_test = self.pool._normalize_by_day(self.target.evaluate(self.test_data))

    def log_metrics(self, episode: int):
        self.writer.add_scalar('pool/size', self.pool.size, episode)
        if self.pool.size > 0:
            self.writer.add_scalar('pool/best_single_ic', np.max(self.pool.single_ics[:self.pool.size]), episode)
            ic_test, rank_ic_test = self.pool.test_ensemble(self.test_data, self.target)
            self.writer.add_scalar('test/ic', ic_test, episode)
            self.writer.add_scalar('test/rank_ic', rank_ic_test, episode)
        self.writer.add_scalar('pool/eval_cnt', self.pool.eval_cnt, episode)

    def save_checkpoint(self, episode: int):
        model_path = os.path.join(self.log_dir, f'model_{episode}.pt')
        pool_path = os.path.join(self.log_dir, f'pool_{episode}.json')
        torch.save(self.model.state_dict(), model_path)
        with open(pool_path, 'w') as f:
            json.dump(self.pool.to_dict(), f, indent=4)

    def show_pool_state(self):
        state = self.pool.state
        exprs = state.get('exprs', [])
        n = len(exprs)
        print('---------------------------------------------')
        for i in range(n):
            expr = state['exprs'][i]
            expr_str = str(expr)
            ic_ret = state['ics_ret'][i]

            # Calculate test IC
            value_test = self.pool._normalize_by_day(expr.evaluate(self.test_data))
            ic_test = batch_pearsonr(value_test, self.target_test).mean().item()

            print(f'> Alpha #{i}: ic={ic_ret:.4f}, test_ic={ic_test:.4f}, expr={expr_str}')
        if self.pool.size > 0:
            print(f'>> Best single ic: {np.max(self.pool.single_ics[:self.pool.size]):.4f}')
        print('---------------------------------------------')

    def close(self):
        self.writer.close()


def train(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize StockData and target expression
    data = StockData(instrument=args.instrument, start_time='2010-01-01', end_time='2020-12-31', qlib_path=QLIB_PATH)
    data_test = StockData(instrument=args.instrument, start_time='2022-01-01', end_time='2024-12-31', qlib_path=QLIB_PATH)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    
    # Initialize AlphaPoolGFN
    pool = AlphaPoolGFN(capacity=args.pool_capacity, stock_data=data, target=target)

    # Initialize environment and model
    env = GFNEnvCore(pool=pool, device=device)
    
    n_tokens = len(FEATURES) + len(OPERATORS) + len(DELTA_TIMES) + len(CONSTANTS)
    
    backbone = SequenceEncoder(
        n_tokens,
        args.encoder_type
    )
    
    pf_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    pb_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0) # pb does not predict exit action
    
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)

    loss_fn = TBGFlowNet(pf=pf, pb=pb)
    loss_fn.to(device)
    sampler = Sampler(estimator=pf)
    
    params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
    optimizer = Adam(params, lr=LEARNING_RATE)


    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = os.path.join(
        'data/gfn_logs',
        f'pool_{args.pool_capacity}',
        f'gfn_{args.instrument}_{args.pool_capacity}_{args.seed}-{timestamp}'
    )
    os.makedirs(log_dir, exist_ok=True)
    logger = GFNLogger(pf, pool, log_dir, data_test, target)

    # Training loop
    losses = []
    minibatch_loss = 0
    update_freq = args.update_freq
    n_episodes = args.n_episodes

    for episode in tqdm(range(n_episodes)):
        trajectories = sampler.sample_trajectories(env=env, n_trajectories=1)
        loss = loss_fn.loss(env=env, trajectories=trajectories)

        if loss is not None and torch.isfinite(loss):
            minibatch_loss += loss
            
        if episode > 0 and episode % update_freq == 0 and minibatch_loss != 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = 0

        if episode > 0 and episode % args.log_freq == 0:
            logger.log_metrics(episode)
            logger.save_checkpoint(episode)
            logger.show_pool_state()
            print(f"----Episode {episode}/{n_episodes} done----")
            
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instrument', type=str, default='csi300')
    parser.add_argument('--pool_capacity', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--eval_prob', type=float, default=0.3)
    parser.add_argument('--update_freq', type=int, default=128)
    parser.add_argument('--n_episodes', type=int, default=2_000)
    parser.add_argument('--encoder_type', type=str, default='lstm', choices=['transformer', 'lstm', 'gnn'])
    args = parser.parse_args()
    train(args)
