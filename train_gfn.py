import torch
import random
import numpy as np
import argparse
import os
import json
from datetime import datetime
from torch.optim import Adam
from torch.distributions import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from alphagen.rl.env.wrapper import action2token
from src.gfn.config import *
from src.gfn.env import GFNEnv, GFNEnvCore
from src.gfn.model import GFNet
from src.gfn.loss import trajectory_balance_loss
from src.gfn.alpha_pool import AlphaPoolGFN
from src.alphagen.data.expression import *
from src.alphagen_qlib.stock_data import StockData
from src.alphagen.utils.correlation import batch_pearsonr

QLIB_PATH = '/DATA1/home/chenbq/AlphaStruct/data/qlib_data/cn_data_rolling'

def tokens_to_tensor(tokens: list, token_to_action):
    # The `+1` is for the padding value 0
    token_indices = [token_to_action(token) + 1 for token in tokens]
    # Pad to max length
    padded_tokens = token_indices + [0] * (MAX_EXPR_LENGTH - len(token_indices))
    return torch.LongTensor(padded_tokens).unsqueeze(0)


class GFNLogger:
    def __init__(self, model: GFNet, pool: AlphaPoolGFN, log_dir: str, test_data: StockData, target: Expression):
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

    # Initialize StockData and target expression
    data = StockData(instrument=args.instrument, start_time='2010-01-01', end_time='2020-12-31', qlib_path=QLIB_PATH)
    data_test = StockData(instrument=args.instrument, start_time='2022-01-01', end_time='2024-12-31', qlib_path=QLIB_PATH)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    
    # Initialize AlphaPoolGFN
    pool = AlphaPoolGFN(capacity=args.pool_capacity, stock_data=data, target=target)

    # Initialize environment and model
    env = GFNEnv(GFNEnvCore(pool))
    model = GFNet(
        n_features=len(FEATURES),
        n_operators=len(OPERATORS),
        n_delta_times=len(DELTA_TIMES),
        n_constants=len(CONSTANTS)
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = os.path.join(
        'data/gfn_logs',
        f'pool_{args.pool_capacity}',
        f'gfn_{args.instrument}_{args.pool_capacity}_{args.seed}-{timestamp}'
    )
    os.makedirs(log_dir, exist_ok=True)
    logger = GFNLogger(model, pool, log_dir, data_test, target)

    # Training loop
    losses = []
    minibatch_loss = 0
    update_freq = 128
    n_episodes = 200_000

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        total_log_P_F = 0
        total_log_P_B = 0
        
        for t in range(MAX_EXPR_LENGTH):
            state_tensor = tokens_to_tensor(obs, env.token_to_action)
            pf_logits, pb_logits = model(state_tensor)
            mask = torch.tensor(info['valid_actions'])
            pf_logits = torch.where(mask, pf_logits, torch.tensor(-1e6))
            
            categorical = Categorical(logits=pf_logits)
            action = categorical.sample()
            total_log_P_F += categorical.log_prob(action)
            
            obs, reward, done, _, info = env.step(action.item())

            new_state_tensor = tokens_to_tensor(obs, env.token_to_action)
            _, new_pb_logits = model(new_state_tensor)
            total_log_P_B += Categorical(logits=new_pb_logits).log_prob(action)

            if done:
                break
        
        if done and reward > -1:
            log_reward = torch.log(torch.clamp(torch.tensor(reward), min=1e-9))
            minibatch_loss += trajectory_balance_loss(
                model.logZ, total_log_P_F, total_log_P_B, log_reward
            )
            
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
    parser.add_argument('--pool_capacity', type=int, default=20)
    parser.add_argument('--log_freq', type=int, default=100)
    args = parser.parse_args()
    train(args)
