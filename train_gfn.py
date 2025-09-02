import torch
import random
import numpy as np
import argparse
import os
import json
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, PolynomialLR
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
from src.alpha_gfn.gflownet import EntropyTBGFlowNet

from gfn.samplers import Sampler
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.modules import NeuralNet


QLIB_PATH = '/DATA1/home/chenbq/AlphaStruct/data/qlib_data/cn_data_rolling'



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


class WeightScheduler:
    """A scheduler for managing weight decay using PyTorch schedulers"""
    
    def __init__(self, initial_ssl_weight, initial_nov_weight, final_ratio, total_steps, scheduler_type='linear'):
        self.initial_ssl_weight = initial_ssl_weight
        self.initial_nov_weight = initial_nov_weight
        self.final_ssl_weight = initial_ssl_weight * final_ratio
        self.final_nov_weight = initial_nov_weight * final_ratio
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        
        # Create dummy parameters and optimizers for using PyTorch schedulers
        self.ssl_param = nn.Parameter(torch.tensor(initial_ssl_weight))
        self.nov_param = nn.Parameter(torch.tensor(initial_nov_weight))
        
        # Create dummy optimizers (we won't actually use them for optimization)
        self.ssl_optimizer = Adam([self.ssl_param], lr=1.0)
        self.nov_optimizer = Adam([self.nov_param], lr=1.0)
        
        # Create schedulers based on type
        if scheduler_type == 'linear':
            # LinearLR decays from start_factor to end_factor
            start_factor = 1.0
            end_factor = final_ratio
            self.ssl_scheduler = LinearLR(self.ssl_optimizer, start_factor=start_factor, 
                                        end_factor=end_factor, total_iters=total_steps)
            self.nov_scheduler = LinearLR(self.nov_optimizer, start_factor=start_factor, 
                                        end_factor=end_factor, total_iters=total_steps)
        elif scheduler_type == 'exponential':
            # ExponentialLR multiplies by gamma each step
            gamma = (final_ratio) ** (1.0 / total_steps)
            self.ssl_scheduler = ExponentialLR(self.ssl_optimizer, gamma=gamma)
            self.nov_scheduler = ExponentialLR(self.nov_optimizer, gamma=gamma)
        elif scheduler_type == 'polynomial':
            # PolynomialLR with power=1 is linear, power=2 is quadratic, etc.
            self.ssl_scheduler = PolynomialLR(self.ssl_optimizer, total_iters=total_steps, 
                                           power=2.0, end_factor=final_ratio)
            self.nov_scheduler = PolynomialLR(self.nov_optimizer, total_iters=total_steps, 
                                           power=2.0, end_factor=final_ratio)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self):
        """Step both schedulers"""
        self.ssl_scheduler.step()
        self.nov_scheduler.step()
    
    def get_current_weights(self):
        """Get current weight values"""
        ssl_lr = self.ssl_scheduler.get_last_lr()[0]
        nov_lr = self.nov_scheduler.get_last_lr()[0]
        
        current_ssl_weight = self.initial_ssl_weight * ssl_lr
        current_nov_weight = self.initial_nov_weight * nov_lr
        
        return current_ssl_weight, current_nov_weight

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

    # Initialize model
    n_tokens = len(FEATURES) + len(OPERATORS) + len(DELTA_TIMES) + len(CONSTANTS)
    
    backbone = SequenceEncoder(
        n_tokens,
        args.encoder_type
    )
    
    # Initialize environment with encoder and mask dropout
    env = GFNEnvCore(pool=pool,
                     encoder=backbone, 
                     device=device, 
                     mask_dropout_prob=args.mask_dropout_prob,
                     ssl_weight=args.ssl_weight,
                     nov_weight=args.nov_weight)
    
    pf_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    pb_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0) # pb does not predict exit action
    
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)

    loss_fn = EntropyTBGFlowNet(
        pf=pf,
        pb=pb,
        entropy_coef=args.entropy_coef,
        entropy_temperature=args.entropy_temperature
    )
    loss_fn.to(device)
    sampler = Sampler(estimator=pf)
    
    params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
    optimizer = Adam(params, lr=LEARNING_RATE)


    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = os.path.join(
        'data/gfn_logs',
        f'pool_{args.pool_capacity}',
        f'gfn_{args.encoder_type}_{args.instrument}_{args.pool_capacity}_{args.seed}-{args.entropy_coef}-{args.entropy_temperature}-{args.ssl_weight}-{args.nov_weight}-{args.weight_decay_type}-{args.final_weight_ratio}'
    )
    os.makedirs(log_dir, exist_ok=True)
    logger = GFNLogger(pf, pool, log_dir, data_test, target)

    # Training loop
    losses = []
    minibatch_loss = 0
    update_freq = args.update_freq
    n_episodes = args.n_episodes
    
    # Initialize weight scheduler using PyTorch schedulers
    weight_scheduler = WeightScheduler(
        initial_ssl_weight=args.ssl_weight,
        initial_nov_weight=args.nov_weight,
        final_ratio=args.final_weight_ratio,
        total_steps=n_episodes,
        scheduler_type=args.weight_decay_type
    )
    
    print(f"Weight decay strategy: {args.weight_decay_type} (using PyTorch schedulers)")
    print(f"SSL weight: {args.ssl_weight:.4f} -> {args.ssl_weight * args.final_weight_ratio:.4f}")
    print(f"Novelty weight: {args.nov_weight:.4f} -> {args.nov_weight * args.final_weight_ratio:.4f}")
    print(f"Training episodes: {n_episodes}")
    print("=" * 50)

    for episode in tqdm(range(n_episodes)):
        # Get current weights from scheduler
        current_ssl_weight, current_nov_weight = weight_scheduler.get_current_weights()
        
        # Update environment weights
        env.ssl_weight = current_ssl_weight
        env.nov_weight = current_nov_weight
        
        save_estimator_outputs = args.entropy_coef > 0
        trajectories = sampler.sample_trajectories(
            env=env, n_trajectories=1, save_estimator_outputs=save_estimator_outputs
        )
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
            print(f"Current weights: SSL={current_ssl_weight:.4f}, Novelty={current_nov_weight:.4f}")
            
            # Log weight decay to tensorboard
            logger.writer.add_scalar('weights/ssl_weight', current_ssl_weight, episode)
            logger.writer.add_scalar('weights/nov_weight', current_nov_weight, episode)
        
        # Step the weight scheduler at the end of each episode
        weight_scheduler.step()
    # last save
    print(f"Saving checkpoint at episode {n_episodes - 1}")
    logger.save_checkpoint(n_episodes - 1)
    logger.show_pool_state()
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instrument', type=str, default='csi300')
    parser.add_argument('--pool_capacity', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=1000)
    parser.add_argument('--update_freq', type=int, default=128)
    parser.add_argument('--n_episodes', type=int, default=1_000)
    parser.add_argument('--encoder_type', type=str, default='lstm', choices=['transformer', 'lstm', 'gnn'])
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Coefficient for entropy regularization')
    parser.add_argument('--entropy_temperature', type=float, default=1.0, help='Temperature for entropy calculation')
    parser.add_argument('--mask_dropout_prob', type=float, default=0.5, help='Probability of masking out valid actions based on expression length')
    parser.add_argument('--ssl_weight', type=float, default=0.5, help='Initial weight for SSL reward (will decay during training)')
    parser.add_argument('--nov_weight', type=float, default=0.5, help='Initial weight for novelty reward (will decay during training)')
    parser.add_argument('--weight_decay_type', type=str, default='linear', choices=['linear', 'exponential', 'polynomial'], help='Type of weight decay to apply')
    parser.add_argument('--final_weight_ratio', type=float, default=0.0, help='Final weight as ratio of initial weight (e.g., 0.1 means decay to 10% of initial)')
    args = parser.parse_args()
    train(args)
