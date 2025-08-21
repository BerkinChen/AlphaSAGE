from abc import ABC, abstractmethod
import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fqf_iqn_qrdqn.memory import LazyMultiStepMemory, \
    LazyPrioritizedMultiStepMemory
from fqf_iqn_qrdqn.utils import RunningMeanStats, LinearAnneaer


class BaseAgent(ABC):

    def __init__(self, env, data_valid, data_test, target, log_dir, num_steps=5*(10**7),
                 batch_size=32, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True, seed=0):

        self.env = env
        self.data_valid = data_valid
        self.data_test = data_test
        self.target = target

        torch.manual_seed(seed)
        np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None
        self.online_mean_net = None
        self.target_mean_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step, beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.best_test_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.use_per = use_per

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_random(self, eval=False):
        # Use e-greedy for evaluation.
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        if self.noisy_net:
            return False
        return np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self):
        # Act with randomness.
        allowed_action = self.env.action_masks()
        action = self.env.action_space.sample()
        while not allowed_action[action]:
            action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(state).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            q_values = self.online_net.calculate_q(states=state)
            forbid_action = torch.BoolTensor(
                ~self.env.action_masks()).to(self.device)
            q_values[:, forbid_action] = -1e6
            action = q_values.argmax().item()
        return action

    @abstractmethod
    def learn(self):
        pass

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))
        if (self.online_mean_net is not None) & (self.target_mean_net is not None):
            torch.save(
                self.online_mean_net.state_dict(),
                os.path.join(save_dir, 'online_mean_net.pth'))
            torch.save(
                self.target_mean_net.state_dict(),
                os.path.join(save_dir, 'target_mean_net.pth'))
        

    def load_models(self, save_dir, require_mean = False):
        self.online_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))
        
        if require_mean:
            self.online_net.load_state_dict(torch.load(
                os.path.join(save_dir, 'online_mean_net.pth')))
            self.target_net.load_state_dict(torch.load(
                os.path.join(save_dir, 'target_mean_net.pth')))

    def save_exprs(self, save_dir, valid_ic, test_ic, set_indice):
        state = self.env.env.pool.state
        n = len(state['exprs'])

        log_table = pd.DataFrame(
            columns=['exprs', 'ic', 'weight'], index=range(n + 1))

        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]

            log_table.loc[i, :] = (expr_str, ic_ret, weight)

        if set_indice == 'test':
            
            log_table.loc[n, :] = ('Ensemble', valid_ic, test_ic)
            log_table.to_csv(f'{save_dir}/test_best_table.csv')
        elif set_indice == 'valid':
            log_table.loc[n, :] = ('Ensemble', valid_ic, test_ic)
            log_table.to_csv(f'{save_dir}/valid_best_table.csv')

    # def save_agent(self, save_dir):
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     torch.save({'train_return': self.train_return, 'steps': self.steps,
    #                 'learning_steps': self.learning_steps, 'episodes': self.episodes,
    #                 'best_eval_score': self.best_eval_score, 'epsilon_train': self.epsilon_train,
    #                 'optim_online': self.optim_online.state_dict()},
    #                os.path.join(save_dir, 'agent.pkl'))

    # def load_agent(self, save_dir):
    #     checkpoint = torch.load(os.path.join(save_dir, 'agent.pkl'))
    #     self.train_return = checkpoint['train_return']
    #     self.steps = checkpoint['steps']
    #     self.learning_steps = checkpoint['learning_steps']
    #     self.episodes = checkpoint['episodes']
    #     self.best_eval_score = checkpoint['best_eval_score']
    #     self.epsilon_train = checkpoint['epsilon_train']
    #     self.optim_online.load_state_dict(checkpoint['optim_online'])

    def train_episode(self):
        # starttime = time.time()
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state, info = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better
            # performances.
            self.online_net.sample_noise()

            if self.is_random(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _, info = self.env.step(action)

            # To calculate efficiently, I just set priority=max_priority here.
            self.memory.append(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training steps.
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'ic/train', self.env.env.pool.state['best_ic_ret'], self.steps)

    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()

        if (self.steps % self.eval_interval == 0) & (len(self.env.env.pool.state['exprs']) >= 1):
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))
            # self.online_net.train()

    def evaluate(self):

        valid_ic, _ = self.env.env.pool.test_ensemble(self.data_valid, self.target)
        test_ic, _ = self.env.env.pool.test_ensemble(self.data_test, self.target)

        if valid_ic > self.best_eval_score:
            self.best_eval_score = valid_ic
            self.save_models(os.path.join(self.model_dir, 'best'))
            self.save_exprs(self.log_dir, valid_ic, test_ic, 'valid')
        
        if test_ic > self.best_test_score:
            self.best_test_score = test_ic
            self.save_exprs(self.log_dir, valid_ic, test_ic, 'test')

        # We log evaluation results along with training steps.
        self.writer.add_scalar('ic/valid', valid_ic, self.steps)
        self.writer.add_scalar('ic/test', test_ic, self.steps)

    def __del__(self):
        self.env.close()
        self.writer.close()
