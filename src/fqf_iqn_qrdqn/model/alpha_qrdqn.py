from torch import nn
import torch

from .base_model import BaseModel
from fqf_iqn_qrdqn.network import DQNBase, NoisyLinear, LSTMBase


class QRDQN(BaseModel):

    def __init__(self, 
                 num_actions,
                 N=200, 
                 embedding_dim=128,
                 dueling_net=False,
                 noisy_net=False,
                 require_QCM=False):
        super(QRDQN, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        # Feature extractor of DQN.
        self.dqn_net = LSTMBase(n_actions = num_actions, 
                                embedding_dim = embedding_dim)
        # Quantile network.
        if not dueling_net:
            self.q_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, num_actions * N),
            )
        else:
            self.advantage_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, num_actions * N),
            )
            self.baseline_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, N),
            )

        self.N = N
        # self.num_channels = num_channels
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        
        if require_QCM:
            self.initilize_qcm_x()
        
    def initilize_qcm_x(self):
        taus = torch.arange(0, self.N+1, dtype=torch.float32) / self.N
        tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(self.N, 1)
        norm_dist = torch.distributions.normal.Normal(0, 1)
        qcm_x = norm_dist.icdf(tau_hats)
        qcm_X = torch.concat([torch.ones([self.N, 1]), qcm_x, qcm_x**2 - 1, qcm_x**3 - 3 * qcm_x], dim = 1)
        qcm_trans = (qcm_X.T @ qcm_X).inverse() @ qcm_X.T
        self.register_buffer('qcm_trans', qcm_trans)
        self.qcm_trans.requires_grad = False

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        if not self.dueling_net:
            quantiles = self.q_net(state_embeddings).view(batch_size, self.N, self.num_actions)
        else:
            advantages = self.advantage_net(state_embeddings).view(batch_size, self.N, self.num_actions)
            baselines = self.baseline_net(state_embeddings).view(batch_size, self.N, 1)
            quantiles = baselines + advantages - advantages.mean(dim=2, keepdim=True)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)

        return quantiles

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]

        # Calculate quantiles.
        quantiles = self(states=states, state_embeddings=state_embeddings)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q

    def calculate_higher_moments(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]
        
        quantiles = self(states=states, state_embeddings=state_embeddings)
        
        higher_moments = torch.matmul(self.qcm_trans.unsqueeze(0).repeat(batch_size, 1, 1),
                                      quantiles)
        
        assert higher_moments.shape == (batch_size, 4, self.num_actions)
        
        std = higher_moments[:, 1, :]
        skewness = 6 * higher_moments[:, 2, :]/higher_moments[:, 1, :]
        kurtosis = 24 * higher_moments[:, 3, :]/higher_moments[:, 1, :] + 3
        
        return std, skewness, kurtosis