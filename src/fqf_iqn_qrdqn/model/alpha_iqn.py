import torch

from .base_model import BaseModel
from fqf_iqn_qrdqn.network import DQNBase, CosineEmbeddingNetwork, QuantileNetwork, LSTMBase


class IQN(BaseModel):

    def __init__(self, num_actions, K=32, num_cosines=32,
                 embedding_dim=128, dueling_net=False, noisy_net=False,
                 require_QCM=False):
        super(IQN, self).__init__()

        # Feature extractor of DQN.
        self.dqn_net = LSTMBase(n_actions=num_actions,
                                embedding_dim=embedding_dim)
        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines, embedding_dim=embedding_dim,
                                                 noisy_net=noisy_net)
        # Quantile network.
        self.quantile_net = QuantileNetwork(
            num_actions=num_actions, dueling_net=dueling_net,
            noisy_net=noisy_net)

        self.K = K
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        
        if require_QCM:
            self.norm_dist = torch.distributions.normal.Normal(0, 1)

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        # Sample fractions.
        taus = torch.rand(
            batch_size, self.K, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q
    
    def calculate_higher_moments(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        # Sample fractions.
        taus = torch.rand(
            batch_size, self.K, dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        
        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)

        qcm_x = self.norm_dist.icdf(taus).to(state_embeddings.device)
        qcm_x = qcm_x.unsqueeze(-1)
        
        qcm_X = torch.concat([torch.ones([batch_size, self.K, 1], device = state_embeddings.device),
                              qcm_x, qcm_x**2 - 1, qcm_x**3 - 3 * qcm_x], dim = 2)
        qcm_trans = (qcm_X.mT @ qcm_X).inverse() @ qcm_X.mT
        
        higher_moments = qcm_trans @ quantiles
        
        assert higher_moments.shape == (batch_size, 4, self.num_actions)
        
        std = higher_moments[:, 1, :]
        skewness = 6 * higher_moments[:, 2, :]/higher_moments[:, 1, :]
        kurtosis = 24 * higher_moments[:, 3, :]/higher_moments[:, 1, :] + 3
        
        return std, skewness, kurtosis
        
