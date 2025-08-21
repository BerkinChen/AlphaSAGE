from torch import nn

from .base_model import BaseModel
from fqf_iqn_qrdqn.network import NoisyLinear, LSTMBase


class MeanNetwork(BaseModel):

    def __init__(self, num_actions,
                 embedding_dim=128,
                 dueling_net=False,
                 noisy_net=False):
        super(MeanNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        # Feature extractor of DQN.
        self.dqn_net = LSTMBase(n_actions = num_actions, 
                                embedding_dim = embedding_dim)
        # Quantile network.
        if not dueling_net:
            self.q_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, num_actions),
            )
        else:
            self.advantage_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, num_actions),
            )
            self.baseline_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, 1),
            )

        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        if not self.dueling_net:
            mean_pred = self.q_net(state_embeddings).view(batch_size, 1, self.num_actions)
        else:
            advantages = self.advantage_net(state_embeddings).view(batch_size, 1, self.num_actions)
            baselines = self.baseline_net(state_embeddings).view(batch_size, 1, 1)
            mean_pred = baselines + advantages - advantages.mean(dim=2, keepdim=True)

        assert mean_pred.shape == (batch_size, 1, self.num_actions)

        return mean_pred

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]
        
        mean_pred = self(states=states, state_embeddings=state_embeddings)
        q = mean_pred.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q