import torch
import torch.nn as nn
from torch import Tensor

from .config import *
from alphagen.data.tokens import *

class GFNet(nn.Module):
    def __init__(self, n_features: int, n_operators: int, n_delta_times: int, n_constants: int):
        super().__init__()
        
        self.n_features = n_features
        self.n_operators = n_operators
        self.n_delta_times = n_delta_times
        self.n_constants = n_constants
        self.n_tokens = n_features + n_operators + n_delta_times + n_constants
        
        self.token_embedding = nn.Embedding(self.n_tokens + 1, HIDDEN_DIM, padding_idx=0)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        
        self.pf_head = nn.Linear(HIDDEN_DIM, self.n_tokens)
        self.pb_head = nn.Linear(HIDDEN_DIM, self.n_tokens)
        
        self.logZ = nn.Parameter(torch.tensor(0.))
        self.initial_state_embedding = nn.Parameter(torch.randn(1, HIDDEN_DIM))

    def _make_padding_mask(self, tokens: Tensor) -> Tensor:
        return tokens == 0
        
    def forward(self, state_tokens: Tensor):
        padding_mask = self._make_padding_mask(state_tokens)
        
        state_embedding = self.token_embedding(state_tokens)
        
        hidden_states = self.transformer_encoder(state_embedding, src_key_padding_mask=padding_mask)

        # Get length of each sequence in the batch
        lengths = (~padding_mask).sum(dim=1)
        
        # Create a tensor to hold the final hidden states for the heads
        final_hidden_states = torch.zeros((state_tokens.size(0), HIDDEN_DIM), device=state_tokens.device)

        # For non-empty sequences, get the hidden state of the last token
        non_empty_mask = lengths > 0
        if non_empty_mask.any():
            last_indices = lengths[non_empty_mask] - 1
            batch_indices = torch.arange(state_tokens.size(0), device=state_tokens.device)[non_empty_mask]
            final_hidden_states[non_empty_mask] = hidden_states[batch_indices, last_indices]

        # For empty sequences, use the initial state embedding
        empty_mask = ~non_empty_mask
        if empty_mask.any():
            final_hidden_states[empty_mask] = self.initial_state_embedding.expand(empty_mask.sum(), -1)

        pf_logits = self.pf_head(final_hidden_states)
        pb_logits = self.pb_head(final_hidden_states)
        
        return pf_logits, pb_logits
