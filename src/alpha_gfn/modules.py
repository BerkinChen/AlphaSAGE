import torch
import torch.nn as nn
from torch import Tensor
import math

from .config import *
from alphagen.data.tokens import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self._pe[:seq_len]

class SequenceTransformer(nn.Module):
    def __init__(self, n_tokens: int, encoder_type: str = 'lstm'):
        super().__init__()
        self.n_tokens = n_tokens
        self.beg_token_id = self.n_tokens + 1
        
        self.token_embedding = nn.Embedding(self.n_tokens + 2, HIDDEN_DIM, padding_idx=0)
        self.pos_enc = PositionalEncoding(HIDDEN_DIM)
        
        if encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=NUM_HEADS, dropout=DROPOUT, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        elif encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=HIDDEN_DIM,
                hidden_size=HIDDEN_DIM,
                num_layers=NUM_ENCODER_LAYERS,
                batch_first=True,
                dropout=DROPOUT
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _make_padding_mask(self, tokens: Tensor) -> Tensor:
        return tokens == 0
        
    def forward(self, state_tokens: Tensor):
        bs = state_tokens.shape[0]
        beg_tokens = torch.full((bs, 1), fill_value=self.beg_token_id, dtype=torch.long, device=state_tokens.device)
        
        state_tokens = torch.cat([beg_tokens, state_tokens], dim=1)
        padding_mask = self._make_padding_mask(state_tokens)
        
        state_embedding = self.pos_enc(self.token_embedding(state_tokens))
        
        if isinstance(self.encoder, nn.TransformerEncoder):
            hidden_states = self.encoder(state_embedding, src_key_padding_mask=padding_mask)
        else:
            hidden_states, _ = self.encoder(state_embedding)

        lengths = (~padding_mask).sum(dim=1)
        last_indices = lengths - 1
        batch_indices = torch.arange(state_tokens.size(0), device=state_tokens.device)
        final_hidden_states = hidden_states[batch_indices, last_indices]
        
        return final_hidden_states
