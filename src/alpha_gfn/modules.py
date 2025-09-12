import torch
import torch.nn as nn
from torch import Tensor
import math
from torch_geometric.nn import global_mean_pool, RGCNConv
from torch_geometric.data import Data, Batch

from .config import *
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.rl.env.wrapper import action2token

def _build_graph_from_rpn(token_ids: list[int], token_embedding_layer: nn.Embedding, beg_token_id: int) -> Data:
    device = token_embedding_layer.weight.device
    tokens = []
    for tid in token_ids:
        if tid == beg_token_id:
            tokens.append(BEG_TOKEN)
        else:
            tokens.append(action2token(tid))
    
    edges = []
    edge_types = []
    stack = []
    UNARY_OPS = ['Abs', 'SLog1p', 'Inv', 'Sign', 'Log', 'Rank']
    COMMUTATIVE_OPS = ['Add', 'Mul']
    NON_COMMUTATIVE_OPS = ['Sub', 'Div', 'Pow', 'Greater', 'Less']
    ROLLING_OPS = ['Ref', 'TsMean', 'TsSum', 'TsStd', 'TsIr', 'TsMinMaxDiff', 'TsMaxDiff', 'TsMinDiff', 'TsVar', 'TsSkew', 'TsKurt', 'TsMax', 'TsMin',
        'TsMed', 'TsMad', 'TsRank', 'TsDelta', 'TsDiv', 'TsPctChange', 'TsWMA', 'TsEMA']
    PAIR_ROLLING_OPS = ['TsCov', 'TsCorr']

    for j, token in enumerate(tokens):
        if isinstance(token, OperatorToken):
            op = token.operator
            n_args = op.n_args()
            if len(stack) < n_args: continue
            
            for i in range(n_args):
                child_node_idx = stack.pop()
                edges.append((child_node_idx, j))
                
                if op.__name__ in UNARY_OPS:
                    edge_types.append(0)  # Unary operand
                elif op.__name__ in COMMUTATIVE_OPS:
                    edge_types.append(1)  # Commutative binary operand
                elif op.__name__ in NON_COMMUTATIVE_OPS:  # Non-commutative binary
                    # The first pop (i=0) is the right-most operand in an expression
                    if i == 0:
                        edge_types.append(3)  # Right operand
                    else:
                        edge_types.append(2)  # Left operand
                elif op.__name__ in ROLLING_OPS or op.__name__ in PAIR_ROLLING_OPS:
                    if i == 0:
                        edge_types.append(5)  # Time series operand
                    else:
                        edge_types.append(4)  # Feature operand
                else:
                    raise TypeError(f"Unknown operator: {op.__name__}")
        stack.append(j)

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
    
    node_feature_ids = torch.tensor(token_ids, device=device)
    x = token_embedding_layer(node_feature_ids)
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)

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

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.num_relations = 6  # 0: unary, 1: commutative, 2: left, 3: right, 4: feature, 5: time series
        self.layers = nn.ModuleList()
        self.layers.append(RGCNConv(input_dim, hidden_dim, self.num_relations))
        for _ in range(n_layers - 1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, self.num_relations))
        
    def forward(self, data: Batch):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index, edge_type))
        
        return global_mean_pool(x, batch)

class SequenceEncoder(nn.Module):
    def __init__(self, n_tokens: int, encoder_type: str = 'lstm'):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_tokens = n_tokens
        self.beg_token_id = 0
        
        # Reserve one extra id for padding; use a valid non-negative padding index
        self.padding_id = self.n_tokens + 1
        self.token_embedding = nn.Embedding(self.n_tokens + 2, HIDDEN_DIM, padding_idx=self.padding_id)
        
        if encoder_type == 'transformer':
            self.pos_enc = PositionalEncoding(HIDDEN_DIM)
            encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=NUM_HEADS, dropout=DROPOUT, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        elif encoder_type == 'lstm':
            self.pos_enc = PositionalEncoding(HIDDEN_DIM)
            self.encoder = nn.LSTM(
                input_size=HIDDEN_DIM,
                hidden_size=HIDDEN_DIM,
                num_layers=NUM_ENCODER_LAYERS,
                batch_first=True,
                dropout=DROPOUT
            )
        elif encoder_type == 'gnn':
            self.encoder = GNNEncoder(input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _make_padding_mask(self, tokens: Tensor) -> Tensor:
        return tokens == -1
        
    def forward(self, state_tokens: Tensor):
        bs = state_tokens.shape[0]

        if self.encoder_type == 'gnn':
            data_list = []
            for i in range(bs):
                token_ids = [tid for tid in state_tokens[i].tolist() if tid > -1]
                graph_data = _build_graph_from_rpn(
                    token_ids,
                    self.token_embedding,
                    self.beg_token_id
                )
                data_list.append(graph_data)
            
            batched_graph = Batch.from_data_list(data_list)
            return self.encoder(batched_graph)

        else:
            beg_tokens = torch.full((bs, 1), fill_value=self.beg_token_id, dtype=torch.long, device=state_tokens.device)
            
            state_tokens = torch.cat([beg_tokens, state_tokens], dim=1)
            padding_mask = self._make_padding_mask(state_tokens)
            
            # Remap -1 paddings to a valid padding id before embedding to avoid CUDA asserts
            state_tokens_for_embedding = state_tokens.clone()
            state_tokens_for_embedding[state_tokens_for_embedding == -1] = self.padding_id
            
            state_embedding = self.pos_enc(self.token_embedding(state_tokens_for_embedding))
            
            if isinstance(self.encoder, nn.TransformerEncoder):
                hidden_states = self.encoder(state_embedding, src_key_padding_mask=padding_mask)
            else:
                hidden_states, _ = self.encoder(state_embedding)

            lengths = (~padding_mask).sum(dim=1)
            last_indices = lengths - 1
            batch_indices = torch.arange(state_tokens.size(0), device=state_tokens.device)
            final_hidden_states = hidden_states[batch_indices, last_indices]
            
            return final_hidden_states
