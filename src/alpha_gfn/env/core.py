import random
from typing import List, Tuple, Optional
import torch
import numpy as np
import math
from torch import nn
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.actions import Actions

from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder, OutOfDataRangeError
from ..config import *
from ..alpha_pool import AlphaPoolGFN
from ..preprocessors import IntegerPreprocessor

class GFNEnvCore(DiscreteEnv):
    def __init__(self, pool: AlphaPoolGFN, 
                 encoder: nn.Module = None, 
                 device: torch.device = torch.device('cuda:0'), 
                 mask_dropout_prob: float = 0.1,
                 ssl_weight: float = 0.1,
                 nov_weight: float = 0.1):
        self.pool = pool
        self.encoder = encoder
        self.mask_dropout_prob = mask_dropout_prob
        self.ssl_weight = ssl_weight
        self.nov_weight = nov_weight
        self.builder = ExpressionBuilder()
        
        self.beg_token = [BEG_TOKEN]
        self.operators = [OperatorToken(op) for op in OPERATORS]
        self.features = [FeatureToken(feat) for feat in FEATURES]
        self.delta_times = [DeltaTimeToken(dt) for dt in DELTA_TIMES]
        self.constants = [ConstantToken(c) for c in CONSTANTS]
        self.sep_token = [SEP_TOKEN]
        self.action_list: List[Token] = self.beg_token + self.operators + self.features + self.delta_times + self.constants + self.sep_token
        self.id_to_token_map = {i: token for i, token in enumerate(self.action_list)}
        n_actions = len(self.action_list)
        print(self.token_to_id_map)
        s0 = torch.tensor([self.token_to_id_map[BEG_TOKEN]] + [-1] * (MAX_EXPR_LENGTH - 1), dtype=torch.long, device=device)
        # Sink state: a special state that represents completed trajectories
        # We use -2 to distinguish it from the padding value -1
        sf = torch.full((MAX_EXPR_LENGTH,), self.token_to_id_map[SEP_TOKEN], dtype=torch.long, device=device)
        preprocessor = IntegerPreprocessor(output_dim=MAX_EXPR_LENGTH)
        
        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            state_shape=(MAX_EXPR_LENGTH,),
            dummy_action=torch.tensor([-1], dtype=torch.long, device=device),
            exit_action=torch.tensor([self.token_to_id_map[SEP_TOKEN]], dtype=torch.long, device=device),
            device_str=str(device),
            preprocessor=preprocessor
        )

    @property
    def token_to_id_map(self):
        # The last action is the exit action
        mapping = {token: i for i, token in enumerate(self.action_list)}
        return mapping

    def set_encoder(self, encoder: nn.Module):
        self.encoder = encoder

    def tensor_to_tokens(self, tensor: torch.Tensor) -> List[Optional[Token]]:
        return [self.id_to_token_map.get(i.item()) for i in tensor]
        
    def step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        next_states_tensor = states.tensor.clone()
        for i, (state_tensor, action_id_tensor) in enumerate(zip(states.tensor, actions.tensor.squeeze(-1))):
            action_id = action_id_tensor.item()
            if self.id_to_token_map[action_id] == SEP_TOKEN: # Exit action - transition to sink state
                next_states_tensor[i] = self.sf
            else: # Not an exit action
                non_padded_len = (state_tensor != -1).sum()
                if non_padded_len < MAX_EXPR_LENGTH:
                    next_states_tensor[i, non_padded_len] = action_id
        return next_states_tensor

    def backward_step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        # Implement backward step if needed
        raise NotImplementedError

    def update_masks(self, states: DiscreteStates):
        batch_masks = []
        for state_tensor in states.tensor:
            if torch.all(state_tensor == self.sf):  # This is a sink state
                batch_masks.append([False] * self.n_actions)
                continue

            builder = ExpressionBuilder()
            token_ids = [tid.item() for tid in state_tensor if tid >= 0]
            for token_id in token_ids[1:]:
                builder.add_token(self.id_to_token_map[token_id])

            valid_actions = [False] * self.n_actions
            
            # Account for BEG_TOKEN at index 0
            beg_offset = len(self.beg_token)  # = 1
            n_ops = len(self.operators)
            n_features = len(self.features)
            n_dts = len(self.delta_times)

            for i, op_token in enumerate(self.operators):
                valid_actions[beg_offset + i] = builder.validate(op_token)
            for i, feature_token in enumerate(self.features):
                valid_actions[beg_offset + n_ops + i] = builder.validate(feature_token)
            for i, dt_token in enumerate(self.delta_times):
                valid_actions[beg_offset + n_ops + n_features + i] = builder.validate(dt_token)
            for i, const_token in enumerate(self.constants):
                valid_actions[beg_offset + n_ops + n_features + n_dts + i] = builder.validate(const_token)

            if len(token_ids) < MAX_EXPR_LENGTH:
                if builder.is_valid():
                    valid_actions[-1] = True
            else:
                valid_actions[-1] = True
            
            # Apply random mask dropout based on expression length
            # Longer expressions get higher dropout probability
            expr_length = len(token_ids)
            length_based_dropout_prob = self.mask_dropout_prob * (expr_length / MAX_EXPR_LENGTH)
            
            # set True actions to False (except the last action which is SEP)
            true_indices = [i for i in range(len(valid_actions) - 1) if valid_actions[i]]  # Exclude last action
            if valid_actions[-1] == True:
                if np.random.rand() < length_based_dropout_prob:
                    for idx in true_indices:
                        valid_actions[idx] = False
                    #print(f"[Mask Debug] Expr length: {expr_length}, Masked all {len(true_indices)} actions")
            

            
            batch_masks.append(valid_actions)
        
        states.forward_masks = torch.tensor(batch_masks, dtype=torch.bool, device=self.device)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        rewards = []
        for state_tensor in final_states.tensor:
            builder = ExpressionBuilder()
            token_ids = [tid.item() for tid in state_tensor if tid >= 0]
            
            # Reconstruct the expression for reward calculation
            for token_id in token_ids[1:]:
                builder.add_token(self.id_to_token_map[token_id])

            reward = 0.0
            if builder.is_valid():
                try:
                    expr = builder.get_tree()
                    
                    # Compute embedding only for this valid expression
                    embedding = None
                    if self.encoder is not None:
                        with torch.no_grad():
                            # Add batch dimension for single state
                            single_state = state_tensor.unsqueeze(0)
                            embedding = self.encoder(single_state).squeeze(0)
                    ic_reward, nov_reward, ssl_reward = self.pool.try_new_expr_with_ssl(expr, embedding)
                    reward = ic_reward + self.ssl_weight * ssl_reward + self.nov_weight * nov_reward
                except OutOfDataRangeError:
                    reward = 0.0
            rewards.append(np.maximum(reward, np.exp(-10)))
        
        return torch.tensor(rewards, dtype=torch.float, device=self.device)

