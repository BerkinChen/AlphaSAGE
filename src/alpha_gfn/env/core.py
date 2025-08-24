import random
from typing import List, Tuple, Optional
import torch
import math
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.actions import Actions

from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder, OutOfDataRangeError
from ..config import *
from ..alpha_pool import AlphaPoolGFN
from ..preprocessors import IntegerPreprocessor

class GFNEnvCore(DiscreteEnv):
    def __init__(self, pool: AlphaPoolGFN, device: torch.device = torch.device('cuda:0')):
        self.pool = pool
        self.builder = ExpressionBuilder()
        
        self.operators = [OperatorToken(op) for op in OPERATORS]
        self.features = [FeatureToken(feat) for feat in FEATURES]
        self.delta_times = [DeltaTimeToken(dt) for dt in DELTA_TIMES]
        self.constants = [ConstantToken(c) for c in CONSTANTS]
        self.action_list: List[Token] = self.operators + self.features + self.delta_times + self.constants
        self.id_to_token_map = {i: token for i, token in enumerate(self.action_list)}
        n_actions = len(self.action_list) + 1  # Add 1 for the exit action
        
        s0 = torch.tensor([self.token_to_id_map[BEG_TOKEN]] + [0] * (MAX_EXPR_LENGTH - 1), dtype=torch.long, device=device)
        sf = torch.full((MAX_EXPR_LENGTH,), -1, dtype=torch.long, device=device)
        preprocessor = IntegerPreprocessor(output_dim=MAX_EXPR_LENGTH)
        
        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            state_shape=(MAX_EXPR_LENGTH,),
            device_str=str(device),
            preprocessor=preprocessor
        )

    @property
    def token_to_id_map(self):
        # The last action is the exit action
        mapping = {token: i for i, token in enumerate(self.action_list)}
        mapping[BEG_TOKEN] = len(self.action_list) + 1 
        return mapping

    def tensor_to_tokens(self, tensor: torch.Tensor) -> List[Optional[Token]]:
        return [self.id_to_token_map.get(i.item()) for i in tensor]
        
    def step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        next_states_tensor = states.tensor.clone()
        for i, (state_tensor, action_id_tensor) in enumerate(zip(states.tensor, actions.tensor.squeeze(-1))):
            action_id = action_id_tensor.item()
            if action_id < len(self.action_list): # Not an exit action
                non_padded_len = (state_tensor != 0).sum()
                if non_padded_len < MAX_EXPR_LENGTH:
                    next_states_tensor[i, non_padded_len] = action_id
        return next_states_tensor

    def backward_step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        # Implement backward step if needed
        raise NotImplementedError

    def update_masks(self, states: DiscreteStates):
        batch_masks = []
        for state_tensor in states.tensor:
            if torch.all(state_tensor == self.sf):
                batch_masks.append([False] * self.n_actions)
                continue

            builder = ExpressionBuilder()
            token_ids = [tid.item() for tid in state_tensor if tid > 0]
            for token_id in token_ids[1:]:
                builder.add_token(self.id_to_token_map[token_id])

            valid_actions = [False] * self.n_actions
            
            n_ops = len(self.operators)
            n_features = len(self.features)
            n_dts = len(self.delta_times)

            for i, op_token in enumerate(self.operators):
                valid_actions[i] = builder.validate_op(op_token.operator)
            for i in range(len(self.features)):
                valid_actions[n_ops + i] = builder.validate_feature()
            for i in range(len(self.delta_times)):
                valid_actions[n_ops + n_features + i] = builder.validate_dt()
            for i in range(len(self.constants)):
                valid_actions[n_ops + n_features + n_dts + i] = builder.validate_const()

            if len(token_ids) < MAX_EXPR_LENGTH:
                if builder.is_valid():
                    valid_actions[-1] = True
            else:
                valid_actions[-1] = True
            
            batch_masks.append(valid_actions)
        
        states.forward_masks = torch.tensor(batch_masks, dtype=torch.bool, device=self.device)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        rewards = []
        for state_tensor in final_states.tensor:
            builder = ExpressionBuilder()
            token_ids = [tid.item() for tid in state_tensor if tid > 0]
            
            # Reconstruct the expression for reward calculation
            for token_id in token_ids[1:]:
                builder.add_token(self.id_to_token_map[token_id])

            reward = 0.0
            if builder.is_valid():
                try:
                    expr = builder.get_tree()
                    reward = self.pool.try_new_expr(expr)
                except OutOfDataRangeError:
                    reward = 0.0
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float, device=self.device)
