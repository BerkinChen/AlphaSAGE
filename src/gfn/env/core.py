from typing import List, Tuple, Optional
import gymnasium as gym
import torch
import math
from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder, OutOfDataRangeError
from ..config import *
from ..alpha_pool import AlphaPoolGFN


class GFNEnvCore(gym.Env):
    def __init__(self, pool: AlphaPoolGFN, device: torch.device = torch.device('cuda:0')):
        super().__init__()
        self._device = device
        self.pool = pool
        self.eval_cnt = 0
        self.reset()

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        self._tokens: List[Token] = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.
        truncated = False
        self.pool.record_reward.append(reward)
        return self._tokens, reward, done, truncated, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        print(expr)
        try:
            ret = self.pool.try_new_expr(expr)
            self.eval_cnt += 1
            return ret
        except OutOfDataRangeError:
            return 0.

    def _valid_action_types(self) -> dict:
        n_ops = len(OPERATORS)
        n_features = len(FEATURES)
        n_dts = len(DELTA_TIMES)
        n_consts = len(CONSTANTS)
        
        valid_actions = [False] * (n_ops + n_features + n_dts + n_consts)
        
        for i, op in enumerate(OPERATORS):
            valid_actions[i] = self._builder.validate_op(op)
            
        for i in range(n_features):
            valid_actions[n_ops + i] = self._builder.validate_feature()

        for i in range(n_dts):
            valid_actions[n_ops + n_features + i] = self._builder.validate_dt()
            
        for i in range(n_consts):
            valid_actions[n_ops + n_features + n_dts + i] = self._builder.validate_const()
            
        return {'valid_actions': valid_actions}

    def render(self, mode='human'):
        pass
