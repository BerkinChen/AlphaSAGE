import gymnasium as gym
from typing import List
from ..config import *
from alphagen.data.tokens import *

def token_to_action(token: Token) -> int:
    if isinstance(token, SequenceIndicatorToken):
        if token.indicator in [SequenceIndicatorType.BEG, SequenceIndicatorType.SEP]:
            return -1
    if isinstance(token, OperatorToken):
        return OPERATORS.index(token.operator)
    elif isinstance(token, FeatureToken):
        return len(OPERATORS) + FEATURES.index(token.feature)
    elif isinstance(token, DeltaTimeToken):
        return len(OPERATORS) + len(FEATURES) + DELTA_TIMES.index(token.delta_time)
    elif isinstance(token, ConstantToken):
        return len(OPERATORS) + len(FEATURES) + len(DELTA_TIMES) + CONSTANTS.index(token.constant)
    else:
        raise ValueError(f"Unknown token type: {token}")

class GFNEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        self.operators = [OperatorToken(op) for op in OPERATORS]
        self.features = [FeatureToken(feat) for feat in FEATURES]
        self.delta_times = [DeltaTimeToken(dt) for dt in DELTA_TIMES]
        self.constants = [ConstantToken(c) for c in CONSTANTS]
        self.action_list: List[Token] = self.operators + self.features + self.delta_times + self.constants

        self.action_space = gym.spaces.Discrete(len(self.action_list))
        
    def step(self, action: int):
        token = self.action_list[action]
        return self.env.step(token)

    @property
    def token_to_action(self):
        return token_to_action
