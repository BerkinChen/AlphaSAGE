import torch
from gfn.preprocessors import Preprocessor
from gfn.states import States

class IntegerPreprocessor(Preprocessor):
    "A preprocessor that returns the state tensor as is, without casting to float."
    def __init__(self, output_dim: int):
        super().__init__(output_dim=output_dim)

    def preprocess(self, states: States) -> torch.Tensor:
        return states.tensor
