import torch
from torch import nn
from gfn.preprocessors import Preprocessor
from gfn.states import States


class IntegerPreprocessor(Preprocessor):
    "A preprocessor that returns the state tensor as is, without casting to float."
    def __init__(self, output_dim: int):
        super().__init__(output_dim=output_dim)

    def preprocess(self, states: States) -> torch.Tensor:
        return states.tensor


class EncoderPreprocessor(Preprocessor):
    """
    A preprocessor that wraps a SequenceEncoder to transform integer token states
    into hidden representations suitable for policy heads.
    """
    def __init__(self, encoder: nn.Module, output_dim: int):
        super().__init__(output_dim=output_dim)
        self.encoder = encoder

    def preprocess(self, states: States) -> torch.Tensor:
        return self.encoder(states.tensor)
