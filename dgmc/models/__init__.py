from .mlp import MLP
from .gin import GIN
from .spline import SplineCNN
from .rel import RelCNN
from .dgmc import DGMC
from .dgmc_modified import DGMC_modified

__all__ = [
    'MLP',
    'GIN',
    'SplineCNN',
    'RelCNN',
    'DGMC',
    'DGMC_modified',
]
