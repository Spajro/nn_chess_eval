from torch import nn

from src.models.nnue import NNUE
from src.models.snnue import SNNUE


def get_model(name: str) -> nn.Module:
    match name:
        case 'nnue':
            return NNUE()
        case 'snnue':
            return SNNUE()
        case _:
            raise NameError
