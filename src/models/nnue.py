import torch
from torch import nn

from src.loading.data_loading_halfkp import FEATURES_COUNT

M = 2 ** 7
N = 2 ** 5


class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.layer1 = nn.Linear(FEATURES_COUNT, M)
        self.layer2 = nn.Linear(2 * M, N)
        self.layer3 = nn.Linear(N, 1)
        self.classifier = nn.Sequential(self.layer1,
                                        self.layer2,
                                        self.layer3)

    def forward(self, x: list[tuple[torch.Tensor, torch.Tensor]], color: torch.Tensor) -> torch.Tensor:
        result = []
        for i in range(0, len(x)):
            result.append(self.fwd(x[i], color[i].item()))
        return torch.stack(result)

    def fwd(self, x: tuple[torch.Tensor, torch.Tensor], color: int) -> torch.Tensor:
        white = x[0].reshape(1, -1)
        black = x[1].reshape(1, -1)

        w = self.layer1(white)
        b = self.layer1(black)

        accumulator = (color * torch.cat([w, b], dim=1)) + ((1 - color) * torch.cat([b, w], dim=1))

        relu1 = torch.clamp(accumulator, 0.0, 1.0)
        relu2 = torch.clamp(self.layer2(relu1), 0.0, 1.0)
        return torch.clamp(self.layer3(relu2), 0.0, 1.0)
