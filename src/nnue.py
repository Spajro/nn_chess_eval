from torch import nn

from src.data_loading_halfkp import FEATURES_COUNT

x0 = 2 * FEATURES_COUNT
x1 = 2 ** 8
x2 = 2 ** 5


class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.layer1 = nn.Linear(x0, x1)
        self.layer2 = nn.Linear(x1, x2)
        self.layer3 = nn.Linear(x2, 1)
        self.classifier = nn.Sequential(self.layer1,
                                        nn.ReLU(),
                                        self.layer2,
                                        nn.ReLU(),
                                        self.layer3)

    def forward(self, X):
        return self.classifier.forward(X)