import torch
from torch import nn

from src.core import train
from src.data_loading import wdl_to_cp
from src.data_loading_halfkp import HalfKpDataset, M

BATCH_SIZE = 64

from src.patches import TRAIN_DATASET_PATCH, TEST_DATASET_PATCH

train_dataset = HalfKpDataset(TRAIN_DATASET_PATCH, BATCH_SIZE)
test_dataset = HalfKpDataset(TEST_DATASET_PATCH, BATCH_SIZE)
len(train_dataset), len(test_dataset)


def accuracy(out, truth):
    return wdl_to_cp(torch.abs(truth - out))


x0 = 2 * M
x1 = 2 ** 8
x2 = 2 ** 5


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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


model = Model()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001)

LOAD_FLAG = False

if LOAD_FLAG:
    checkpoint = torch.load('halfkp_checkpoint.pt')
else:
    checkpoint = {'epoch': 0,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'history': []}

train(train_dataset,
      test_dataset,
      model,
      nn.MSELoss(),
      optimizer,
      accuracy,
      300,
      checkpoint)

torch.save(checkpoint, 'halfkp_checkpoint.pt')

torch.save([[model.layer1.weight.tolist(), model.layer1.bias.tolist()],
            [model.layer2.weight.tolist(), model.layer2.bias.tolist()],
            [model.layer3.weight.tolist(), model.layer3.bias.tolist()]],
           "halfkp_wb.pt")
