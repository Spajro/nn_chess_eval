import argparse

import torch
from torch import nn

from src.core import train
from src.loading.data_loading import wdl_to_cp
from src.loading.data_loading_halfkp import HalfKpDataset
from src.models.snnue import SNNUE
from src.patches import TRAIN_DATASET_PATCH, TEST_DATASET_PATCH


def accuracy(out, truth):
    return wdl_to_cp(torch.abs(truth - out))


parser = argparse.ArgumentParser(description='Halfkp NNUE training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--prefix', type=str, default='halfkp_train', help='checkpoint prefix')
parser.add_argument('--device', type=str, default='cpu', help='cuda:X or cpu')
parser.add_argument('--cp-interval', type=int, default=25, help='checkpoint interval')
parser.add_argument('--load-cp', type=str, help='name of checkpoint to load')
parser.add_argument('--san-check', action='store_true', help='test on training set after every epoch')
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
prefix = args.prefix
device = torch.device(args.device)
cp_interval = args.cp_interval
load_cp = args.load_cp
san_check = args.san_check

train_dataset = HalfKpDataset(TRAIN_DATASET_PATCH, batch_size, device)
test_dataset = HalfKpDataset(TEST_DATASET_PATCH, batch_size, device)
len(train_dataset), len(test_dataset)

model = SNNUE()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr)

LOAD_FLAG = False

if load_cp:
    checkpoint = torch.load(load_cp)
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
      epochs,
      device,
      checkpoint=checkpoint,
      save_checkpoint_every=cp_interval,
      prefix=prefix,
      san_check=san_check)

torch.save(checkpoint, prefix + '-final.pt')
