import argparse
import pickle

import torch

from src.core import log

parser = argparse.ArgumentParser(description='Halfkp NNUE training')
parser.add_argument('name', type=str, help='checkpoint to extract')
args = parser.parse_args()

name = args.name
checkpoint = torch.load(name)

for epoch, passed_time, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc in checkpoint['history']:
    log([('train', train_loss, train_acc), ('san', val_loss, val_acc), ('test', test_loss, test_acc)],
        passed_time,
        (epoch, checkpoint['epoch']))

nname = name + '_wb.pt'

pickle.dump({
    'l1w': checkpoint['model']['layer1.weight'].tolist(),
    'l1b': checkpoint['model']['layer1.bias'].tolist(),
    'l2w': checkpoint['model']['layer2.weight'].tolist(),
    'l2b': checkpoint['model']['layer2.bias'].tolist(),
    'l3w': checkpoint['model']['layer3.weight'].tolist(),
    'l3b': checkpoint['model']['layer3.bias'].tolist(),
},
    open(nname, 'wb'))
