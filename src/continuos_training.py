import math
import time
from collections.abc import Callable
from pathlib import Path

import torch

from src.core import log, iterate
from src.patches import CHECKPOINTS_PATCH
from src.loading.data_loading import Dataset


def train(train_data: Dataset,
          test_data: Dataset,
          model,
          criterion,
          optimizer: torch.optim.Optimizer,
          accuracy: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          epoch: int,
          device: str,
          prefix: str = "train",
          san_check: bool = True,
          checkpoint: dict = None,
          interpolate: bool = False,
          interpolation_lambda: float = 0.5,
          checkpoint_every: int = 1e4,
          save_checkpoint_every: int = 25,
          ):
    Path(CHECKPOINTS_PATCH).mkdir(parents=True, exist_ok=True)
    if checkpoint: #TODO move to point in dataset
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for i, passed_time, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc in checkpoint['history']:
            log([('train', train_loss, train_acc), ('san', val_loss, val_acc), ('test', test_loss, test_acc)],
                passed_time,
                (i, epoch))
        print("Checkpoint loaded")

    model.train(True)
    model.to(device)
    criterion.to(device)
    i=0
    while True:
        time_started = time.time() * 1000
        loss_sum = 0.0
        accuracy_sum = 0.0
        count = 0
        train_data_iter=iter(train_data)
        while count < checkpoint_every:
            batch, color, interpolation, truth = next(train_data_iter)
            optimizer.zero_grad()
            out = model.forward(batch, color).reshape(train_data.batch_size())
            if interpolate:
                out = (1.0 - interpolation_lambda) * out + interpolation_lambda * interpolation
            loss = criterion(out, truth)
            accuracy_value = accuracy(out, truth).sum() / train_data.batch_size()
            loss.backward()
            optimizer.step()
            count+=1

            loss_sum += loss.item()
            accuracy_sum += accuracy_value.item()

        i+=1
        train_loss = loss_sum / checkpoint_every
        train_acc = accuracy_sum / checkpoint_every

        test_loss, test_acc = iterate(test_data, model, criterion, accuracy)
        if san_check:
            val_loss, val_acc = iterate(train_data, model, criterion, accuracy)
        else:
            val_loss, val_acc = -1, -1

        passed_time = math.ceil(time.time() * 1000 - time_started)
        if checkpoint:
            checkpoint['history'].append(
                [i + 1, passed_time / 1000, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])
            checkpoint = {'epoch': i + 1,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'history': checkpoint['history']}
            if save_checkpoint_every > 0 and i % save_checkpoint_every == 0:
                torch.save(checkpoint, CHECKPOINTS_PATCH + prefix + "-cp-" + str(i) + '.pth')
        log([('train', train_loss, train_acc), ('san', val_loss, val_acc), ('test', test_loss, test_acc)],
            passed_time / 1000,
            (i, epoch))
