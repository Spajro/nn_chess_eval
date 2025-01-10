import math
import time
import torch


def train(train_data, test_data, model, criterion, optimizer, accuracy, epoch, checkpoint=None):
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch']
        for i, passed_time, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc in checkpoint['history']:
            print(
                f"Epoch [{i}/{epoch}],  train: {train_loss:.5f},{train_acc:.1f}    san_check: {val_loss:.5f}, {val_acc:.1f}  test: {test_loss:.5f},{test_acc:.1f},  time: {passed_time / 1000}s")
        print("Checkpoint loaded")
    else:
        start = 0
    model.train(True)
    model.cuda()
    criterion.cuda()
    for i in range(start, epoch):
        time_started = time.time() * 1000
        loss_sum = 0.0
        accuracy_sum = 0.0
        for batch, truth in train_data:
            optimizer.zero_grad()
            out = model.forward(batch).reshape(train_data.batch_size())
            loss = criterion(out, truth)
            accuracy_value = accuracy(out, truth).sum() / train_data.batch_size()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            accuracy_sum += accuracy_value.item()

        train_loss = loss_sum / len(train_data)
        train_acc = accuracy_sum / len(train_data)

        test_loss, test_acc = iterate(test_data, model, criterion, accuracy)
        val_loss, val_acc = iterate(train_data, model, criterion, accuracy)

        passed_time = math.ceil(time.time() * 1000 - time_started)
        checkpoint['history'].append(
            [i + 1, passed_time / 1000, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])
        checkpoint = {'epoch': i+1,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'history': checkpoint['history']}
        print(
            f"Epoch [{i + 1}/{epoch}],  train: {train_loss:.5f},{train_acc:.1f}    san_check: {val_loss:.5f}, {val_acc:.1f}  test: {test_loss:.5f},{test_acc:.1f},  time: {passed_time / 1000}s")


def test(data, model, criterion, accuracy):
    model.cuda()
    model.eval()
    criterion.cuda()
    time_started = time.time() * 1000
    loss_average, accuracy_average = iterate(data, model, criterion, accuracy)
    passed_time = math.ceil(time.time() * 1000 - time_started)
    print(
        f"test_loss: {loss_average}, test_accuracy: {accuracy_average}, time: {passed_time / 1000}s")


def iterate(data, model, criterion, accuracy):
    loss_sum = 0.0
    accuracy_sum = 0.0
    with torch.no_grad():
        for batch, truth in data:
            out = model.forward(batch).reshape(data.batch_size())
            loss = criterion(out, truth)
            accuracy_value = accuracy(out, truth).sum() / data.batch_size()

            loss_sum += loss.item()
            accuracy_sum += accuracy_value.item()

    loss_average = loss_sum / len(data)
    accuracy_average = accuracy_sum / len(data)
    return loss_average, accuracy_average
