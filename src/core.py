import math
import time
import torch


def train(train_data, test_data, model, criterion, optimizer, accuracy, epoch, checkpoint=None, san_check=True):
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch']
        for i, passed_time, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc in checkpoint['history']:
            log([('train', train_loss, train_acc), ('san', val_loss, val_acc), ('test', test_loss, test_acc)],
                passed_time,
                (i, epoch))
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
        if san_check:
            val_loss, val_acc = iterate(train_data, model, criterion, accuracy)
        else:
            val_loss, val_acc = -1, -1

        passed_time = math.ceil(time.time() * 1000 - time_started)
        checkpoint['history'].append(
            [i + 1, passed_time / 1000, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])
        checkpoint = {'epoch': i + 1,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'history': checkpoint['history']}
        log([('train', train_loss, train_acc), ('san', val_loss, val_acc), ('test', test_loss, test_acc)],
            passed_time / 1000,
            (i, epoch))


def test(data, model, criterion, accuracy):
    model.cuda()
    model.eval()
    criterion.cuda()
    time_started = time.time() * 1000
    loss_average, accuracy_average = iterate(data, model, criterion, accuracy)
    passed_time = math.ceil(time.time() * 1000 - time_started)
    log([('test', loss_average, accuracy_average)], passed_time / 1000)


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


def log(data: (str, float, float), passed_time: float, epoch: (int, int) = None):
    result = ""
    if epoch:
        result += f"Epoch [{epoch[0]}/{epoch[1]}], "
    for text, loss, acc in data:
        result += f"{text}: {loss:.5f} {acc:.2f}, "
    result += f" time: {passed_time:.2f}s"
    print(result)
