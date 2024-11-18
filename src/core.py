import math
import time
import torch


def train(train_data, test_data, batch_size, model, criterion, optimizer, accuracy, epoch):
    model.train(True)
    model.cuda()
    criterion.cuda()
    size = len(train_data)
    test_size = len(test_data)
    print("Dataset size:", size)
    for i in range(epoch):
        time_started = time.time() * 1000
        loss_sum = 0.0
        accuracy_sum = 0.0
        for batch, truth in train_data:
            optimizer.zero_grad()
            out = model.forward(batch).reshape(batch_size)
            loss = criterion(out, truth)
            accuracy_value = accuracy(out, truth).sum() / batch_size
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            accuracy_sum += accuracy_value.item()

        loss_average = loss_sum / size
        accuracy_average = accuracy_sum / size

        loss_sum = 0.0
        accuracy_sum = 0.0
        with torch.no_grad():
            model.eval()
            for batch, truth in test_data:
                out = model.forward(batch).reshape(batch_size)
                loss = criterion(out, truth)
                accuracy_value = accuracy(out, truth).sum() / batch_size

                loss_sum += loss.item()
                accuracy_sum += accuracy_value.item()
            model.train(True)
            test_loss = loss_sum / test_size
            test_accuracy = accuracy_sum / test_size

        loss_sum = 0.0
        accuracy_sum = 0.0
        with torch.no_grad():
            model.eval()
            for batch, truth in train_data:
                out = model.forward(batch).reshape(batch_size)
                loss = criterion(out, truth)
                accuracy_value = accuracy(out, truth).sum() / batch_size

                loss_sum += loss.item()
                accuracy_sum += accuracy_value.item()
            model.train(True)
            val_loss = loss_sum / size
            val_acc = accuracy_sum / size

        passed_time = math.ceil(time.time() * 1000 - time_started)
        print(f"Epoch [{i + 1}/{epoch}],  train: {loss_average:.5f},{accuracy_average:.1f}    val: {val_loss:.5f}, {val_acc:.1f}  test: {test_loss:.5f},{test_accuracy:.1f},  time: {passed_time / 1000}s")


def test(data, batch_size, model, criterion, accuracy):
    model.cuda()
    model.eval()
    criterion.cuda()
    size = len(data)
    print("Dataset size:", size)
    time_started = time.time() * 1000
    loss_sum = 0.0
    accuracy_sum = 0.0
    with torch.no_grad():
        for batch, truth in data:
            out = model.forward(batch).reshape(batch_size)
            loss = criterion(out, truth)
            accuracy_value = accuracy(out, truth).sum() / batch_size

            loss_sum += loss.item()
            accuracy_sum += accuracy_value.item()

    passed_time = math.ceil(time.time() * 1000 - time_started)
    loss_average = loss_sum / size
    accuracy_average = accuracy_sum / size
    print(
        f"test_loss: {loss_average}, test_accuracy: {accuracy_average}, time: {passed_time / 1000}s")
