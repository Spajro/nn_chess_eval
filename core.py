import math
import time


def train(data, batch_size, model, criterion, optimizer, accuracy, epoch):
    model.cuda()
    criterion.cuda()
    size = len(data)
    print("Dataset size:", size)
    for i in range(epoch):
        time_started = time.time() * 1000
        loss_sum = 0.0
        accuracy_sum = 0.0
        for batch, truth in data:
            optimizer.zero_grad()
            out = model.forward(batch).reshape(batch_size)
            loss = criterion(out, truth)
            loss.backward()
            optimizer.step()
            accuracy_value = accuracy(out, truth).sum() / batch_size

            loss_sum += loss.item()
            accuracy_sum += accuracy_value.item()

        passed_time = math.ceil(time.time() * 1000 - time_started)
        loss_average = loss_sum / size
        accuracy_average = accuracy_sum / size
        print(
            f"Epoch [{i + 1}/{epoch}], train_loss: {loss_average}, train_accuracy: {accuracy_average}, time: {passed_time / 1000}s")


def test(data, batch_size, model, criterion, accuracy):
    model.cuda()
    criterion.cuda()
    size = len(data)
    print("Dataset size:", size)
    time_started = time.time() * 1000
    loss_sum = 0.0
    accuracy_sum = 0.0
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
