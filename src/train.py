import time
import torch
import pt_util
import torch.optim as optim
import numpy as np

from os import path
from visualizable_vgg import VisualizableVgg, VisualizableAlexNet
from data_augment import data_import


# Play around with these constants, you may find a better setting.
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 200
LEARNING_RATE = 0.01
MOMENTUM = 0.9
USE_CUDA = True
SEED = 0
PRINT_INTERVAL = 2
WEIGHT_DECAY = 0.0005
DATA_PATH = "./data/"
LOG_PATH = path.join(DATA_PATH, "log.pkl")


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    time.ctime(time.time()),
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return np.mean(losses)


def test(model, device, test_loader, return_images=False, log_interval=None):
    model.eval()
    test_loss = 0
    correct = 0

    correct_images = []
    correct_values = []

    error_images = []
    predicted_values = []
    gt_values = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            print(str(batch_idx) + "/" + str(len(test_loader)))
            data, label = data.to(device), label.to(device)
            label = label.view(-1, 1)
            output = model(data)
            test_loss_on = model.loss(output, label, reduction="sum").item()
            test_loss += test_loss_on
            pred = output.max(1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            if return_images:
                if num_correct > 0:
                    correct_images.append(data[correct_mask, ...].data.cpu().numpy())
                    correct_value_data = label[correct_mask].data.cpu().numpy()[:, 0]
                    correct_values.append(correct_value_data)
                if num_correct < len(label):
                    error_data = data[~correct_mask, ...].data.cpu().numpy()
                    error_images.append(error_data)
                    predicted_value_data = pred[~correct_mask].data.cpu().numpy()
                    predicted_values.append(predicted_value_data)
                    gt_value_data = label[~correct_mask].data.cpu().numpy()[:, 0]
                    gt_values.append(gt_value_data)
            if log_interval is not None and batch_idx % log_interval == 0:
                print(
                    "{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        time.ctime(time.time()),
                        batch_idx * len(data),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        test_loss_on,
                    )
                )
    if return_images:
        correct_images = np.concatenate(correct_images, axis=0)
        error_images = np.concatenate(error_images, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)
        correct_values = np.concatenate(correct_values, axis=0)
        gt_values = np.concatenate(gt_values, axis=0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    if return_images:
        return (
            test_loss,
            test_accuracy,
            correct_images,
            correct_values,
            error_images,
            predicted_values,
            gt_values,
        )
    else:
        return test_loss, test_accuracy


def main():
    # Now the actual training code
    use_cuda = USE_CUDA and torch.cuda.is_available()

    # torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device", device)

    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}

    class_names = [
        line.strip().split(", ")
        for line in open(path.join(DATA_PATH, "class_names.txt"))
    ]
    # name_to_class = {line[1]: line[0] for line in class_names}
    class_names = [line[1] for line in class_names]

    # TODO: add neew loader!!!
    data_train = data_import(path.join(DATA_PATH, 'flowers_train'))
    data_test = data_import(path.join(DATA_PATH, 'flowers_test'))
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs
    )

    model = VisualizableAlexNet(visualizable=False).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    start_epoch = model.load_last_model(path.join(DATA_PATH, "checkpoints"))

    train_losses, test_losses, test_accuracies = pt_util.read_log(
        LOG_PATH, ([], [], [])
    )
    test_loss, test_accuracy, correct_images, correct_val, error_images, predicted_val, gt_val = test(
        model, device, test_loader, True
    )

    correct_images = pt_util.to_scaled_uint8(correct_images.transpose(0, 2, 3, 1))
    error_images = pt_util.to_scaled_uint8(error_images.transpose(0, 2, 3, 1))
    pt_util.show_images(
        correct_images, ["correct: %s" % class_names[aa] for aa in correct_val]
    )
    pt_util.show_images(
        error_images,
        [
            "pred: %s, actual: %s" % (class_names[aa], class_names[bb])
            for aa, bb in zip(predicted_val, gt_val)
        ],
    )

    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            train_loss = train(
                model, device, train_loader, optimizer, epoch, PRINT_INTERVAL
            )
            test_loss, test_accuracy, correct_images, correct_val, error_images, predicted_val, gt_val = test(
                model, device, test_loader, True
            )
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            test_accuracies.append((epoch, test_accuracy))
            pt_util.write_log(LOG_PATH, (train_losses, test_losses, test_accuracies))
            model.save_best_model(
                test_accuracy, path.join(DATA_PATH, "checkpoints/%03d.pt") % epoch
            )

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        model.save_model(path.join(DATA_PATH, "checkpoints/%03d.pt") % epoch, 0)
        ep, val = zip(*train_losses)
        pt_util.plot(ep, val, "Train loss", "Epoch", "Error")
        ep, val = zip(*test_losses)
        pt_util.plot(ep, val, "Test loss", "Epoch", "Error")
        ep, val = zip(*test_accuracies)
        pt_util.plot(ep, val, "Test accuracy", "Epoch", "Accuracy")
        correct_images = pt_util.to_scaled_uint8(correct_images.transpose(0, 2, 3, 1))
        error_images = pt_util.to_scaled_uint8(error_images.transpose(0, 2, 3, 1))
        pt_util.show_images(
            correct_images, ["correct: %s" % class_names[aa] for aa in correct_val]
        )
        pt_util.show_images(
            error_images,
            [
                "pred: %s, actual: %s" % (class_names[aa], class_names[bb])
                for aa, bb in zip(predicted_val, gt_val)
            ],
        )


if __name__ == "__main__":
    main()
