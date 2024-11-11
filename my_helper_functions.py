import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

def plot_decision_boundary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    """
    Plots decision boundaries of model predicting on x in comparison to y.
    Source :
      - https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py (with modifications)
      - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    x, y = x.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_0_min, x_0_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    x_1_min, x_1_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    x0x, x1x = np.meshgrid(np.linspace(x_0_min, x_0_max, 101), np.linspace(x_1_min, x_1_max, 101))

    # Make features
    x_to_pred_on = torch.from_numpy(np.column_stack((x0x.ravel(), x1x.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(x_to_pred_on)

    softmax = nn.Softmax(dim=1)
    y_log_proba = softmax(y_logits)
    y_pred = torch.argmax(y_log_proba, dim=1)

    # Reshape preds and plot
    y_pred = y_pred.reshape(x0x.shape).detach().numpy()
    plt.contourf(x0x, x1x, y_pred, alpha=0.6)
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.6, s=40)
    plt.xlim(x0x.min(), x0x.max())
    plt.ylim(x1x.min(), x1x.max())


def accuracy_fn(y_pred, y_true):
    """
    Calculates accuracy between predictions and truth labels.
    """
    try:
        if (len(y_pred) != len(y_true)):
            raise ValueError("Size Error")
    except ValueError:
        print("y_pred and y_true have not the same size !")
    else:
        size = len(y_pred)
        correct = torch.eq(y_pred, y_true).sum().item()
        acc = (correct / size) * 100
        return acc


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    train_loss, train_acc = 0, 0
    softmax = nn.Softmax(dim=1)

    for batch, (x, y) in enumerate(dataloader):
        model.train()
        # Envoyer les data au GPU
        x, y = x.to(device), y.to(device)

        y_pred_logits = model(x)
        # Calcul de la loss
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss

        # Calcul de l'accuracy
        y_log_proba = softmax(y_pred_logits)
        y_pred = torch.argmax(y_log_proba, dim=1)
        accuracy = accuracy_fn(y_true=y, y_pred=y_pred)
        train_acc += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calcul de la loss/acc moyenne par batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return (train_loss, train_acc)


def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    test_loss, test_acc = 0, 0
    softmax = nn.Softmax(dim=1)

    model.eval()
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred_logits = model(x)
            # Calcul de la loss
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss

            # Calcul de l'accuracy
            y_log_proba = softmax(y_pred_logits)
            y_pred = torch.argmax(y_log_proba, dim=1)
            accuracy = accuracy_fn(y_true=y, y_pred=y_pred)
            test_acc += accuracy

        # Calcul de la loss/acc moyenne par batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return (test_loss, test_acc)


def plot_training_data(typology: str, train_data: list, test_data: list):
    plt.figure()
    plt.title(f"{typology}")
    plt.plot(range(len(train_data)), train_data, label=f"Training {typology}")
    plt.plot(range(len(test_data)), test_data, label=f"Test {typology}")
    plt.xlabel("epochs")
    plt.ylabel(f"{typology}")
    plt.legend()
    plt.show()