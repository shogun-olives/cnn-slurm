from ..utils import ProgressBar
import torch
import numpy as np


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the model on.
        epoch (int): Current epoch.

    Returns:
        Average loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in ProgressBar(
        train_loader, title=f"Epoch {epoch}", write_end=False
    ):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), 100.0 * correct / total


def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, np.array, np.array]:
    """
    Evaluate the model.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on.

    Returns:
        Average loss, accuracy, all predictions, and all labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pred, target = outputs.cpu().numpy(), targets.cpu().numpy()
    return running_loss / len(test_loader), 100.0 * correct / total, pred, target
