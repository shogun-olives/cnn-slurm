from ..utils import get_module
from torchvision import models
from torch import nn
from torch import optim


def get_model(name: str, depth: int, num_classes: int) -> nn.Module:
    """
    Load the model.

    Args:
        name (str): The name of the model.
        depth (int): The depth of the input image.
        size (tuple[int, int]): The size of the input image.
        num_classes (int): The number of classes.

    Returns:
        nn.Module: The model.
    """
    model_module = get_module(models, name)
    weights = get_module(models, f"{name}_Weights")
    model = model_module(weights=weights.DEFAULT)

    # Modify input layer to match input dimensions
    old_conv1 = model.conv1

    model.conv1 = nn.Conv2d(
        depth,
        old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=old_conv1.bias is not None,
    )

    # remove maxpool layer
    model.maxpool = nn.Identity()

    # Modify final fully connected layer for specified number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_optimizer(model: nn.Module, name: str, **kwargs) -> optim.Optimizer:
    """
    Get the optimizer.

    Args:
        model (nn.Module): The model.
        name (str): The name of the optimizer.
        **kwargs: Additional optimizer arguments.

    Returns:
        optim.Optimizer: The optimizer.
    """
    # get and set optimizer
    opt_module = get_module(optim, name)
    return opt_module(model.parameters(), **kwargs)


def get_scheduler(
    optimizer: optim.Optimizer, name: str, **kwargs
) -> optim.lr_scheduler._LRScheduler:
    """
    Get the learning rate scheduler.

    Args:
        optimizer (optim.Optimizer): The optimizer.
        name (str): The name of the scheduler.
        **kwargs: Additional scheduler arguments.

    Returns:
        optim.lr_scheduler._LRScheduler: The scheduler.
    """
    # get and set scheduler
    sched_module = get_module(optim.lr_scheduler, name)
    return sched_module(optimizer, **kwargs)
