from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.types.common import IsolatedRNG


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    # print(f"Length of trainloader dataset: {len(trainloader.dataset)}")
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
    )

    final_epoch_per_sample_loss = 0.0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        for sequence, target in trainloader:
            sequence, target = (
                sequence.to(config.device),
                target.to(config.device),
            )
            optimizer.zero_grad()
            output = net(sequence)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            loss.backward()
            optimizer.step()

    # print(f"Final epoch per sample loss: {final_epoch_per_sample_loss}")
    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss
        / len(cast(Sized, trainloader.dataset)),
    }


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior


    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    # print(f"Length of testloader dataset: {len(testloader.dataset)}")
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.MSELoss()
    per_sample_loss = 0.0

    with torch.no_grad():
        for sequences, target in testloader:
            sequences, target = (
                sequences.to(config.device),
                target.to(config.device),
            )
            output = net(sequences)
            per_sample_loss += criterion(
                output,
                target,
            ).item()
    # print(f"Per Sample Loss: {per_sample_loss}")
    # print(f"Test Loss: {per_sample_loss / len(cast(Sized, testloader.dataset))}")
    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {},
    )


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
