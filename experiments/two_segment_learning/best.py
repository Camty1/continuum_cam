#!/bin/python3
"""
Trains [128, 128] size models on a dataset with 2^15 points (the best performance)
"""
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, random_split

from ANN import Dataset, Model, OrientationLoss, PoseLoss, PositionLoss


def train(
    dataset: Dataset, iterations: int = 10
) -> Tuple[List[Model], List[Tuple[float, float]]]:
    """
    Trains models on the given dataset

    Args:
        dataset: The training dataset
        iterations: Number of models to train

    Returns:
        A list of models, and a list of their training and validation losses
    """

    # pylint: disable-next=redefined-outer-name
    models = []
    losses = []
    power = int(np.log(len(dataset)) / np.log(2))
    for i in range(iterations):
        print(f"Power {power} | Model {i+1}")
        model = Model(
            8,
            6,
            [128, 128],
            loss=PoseLoss(),
            save_path=f"models/best/{i}.pt",
        )
        train_split, validation_split = random_split(dataset, [0.75, 0.25])

        train_dataloader = DataLoader(train_split, batch_size=64, shuffle=True)
        validation_dataloader = DataLoader(
            validation_split, batch_size=64, shuffle=True
        )

        # pylint: disable-next=redefined-outer-name
        loss = model.train(
            train_dataloader,
            validation_dataloader,
            checkpoints=True,
            save_model=True,
        )

        models.append(model)
        losses.append(loss)

    return models, losses


def test(
    dataset: Dataset, models: List[Model]  # pylint: disable=redefined-outer-name
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Tests the models on the given dataset

    Args:
        dataset: The test dataset
        models: A list of the models to test

    Returns:
        A list of the postition and oreitation losses
    """

    test_losses = []
    for model in models:

        model.loss = PositionLoss()
        pos_error = model.test_dataset(dataset)

        model.loss = OrientationLoss()
        tang_error = model.test_dataset(dataset)

        test_losses.append((pos_error, tang_error))

    return test_losses


if __name__ == "__main__":

    TRAIN = False

    # Load data
    test_dataset = Dataset("./test_data/9_clean_2024_09_21_10_02_14.dat")

    if TRAIN:
        train_dataset = Dataset("./training_data/15_2024_09_21_10_02_12.dat")

        # Train models
        models, loss = train(train_dataset)

        # Extract losses
        train_loss = np.array([x[0] for x in loss])
        validation_loss = np.array([x[1] for x in loss])

        # Output losses to files
        np.savetxt("output/best/train_loss.dat", train_loss, delimiter=",")
        np.savetxt("output/best/validation_loss.dat", validation_loss, delimiter=",")

    else:
        models = []
        for i in range(10):
            model = Model(8, 6, [128, 128])
            model.load(f"./models/best/{i}.pt")
            models.append(model)

    # Test models
    test_loss = test(test_dataset, models)

    # Extract losses
    pos_loss = np.concatenate([x[0].reshape((1, -1)) for x in test_loss], axis=0)
    tang_loss = np.concatenate([x[1].reshape((1, -1)) for x in test_loss], axis=0)

    # Output losses to files
    np.savetxt("output/best/pos_loss.dat", pos_loss, delimiter=",")
    np.savetxt("output/best/tang_loss.dat", tang_loss, delimiter=",")
