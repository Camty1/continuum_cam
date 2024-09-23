#!/bin/python3
"""
Tests the effect of data size on test performance by training on datasets
with size 2^13, 2^14, and 2^15 points.
"""
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, random_split

from ANN import Dataset, Model, OrientationLoss, PoseLoss, PositionLoss


def train(
    dataset: Dataset, iterations: int = 10
) -> Tuple[List[Model], List[Tuple[float, float]]]:
    """
    Performs training of models on a given dataset

    Returns:
        A list of models, and the corresponding training and validation loss of the models
    """

    models = []
    losses = []
    power = int(np.log(len(dataset)) / np.log(2))
    for i in range(iterations):
        print(f"Power {power} | Model {i+1}")
        model = Model(
            8,
            6,
            [32, 32],
            loss=PoseLoss(),
            save_path=f"models/data_size/{power}_{i}.pt",
        )
        train_dataset, validation_dataset = random_split(dataset, [0.75, 0.25])

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        validation_dataloader = DataLoader(
            validation_dataset, batch_size=64, shuffle=True
        )

        loss = model.train(
            train_dataloader,
            validation_dataloader,
            checkpoints=True,
            save_model=True,
        )

        models.append(model)
        losses.append(loss)

    return models, losses


def test(dataset: Dataset, models: List[Model]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Tests all of the models in models using the given dataset

    Args:
        dataset: The training dataset
        models: The models

    Returns:
        A list containing the position and orientation loss of each model
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

    # Train models
    if TRAIN:
        dataset_13 = Dataset("./training_data/13_2024_09_21_10_00_02.dat")
        dataset_14 = Dataset("./training_data/14_2024_09_21_10_01_06.dat")
        dataset_15 = Dataset("./training_data/15_2024_09_21_10_02_12.dat")

        models_13, loss_13 = train(dataset_13)
        models_14, loss_14 = train(dataset_14)
        models_15, loss_15 = train(dataset_15)

        # Extract losses
        train_loss_13 = np.array([x[0] for x in loss_13])
        validation_loss_13 = np.array([x[1] for x in loss_13])

        train_loss_14 = np.array([x[0] for x in loss_14])
        validation_loss_14 = np.array([x[1] for x in loss_14])

        train_loss_15 = np.array([x[0] for x in loss_15])
        validation_loss_15 = np.array([x[1] for x in loss_15])

        # Output losses to files
        np.savetxt("output/data_size/train_loss_13.dat", train_loss_13, delimiter=",")
        np.savetxt(
            "output/data_size/validation_loss_13.dat", validation_loss_13, delimiter=","
        )

        np.savetxt("output/data_size/train_loss_14.dat", train_loss_14, delimiter=",")
        np.savetxt(
            "output/data_size/validation_loss_14.dat", validation_loss_14, delimiter=","
        )

        np.savetxt("output/data_size/train_loss_15.dat", train_loss_15, delimiter=",")
        np.savetxt(
            "output/data_size/validation_loss_15.dat", validation_loss_15, delimiter=","
        )

    else:
        models_13, models_14, models_15 = [], [], []
        for i in range(10):
            loaded_model = Model(8, 6, [32, 32])
            loaded_model.load(f"./models/data_size/13_{i}.pt")
            models_13.append(loaded_model)
            loaded_model = Model(8, 6, [32, 32])
            loaded_model.load(f"./models/data_size/14_{i}.pt")
            models_14.append(loaded_model)
            loaded_model = Model(8, 6, [32, 32])
            loaded_model.load(f"./models/data_size/15_{i}.pt")
            models_15.append(loaded_model)

    # Test models
    test_loss_13 = test(test_dataset, models_13)
    test_loss_14 = test(test_dataset, models_14)
    test_loss_15 = test(test_dataset, models_15)

    # Extract losses
    pos_loss_13 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_13], axis=0)
    tang_loss_13 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_13], axis=0)

    pos_loss_14 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_14], axis=0)
    tang_loss_14 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_14], axis=0)

    pos_loss_15 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_15], axis=0)
    tang_loss_15 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_15], axis=0)

    # Output losses to files
    np.savetxt("output/data_size/pos_loss_13.dat", pos_loss_13, delimiter=",")
    np.savetxt("output/data_size/tang_loss_13.dat", tang_loss_13, delimiter=",")

    np.savetxt("output/data_size/pos_loss_14.dat", pos_loss_14, delimiter=",")
    np.savetxt("output/data_size/tang_loss_14.dat", tang_loss_14, delimiter=",")

    np.savetxt("output/data_size/pos_loss_15.dat", pos_loss_15, delimiter=",")
    np.savetxt("output/data_size/tang_loss_15.dat", tang_loss_15, delimiter=",")
