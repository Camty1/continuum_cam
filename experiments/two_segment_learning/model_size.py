#!/bin/python3
"""
Trains and tests models of different size to determine the effect of model size on performance.
"""
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, random_split

from ANN import Dataset, Model, OrientationLoss, PoseLoss, PositionLoss


def train(
    dataset: Dataset,
    hidden_layers: List[float],
    iterations: int = 10,
) -> Tuple[List[Model], List[Tuple[float, float]]]:
    """
    For a given dataset and size hidden layer size, trains iterations many models

    Args:
        dataset: Training dataset
        hidden_layers: A list containing the number of neurons per hidden layer
        iterations: Number of models to train

    Returns:
        A list of the models, and a list containing the training and validation losses for each
    """

    models = []
    losses = []
    for iteration in range(iterations):
        print(f"Size {hidden_layers} | Model {iteration+1}")
        model_string = "_".join([str(x) for x in hidden_layers])
        model = Model(
            8,
            6,
            hidden_layers,
            loss=PoseLoss(),
            save_path=f"models/model_size/{model_string}_{iteration}.pt",
        )
        train_split, validation_split = random_split(dataset, [0.75, 0.25])

        train_dataloader = DataLoader(train_split, batch_size=64, shuffle=True)
        validation_dataloader = DataLoader(
            validation_split, batch_size=64, shuffle=True
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
    Tests the models on a given dataset

    Args:
        dataset: The test dataset
        models: The list of models

    Returns:
        The position and orientation loss for each model
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
        train_dataset = Dataset("./training_data/14_2024_09_21_10_01_06.dat")

        # Train models
        models_32, loss_32 = train(train_dataset, [32, 32])
        models_64, loss_64 = train(train_dataset, [64, 64])
        models_128, loss_128 = train(train_dataset, [128, 128])

        # Extract losses
        train_loss_32 = np.array([x[0] for x in loss_32])
        validation_loss_32 = np.array([x[1] for x in loss_32])

        train_loss_64 = np.array([x[0] for x in loss_64])
        validation_loss_64 = np.array([x[1] for x in loss_64])

        train_loss_128 = np.array([x[0] for x in loss_128])
        validation_loss_128 = np.array([x[1] for x in loss_128])

        # Output losses to files
        np.savetxt("output/model_size/train_loss_32.dat", train_loss_32, delimiter=",")
        np.savetxt(
            "output/model_size/validation_loss_32.dat",
            validation_loss_32,
            delimiter=",",
        )

        np.savetxt("output/model_size/train_loss_64.dat", train_loss_64, delimiter=",")
        np.savetxt(
            "output/model_size/validation_loss_64.dat",
            validation_loss_64,
            delimiter=",",
        )

        np.savetxt(
            "output/model_size/train_loss_128.dat", train_loss_128, delimiter=","
        )
        np.savetxt(
            "output/model_size/validation_loss_128.dat",
            validation_loss_128,
            delimiter=",",
        )

    else:
        models_32, models_64, models_128 = [], [], []
        for i in range(10):
            loaded_model = Model(8, 6, [32, 32])
            loaded_model.load(f"./models/model_size/32_32_{i}.pt")
            models_32.insert(0, loaded_model)
            loaded_model = Model(8, 6, [64, 64])
            loaded_model.load(f"./models/model_size/64_64_{i}.pt")
            models_64.insert(0, loaded_model)
            loaded_model = Model(8, 6, [128, 128])
            loaded_model.load(f"./models/model_size/128_128_{i}.pt")
            models_128.insert(0, loaded_model)

    # Test models
    test_loss_32 = test(test_dataset, models_32)
    test_loss_64 = test(test_dataset, models_64)
    test_loss_128 = test(test_dataset, models_128)

    # Extract losses
    pos_loss_32 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_32], axis=0)
    tang_loss_32 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_32], axis=0)

    pos_loss_64 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_64], axis=0)
    tang_loss_64 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_64], axis=0)

    pos_loss_128 = np.concatenate(
        [x[0].reshape((1, -1)) for x in test_loss_128], axis=0
    )
    tang_loss_128 = np.concatenate(
        [x[1].reshape((1, -1)) for x in test_loss_128], axis=0
    )

    # Output losses to files
    np.savetxt("output/model_size/pos_loss_32.dat", pos_loss_32, delimiter=",")
    np.savetxt("output/model_size/tang_loss_32.dat", tang_loss_32, delimiter=",")

    np.savetxt("output/model_size/pos_loss_64.dat", pos_loss_64, delimiter=",")
    np.savetxt("output/model_size/tang_loss_64.dat", tang_loss_64, delimiter=",")

    np.savetxt("output/model_size/pos_loss_128.dat", pos_loss_128, delimiter=",")
    np.savetxt("output/model_size/tang_loss_128.dat", tang_loss_128, delimiter=",")
