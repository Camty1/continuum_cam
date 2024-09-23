#!/bin/python3
"""
Trains models on simulation data for both one and two segments, along with noise free
and noisy datasets.  Then tests the models to determine position and orientation loss.
"""
import datetime
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils_data
from ANN import Dataset, Model, OrientationLoss, PoseLoss, PositionLoss

TRAINING_ITERATIONS = 10


def train(
    dataset: Dataset,
    noise: str,
    num_segments: int,
    iteration: int,
    model_folder: str,
) -> Tuple[Model, np.ndarray, np.ndarray]:

    print(
        f"Noise: {noise}. Number of segments: {num_segments}. Iteration: {iteration+1}"
    )

    train_set, validation_set = torch.utils.data.random_split(dataset, [0.75, 0.25])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=True)

    model = Model(
        input_dim=4 * num_segments,
        output_dim=6,
        hidden_layers=[32, 32],
        loss=PoseLoss(),
        save_path=f"{model_folder}/{noise}_{num_segments}/{iteration}.pt",
    )

    train_loss, validation_loss = model.train(
        train_loader,
        validation_loader,
        checkpoints=True,
        save_model=True,
    )

    return np.array(train_loss), np.array(validation_loss), model


def training(
    iterations: int, model_folder: str, output_folder: str
) -> Dict[str, List[Model]]:

    # Input filenames
    train_clean_one_seg_filename = "training_data/clean_1_seg_2024_05_09_15_55_05.dat"
    train_noisy_one_seg_filename = "training_data/noisy_1_seg_2024_05_09_15_55_05.dat"
    train_clean_two_seg_filename = "./training_data/14_clean_2024_09_21_10_00_34.dat"
    train_noisy_two_seg_filename = "./training_data/14_2024_09_21_10_01_06.dat"

    # Data loading
    # Clean one seg
    clean_train_container_one_seg = utils_data.DataContainer()
    clean_train_container_one_seg.file_import(train_clean_one_seg_filename)
    clean_train_dataset_one_seg = Dataset()
    clean_train_dataset_one_seg.load_from_DataContainer(clean_train_container_one_seg)

    # Noisy one seg
    noisy_train_container_one_seg = utils_data.DataContainer()
    noisy_train_container_one_seg.file_import(train_noisy_one_seg_filename)
    noisy_train_dataset_one_seg = Dataset()
    noisy_train_dataset_one_seg.load_from_DataContainer(noisy_train_container_one_seg)

    # Clean two seg
    clean_train_container_two_seg = utils_data.DataContainer()
    clean_train_container_two_seg.file_import(train_clean_two_seg_filename)
    clean_train_dataset_two_seg = Dataset()
    clean_train_dataset_two_seg.load_from_DataContainer(clean_train_container_two_seg)

    # Noisy two seg
    noisy_train_container_two_seg = utils_data.DataContainer()
    noisy_train_container_two_seg.file_import(train_noisy_two_seg_filename)
    noisy_train_dataset_two_seg = Dataset()
    noisy_train_dataset_two_seg.load_from_DataContainer(noisy_train_container_two_seg)

    clean_one_seg_training = []
    clean_one_seg_validation = []
    clean_one_seg_models = []

    noisy_one_seg_training = []
    noisy_one_seg_validation = []
    noisy_one_seg_models = []

    clean_two_seg_training = []
    clean_two_seg_validation = []
    clean_two_seg_models = []

    noisy_two_seg_training = []
    noisy_two_seg_validation = []
    noisy_two_seg_models = []

    for iteration in range(TRAINING_ITERATIONS):
        training, validation, model = train(
            clean_train_dataset_one_seg, "clean", 1, iteration, model_folder
        )
        clean_one_seg_training.append(training)
        clean_one_seg_validation.append(validation)
        clean_one_seg_models.append(model)

        training, validation, model = train(
            noisy_train_dataset_one_seg, "noisy", 1, iteration, model_folder
        )
        noisy_one_seg_training.append(training)
        noisy_one_seg_validation.append(validation)
        noisy_one_seg_models.append(model)

        training, validation, model = train(
            clean_train_dataset_two_seg, "clean", 2, iteration, model_folder
        )
        clean_two_seg_training.append(training)
        clean_two_seg_validation.append(validation)
        clean_two_seg_models.append(model)

        training, validation, model = train(
            noisy_train_dataset_two_seg, "noisy", 2, iteration, model_folder
        )
        noisy_two_seg_training.append(training)
        noisy_two_seg_validation.append(validation)
        noisy_two_seg_models.append(model)

    clean_one_seg_training = np.array(clean_one_seg_training)
    clean_one_seg_validation = np.array(clean_one_seg_validation)
    noisy_one_seg_training = np.array(noisy_one_seg_training)
    noisy_one_seg_validation = np.array(noisy_one_seg_validation)
    clean_two_seg_training = np.array(clean_two_seg_training)
    clean_two_seg_validation = np.array(clean_two_seg_validation)
    noisy_two_seg_training = np.array(noisy_two_seg_training)
    noisy_two_seg_validation = np.array(noisy_two_seg_validation)

    np.savetxt(
        f"{output_folder}/clean_1_train.dat", clean_one_seg_training, delimiter=","
    )
    np.savetxt(
        f"{output_folder}/noisy_1_train.dat", noisy_one_seg_training, delimiter=","
    )
    np.savetxt(
        f"{output_folder}/clean_2_train.dat", clean_two_seg_training, delimiter=","
    )
    np.savetxt(
        f"{output_folder}/noisy_2_train.dat", noisy_two_seg_training, delimiter=","
    )
    np.savetxt(
        f"{output_folder}/clean_1_validation.dat",
        clean_one_seg_validation,
        delimiter=",",
    )
    np.savetxt(
        f"{output_folder}/noisy_1_validation.dat",
        noisy_one_seg_validation,
        delimiter=",",
    )
    np.savetxt(
        f"{output_folder}/clean_2_validation.dat",
        clean_two_seg_validation,
        delimiter=",",
    )
    np.savetxt(
        f"{output_folder}/noisy_2_validation.dat",
        noisy_two_seg_validation,
        delimiter=",",
    )

    return {
        "clean_1": clean_one_seg_models,
        "noisy_1": noisy_one_seg_models,
        "clean_2": clean_two_seg_models,
        "noisy_2": noisy_two_seg_models,
    }


def testing(models: Dict[str, List[Model]], output_folder: str):

    one_seg_test_file = "training_data/clean_1_seg_2024_05_05_14_24_57.dat"
    two_seg_test_file = "./test_data/9_clean_2024_09_21_10_02_14.dat"

    one_seg_dataset = Dataset()
    one_seg_dataset.load_from_file(one_seg_test_file)
    two_seg_dataset = Dataset()
    two_seg_dataset.load_from_file(two_seg_test_file)

    pos_dict = {}
    tang_dict = {}

    for key in models.keys():
        pos_dict[key] = []
        tang_dict[key] = []

    for key, model_list in models.items():
        if "1" in key:
            dataset = one_seg_dataset

        else:
            dataset = two_seg_dataset

        for idx, model in enumerate(model_list):
            print(f"{key}, {idx+1}")
            model.loss = PositionLoss()
            pos_dict[key].append(model.test_dataset(dataset))

            model.loss = OrientationLoss()
            tang_dict[key].append(model.test_dataset(dataset))

    for key in models.keys():
        pos_loss = np.array(pos_dict[key])
        tang_loss = np.array(tang_dict[key])

        np.savetxt(f"{output_folder}/{key}_pos.dat", pos_loss, delimiter=",")
        np.savetxt(f"{output_folder}/{key}_tang.dat", tang_loss, delimiter=",")


if __name__ == "__main__":

    TRAIN = False

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_folder = f"./models/{now}"
    os.mkdir(f"{model_folder}")
    os.mkdir(f"{model_folder}/clean_1")
    os.mkdir(f"{model_folder}/noisy_1")
    os.mkdir(f"{model_folder}/clean_2")
    os.mkdir(f"{model_folder}/noisy_2")
    output_folder = f"./output/{now}"
    os.mkdir(output_folder)

    if TRAIN:
        models = training(TRAINING_ITERATIONS, model_folder, output_folder)

    else:
        MODEL_FOLDER = "./models/2024_09_21_11_59_03"
        models = {"clean_1": [], "noisy_1": [], "clean_2": [], "noisy_2": []}

        for key in models.keys():
            for i in range(10):
                if "1" in key:
                    model = Model(4, 6, [32, 32])
                else:
                    model = Model(8, 6, [32, 32])

                model.load(f"{MODEL_FOLDER}/{key}/{i}.pt")
                models[key].append(model)

    testing(models, output_folder)
