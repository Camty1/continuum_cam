#!/bin/python3
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
from typing import List, Tuple
import os
import numpy as np
import datetime
import ANN
import camarillo_cc
import utils_cc
import utils_data
import kinematics

TRAINING_ITERATIONS = 3

# Input filenames
data_filename = "training_data/kinematic_2024_05_09_20_11_13.dat"

# Data loading
container = utils_data.DataContainer()
container.file_import(data_filename)
dataset = ANN.Dataset()
dataset.load_from_DataContainer(container)
dataset.clean()
print(f"Removed {2**14 - len(dataset)} faulty measurements from dataset")
dataset.plot_pos()


def train():
    now = datetime.datetime.now()
    save_dir = f"models/{now.year}_{now.month:02n}_{now.day:02n}_{now.hour:02n}_{now.minute:02n}_{now.second:02n}/"
    # Training
    # Clean one seg
    split_datasets = torch.utils.data.random_split(dataset, [0.75, 0.25])
    train_dataloader = DataLoader(split_datasets[0], batch_size=64)
    validation_dataloader = DataLoader(split_datasets[1], batch_size=64)

    model = ANN.Model(
        input_dim=4,
        output_dim=6,
        hidden_layers=[32, 32],
        loss=ANN.PoseLoss(),
        save_path=f"models/{now.year}_{now.month:02n}_{now.day:02n}_{now.hour:02n}_{now.minute:02n}_{now.second:02n}.pt",
    )
    train_loss, validation_loss = model.train(
        train_dataloader,
        validation_dataloader,
        checkpoints=True,
        save_model=True,
    )

    # Loss data restructuring
    train_loss = np.array(train_loss)
    validation_loss = np.array(validation_loss)

    # Save loss as csv
    file_num = 0
    while os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath("__file__")),
            f"output/real_train_loss_{file_num}.dat",
        )
    ):
        file_num += 1

    np.savetxt(
        f"output/real_train_loss_{file_num}.dat",
        train_loss,
        delimiter=",",
    )
    np.savetxt(
        f"output/real_validation_loss_{file_num}.dat",
        validation_loss,
        delimiter=",",
    )


for i in range(TRAINING_ITERATIONS):
    print(f"TRAINING ITERATION {i+1}")
    # train()
