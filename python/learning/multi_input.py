import datetime
import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils_data
from torch import nn
from torch.utils.data import DataLoader

from ANN import Dataset, Model


class MultiInputModel(Model):

    def __init__(
        self,
        num_cables: int,
        num_coils: int,
        num_previous_inputs: int,
        hidden_layers: List[int],
        loss: nn.Module = nn.MSELoss(),
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module = None,
        lr: float = 1e-3,
        checkpoints_path: str = None,
        save_path: str = None,
    ):
        self.num_cables = num_cables
        self.num_coils = num_coils
        self.num_previous_inputs = num_previous_inputs

        super().__init__(
            (self.num_previous_inputs + 1) * self.num_cables,
            6 * self.num_coils,
            hidden_layers,
            loss,
            activation,
            output_activation,
            lr,
            checkpoints_path,
            save_path,
        )


class MultiInputDataset(Dataset):

    def __init__(self, num_previous_inputs: int, filename: str = None):
        self.num_previous_inputs = num_previous_inputs

        super().__init__(filename)

    def load_from_file(self, filename: str):
        """
        Imports data from a file.

        Loads file into a DataContainer from utils_data, and then loads from the container.

        Args:
            filename: The name of the file to load from
        """
        data = utils_data.DataContainer()
        data.file_import(filename)
        self.load_from_DataContainer(data)

    def load_from_DataContainer(self, data: utils_data.DataContainer):
        """
        Imports data from a DataContainer, setting all properties of the object.

        Args:
            data: a DataContainer object
        """
        self.date = data.date
        self.time = data.time
        self.num_cables = data.num_cables
        self.num_coils = data.num_coils
        self.num_measurements = data.num_measurements - self.num_previous_inputs

        # Convert numpy arrays to tensors and put them in the input and output arrays

        inputs = []
        for i in range(self.num_previous_inputs, len(data.inputs)):

            if i == self.num_previous_inputs:
                inputs.append(
                    torch.from_numpy(np.stack(data.inputs[i::-1]).flatten()).to(
                        self.device
                    )
                )
            else:
                inputs.append(
                    torch.from_numpy(
                        np.stack(
                            data.inputs[i : i - self.num_previous_inputs - 1 : -1]
                        ).flatten()
                    ).to(self.device)
                )

        self.inputs = torch.stack(inputs)

        self.outputs = torch.stack(
            [
                torch.from_numpy(data.outputs[i]).to(self.device)
                for i in range(self.num_previous_inputs, len(data.outputs))
            ]
        )

    def from_raw(
        self,
        date: Tuple[int, int, int],
        time: Tuple[int, int, int],
        num_cables: int,
        num_coils: int,
        inputs: List[np.ndarray],
        outputs: List[np.ndarray],
    ):
        """
        Takes raw data and uses it to fill the Dataset.

        Args:
            date: The date of data collection in the following form (Y, M, D)
            time: The time of data collection in the following form (H, M, S)
            num_cables: The number of cables () of the system
            num_coils: The number of coils (3 position and 3 orientation measurements per coil)
            inputs: The cable displacements
            outputs: Pose data
        """
        raise NotImplementedError

    def from_numpy(
        self,
        date: Tuple[int, int, int],
        time: Tuple[int, int, int],
        num_cables: int,
        num_coils: int,
        inputs: np.ndarray,
        outputs: np.ndarray,
    ):
        """
        Populates dataset from input and output numpy arrays.

        Args:
            date: The date of data collection in the following form (Y, M, D)
            time: The time of data collection in the following form (H, M, S)
            num_cables: The number of cables () of the system
            num_coils: The number of coils (3 position and 3 orientation measurements per coil)
            inputs: The cable displacements
            outputs: Pose data
        """
        raise NotImplementedError()
