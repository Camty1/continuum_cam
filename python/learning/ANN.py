"""
A standard neural network implementation using pytorch
"""
import datetime
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils_data
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Model(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    A simple implementation of an Artificial Neural Network (ANN).
    """
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        loss: nn.Module = nn.MSELoss(),
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module = None,
        lr: float = 1e-3,
        checkpoints_path: str = None,
        save_path: str = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = "cpu"
        # self.device = (
        #    "cuda"
        #    if torch.cuda.is_available()
        #    else "mps" if torch.backends.mps.is_available() else "cpu"
        # )
        print(
            f"Using {self.device} device for model (this was set intentionally due to the sizes of the matrices used.  If this needs to be changed, go to line 30 of ANN.py"  # pylint: disable=line-too-long
        )

        self.model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "input",
                        nn.Linear(
                            self.input_dim,
                            hidden_layers[0],
                        ).double(),
                    ),
                    ("input_activation", activation),
                ]
            )
        )

        for i in range(len(hidden_layers) - 1):
            self.model.add_module(
                f"hidden{i+1}",
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]).double(),
            )
            self.model.add_module(f"activation{i+1}", activation)

        self.model.add_module(
            "output", nn.Linear(hidden_layers[-1], output_dim).double()
        )

        if output_activation:
            self.model.add_module("output_activation", output_activation)

        self.model = self.model.to(self.device)

        self.loss = loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.checkpoints_path = checkpoints_path
        self.save_path = save_path

    def forward(self, x):
        """
        Evaluation of the neural network

        Args:
            x (tensor): Input with dimensions (batches x inputs)

        Returns:
            tensor: Output of model
        """
        return self.model(x)

    def train_epoch(self, dataloader: DataLoader, print_output=True) -> float:
        """
        Performs one epoch of training on the model

        Args:
            dataloader: A pytorch DataLoader that contains the training data

        Returns:
            The average training loss of the model over the epoch
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0

        self.model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            if print_output:
                if batch % 32 == 0:
                    current = (batch + 1) * len(X)
                    print(
                        f"Avg train loss: {train_loss / (batch+1):>7f} [{current:>5d}/{size:>5d}]",
                        flush=True,
                        end="\r",
                    )
        train_loss /= num_batches
        if print_output:
            print(
                f"Avg train loss: {train_loss:>7f} [{size:>5d}/{size:>5d}]",
                flush=True,
            )

        return train_loss

    def test_epoch(self, dataloader: DataLoader, print_output=True) -> float:
        """
        Performs one epoch (all batches of test data) of testing on the model

        Args:
            dataloader: A pytorch DataLoader that contains the test data

        Returns:
            The average test loss of the model over the epoch
        """
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss(pred, y).item()

        test_loss /= num_batches
        if print_output:
            print(f"Avg test loss: {test_loss:>7f}", flush=True)

        return test_loss

    def test_dataset(self, dataset: Dataset):
        """
        Tests model output on all inputs in a Dataset.

        Args:
            dataset: A Pytorch Dataset containing all test data

        Returns:
            The test loss of each item in the dataset
        """
        dataloader = DataLoader(dataset)

        self.model.eval()
        test_loss = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss.append(self.loss(pred, y).item())

        return np.array(test_loss)

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        num_epochs: int = 2048,
        checkpoints: bool = False,
        save_model: bool = False,
        print_output: bool = True,
    ) -> Tuple[List[float], ...]:
        """
        Trains the model

        Args:
            train_dataloader: A dataloader containing the training data
            test_dataloader: A dataloader containing the validation/ test data
            num_epochs: The number of epochs to train the model
            checkpoints: Will the model create checkpoints as it trains?
            save_model: Will the model be saved to a .pt after training?

        Returns:
            The training and validation losses after each epoch
        """
        train_loss = []
        test_loss = []

        if checkpoints:
            now = datetime.datetime.now()

            checkpoints_dir = Path("checkpoints/")

            if not checkpoints_dir.exists():
                os.mkdir(checkpoints_dir)

            checkpoints_path = f"{now.year}_{now.month:02n}_{now.day:02n}_{now.hour:02n}_{now.minute:02n}_{now.second:02n}/"

            checkpoints_path = checkpoints_dir / checkpoints_path

            if not checkpoints_path.exists():
                os.mkdir(checkpoints_path)

        for epoch in range(num_epochs):
            if print_output:
                print(f"Epoch {epoch+1}\n-------------------------------", flush=True)
            train_loss.append(
                self.train_epoch(train_dataloader, print_output=print_output)
            )

            if test_dataloader:
                test_loss.append(
                    self.test_epoch(test_dataloader, print_output=print_output)
                )

            if checkpoints:
                file_path = checkpoints_path / f"epoch_{epoch+1}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "validation_loss": test_loss[-1],
                        "model_state_dict": self.model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                    },
                    file_path,
                )

        if checkpoints:
            min_val_loss = min(enumerate(test_loss), key=lambda x: x[1])
            epoch = min_val_loss[0] + 1

            checkpoint = torch.load(checkpoints_path / f"epoch_{epoch}.pt")
            assert (
                checkpoint["epoch"] == epoch
                and checkpoint["validation_loss"] == min_val_loss[1]
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])
            if print_output:
                print(
                    f"Epoch with lowest validation loss (epoch {epoch}) loaded into model"
                )

            # Remove checkpoints to save storage
            shutil.rmtree(checkpoints_path)

            self.model.eval()

            if save_model:
                self.save()

        # No checkpoints, so save last training iteration
        else:
            if save_model:
                self.save()

        if test_dataloader:
            return train_loss, test_loss

        return train_loss, None

    def save(self, model_save_path: str = None):
        """
        Saves the model to a .pt file

        Args:
            model_save_path: The path to save the model at.  If none is given, then it defaults to the save_path defined in initialization
        """

        if model_save_path:
            self.save_path = model_save_path

        assert (
            self.save_path
        ), "No model save path specified in initialization or in save function"
        if not self.save_path.endswith(".pt"):
            self.save_path += ".pt"
        torch.save(self.model.state_dict(), Path(self.save_path))

    def load(self, model_load_path: str):
        """
        Loads the state_dict (parameter values) into the model

        Args:
            model_load_path: The location of the .pt file to load from
        """

        if not model_load_path.endswith(".pt"):
            model_load_path += ".pt"
        self.model.load_state_dict(torch.load(Path(model_load_path)))
        self.model.to(self.device)


class Dataset(Dataset):
    """
    Houses data for training.  Built on top of DataContainer class from utils_data
    """

    def __init__(self, filename: str = None):
        """
        Creates an empty Dataset object, and imports from a file if it is given

        Args:
            filename: The path to a file to import from
        """
        self.date = None
        self.time = None
        self.num_cables = None
        self.num_coils = None

        self.device = "cpu"
        # self.device = (
        #    "cuda"
        #    if torch.cuda.is_available()
        #    else "mps" if torch.backends.mps.is_available() else "cpu"
        # )
        print(
            f"Using {self.device} device for dataset (this was set intentionally due to the sizes of the matrices used.  If this needs to be changed, go to line 200 of ANN.py"
        )

        self.inputs = []
        self.outputs = []

        if filename:
            self.load_from_file(filename)

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
        self.num_measurements = data.num_measurements

        # Convert numpy arrays to tensors and put them in the input and output arrays
        self.inputs = torch.stack(
            [torch.from_numpy(input).to(self.device) for input in data.inputs]
        )
        self.outputs = torch.stack(
            [torch.from_numpy(output).to(self.device) for output in data.outputs]
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
        assert len(inputs) == len(outputs), "Input and output must be same length"
        for input, output in zip(inputs, outputs):
            assert (
                len(input) == num_cables
            ), "All inputs must match the number of cables"
            assert (
                len(output) == 6 * num_coils
            ), "All outputs must match the number of coils * 6 (3 position and 3 orientation measurements)"

        self.date = date
        self.time = time
        self.num_cables = num_cables
        self.num_coils = num_coils
        self.num_measurements = len(inputs)
        # Convert lists of numpy arrays to lists of tensors
        self.inputs = torch.stack(
            [torch.from_numpy(input).to(self.device) for input in inputs]
        )
        self.outputs = torch.stack(
            [torch.from_numpy(output).to(self.device) for output in outputs]
        )

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

        assert (
            inputs.shape[1] == outputs.shape[1]
        ), "Make sure there are the same number of measurements"
        assert (
            inputs.shape[0] == num_cables
        ), "Ensure input size matches number of cables"
        assert (
            outputs.shape[0] == 6 * num_coils
        ), "Ensure output size matches number of measurements from coils"

        self.date = date
        self.time = time
        self.num_cables = num_cables
        self.num_coils = num_coils
        self.num_measurements = len(inputs)

        # Converts numpy arrays to lists of tensors
        self.inputs = torch.stack(
            [
                torch.from_numpy(inputs[:, i]).to(self.device)
                for i in range(self.num_measurements)
            ]
        )
        self.outputs = torch.stack(
            [
                torch.from_numpy(outputs[:, i]).to(self.device)
                for i in range(self.num_measurements)
            ]
        )

    def __len__(self) -> int:
        """
        Returns number of items in Dataset
        """
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        """
        Gets the (input, output) datapoint at the given index
        """
        return self.inputs[idx, :], self.outputs[idx, :]

    def save(self, filename: str = "dataset_out.txt"):
        """
        Saves the dataset to a file.
        """
        raise NotImplementedError

    def clean(self, pos_threshold=128, tang_threshold=np.pi):
        """
        Removes all NaNs and faulty measurements from dataset

        Args:
            pos_threshold: Maximum position coordinate value before point is discarded
            tang_threshold: Maximum orientation tang value before point is discarded
        """

        bad_indices = []
        for i in range(len(self)):
            # Check for nan
            has_nan = (
                np.isnan(self.inputs[i, :]).any() or np.isnan(self.outputs[i, :]).any()
            )
            # Position is outside valid domain
            bad_pos = (np.abs(self.outputs[i, :][:3]) > pos_threshold).any()
            # orientation is outside valid domain
            bad_tang = (np.abs(self.outputs[i, :][3:]) > tang_threshold).any()

            if has_nan or bad_pos or bad_tang:
                bad_indices.append(i)

        # Remove all bad indices from Dataset, taking into account number of measurements removed
        good_indices = np.logical_not(np.isin(np.arange(len(self)), bad_indices))

        self.inputs = self.inputs[good_indices, :]
        self.outputs = self.outputs[good_indices, :]
        self.num_measurements = len(good_indices)

    def plot_pos(self, decimation: int = 10) -> plt.figure:
        """
        Create a scatter plot of the (x, y) positions of the points in the Dataset

        Args:
            decimation: Only plot one point for every decimation points in the Dataset

        Returns:
            The matplotlib figure
        """
        fig = plt.figure()
        for i in range(0, self.num_measurements, decimation):
            plt.scatter(self.outputs[i][0], self.outputs[i][1])
        plt.show()

        return fig


class PoseLoss(nn.Module):
    """
    A weighted MSE loss function used to consider both position error and orientation error
    """

    def __init__(self, scale: float = 10, num_coils: int = 1):
        """
        Creates PoseLoss object, which can be passed to a Model for its loss function.

        Args:
            scale: How much to scale up orientation error relative to position error
            num_coils: Number of aurora coils (poses) of system
        """
        super(PoseLoss, self).__init__()

        self.device = "cpu"
        # self.device = (
        #    "cuda"
        #    if torch.cuda.is_available()
        #    else "mps" if torch.backends.mps.is_available() else "cpu"
        # )
        print(
            f"Using {self.device} device for PoseLoss (this was set intentionally due to the sizes of the matrices used.  If this needs to be changed, go to line 312 of ANN.py"
        )
        self.num_coils = num_coils
        self.weights = torch.from_numpy(
            np.concatenate((np.ones(3 * num_coils), scale * np.ones(3 * num_coils)))
        ).to(self.device)

    def forward(self, pred, target):
        """
        Calculates loss

        Args:
            pred (tensor): The prediction of the model
            target (tensor): The true value

        Returns:
            Loss
        """
        expanded_weights = self.weights.expand(pred.size(0), -1)

        return nn.functional.mse_loss(
            pred * expanded_weights, target * expanded_weights
        )


class PositionLoss(nn.Module):
    """
    PositionLoss is a MSE loss function that only considers position
    """

    def __init__(self):
        """
        Creates PositionLoss object, which can be passed to a Model for its loss function.
        """
        super(PositionLoss, self).__init__()

    def forward(self, pred, target):
        """
        Calculates loss

        Args:
            pred (tensor): The prediction of the model
            target (tensor): The true value

        Returns:
            Loss
        """
        return torch.sqrt(nn.functional.mse_loss(pred[:, :3], target[:, :3]) * 3)


class OrientationLoss(nn.Module):
    """
    OrientationLoss is a MSE loss function that only considers orientation
    """

    def __init__(self):
        """
        Creates OrientationLoss object, which can be passed to a Model for its loss function.
        """
        super(OrientationLoss, self).__init__()

    def forward(self, pred, target):
        """
        Calculates loss

        Args:
            pred (tensor): The prediction of the model
            target (tensor): The true value

        Returns:
            Loss
        """
        return torch.sqrt(nn.functional.mse_loss(pred[:, 3:], target[:, 3:]) * 3)
