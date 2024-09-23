# pylint: disable=invalid-name
"""
A neural network based model which uses current and previous inputs/ states to predict
the next state.
"""
from collections import OrderedDict
from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ANN import Dataset, Model
from utils_data import DataContainer


class NARXModel(Model):  # pylint: disable=too-many-instance-attributes
    """
    The class for using a NARX model
    """

    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_layers: List[int],
        loss: nn.Module = nn.MSELoss(),
        activation=nn.ReLU(),
        output_activation: nn.Module = None,
        num_previous_actions: int = 1,
        num_previous_observations: int = 0,
        lr: float = 1e-3,
        checkpoints_path: str = None,
        save_path: str = None,
    ):
        input_dim = observation_dim * (1 + num_previous_observations) + action_dim * (
            1 + num_previous_actions
        )

        super().__init__(
            input_dim=input_dim,
            output_dim=observation_dim,
            hidden_layers=hidden_layers,
            loss=loss,
            activation=activation,
            output_activation=output_activation,
            lr=lr,
            checkpoints_path=checkpoints_path,
            save_path=save_path,
        )


class BabbleEnum(Enum):
    """
    An enum used to define whether babble data is dynamic or kinematic
        Dynamic: y(k) = x(k)
        Kinematic: y(k) = x(k+1)
    """

    DYNAMIC = 0
    KINEMATIC = 1


# pylint: disable=attribute-defined-outside-init
class NARXDataset(Dataset):
    """
    A dataset to be used with NARX models
    """

    def __init__(
        self,
        babble_type: BabbleEnum,
        num_previous_observations: int,
        num_previous_actions: int,
        filename: str | None = None,
    ):
        self.babble_type = babble_type
        self.num_previous_observations = num_previous_observations
        self.num_previous_actions = num_previous_actions

        super().__init__(filename)

    def load_from_DataContainer(self, data: DataContainer):
        """
        Imports data from a DataContainer, setting all properties of the object.

        Args:
            data: a DataContainer object
        """
        self.date = data.date
        self.time = data.time
        self.num_cables = data.num_cables
        self.num_coils = data.num_coils

        inputs = []
        if self.babble_type == BabbleEnum.DYNAMIC:
            offset = 1 + max(self.num_previous_observations, self.num_previous_actions)
            self.num_measurements = data.num_measurements - offset
            for i in range(offset - 1, data.num_measurements - 1):
                observation_start = i - (1 + self.num_previous_observations)
                action_start = i - (1 + self.num_previous_actions)

                if observation_start == -1:
                    observation_start = None

                if action_start == -1:
                    action_start = None

                inputs.append(
                    torch.from_numpy(
                        np.concatenate(
                            [
                                np.stack(
                                    data.outputs[i:observation_start:-1]
                                ).flatten(),
                                np.stack(data.inputs[i:action_start:-1]).flatten(),
                            ],
                            axis=0,
                        )
                    ).to(self.device)
                )
            self.inputs = torch.stack(inputs)

            self.outputs = torch.stack(
                [
                    torch.from_numpy(data.outputs[i]).to(self.device)
                    for i in range(offset, len(data.outputs))
                ]
            )

        else:
            offset = max(1 + self.num_previous_observations, self.num_previous_actions)
            self.num_measurements = data.num_measurements - offset

            for i in range(offset, len(data.outputs)):
                observation_start = i - (1 + self.num_previous_observations) - 1
                action_start = i - self.num_previous_actions - 1

                if observation_start == -1:
                    observation_start = None

                if action_start == -1:
                    action_start = None

                inputs.append(
                    torch.from_numpy(
                        np.concatenate(
                            [
                                np.stack(
                                    data.outputs[i - 1 : observation_start : -1]
                                ).flatten(),
                                np.stack(data.inputs[i:action_start:-1]).flatten(),
                            ],
                            axis=0,
                        )
                    ).to(self.device)
                )

            self.inputs = torch.stack(inputs)

            self.outputs = torch.stack(
                [
                    torch.from_numpy(data.outputs[i]).to(self.device)
                    for i in range(offset, len(data.outputs))
                ]
            )
