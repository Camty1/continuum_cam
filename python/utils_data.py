"""
A module that is used to interact with continuum data
"""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataContainer:  # pylint: disable=too-many-instance-attributes
    """
    Used to store, import and export continuum robot data
    """

    date: Tuple[int, int, int] = None
    time: Tuple[int, int, int] = None
    num_cables: int = None
    num_measurements: int = None
    inputs: List[np.ndarray] = field(default_factory=list)
    outputs: List[np.ndarray] = field(default_factory=list)
    num_coils: int = 1
    prefix: str = "data"

    def file_export(self, filename: str | None = None):
        """
        Exports data in the container to a file

        Args:
            filename: An optional filename to save the container as

        """
        if not filename:
            filename = (
                self.prefix
                + f"_{self.date[0]:02n}"
                + f"_{self.date[1]:02n}"
                + f"_{self.date[2]:02n}"
                + f"_{self.time[0]:02n}"
                + f"_{self.time[1]:02n}"
                + f"_{self.time[2]:02n}.dat"
            )

        with open(Path(filename), "w", encoding="UTF-8") as file:
            file.write(f"DATE: {self.date[0]}-{self.date[1]}-{self.date[2]}\n")
            file.write(f"TIME: {self.time[0]}-{self.time[1]}-{self.time[2]}\n")
            file.write(f"NUM_CABLES: {self.num_cables}\n")
            file.write(f"num_coils: {self.num_coils}\n")
            file.write(f"NUM_MEASUREMENTS: {self.num_measurements}\n")
            file.write("---\n")

            counter = 0
            for cable_input, output in zip(self.inputs, self.outputs):
                file.write(f"{counter},")

                for input_val in cable_input:
                    file.write(f"{input_val},")

                for i, _ in enumerate(output):
                    if i < len(output) - 1:
                        file.write(f"{output[i]},")
                    else:
                        file.write(f"{output[i]}\n")

                counter += 1

    def file_import(self, filename: str):
        """
        Reads the data from a file.

        Args:
            filename: The name of the file
        """
        with open(Path(filename), "r", encoding="utf-8") as file:
            date_list = file.readline().split(":")
            assert date_list[0] == "DATE"

            self.date = tuple(int(x) for x in date_list[1].split("-"))

            time_line = file.readline()
            time_list = time_line.split(":")
            assert time_list[0] == "TIME"

            self.time = tuple(int(x) for x in time_list[1].split("-"))

            num_cables_line = file.readline()
            self.num_cables = int(num_cables_line.split(":")[1])

            num_coils_line = file.readline()
            num_coils_list = num_coils_line.split(":")
            self.num_coils = int(num_coils_list[1])

            num_measurements_line = file.readline()
            num_measurements_list = num_measurements_line.split(":")
            assert num_measurements_list[0] == "NUM_MEASUREMENTS"

            self.num_measurements = int(num_measurements_list[1])

            spacer = file.readline()
            assert spacer.strip() == "---"

            while line := file.readline():
                row = line.split(",")
                self.inputs.append(
                    np.array(
                        [float(x) for x in row[1 : self.num_cables + 1]],
                        dtype=float,
                    )
                )
                self.outputs.append(
                    np.array(
                        [float(x) for x in row[self.num_cables + 1 :]],
                        dtype=float,
                    )
                )

            assert len(self.inputs) == len(self.outputs) == self.num_measurements

    # pylint: disable-next=too-many-positional-arguments, too-many-arguments
    def from_raw_data(
        self,
        date: Tuple[int, int, int],
        time: Tuple[int, int, int],
        num_cables: int,
        num_measurements: int,
        cable_deltas: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        num_coils: int = 1,
    ):
        """
        Populates the data container with numpy arrays

        Args:
            date: (year, month, day)
            time: (hour, minute, second)
            num_cables: The number of cables in the robot
            num_measurements: The number of coils the robot has
            cable_deltas: Used to command the robot
            positions: Tip positions
            orientations: Tip orientations
            num_coils: How many aurora coils used
        """
        assert cable_deltas.shape == (num_cables, num_measurements)
        assert positions.shape == (3 * num_coils, num_measurements)
        assert orientations.shape == (3 * num_coils, num_measurements)

        self.date = date
        self.time = time
        self.num_cables = num_cables
        self.num_measurements = num_measurements
        self.num_coils = num_coils

        self.inputs = []
        self.outputs = []

        for i in range(num_measurements):
            self.inputs.append(cable_deltas[:, i].flatten())
            self.outputs.append(
                np.concatenate([positions[:, i], orientations[:, i]], axis=0).flatten()
            )

    def set_date_and_time(self):
        """
        Get the current date and time to update the Datacontainer
        """
        now = datetime.datetime.now()

        self.date = (now.year, now.month, now.day)
        self.time = (now.hour, now.minute, now.second)

    def to_numpy(self):
        """
        Takes the data in the Datacontainer and outputs them as numpy arrays

        Returns:
            cable_deltas: The cable inputs
            pos: The tip positions
            tang: The tip orientations
        """
        cable_deltas = np.concatenate([x.reshape((-1, 1)) for x in self.inputs], axis=1)

        pos = np.concatenate([x[:3].reshape((-1, 1)) for x in self.outputs], axis=1)
        tang = np.concatenate([x[3:].reshape((-1, 1)) for x in self.outputs], axis=1)

        return cable_deltas, pos, tang

    def clean(self, pos_threshold=128, tang_threshold=np.pi):
        """
        Removes all NaNs and invalid measurements from the data_container.
        Should not be used on multi_input_learning datasets.

        Args:
            pos_threshold: The distance from the origin for an invalid measurement
            tang_threshold: The rotation angle from the origin for an invalid measurement
        """
        bad_indices = []
        for i in range(self.num_measurements):
            has_nan = np.isnan(self.inputs[i]).any() or np.isnan(self.outputs[i]).any()
            bad_pos = (np.abs(self.outputs[i][:3]) > pos_threshold).any()
            bad_tang = (np.abs(self.outputs[i][3:]) > tang_threshold).any()
            if has_nan or bad_pos or bad_tang:
                bad_indices.append(i)

        for idx in reversed(sorted(bad_indices)):
            self.inputs.pop(idx)
            self.outputs.pop(idx)
            self.num_measurements -= 1


def parse_aurora_csv(
    filename: str,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    Parses aurora data into a dictionary containing the quaternions, positions, and
    rms errors for each probe.

    Args:
        filename: The file to parse
    """

    df = pd.read_csv(Path(filename), header=None)

    probes = pd.unique(df.iloc[:, 2])

    output = {}
    for probe in probes:
        output[probe] = []
        probe_df = df[df[2] == probe]
        qs = np.transpose(probe_df.iloc[:, 3:7].to_numpy())
        ts = np.transpose(probe_df.iloc[:, 7:10].to_numpy())
        rms = probe_df.iloc[:, 13].to_numpy()

        for i in range(qs.shape[1]):
            output[probe].append((qs[:, i], ts[:, i], rms[i].item()))

    return output
