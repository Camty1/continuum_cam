#!/bin/python3
from typing import List, Tuple
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from camarillo_cc import CamarilloModel
from utils_cc import camarillo_2_webster_params, calculate_transforms
import utils_data
import kinematics


def generate_babble_data(
    model: CamarilloModel,
    num_measurements: int = 2**16,
    cable_range: float = 12,
    pos_noise_std: float = 0.5,
    tang_noise_std: float = 0.05,
) -> List[utils_data.DataContainer]:

    now = datetime.datetime.now()
    date = (now.year, now.month, now.day)
    time = (now.hour, now.minute, now.second)

    rng = np.random.default_rng()

    cable_deltas = (
        2 * cable_range * (rng.random((model.num_cables, num_measurements)) - 0.5)
    )
    positions = np.zeros((3, num_measurements))
    orientations = np.zeros((3, num_measurements))

    for i in tqdm(range(num_measurements)):

        camarillo_params = model.forward(cable_deltas[:, i])
        webster_params = camarillo_2_webster_params(
            camarillo_params, model.segment_lengths
        )

        T_list = calculate_transforms(webster_params)

        positions[:, i] = T_list[-1][0:3, 3]
        orientations[:, i] = kinematics.dcm_2_tang(T_list[-1][0:3, 0:3])

    noisy_positions = positions + np.random.normal(0, pos_noise_std, positions.shape)
    noisy_orientations = orientations + np.random.normal(
        0, tang_noise_std, orientations.shape
    )

    plt.figure()
    plt.scatter(noisy_positions[0, :], noisy_positions[1, :], alpha=0.3)
    plt.title("Noisy Simulation Data w/ Slack Cables")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.show()

    pure_container = utils_data.DataContainer()
    pure_container.from_raw_data(
        date, time, num_cables, num_measurements, cable_deltas, positions, orientations
    )
    pure_container.prefix = f"training_data/clean_{model.num_segments}_seg"

    noisy_container = utils_data.DataContainer()
    noisy_container.from_raw_data(
        date,
        time,
        num_cables,
        num_measurements,
        cable_deltas,
        noisy_positions,
        noisy_orientations,
    )
    noisy_container.prefix = f"training_data/noisy_{model.num_segments}_seg"

    return pure_container, noisy_container


if __name__ == "__main__":

    camarillo_stiffness = np.loadtxt("../../tools/camarillo_stiffness", delimiter=",")
    ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
    cable_positions_one_seg = [
        ((4, 0), (0, 4), (-4, 0), (0, -4)),
    ]
    segment_stiffness_vals_one_seg = [(ka, kb)]
    cable_stiffness_vals_one_seg = [(kt, kt, kt, kt)]
    segment_lengths_one_seg = [64]

    one_seg_model = CamarilloModel(
        cable_positions_one_seg,
        segment_stiffness_vals_one_seg,
        cable_stiffness_vals_one_seg,
        segment_lengths_one_seg,
        50,
    )

    cable_positions_two_seg = cable_positions_one_seg + cable_positions_one_seg
    segment_stiffness_vals_two_seg = (
        segment_stiffness_vals_one_seg + segment_stiffness_vals_one_seg
    )
    cable_stiffness_vals_two_seg = (
        cable_stiffness_vals_one_seg + cable_stiffness_vals_one_seg
    )
    segment_lengths_two_seg = segment_lengths_one_seg + segment_lengths_one_seg

    two_seg_model = CamarilloModel(
        cable_positions_two_seg,
        segment_stiffness_vals_two_seg,
        cable_stiffness_vals_two_seg,
        segment_lengths_two_seg,
        50,
    )

    pure_container_one_seg, noisy_container_one_seg = generate_babble_data(
        one_seg_model,
        num_measurements=2**14,
    )

    pure_container_two_seg, noisy_container_two_seg = generate_babble_data(
        two_seg_model,
        num_measurements=2**14,
    )

    pure_container_one_seg.file_export()
    noisy_container_one_seg.file_export()
    pure_container_two_seg.file_export()
    noisy_container_two_seg.file_export()
