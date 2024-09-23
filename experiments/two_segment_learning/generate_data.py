#!/bin/python3
import numpy as np
from utils_data import DataContainer
from camarillo_cc import CamarilloModel
from utils_cc import camarillo_2_webster_params, calculate_transforms
from kinematics import dcm_2_tang
import datetime
from tqdm import tqdm


def generate_data(
    model: CamarilloModel,
    num_points: int = 2**14,
    pos_noise: float = 0.5,
    tang_noise: float = 0.05,
    prefix: str = None,
):
    rng = np.random.default_rng()

    cable_deltas = rng.uniform(-12, 12, (8, num_points))
    pos = np.zeros((3, num_points))
    tang = np.zeros((3, num_points))

    for i in tqdm(range(num_points)):
        camarillo_params = model.forward(cable_deltas[:, i])
        webster_params = camarillo_2_webster_params(
            camarillo_params, model.segment_lengths
        )
        T = calculate_transforms(webster_params)[-1]
        pos[:, i] = T[:3, 3]
        tang[:, i] = dcm_2_tang(T[:3, :3])

    pos += rng.normal(0, pos_noise, pos.shape)
    tang += rng.normal(0, tang_noise, tang.shape)

    container = DataContainer()
    if prefix:
        container.prefix = prefix

    now = datetime.datetime.now()

    container.from_raw_data(
        (now.year, now.month, now.day),
        (now.hour, now.minute, now.second),
        8,
        num_points,
        cable_deltas,
        pos,
        tang,
    )

    container.file_export()


if __name__ == "__main__":
    camarillo_stiffness = np.loadtxt("../../tools/camarillo_stiffness", delimiter=",")
    ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
    cable_positions = [((4, 0), (0, 4), (-4, 0), (0, -4))] * 2
    segment_stiffness_vals = [(ka, kb)] * 2
    cable_stiffness_vals = [(kt, kt, kt, kt)] * 2
    segment_lengths = [64, 64]
    additional_cable_length = 50

    model = CamarilloModel(
        cable_positions,
        segment_stiffness_vals,
        cable_stiffness_vals,
        segment_lengths,
        additional_cable_length,
    )

    generate_data(
        model=model,
        num_points=2**13,
        pos_noise=0.5,
        tang_noise=0.05,
        prefix="training_data/13",
    )
    generate_data(
        model=model,
        num_points=2**14,
        pos_noise=0,
        tang_noise=0,
        prefix="training_data/14_clean",
    )
    generate_data(
        model=model,
        num_points=2**14,
        pos_noise=0.5,
        tang_noise=0.05,
        prefix="training_data/14",
    )
    generate_data(
        model=model,
        num_points=2**15,
        pos_noise=0.5,
        tang_noise=0.05,
        prefix="training_data/15",
    )
    generate_data(
        model=model,
        num_points=2**9,
        pos_noise=0,
        tang_noise=0,
        prefix="./test_data/9_clean",
    )
