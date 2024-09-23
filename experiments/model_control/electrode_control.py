#!/bin/python3
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
from math import factorial
import ANN
from pathlib import Path
import camarillo_cc
import kinematics
import utils_cc
import utils_data

trajectory = np.loadtxt(
    Path("output/nathan_trajectory_v3.dat"), delimiter=",", dtype=np.float64
)
trajectory_tensor = torch.tensor(trajectory)
num_points = trajectory.shape[1]
num_closed_loop_steps = 11
step_size = 1

T_electrode_2_tip = np.loadtxt(
    Path("../../tools/T_electrode_2_tip"), delimiter=",", dtype=np.float64
)

T_electrode_2_tip_tensor = torch.from_numpy(T_electrode_2_tip)


model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)

model.load("../model_learning/models/real_07_17_2024/2024_07_17_19_42_23.pt")
model.model.eval()


def loss_fcn(
    dl: np.ndarray,
    model: ANN.Model,
    x_star: torch.tensor,
    dl_0: torch.tensor,
    weighting: Tuple[float, float],
    T_electrode_2_tip: torch.tensor,
) -> Tuple[float, np.ndarray]:
    dl_tensor = torch.tensor(dl, requires_grad=True)
    model.zero_grad()

    out = model(dl_tensor)

    def skew(vec: torch.tensor) -> torch.tensor:
        """
        Generates a skew symmetric matrix (used for tang map)

        Args:
            vec: A vector of dimension 3

        Returns:
            A skew symmetric 3 x 3 matrix
        """
        assert len(vec) == 3
        out = torch.zeros((3, 3))

        out[0, 1] = -vec[2]
        out[0, 2] = vec[1]
        out[1, 0] = vec[2]
        out[1, 2] = -vec[0]
        out[2, 0] = -vec[1]
        out[2, 1] = vec[0]

        return out

    def matrix_exponential(tang: torch.tensor) -> torch.tensor:
        tilde = skew(tang)

        out = torch.eye(3, 3)
        for i in range(10):
            out += torch.linalg.matrix_power(tilde, i + 1) / factorial(i + 1)

        return out

    T_tip_2_model = torch.eye(4, dtype=torch.double)
    T_tip_2_model[:3, 3] = out[:3]
    T_tip_2_model[:3, :3] = matrix_exponential(out[3:])
    T_electrode_2_model = T_tip_2_model @ T_electrode_2_tip
    x_hat = T_electrode_2_model[:3, 3]

    e = x_hat - x_star
    delta_dl = dl_tensor - dl_0

    loss = torch.dot(e, e) * weighting[0] + torch.dot(delta_dl, delta_dl) * weighting[1]

    loss.backward()
    grad = dl_tensor.grad.numpy()

    return (loss.item(), grad)


open_loop_dls = np.zeros((4, num_points))

for i in range(num_points):
    print(f"{i+1} of {num_points}")
    dl_0 = 0
    if i == 0:
        dl_0 = torch.tensor(np.array([0.0] * 4))
    else:
        dl_0 = torch.tensor(open_loop_dls[:, i - 1].flatten())

    x_star = trajectory_tensor[:, i].flatten()

    optim_func = lambda x: loss_fcn(
        x, model, x_star, dl_0, (5.0, 1.0), T_electrode_2_tip_tensor
    )

    result = opt.minimize(
        optim_func,
        dl_0.numpy(),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-12, 12), (-12, 12), (-12, 12), (-12, 12)],
    )
    open_loop_dls[:, i] = result["x"]

np.savetxt("output/nathan_cable_trajectory_v3.dat", open_loop_dls, delimiter=",")
