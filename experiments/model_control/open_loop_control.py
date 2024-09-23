#!/bin/python3
import time
import serial
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
import kinematics
import utils_data
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

wait_time = 0
repetitions = 1
ns2s = 10 ** (-9)
trajectory = np.loadtxt("output/nathan_trajectory_v3.dat", delimiter=",")
cable_trajectory = np.loadtxt("output/nathan_cable_trajectory_v3.dat", delimiter=",")
num_points = cable_trajectory.shape[1]

aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)
probe_list = ["0A"]
arduino = ContinuumArduino()

arduino.write_dls(np.zeros(4, dtype=float))
time.sleep(2)

model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)
model.load("../model_learning/models/real_07_17_2024/2024_07_17_19_42_23.pt")
model.model.eval()

model_pos = np.nan * np.zeros((3, num_points * repetitions))
model_tang = np.nan * np.zeros_like(model_pos)
true_pos = np.nan * np.zeros_like(model_pos)
true_tang = np.nan * np.zeros_like(model_pos)

for j in range(repetitions):
    for i in range(num_points):
        idx = num_points * j + i
        print(f"{idx+1} of {repetitions * num_points}")
        arduino.write_dls(cable_trajectory[:, i])
        dls_tensor = torch.tensor(cable_trajectory[:, i])
        output_tensor = model(dls_tensor)
        model_pos[:, idx] = output_tensor[:3].detach().numpy()
        model_tang[:, idx] = output_tensor[3:].detach().numpy()
        time.sleep(wait_time)
        trans = aurora.get_aurora_transforms(probe_list)
        T = aurora.get_T_tip_2_model(trans["0A"])
        true_pos[:, idx] = T[0:3, 3]
        true_tang[:, idx] = kinematics.dcm_2_tang(T[0:3, 0:3])

arduino.write_dls(np.zeros(4, dtype=float))

np.savetxt("output/model_trajectory.dat", model_pos, delimiter=",")
np.savetxt("output/true_trajectory.dat", true_pos, delimiter=",")

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(trajectory[0, :], trajectory[1, :], trajectory[2, :], label="Trajectory")
ax.scatter(model_pos[0, :], model_pos[1, :], model_pos[2, :], label="Model Prediction")
ax.scatter(true_pos[0, :], true_pos[1, :], true_pos[2, :], label="Truth Position")
plt.xlim((-32, 32))
plt.ylim((-32, 32))
ax.set_zlim((0, 64))
plt.title("Open Loop Trajectory Tracking")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")
plt.legend()
ax.view_init(90, -90, 0)
plt.show()
