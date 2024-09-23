import numpy as np
import torch
from pathlib import Path

trajectory = np.loadtxt(
    Path("../model_control/output/nathan_trajectory_v2.dat"), delimiter=",", dtype=np.float64
)

trajectory_tensor = torch.tensor(trajectory)
num_points = trajectory.shape[1]
increments = 10

path = np.zeros((3, (num_points-1)*increments + 1))


for i in range(num_points - 1):
    p1 = trajectory_tensor[:,i]
    p2 = trajectory_tensor[:,i+1]
    print(f"point 1 = {p1}\npoint 2 = {p2}")
    diff = p2 - p1
    print(f"distance from p1 to p2 = {diff}")
    diff_inc = diff / increments
    print(f"diff inc = {diff_inc}")
    for j in range(increments + 1):
        path[:,increments*i + j] = p1 + j*diff_inc
        print(f"pos {increments*i + j} = {path[:,increments*i + j]}")
    print("\n")

np.savetxt(Path("../model_control/output/nathan_trajectory_v3.dat"), path, delimiter=",")
