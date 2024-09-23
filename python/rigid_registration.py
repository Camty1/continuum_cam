#!/bin/python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import kinematics
import utils_data

# Parameters
REG_FILE = "../data/regs/reg_08_09_24d.csv"
TIP_FILE = "../tools/penprobe_08_09_24c"
output = True
num_repetitions = 5

# Setup
# Truth filenames
SW_MODEL_POS_FILE = "../tools/12_model_registration_points_in_sw"
SW_TIP_POS_FILE = "../tools/all_tip_registration_points_in_sw"
T_SW_2_MODEL_FILE = "../tools/T_sw_2_model"
T_SW_2_TIP_FILE = "../tools/T_sw_2_tip"

# FRE (fiducial regisitration error) dictionary initialization
fre = {}

# File inputs
model_truth_in_sw = np.loadtxt(Path(SW_MODEL_POS_FILE), delimiter=",")
tip_truth_in_sw = np.loadtxt(Path(SW_TIP_POS_FILE), delimiter=",")
T_sw_2_model = np.loadtxt(Path(T_SW_2_MODEL_FILE), delimiter=",")
T_sw_2_tip = np.loadtxt(Path(T_SW_2_TIP_FILE), delimiter=",")
penprobe = np.loadtxt(Path(TIP_FILE), delimiter=",")

aurora_transforms = utils_data.parse_aurora_csv(REG_FILE)

model_truth_in_model = kinematics.Tmult(T_sw_2_model, model_truth_in_sw)
tip_truth_in_tip = kinematics.Tmult(T_sw_2_tip, tip_truth_in_sw)
print(tip_truth_in_tip)

temp_model = np.zeros((3, model_truth_in_model.shape[1] * num_repetitions))
temp_tip = np.zeros((3, tip_truth_in_tip.shape[1] * num_repetitions))

for i in range(num_repetitions):
    for j in range(model_truth_in_model.shape[1]):
        temp_model[:, j * num_repetitions + i] = model_truth_in_model[:, j]
    for j in range(tip_truth_in_tip.shape[1]):
        temp_tip[:, j * num_repetitions + i] = tip_truth_in_tip[:, j]

model_truth_in_model = temp_model
tip_truth_in_tip = temp_tip

num_model_points = model_truth_in_model.shape[1]
num_tip_points = tip_truth_in_tip.shape[1]

# Get truth positions in model and tip frames
# Get penprobe positions
meas_in_aurora = kinematics.penprobe_transform(penprobe, aurora_transforms["0A"])

model_meas_in_aurora = meas_in_aurora[:, :num_model_points]
tip_meas_in_aurora = meas_in_aurora[
    :, num_model_points : num_model_points + num_tip_points
]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(
    model_meas_in_aurora[0, :], model_meas_in_aurora[1, :], model_meas_in_aurora[2, :]
)
ax.plot(tip_meas_in_aurora[0, :], tip_meas_in_aurora[1, :], tip_meas_in_aurora[2, :])
plt.show()

# Perform registrations
model_meas_in_model, T_aurora_2_model, fre["aurora_2_model"] = (
    kinematics.rigid_align_svd(model_meas_in_aurora, model_truth_in_model)
)
tip_meas_in_tip, T_aurora_2_tip, fre["aurora_2_tip"] = kinematics.rigid_align_svd(
    tip_meas_in_aurora, tip_truth_in_tip
)

# Get average coil transform
coil_qs = np.concatenate(
    [transform[0].reshape((4, 1)) for transform in aurora_transforms["0B"]], axis=1
)
coil_ts = np.concatenate(
    [transform[1].reshape((3, 1)) for transform in aurora_transforms["0B"]], axis=1
)

q_mean = coil_qs.mean(axis=1)
t_mean = coil_ts.mean(axis=1)

T_coil_2_aurora = np.identity(4)
T_coil_2_aurora[0:3, 0:3] = kinematics.quat_2_dcm(q_mean)
T_coil_2_aurora[0:3, 3] = t_mean.flatten()

T_tip_2_coil = np.linalg.inv(T_coil_2_aurora) @ np.linalg.inv(T_aurora_2_tip)
np.set_printoptions(suppress=True)
print("T_aurora_2_model")
print(T_aurora_2_model)
print("T_aurora_2_tip")
print(T_aurora_2_tip)
print("T_coil_2_aurora")
print(T_coil_2_aurora)
print("T_tip_2_coil")
print(T_tip_2_coil)
print("T_tip_2_model")
print(T_aurora_2_model @ np.linalg.inv(T_aurora_2_tip))
print(fre)

fle = {}

for key, value in fre.items():
    fle[key] = np.sqrt(value**2 * (1 - 2 / (12 * num_repetitions)))

tre = {}

model_cov = model_truth_in_model @ np.transpose(model_truth_in_model)
model_tre_factor = 1 / (12 * num_repetitions) + 1 / 3 * 5**2 * (
    1 / (model_truth_in_model[0, :] ** 2).mean()
    + 1 / (model_truth_in_model[1, :] ** 2).mean()
)

tip_tre_factor = 1 / (12 * num_repetitions) + 1 / 2 * 3**2 * (
    1 / (tip_truth_in_tip[0, :] ** 2).mean() + 1 / (tip_truth_in_tip[1, :] ** 2).mean()
)

tre["aurora_2_model"] = fle["aurora_2_model"] * model_tre_factor
tre["aurora_2_tip"] = fle["aurora_2_tip"] * tip_tre_factor
tre["tip_2_model"] = np.sqrt(tre["aurora_2_tip"] ** 2 + tre["aurora_2_model"] ** 2)

print(model_tre_factor, tip_tre_factor, tre)

if output:
    np.savetxt(Path("../tools/T_aurora_2_model"), T_aurora_2_model, delimiter=",")
    np.savetxt(Path("../tools/T_tip_2_coil"), T_tip_2_coil, delimiter=",")
