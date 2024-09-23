from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
import torch
from kinematics import dcm_2_quat, quat_2_dcm
import numpy as np
from pathlib import Path
from scipy.spatial import distance
import time

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")
T_electrode_2_tip = np.loadtxt(Path("../../tools/T_electrode_2_tip"), delimiter=",", dtype=np.float64)

arduino = ContinuumArduino()
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)
probe_list = ["0A"]
gain = 0.1
compression = 3
tolerance = 0.25
set_compression = 2

dls = -set_compression* np.ones(4, dtype=float)
arduino.write_dls(dls)
time.sleep(2)

range = 30

setpoints = np.array([[0, 0], [0, 8], [0, 16], [0, 24], 
                      [8, 0], [16, 0], [24, 0],
                      [0, -8], [0, -16], [0, -24],
                      [-8, 0], [-16, 0], [-24, 0]]) # x,y


def get_error(setpoint):
    raw_trans = aurora.get_aurora_transforms(probe_list)
    T_tip_2_model = aurora.get_T_tip_2_model(raw_trans["0A"])
    T_electrode_2_model = T_tip_2_model @ T_electrode_2_tip
    elec_pos = np.array([T_electrode_2_model[0,3], T_electrode_2_model[1,3]])
    error = (setpoint - elec_pos)
    print(error)
    return error

def get_pos():
    raw_trans = aurora.get_aurora_transforms(probe_list)
    T_tip_2_model = aurora.get_T_tip_2_model(raw_trans["0A"])
    T_electrode_2_model = T_tip_2_model @ T_electrode_2_tip
    elec_pos = np.array(T_electrode_2_model[:3,3], dtype=np.float64)
    print(f"elec pos = {elec_pos}")
    elec_rot = np.array(dcm_2_quat(T_electrode_2_model[:3,:3]), dtype=np.float64)
    print(f"elec_rot = {elec_rot}")
    data = np.concatenate([elec_pos, elec_rot])
    print(f"(combined = {data})")
    return data

def to_setpoint(setpoint, is_compressed):
    global dls
    if is_compressed == 'c':
        dls = dls - 5
    elif is_compressed == 'e':
        dls = dls + 5
    else: pass
    arduino.write_dls(dls)
    error = get_error(setpoint)
    while distance.euclidean([0,0], error) > tolerance:
        # print(f"error = {error}\n")
        ex = error[0]; ey = error[1]
        dls[0] = dls[0] + (-gain * ex); dls[2] = dls[2] + (gain * ex)
        dls[1] = dls[1] + (-gain * ey); dls[3] = dls[3] + (gain * ey)
        arduino.write_dls(dls)
        error = get_error(setpoint)
    else:
        print("error is sufficiently low")

print(f"{len(setpoints)} setpoints")
data = np.zeros([len(setpoints),10])
i = 0

for setpoint in setpoints:
    to_setpoint(setpoint, 'c')
    print("reached setpoint compressed")
    to_setpoint(setpoint, 'e')
    print("reached setpoint extended")
    print("\nPress any key to continue...")
    input()

    data[i] = np.concatenate([[i], setpoint, get_pos()]) # [ID, xset, yset, x, y, z, q1, q2, q3, q4]
    i = i + 1

np.savetxt(Path("output/closed_loop_control_1"), data, delimiter=',')