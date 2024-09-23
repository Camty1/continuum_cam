#!/bin/python3
"""Creates plots to visualize the dataset found at FILENAME"""
import matplotlib.pyplot as plt

from utils_data import DataContainer

FILENAME = "./test_data/9_clean_2024_09_21_10_02_14.dat"

container = DataContainer()
container.file_import(FILENAME)
_, pos, tang = container.to_numpy()

plt.figure()
ax = plt.subplot(projection="3d")
ax.plot(pos[0, :], pos[1, :], pos[2, :], "o", alpha=0.3)
plt.title("3D Position Distribution")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")

plt.figure()
plt.plot(pos[0, :], pos[1, :], "o", alpha=0.3)
plt.title("X-Y Position Distribution")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

plt.figure()
plt.plot(pos[0, :], pos[2, :], "o", alpha=0.3)
plt.title("X-Z Position Distribution")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")

plt.figure()
plt.plot(pos[1, :], pos[2, :], "o", alpha=0.3)
plt.title("Y-Z Position Distribution")
plt.xlabel("y (mm)")
plt.ylabel("z (mm)")

plt.figure()
ax = plt.subplot(projection="3d")
ax.plot(tang[0, :], tang[1, :], tang[2, :], "o", alpha=0.3)
plt.title("3D Orientation Distribution")
plt.xlabel("x (rad)")
plt.ylabel("y (rad)")
ax.set_zlabel("z (rad)")


plt.figure()
plt.plot(tang[0, :], tang[1, :], "o", alpha=0.3)
plt.title("X-Y Orientation Distribution")
plt.xlabel("x (rad)")
plt.ylabel("y (rad)")

plt.figure()
plt.plot(tang[0, :], tang[2, :], "o", alpha=0.3)
plt.title("X-Z Orientation Distribution")
plt.xlabel("x (rad)")
plt.ylabel("z (rad)")

plt.figure()
plt.plot(tang[1, :], tang[2, :], "o", alpha=0.3)
plt.title("Y-Z Orientation Distribution")
plt.xlabel("y (rad)")
plt.ylabel("z (rad)")

plt.show()
