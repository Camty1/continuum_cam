import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt    
import math  

data = np.loadtxt(Path("output/closed_loop_control_1"), delimiter=',')
IDs = data[:][0]; setpoints = data[:][2:3]; pos = data[:][4:6]; quat = data[:][7:10]

# plot all points together
ax = plt.figure().add_subplot()
ax.scatter(setpoints[0], setpoints[1], label="desired position")
ax.scatter(pos[0], pos[1], label="real position")
plt.xlim((-35, 35))
plt.ylim((-35, 35))
plt.title("Closed Loop Imaging Control")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend()
plt.show()

# calculate RMS error
mse = np.square(np.subtract(pos[:2],setpoints)).mean()   
rsme = math.sqrt(mse) 

# plot x rms error
ax = plt.figure().add_subplot()
ax.plot([])
