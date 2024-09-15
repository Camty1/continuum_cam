#!/bin/python3
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from kinematics import quat_2_dcm
from utils_data import parse_aurora_csv

"""
pivot_cal_ransac
Cameron Wolfe 07/03/2024

Performs Algebraic One Step pivot calibration with RANSAC outlier rejection.

RANSAC (Random Sample Consensus) works by performing pivot calibration on a small
subset of the data.  This is then tested on the rest of the dataset to determine
how well the new penprobe fits the rest of the data.  If the error is below a
threshold, it is added to the set of points the model fits.  After going through
all of the points, if a large portion of the dataset fits, then this is considered
a valid penprobe, and it is saved.  The best valid penprobe (lowest RMSE) is found
by iterating this process many times.

Useful resources:
Which pivot calibration? paper: https://doi.org/10.1117/12.2081348
RANSAC overview: https://www.baeldung.com/cs/ransac
"""

# Parameters
pivot_filename = "../data/tip_cals/tip_cal_08_09_24c.csv"
output_filename = "../tools/penprobe_08_09_24c"
show_plots = True

# RANSAC parameters
p_outlier_in_subset = 0.01  # Used for estimating number of iterations
p_outlier_in_data = 0.05  # Used for estimating number of iterations
error_threshold = 0.75  # (mm) Considered a good point if error is below threshold
good_fit_threshold = 0.75  # If we this percentage of the points, its a good fit
subset_size = 100  # Starting random subset size

ransac_iterations = int(
    np.ceil(
        np.log(p_outlier_in_subset) / np.log(1 - (1 - p_outlier_in_data) ** subset_size)
    )
)

num_digits = int(np.ceil(np.log10(ransac_iterations + 1)))
print(f"Number of RANSAC iterations: {ransac_iterations}")


# Local functions
def solve_system(
    coil_quat: np.ndarray,
    coil_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function creates/ solves the overdefined system of eqns for penprobe + pivot

    Args:
        coil_quat: 4 x N array containing coil quaternions
        coil_pos: 3 x N array containing coil positions

    Returns:
        The penprobe and pivot vectors
    """
    num_meas = coil_quat.shape[1]
    assert num_meas == coil_pos.shape[1]

    # Form overdefined system of equations
    A = np.zeros((3 * num_meas, 6))
    b = np.zeros((3 * num_meas))
    for i in range(num_meas):
        A[3 * i : 3 * (i + 1), :] = np.concatenate(
            [quat_2_dcm(coil_quat[:, i]), -np.identity(3)], axis=1
        )
        b[3 * i : 3 * (i + 1)] = -coil_pos[:, i]

    # Solve system
    out = np.linalg.lstsq(A, b, rcond=None)

    # Extract and return
    penprobe = out[0][:3]
    pivot = out[0][3:]
    return penprobe, pivot


def get_error(
    coil_quat: np.ndarray,
    coil_pos: np.ndarray,
    penprobe: np.ndarray,
    pivot: np.ndarray,
) -> np.ndarray:
    """
    Finds error between coil + rotated penprobe vs pivot point.

    Args:
        coil_quat: 4 x N array containing coil quaternions
        coil_pos: 4 x N array containing coil positions
        penprobe: Coil to tip vector
        pivot: Position in Aurora space pen pivots about

    Returns:
        An array containing the position error for each coil measurement
    """
    # Setup
    num_meas = coil_quat.shape[1]
    assert num_meas == coil_pos.shape[1]
    error = np.zeros(num_meas)

    # Go through all points and find deviation from pivot
    for i in range(num_meas):
        tip_pos = coil_pos[:, i] + quat_2_dcm(coil_quat[:, i]) @ penprobe
        error[i] = np.linalg.norm(tip_pos - pivot)

    return error


# Load data
aurora_data = parse_aurora_csv(pivot_filename)["0A"]
num_meas = len(aurora_data)
coil_quat = np.zeros((4, num_meas))
coil_pos = np.zeros((3, num_meas))

for i in range(num_meas):
    coil_quat[:, i] = aurora_data[i][0]
    coil_pos[:, i] = aurora_data[i][1]


# RANSAC algorithm
def ransac(ransac_iterations, subset_size, error_threshold, good_fit_threshold):
    best_penprobe = np.nan * np.zeros(3)
    best_pivot = np.nan * np.zeros(3)
    best_rmse = np.inf
    for i in range(ransac_iterations):

        # Get random subset
        subset = np.random.choice(num_meas, subset_size, replace=False)
        outside_subset = np.setdiff1d(np.arange(num_meas), subset)

        # Find penprobe on random subset
        penprobe, pivot = solve_system(coil_quat[:, subset], coil_pos[:, subset])

        # Test points outside of subset
        error = get_error(
            coil_quat[:, outside_subset], coil_pos[:, outside_subset], penprobe, pivot
        )

        # Get points that model fits
        inside_subset = outside_subset[error < error_threshold]

        # Valid fit
        if len(inside_subset) + len(subset) >= good_fit_threshold * num_meas:

            # Collect all points that fit the model into one set
            new_subset = np.concatenate((subset, inside_subset))

            # Find penprobe on larger set
            penprobe, pivot = solve_system(
                coil_quat[:, new_subset], coil_pos[:, new_subset]
            )

            # Test fit
            error = get_error(
                coil_quat[:, new_subset], coil_pos[:, new_subset], penprobe, pivot
            )

            rmse = np.sqrt(np.mean(error**2))

            # Update if best fit
            if rmse < best_rmse:
                best_rmse = rmse
                best_penprobe = penprobe
                best_pivot = pivot

        # Printing
        if i != ransac_iterations - 1:
            print(
                f"RANSAC iteration {i + 1:0{num_digits}}/{ransac_iterations}: {best_rmse:.4f} mm RMSE",
                end="\r",
            )
        else:
            print(
                f"RANSAC iteration {i + 1:0{num_digits}}/{ransac_iterations}: {best_rmse:.4f} mm RMSE",
            )

    if not np.isnan(best_penprobe).any():
        return best_penprobe, best_pivot
    else:
        print("RANSAC did not converge with current thresholds, relaxing thresholds.")
        return ransac(
            ransac_iterations,
            subset_size,
            error_threshold + 0.25,
            good_fit_threshold,
        )


penprobe, pivot = ransac(
    ransac_iterations, subset_size, error_threshold, good_fit_threshold
)
np.savetxt(Path(output_filename), penprobe, delimiter=",")

# Plotting
if show_plots:

    error = get_error(coil_quat, coil_pos, penprobe, pivot)
    rmse = np.sqrt(np.mean(error**2))

    def error_2_color(
        error: float,
        min_error: float = 0,
        max_error: float = 1.5,
        min_color: np.ndarray = np.array([0.12156863, 0.46666667, 0.70588235]),
        max_color: np.ndarray = np.array([0.83921569, 0.15294118, 0.15686275]),
    ) -> np.ndarray:
        delta_color_vec = max_color - min_color

        return (
            min_color + (error - min_error) / (max_error - min_error) * delta_color_vec
        )

    num_meas = coil_quat.shape[1]
    full_data = np.concatenate([coil_pos, pivot.reshape((3, 1))], axis=1)

    point_color = np.zeros((num_meas, 3))
    min_error = error.min()
    max_error = error.max()
    for i in range(num_meas):
        point_color[i, :] = error_2_color(error[i], min_error, max_error)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(
        coil_pos[0, :],
        coil_pos[1, :],
        coil_pos[2, :],
        c=point_color,
        label="Coil Positions",
    )
    ax.set_box_aspect(
        (np.ptp(full_data[0, :]), np.ptp(full_data[1, :]), np.ptp(full_data[2, :]))
    )
    for i in range(0, num_meas, 2):
        vec = quat_2_dcm(coil_quat[:, i]) @ penprobe
        legend_label = "Penprobe Vectors" if i == 0 else "_Penprobe Vectors"
        ax.quiver(*coil_pos[:, i], *vec, label=legend_label, color=point_color[i, :])

    ax.scatter(pivot[0], pivot[1], pivot[2], label="Pivot Point", color="C1")
    plt.title(f"Pivot Calibration (RMSE: {rmse.item():.3f} mm)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    plt.legend()
    plt.figure()
    plt.plot(error)
    plt.title("Error of Data Points")
    plt.xlabel("Point")
    plt.ylabel("Error (mm)")
    plt.show()
