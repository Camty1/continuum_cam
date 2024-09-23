#!/bin/python3
import matplotlib.pyplot as plt
import numpy as np

import utils_data

if __name__ == "__main__":
    folder = "2024_09_21_11_59_03"
    clean_1_folder = f"./output/{folder}/clean_1"
    noisy_1_folder = f"./output/{folder}/noisy_1"
    clean_2_folder = f"./output/{folder}/clean_2"
    noisy_2_folder = f"./output/{folder}/noisy_2"

    clean_1_train = np.loadtxt(f"{clean_1_folder}_train.dat", delimiter=",")
    noisy_1_train = np.loadtxt(f"{noisy_1_folder}_train.dat", delimiter=",")
    clean_2_train = np.loadtxt(f"{clean_2_folder}_train.dat", delimiter=",")
    noisy_2_train = np.loadtxt(f"{noisy_2_folder}_train.dat", delimiter=",")

    clean_1_validation = np.loadtxt(f"{clean_1_folder}_validation.dat", delimiter=",")
    noisy_1_validation = np.loadtxt(f"{noisy_1_folder}_validation.dat", delimiter=",")
    clean_2_validation = np.loadtxt(f"{clean_2_folder}_validation.dat", delimiter=",")
    noisy_2_validation = np.loadtxt(f"{noisy_2_folder}_validation.dat", delimiter=",")

    clean_1_pos = np.loadtxt(f"{clean_1_folder}_pos.dat", delimiter=",")
    noisy_1_pos = np.loadtxt(f"{noisy_1_folder}_pos.dat", delimiter=",")
    clean_2_pos = np.loadtxt(f"{clean_2_folder}_pos.dat", delimiter=",")
    noisy_2_pos = np.loadtxt(f"{noisy_2_folder}_pos.dat", delimiter=",")

    clean_1_tang = np.loadtxt(f"{clean_1_folder}_tang.dat", delimiter=",") * 180 / np.pi
    noisy_1_tang = np.loadtxt(f"{noisy_1_folder}_tang.dat", delimiter=",") * 180 / np.pi
    clean_2_tang = np.loadtxt(f"{clean_2_folder}_tang.dat", delimiter=",") * 180 / np.pi
    noisy_2_tang = np.loadtxt(f"{noisy_2_folder}_tang.dat", delimiter=",") * 180 / np.pi

    clean_1_train_mean = clean_1_train.mean(axis=0)
    clean_1_train_std = np.std(clean_1_train, axis=0, ddof=1)
    noisy_1_train_mean = noisy_1_train.mean(axis=0)
    noisy_1_train_std = np.std(noisy_1_train, axis=0, ddof=1)
    clean_2_train_mean = clean_2_train.mean(axis=0)
    clean_2_train_std = np.std(clean_2_train, axis=0, ddof=1)
    noisy_2_train_mean = noisy_2_train.mean(axis=0)
    noisy_2_train_std = np.std(noisy_2_train, axis=0, ddof=1)

    clean_1_validation_mean = clean_1_validation.mean(axis=0)
    clean_1_validation_std = np.std(clean_1_validation, axis=0, ddof=1)
    noisy_1_validation_mean = noisy_1_validation.mean(axis=0)
    noisy_1_validation_std = np.std(noisy_1_validation, axis=0, ddof=1)
    clean_2_validation_mean = clean_2_validation.mean(axis=0)
    clean_2_validation_std = np.std(clean_2_validation, axis=0, ddof=1)
    noisy_2_validation_mean = noisy_2_validation.mean(axis=0)
    noisy_2_validation_std = np.std(noisy_2_validation, axis=0, ddof=1)

    clean_1_pos_mean = clean_1_pos.mean()
    noisy_1_pos_mean = noisy_1_pos.mean()
    clean_2_pos_mean = clean_2_pos.mean()
    noisy_2_pos_mean = noisy_2_pos.mean()

    clean_1_tang_mean = clean_1_tang.mean()
    noisy_1_tang_mean = noisy_1_tang.mean()
    clean_2_tang_mean = clean_2_tang.mean()
    noisy_2_tang_mean = noisy_2_tang.mean()

    epochs = np.arange(1, len(clean_1_train_mean) + 1)

    # Plotting
    plt.figure()
    plt.semilogy(epochs, clean_1_train_mean, label="Mean Clean One Segment Loss")
    plt.fill_between(
        epochs,
        clean_1_train_mean - clean_1_train_std,
        clean_1_train_mean + clean_1_train_std,
        alpha=0.3,
        label="_c1std",
    )
    plt.semilogy(epochs, noisy_1_train_mean, label="Mean Noisy One Segment Loss")
    plt.fill_between(
        epochs,
        noisy_1_train_mean - noisy_1_train_std,
        noisy_1_train_mean + noisy_1_train_std,
        alpha=0.3,
        label="_n1std",
    )
    plt.semilogy(epochs, clean_2_train_mean, label="Mean Clean Two Segment Loss")
    plt.fill_between(
        epochs,
        clean_2_train_mean - clean_2_train_std,
        clean_2_train_mean + clean_2_train_std,
        alpha=0.3,
        label="_c2std",
    )
    plt.semilogy(epochs, noisy_2_train_mean, label="Mean Noisy Two Segment Loss")
    plt.fill_between(
        epochs,
        noisy_2_train_mean - noisy_2_train_std,
        noisy_2_train_mean + noisy_2_train_std,
        alpha=0.3,
        label="_n2std",
    )
    plt.title("Average Batch Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.ylim((0.5, 100))

    plt.figure()
    plt.semilogy(epochs, clean_1_validation_mean, label="Mean Clean One Segment Loss")
    plt.fill_between(
        epochs,
        clean_1_validation_mean - clean_1_validation_std,
        clean_1_validation_mean + clean_1_validation_std,
        alpha=0.3,
        label="_c1std",
    )
    plt.semilogy(epochs, noisy_1_validation_mean, label="Mean Noisy One Segment Loss")
    plt.fill_between(
        epochs,
        noisy_1_validation_mean - noisy_1_validation_std,
        noisy_1_validation_mean + noisy_1_validation_std,
        alpha=0.3,
        label="_n1std",
    )
    plt.semilogy(epochs, clean_2_validation_mean, label="Mean Clean Two Segment Loss")
    plt.fill_between(
        epochs,
        clean_2_validation_mean - clean_2_validation_std,
        clean_2_validation_mean + clean_2_validation_std,
        alpha=0.3,
        label="_c2std",
    )
    plt.semilogy(epochs, noisy_2_validation_mean, label="Mean Noisy Two Segment Loss")
    plt.fill_between(
        epochs,
        noisy_2_validation_mean - noisy_2_validation_std,
        noisy_2_validation_mean + noisy_2_validation_std,
        alpha=0.3,
        label="_n2std",
    )
    plt.title("Average Batch Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.ylim((0.5, 100))

    plt.figure()
    plt.boxplot(
        [clean_1_pos.flatten(), noisy_1_pos.flatten()],
        showfliers=False,
        labels=[
            f"Clean (Mean: {clean_1_pos_mean:.3f} mm)",
            f"Noisy (Mean: {noisy_1_pos_mean:.3f} mm)",
        ],
    )
    plt.title("Position Error of One-Segment Models")
    plt.xlabel("Dataset")
    plt.ylabel("Position Error (mm)")

    plt.figure()
    plt.boxplot(
        [clean_1_tang.flatten(), noisy_1_tang.flatten()],
        showfliers=False,
        labels=[
            f"Clean (Mean: ${clean_1_tang_mean:.3f}^\circ$)",
            f"Noisy (Mean: ${noisy_1_tang_mean:.3f}^\circ$)",
        ],
    )
    plt.title("Orientation Error of One-Segment Models")
    plt.xlabel("Dataset")
    plt.ylabel("Orientation Error (degrees)")

    plt.figure()
    plt.boxplot(
        [clean_2_pos.flatten(), noisy_2_pos.flatten()],
        showfliers=False,
        labels=[
            f"Clean (Mean: {clean_2_pos_mean:.3f} mm)",
            f"Noisy (Mean: {noisy_2_pos_mean:.3f} mm)",
        ],
    )
    plt.title("Position Error of Two-Segment Models")
    plt.xlabel("Dataset")
    plt.ylabel("Position Error (mm)")

    plt.figure()
    plt.boxplot(
        [clean_2_tang.flatten(), noisy_2_tang.flatten()],
        showfliers=False,
        labels=[
            f"Clean (Mean: ${clean_2_tang_mean:.3f}^\circ$)",
            f"Noisy (Mean: ${noisy_2_tang_mean:.3f}^\circ$)",
        ],
    )
    plt.title("Orientation Error of Two-Segment Models")
    plt.xlabel("Dataset")
    plt.ylabel("Orientation Error (degrees)")

    plt.show()
