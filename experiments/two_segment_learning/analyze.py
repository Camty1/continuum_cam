#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

data_size_folder = "output/data_size/"
model_size_folder = "output/model_size/"
data_size_pos_bins = np.arange(120) / 120 * 30
data_size_tang_bins = np.arange(120) / 120 * 2 * 180 / np.pi

# Figures that need to be reopened
plt.figure(0)
plt.figure(1)
plt.figure(2)
plt.figure(3)

for i in range(13, 16):
    train_loss = np.loadtxt(f"{data_size_folder}train_loss_{i}.dat", delimiter=",")
    validation_loss = np.loadtxt(
        f"{data_size_folder}validation_loss_{i}.dat", delimiter=","
    )
    pos_loss = np.loadtxt(f"{data_size_folder}pos_loss_{i}.dat", delimiter=",")
    tang_loss = np.loadtxt(f"{data_size_folder}tang_loss_{i}.dat", delimiter=",")
    title_string = f"Loss vs Epoch 2^{i} Points"
    avg_train_loss = train_loss.mean(axis=0)
    avg_validation_loss = validation_loss.mean(axis=0)

    std_train_loss = np.std(train_loss, axis=0, ddof=1)
    std_validation_loss = np.std(validation_loss, axis=0, ddof=1)

    plt.figure()
    x = np.arange(1, len(avg_train_loss) + 1)
    plt.semilogy(
        x,
        avg_train_loss,
        label=f"Training Loss (Avg min: {train_loss.min(axis=1).mean():.3f})",
    )
    plt.fill_between(
        x,
        avg_train_loss - std_train_loss,
        avg_train_loss + std_train_loss,
        alpha=0.3,
        label="_clean one segment std",
    )
    plt.semilogy(
        x,
        avg_validation_loss,
        label=f"Validation Loss (Avg min: {validation_loss.min(axis=1).mean():.3f})",
    )
    plt.fill_between(
        x,
        avg_validation_loss - std_validation_loss,
        avg_validation_loss + std_validation_loss,
        alpha=0.3,
        label="_clean one segment std",
    )
    plt.title(title_string)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim((1, 100))

    plt.figure(0)
    plt.hist(
        pos_loss.flatten(),
        data_size_pos_bins,
        alpha=0.3,
        label=f"2^{i} Points, Mean: {pos_loss.mean():.3f} mm",
    )
    plt.title("Position Error Histogram for Different Training Data Sizes")
    plt.xlabel("Position Error (mm)")
    plt.ylabel("Count")
    plt.legend()

    plt.figure(1)
    plt.hist(
        tang_loss.flatten() * 180 / np.pi,
        data_size_tang_bins,
        alpha=0.3,
        label=f"2^{i} Points, Mean: {tang_loss.mean() * 180 / np.pi:.3f} degrees",
    )
    plt.title("Orientation Error Histogram for Different Training Data Sizes")
    plt.xlabel("Orientation Error (deg)")
    plt.ylabel("Count")
    plt.legend()


for i in range(3):
    size = 32 * 2**i
    train_loss = np.loadtxt(f"{model_size_folder}train_loss_{size}.dat", delimiter=",")
    validation_loss = np.loadtxt(
        f"{model_size_folder}validation_loss_{size}.dat", delimiter=","
    )
    pos_loss = np.loadtxt(f"{model_size_folder}pos_loss_{size}.dat", delimiter=",")
    tang_loss = np.loadtxt(f"{model_size_folder}tang_loss_{size}.dat", delimiter=",")
    title_string = f"Loss vs Epoch with [{size}, {size}] Hidden Neurons"
    avg_train_loss = train_loss.mean(axis=0)
    avg_validation_loss = validation_loss.mean(axis=0)

    std_train_loss = np.std(train_loss, axis=0, ddof=1)
    std_validation_loss = np.std(validation_loss, axis=0, ddof=1)

    plt.figure()
    x = np.arange(1, len(avg_train_loss) + 1)
    plt.semilogy(
        x,
        avg_train_loss,
        label=f"Training Loss (Avg min: {train_loss.min(axis=1).mean():.3f})",
    )
    plt.fill_between(
        x,
        avg_train_loss - std_train_loss,
        avg_train_loss + std_train_loss,
        alpha=0.3,
        label="_clean one segment std",
    )
    plt.semilogy(
        x,
        avg_validation_loss,
        label=f"Validation Loss (Avg min: {validation_loss.min(axis=1).mean():.3f})",
    )
    plt.fill_between(
        x,
        avg_validation_loss - std_validation_loss,
        avg_validation_loss + std_validation_loss,
        alpha=0.3,
        label="_clean one segment std",
    )
    plt.title(title_string)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim((1, 100))

    plt.figure(2)
    plt.hist(
        pos_loss.flatten(),
        data_size_pos_bins,
        alpha=0.3,
        label=f"[{size}, {size}] Hidden Neurons, Mean: {pos_loss.mean():.3f} mm",
    )
    plt.title("Position Error Histogram for Different Model Sizes")
    plt.xlabel("Position Error (mm)")
    plt.ylabel("Count")
    plt.legend()

    plt.figure(3)
    plt.hist(
        tang_loss.flatten() * 180 / np.pi,
        data_size_tang_bins,
        alpha=0.3,
        label=f"[{size}, {size}] Hidden Neurons, Mean: {tang_loss.mean() * 180 / np.pi:.3f} degrees",
    )
    plt.title("Orientation Error Histogram for Different Model Sizes")
    plt.xlabel("Orientation Error (deg)")
    plt.ylabel("Count")
    plt.legend()

best = True
if best:
    train_loss = np.loadtxt(f"output/best/train_loss.dat", delimiter=",")
    validation_loss = np.loadtxt(f"output/best/validation_loss.dat", delimiter=",")
    pos_loss = np.loadtxt(f"output/best/pos_loss.dat", delimiter=",")
    tang_loss = np.loadtxt(f"output/best/tang_loss.dat", delimiter=",")

    avg_train_loss = train_loss.mean(axis=0)
    avg_validation_loss = validation_loss.mean(axis=0)

    std_train_loss = np.std(train_loss, axis=0, ddof=1)
    std_validation_loss = np.std(validation_loss, axis=0, ddof=1)

    plt.figure()
    x = np.arange(1, len(avg_train_loss) + 1)
    plt.semilogy(
        x,
        avg_train_loss,
        label=f"Training Loss (Avg min: {train_loss.min(axis=1).mean():.3f})",
    )
    plt.fill_between(
        x,
        avg_train_loss - std_train_loss,
        avg_train_loss + std_train_loss,
        alpha=0.3,
        label="_clean one segment std",
    )
    plt.semilogy(
        x,
        avg_validation_loss,
        label=f"Validation Loss (Avg min: {validation_loss.min(axis=1).mean():.3f})",
    )
    plt.fill_between(
        x,
        avg_validation_loss - std_validation_loss,
        avg_validation_loss + std_validation_loss,
        alpha=0.3,
        label="_clean one segment std",
    )
    plt.title("Loss vs Epoch for [128, 128] Model with 2^15 Datapoints")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim((1, 100))

    plt.figure()
    plt.hist(
        pos_loss.flatten(),
        data_size_pos_bins,
        label=f"Mean: {pos_loss.mean():.3f} mm",
    )
    plt.title("Position Error Histogram for [128, 128] Model with 2^15 Datapoints")
    plt.xlabel("Position Error (mm)")
    plt.ylabel("Count")
    plt.legend()

    plt.figure()
    plt.hist(
        tang_loss.flatten() * 180 / np.pi,
        data_size_tang_bins,
        label=f"Mean: {tang_loss.mean() * 180 / np.pi:.3f} degrees",
    )
    plt.title("Orientation Error Histogram for [128, 128] Model with 2^15 Datapoints")
    plt.xlabel("Orientation Error (deg)")
    plt.ylabel("Count")
    plt.legend()

plt.show()
