import matplotlib.pyplot as plt
import numpy as np


def visualize(total_loss_file_name, mel_loss_file_name, gate_loss_file_name):
    plt.figure()

    total_loss_arr = np.array(list())
    with open(total_loss_file_name, "r") as f_total_loss:
        for loss in f_total_loss.readlines():
            total_loss_arr = np.append(total_loss_arr, float(loss))

    x = np.array([i for i in range(np.shape(total_loss_arr)[0])])
    y = total_loss_arr

    plt.plot(x, y, color="y", lw=0.7, label="total loss")

    mel_loss_arr = np.array(list())
    with open(mel_loss_file_name, "r") as f_mel_loss:
        for loss in f_mel_loss.readlines():
            mel_loss_arr = np.append(mel_loss_arr, float(loss))

    x = np.array([i for i in range(np.shape(mel_loss_arr)[0])])
    y = mel_loss_arr

    plt.plot(x, y, color="r", lw=0.7, label="mel loss")

    gate_loss_arr = np.array(list())
    with open(gate_loss_file_name, "r") as f_gate_loss:
        for loss in f_gate_loss.readlines():
            gate_loss_arr = np.append(gate_loss_arr, float(loss))

    x = np.array([i for i in range(np.shape(gate_loss_arr)[0])])
    y = gate_loss_arr

    plt.plot(x, y, color="b", lw=0.7, label="gate loss")

    plt.legend()
    plt.xlabel("sequence number")
    plt.ylabel("loss item")
    plt.title("loss")
    plt.savefig("loss.jpg")


if __name__ == "__main__":
    # Test
    visualize("total_loss.txt", "mel_loss.txt", "gate_loss.txt")
