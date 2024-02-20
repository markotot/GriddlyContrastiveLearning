import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    random_files = np.load("../results/Random-CNN-acc.npz")
    ppo_files = np.load("../results/PPO-CNN-acc.npz")
    fal_files = np.load("../results/FAL-CNN-acc.npz")

    random_train, random_test = random_files[random_files.files[0]], random_files[random_files.files[1]]
    ppo_train, ppo_test = ppo_files[ppo_files.files[0]], ppo_files[ppo_files.files[1]]
    fal_train, fal_test = fal_files[fal_files.files[0]], fal_files[fal_files.files[1]]

    x = range(fal_train.shape[0])

    plt.plot(x, random_test, label="Random")
    plt.plot(x, ppo_test, label="PPO")
    plt.plot(x, fal_test, label="FAL")
    plt.title("Classification Accuracy")
    plt.ylim(0.49, 1.01)
    plt.xlim(0, fal_train.shape[0])
    plt.xlabel("Batches Trained")
    plt.legend()
    plt.show()



