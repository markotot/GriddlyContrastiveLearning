import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification  # Example function to generate labeled data

from contrastive_cnn import load_dataset
from core.environment import make_env
from core.network import PPONetwork
import torch


def preprocess_data(agent, full_dataset, num_datapoints):
    labels_background = []
    labels_positive = []
    data = []

    # idx = np.random.permutation(range(num_samples))

    num_samples = len(list(full_dataset.values())[0])
    idx = np.random.choice(range(num_samples), size=num_datapoints, replace=False)


    for key in full_dataset.keys():
        n = 0
        for observation in full_dataset[key]:
            if n in idx:
                data.append(observation)
                labels_background.append(key)
                labels_positive.append(n)
            n += 1

    encodings = agent.get_latent_encoding(torch.tensor(np.array(data)).to(device))
    return encodings, labels_background, labels_positive

def PCA_visualization_by_background(agent, full_dataset, num_datapoints, plot_title):


    print("Creating encodings")
    encodings, labels_background, labels_positive = preprocess_data(agent, full_dataset, num_datapoints)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(encodings.detach().cpu().numpy())

    df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df['label_background'] = labels_background  # Adding labels for background colour to the DataFrame
    df['label_positive'] = labels_positive  # Adding labels for positive pairs to the DataFrame
    # Plotting data points with different labels using different colors

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
    # Plotting each class separately with a different color
    axes[0].set_title("Background")
    for label in np.unique(labels_background):
        class_data = df[df['label_background'] == label]
        axes[0].scatter(class_data['PC1'], class_data['PC2'], label=f'Class {label}', s=5)

    axes[1].set_title("Positive pairs")
    for label in np.unique(labels_positive):
        class_data = df[df['label_positive'] == label]
        axes[1].scatter(class_data['PC1'], class_data['PC2'], label=f'Class {label}', s=5)

    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ppo_agent_ckpt = "ppo_ood_red_white.ckpt"
    env = make_env(f"configs/cluster-1-floor.yaml", 0, 0, 0, 0)()
    env.single_action_space = env.action_space
    agent = PPONetwork(env).to(device)

    # agent.load_checkpoint(ppo_agent_ckpt)
    # agent.load_state_dict(torch.load(f"contrastive_cnn.ckpt"))
    agent.load_checkpoint("contrastive_cnn_97.ckpt")
    env_configs = [
        "cluster-1-floor.yaml",
        "cluster-2-grass-15.yaml",
        "cluster-3-orange.yaml",
        "cluster-4-lbrown.yaml",
        "cluster-5-lblue.yaml",
        "cluster-6-biege.yaml",
        "cluster-7-space.yaml",
        "cluster-8-grey.yaml",
        "cluster-9-red.yaml",
        "cluster-10-fill.yaml",
    ]

    ood_env_configs = [
        "cluster-9-red.yaml",
    ]

    # Load the data
    train_dataset, test_dataset, ood_train_dataset, ood_test_dataset = load_dataset(env_configs, ood_env_configs)

    train_dataset.update(ood_train_dataset)
    test_dataset.update(ood_test_dataset)

    # PCA_visualization_by_background(agent, train_dataset, num_datapoints=10, plot_title="PCA Visualization of Training Data")
    PCA_visualization_by_background(agent, test_dataset, num_datapoints=10, plot_title="PCA Visualization of Testing Data")