from matplotlib import pyplot as plt

from core.environment import make_env
from core.network import PPONetwork
import numpy as np

import torch
import torch.nn.functional as F


def get_agent_actions(num_total_actions, env_config, ckpt_path, label_actions=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(f"../configs/{env_config}", 0, 0, 0, 0)()
    env.single_action_space = env.action_space

    agent = PPONetwork(env).to(device)
    agent.load_state_dict(torch.load(f"../checkpoints/ppo/{ckpt_path}"))

    observations = np.zeros(shape=(num_total_actions,) + env.observation_space.shape, dtype=np.uint8)
    encodings = np.zeros(shape=(num_total_actions, 512))
    actions = np.zeros(shape=(num_total_actions,))

    obs = env.reset()
    obs = torch.tensor(obs).to(device)

    for step in range(0, num_total_actions):

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
            encoding = agent.get_latent_encoding(obs.unsqueeze(0))
        encodings[step] = encoding.cpu().numpy()
        observations[step] = obs.cpu().numpy()

        if label_actions is None:
            obs, reward, done, info = env.step(action.cpu().numpy())
        else:
            obs, reward, done, info = env.step(label_actions[step])

        obs = torch.from_numpy(obs).to(device)

        if done:
            env.reset()
        actions[step] = action

    return actions, observations, encodings

def plot_sample(original_obs, paired_obs, contrastive_loss, plot_name):
    obs = np.moveaxis(original_obs, 0, -1)
    pair = np.moveaxis(paired_obs, 0, -1)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(obs)
    axes[0].set_title("Original")
    axes[1].imshow(pair)

    plt.suptitle(f"{plot_name} Similarity: {contrastive_loss:.5f}")
    if "positive" in plot_name:
        fig.patch.set_facecolor('xkcd:mint green')
        axes[1].set_title("Positive")
    elif "negative" in plot_name:
        fig.patch.set_facecolor('xkcd:dark pink')
        axes[1].set_title("Negative")
    plt.show()


def check_if_action_sequence_is_same():

    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    num_agent_actions = 200
    original_env_name = "cluster-1-floor.yaml"
    ckpt_path = "ppo_1_after_c-cnn-6.ckpt"
    actions_a, observations_a, encodings_a = get_agent_actions(num_agent_actions,
                                        env_config=original_env_name,
                                        ckpt_path=ckpt_path)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    original_env_name = "cluster-9-red-walls.yaml"
    actions_b, observations_b, encodings_b = get_agent_actions(num_agent_actions,
                                        env_config=original_env_name,
                                        ckpt_path=ckpt_path,
                                        label_actions=actions_a)

    equal_actions = (actions_a == actions_b)
    num_equal = np.sum(equal_actions)
    num_diff = num_agent_actions - num_equal

    cosine_similarities = [np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) for a, b in zip(encodings_a, encodings_b)]
    norm_distances = [np.linalg.norm(a-b) for a, b in zip(encodings_a, encodings_b)]
    print(f"Mean cosine similarity: {np.mean(cosine_similarities):.5f}")
    print(f"Mean norm distance: {np.mean(norm_distances):.5f}")
    print(f"Total actions: {num_agent_actions}")
    print(f"Same actions: {num_equal} ({num_equal/num_agent_actions*100:.2f}%)")
    print(f"Diff actions: {num_diff} ({num_diff/num_agent_actions*100:.2f}%)")


def batch_euclidean_distance(anchor_batch, positive_batch, negative_batch):
    # Broadcast the vectors to have the same shape as the matrices

    positive_distances = -1 * np.linalg.norm(anchor_batch - positive_batch, axis=1)

    # Calculate the Euclidean distances
    anchor_broadcasted = np.tile(anchor_batch[:, np.newaxis, :], (1, negative_batch.shape[1], 1))
    negative_distances = -1 * np.linalg.norm(negative_batch - anchor_broadcasted, axis=2)

    # softmax_fn = torch.nn.Softmax(dim=1)
    loss = torch.nn.CrossEntropyLoss()

    logits = np.concatenate((np.expand_dims(positive_distances, axis=1), negative_distances), axis=1)
    # logits = softmax_fn(torch.from_numpy(logits))
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    output = loss(torch.from_numpy(logits), labels)
    return output

def torch_batch_euclidean_distance(temperature, anchor_batch, positive_batch, negative_batch):

    # Calculate the Euclidean distances
    positive_distances = -torch.norm(anchor_batch - positive_batch, dim=1)

    # Broadcast the vectors to have the same shape as the matrices
    anchor_broadcasted = anchor_batch.unsqueeze(1).expand(-1, negative_batch.shape[1], -1)
    negative_distances = -torch.norm(negative_batch - anchor_broadcasted, dim=2)

    # Apply softmax along dimension 1
    logits = torch.cat((positive_distances.unsqueeze(1), negative_distances), dim=1)

    # Create labels tensor
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    # Calculate CrossEntropyLoss
    output = F.cross_entropy(logits / temperature, labels)

    return output

if __name__ == "__main__":
    check_if_action_sequence_is_same()

    enc_length = 512
    batch_size = 64
    num_negatives = 128

    anchor = torch.rand(batch_size, enc_length)
    positive = torch.rand(batch_size, enc_length)
    negative = torch.rand(batch_size, num_negatives, enc_length) * 10


    loss = torch_batch_euclidean_distance(0.5, anchor, positive, negative)
    print(f"Loss: {loss}")





