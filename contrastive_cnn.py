import copy
import time

from info_nce import InfoNCE
from matplotlib import pyplot as plt
from torch import optim

from core.config_generator import get_mixed_env_configs, get_background_env_configs
from core.network import PPONetwork
import numpy as np
import torch
import torch.nn.functional as F
from core.environment import make_env


def get_samples(dataset, num_negative_samples, same_env_percent):
    # Select anchor and test envs
    env_names = list(dataset.keys())
    anchor_env = np.random.choice(env_names)
    env_names.remove(anchor_env)
    test_envs = np.random.choice(env_names)

    # Select anchor, positive, and negative sample idxs
    step = np.random.randint(0, len(dataset[anchor_env]) - 1)
    anchor_observation = dataset[anchor_env][step]
    positive_observation = dataset[test_envs][step]
    negative_observation = np.zeros(shape=(num_negative_samples, *anchor_observation.shape), dtype=np.float32)
    available_idx = np.arange(len(dataset[anchor_env]))
    available_idx = np.delete(available_idx, step)  # Remove the positive sample idx

    for x in range(0, num_negative_samples):
        negative_idx = np.random.choice(range(0, len(available_idx)))
        available_idx = np.delete(available_idx, negative_idx)  # Remove the negative sample idx that was just selected

        # Select negative sample from anchor env or test env
        if np.random.random() < same_env_percent:
            negative_observation[x] = dataset[anchor_env][negative_idx]
        else:
            negative_observation[x] = dataset[test_envs][negative_idx]

    return anchor_observation, positive_observation, negative_observation


def better_plot(observations, losses, plot_name, pairwise=False):
    for n, obs in enumerate(observations):
        observations[n] = np.moveaxis(obs, 0, -1)

    n_cols = 4

    n_rows = max(1, int(np.ceil(len(observations) / n_cols)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))


    n = 0
    for i in range(n_rows):
        for j in range(n_cols):

            if n >= len(observations):
                continue

            if axes.shape[0] > 1:
                axes[i][j].axis('off')
                axes[i][j].imshow(observations[n])
                axes[i][j].set_title(losses[n])
            else:
                axes[j].axis('off')
                axes[j].imshow(observations[n])
                axes[j].set_title(losses[n])
            n += 1


    plt.suptitle(plot_name)

    if "positive" in plot_name:
        fig.patch.set_facecolor('xkcd:mint green')
    elif "negative" in plot_name:
        fig.patch.set_facecolor('xkcd:dark pink')
    plt.show()


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


def test_same_background(agent, anchor_dataset, verbose_num, print_details, plot_name="Test"):

    test_dataset = copy.deepcopy(anchor_dataset)
    num_samples = anchor_dataset.shape[1]

    idx = np.random.permutation(range(num_samples))
    test_dataset = test_dataset[:, idx]

    background_similarities = {}
    for idx in range(anchor_dataset.shape[0]):
        with torch.no_grad():
            anchor_encodigs = agent.get_latent_encoding(torch.from_numpy(anchor_dataset[idx]).to(device))
            test_encodings = agent.get_latent_encoding(torch.from_numpy(test_dataset[idx]).to(device))

        cos_similarities = torch.cosine_similarity(anchor_encodigs, test_encodings, dim=1).cpu().detach().numpy()
        value_similarities = torch.mean(torch.abs(anchor_encodigs - test_encodings), dim=1).cpu().detach().numpy()

        background_similarities[idx] = (cos_similarities, value_similarities)

        if verbose_num > 0:
            plot_observations = []
            plot_similarities = []
            for n in range(verbose_num):
                plot_observations.append(anchor_dataset[idx][n])
                plot_observations.append(test_dataset[idx][n])
                plot_similarities.append(f"anchor")
                plot_similarities.append(f"{np.mean(cos_similarities):.4f}")

            better_plot(plot_observations,
                        plot_similarities,
                        plot_name=f"{plot_name} mean:{np.mean(cos_similarities):.3f}",
                        pairwise=True
                        )

    average_cos_similarities = []
    average_val_similarities = []
    for key in background_similarities.keys():
        average_cos_similarities.append(np.mean(background_similarities[key][0]))
        average_val_similarities.append(np.mean(background_similarities[key][1]))
        if print_details is True:
            print(f"{key:50} average:{np.mean(background_similarities[key][0]):.2f}\t"
                  f" std:{np.std(background_similarities[key][0]):.6f}\t"
                  f" median:{np.median(background_similarities[key][0]):.2f}\t"
                  f" min:{min(background_similarities[key][0]):.2f}\t"
                  f" max:{max(background_similarities[key][0]):.2f}")
    print(f"{plot_name} average cos similarity: {np.mean(average_cos_similarities):.5f}")
    print(f"{plot_name} average abs value difference: {np.mean(average_val_similarities):.5f}")


def test_out_of_distribution(agent, anchor_dataset, test_dataset, shuffled, verbose_num, print_details, plot_name="Test"):

    np.random.seed(0)

    if shuffled:
        test_dataset = np.random.permutation(test_dataset.squeeze())
        test_dataset = np.expand_dims(test_dataset, axis=0)

    average_env_similarities = {}
    for data_idx in range(anchor_dataset.shape[0]):

        with torch.no_grad():
            anchor_env_encodings = agent.get_latent_encoding(torch.from_numpy(anchor_dataset[data_idx]).to(device))
            norm_ancor = F.layer_norm(anchor_env_encodings, anchor_env_encodings.shape[1:])
        for ood_data_idx in range(test_dataset.shape[0]):

            with torch.no_grad():
                ood_env_encodings = agent.get_latent_encoding(torch.from_numpy(test_dataset[ood_data_idx]).to(device))
                norm_ood = F.layer_norm(ood_env_encodings, ood_env_encodings.shape[1:])
            cos_similarities = torch.cosine_similarity(anchor_env_encodings, ood_env_encodings, dim=1).cpu().detach().numpy()
            value_similarities = torch.mean(torch.abs(anchor_env_encodings - ood_env_encodings), dim=1).cpu().detach().numpy()

            average_env_similarities[f"{data_idx}_{ood_data_idx}"] = (cos_similarities, value_similarities)

            if verbose_num > 0:
                plot_observations = []
                plot_similarities = []
                for n in range(verbose_num):
                    plot_observations.append(anchor_dataset[data_idx][n])
                    plot_observations.append(test_dataset[ood_data_idx][n])
                    plot_similarities.append(f"anchor")
                    plot_similarities.append(f"{np.mean(cos_similarities):.4f}")

                better_plot(plot_observations,
                            plot_similarities,
                            plot_name=f"{plot_name} mean:{np.mean(cos_similarities):.3f}",
                            pairwise=True
                            )

    average_cos_similarities = []
    average_value_similarities = []
    for key in average_env_similarities.keys():
        average_cos_similarities.append(np.mean(average_env_similarities[key][0]))
        average_value_similarities.append(np.mean(average_env_similarities[key][1]))
        if print_details is True:
            print(f"{key:50} average:{np.mean(average_env_similarities[key][0]):.2f}\t"
                  f" std:{np.std(average_env_similarities[key][0]):.6f}\t"
                  f" median:{np.median(average_env_similarities[key][0]):.2f}\t"
                  f" min:{min(average_env_similarities[key][0]):.2f}\t"
                  f" max:{max(average_env_similarities[key][0]):.2f}")
    print(f"{plot_name} average cos similarity: {np.mean(average_cos_similarities):.5f}")
    print(f"{plot_name} average absolute value difference: {np.mean(average_value_similarities):.5f}")


def create_batch(dataset, shuffled_idxs, num_envs, num_samples, batch_idx, batch_size, num_negative):
    # Select anchor observations
    anchor_envs = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size) % num_envs
    anchor_obs_ind = shuffled_idxs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    batched_anchor_obs = dataset[anchor_envs, anchor_obs_ind]

    # Select positive observations
    all_envs = np.arange(num_envs).reshape(1, -1) + np.zeros(shape=(batch_size, num_envs), dtype=np.uint8)
    positive_mask = (np.arange(all_envs.shape[1]) != anchor_envs[:, None])  # Create a mask for removing the anchor envs
    positive_envs = all_envs[positive_mask]  # Remove anchor envs
    positive_envs = np.reshape(positive_envs, (all_envs.shape[0], all_envs.shape[1] - 1))
    selected_positive_envs_idx = np.random.randint(0, positive_envs.shape[1], size=batch_size)
    positive_envs = np.take_along_axis(positive_envs, selected_positive_envs_idx[:, np.newaxis], axis=1).reshape(-1)
    batched_positive_obs = dataset[positive_envs, anchor_obs_ind]

    # Select batch_size negative envs
    same_envs_flag = (np.random.random(size=batch_size) < 0.5) * anchor_envs
    negative_mask = (np.arange(all_envs.shape[1]) != anchor_envs[:, None])  # Create a mask for removing the anchor envs
    negative_envs = all_envs[negative_mask]  # Remove anchor envs
    negative_envs = np.reshape(negative_envs, (all_envs.shape[0], all_envs.shape[1] - 1))
    selected_negative_envs_idx = np.random.randint(0, negative_envs.shape[1], size=batch_size)
    selected_negative_envs_idx = np.where(same_envs_flag == 0, selected_negative_envs_idx, same_envs_flag)
    negative_envs = np.take_along_axis(all_envs, selected_negative_envs_idx[:, np.newaxis], axis=1).reshape(-1)

    # Select num_negative observations for each negative env
    all_idx = np.arange(num_samples).reshape(1, -1) + np.zeros(shape=(batch_size, num_samples), dtype=np.int32)
    negative_idx = all_idx[np.arange(num_samples) != anchor_obs_ind[:, None]].reshape((batch_size, -1))
    selected_negative_obs_idx = np.random.randint(0, negative_idx.shape[1], size=(batch_size, num_negative))
    batched_negative_obs = dataset[negative_envs[:, None], selected_negative_obs_idx]
    batched_negative_obs = np.reshape(batched_negative_obs, (batch_size * num_negative, 3, 84, 84))

    return batched_anchor_obs, batched_positive_obs, batched_negative_obs

def train_model(agent, optimizer, loss_fn, batch_size, num_negative, train_dataset, device):
    num_envs = train_dataset.shape[0]
    num_samples = train_dataset.shape[1]

    losses = []
    shuffled_idxs = np.random.permutation(range(num_samples))
    all_logits = np.zeros(shape=(num_negative + 1,))
    for n in range(num_samples // batch_size):
        optimizer.zero_grad()
        anchor_obs, positive_obs, negative_obs = create_batch(dataset=train_dataset,
                                                              shuffled_idxs=shuffled_idxs,
                                                              num_envs=num_envs,
                                                              num_samples=num_samples,
                                                              batch_idx=n,
                                                              batch_size=batch_size,
                                                              num_negative=num_negative
                                                              )

        # Get the encodings
        anchor_encodings = agent.get_latent_encoding(torch.from_numpy(anchor_obs).to(device))
        positive_encodings = agent.get_latent_encoding(torch.from_numpy(positive_obs).to(device))
        negative_encodings = agent.get_latent_encoding(torch.from_numpy(negative_obs).to(device))
        negative_encodings = torch.reshape(negative_encodings, (batch_size, num_negative, 512))

        loss, logits = loss_fn(anchor_encodings, positive_encodings, negative_encodings)
        all_logits += logits.cpu().detach().numpy().mean(axis=0)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())

    avg_logits = all_logits / (num_samples // batch_size)
    return np.array(losses), avg_logits


def test_model(agent, test_dataset, ood_test_dataset, verbose_num, print_details):
    print(f"----------Testing positive----------")
    test_out_of_distribution(agent,
                             anchor_dataset=ood_test_dataset,
                             test_dataset=test_dataset,
                             shuffled=False,
                             verbose_num=verbose_num,
                             print_details=print_details,
                             plot_name="contrastive_positive"
                             )

    print(f"----------Testing negative----------")
    test_out_of_distribution(agent,
                             anchor_dataset=ood_test_dataset,
                             test_dataset=test_dataset,
                             shuffled=True,
                             verbose_num=verbose_num,
                             print_details=print_details,
                             plot_name="contrastive_negative"
                             )

    print(f"----------Testing same background ood----------")
    test_same_background(agent,
                         anchor_dataset=ood_test_dataset,
                         verbose_num=verbose_num,
                         print_details=print_details,
                         plot_name="contrastive_same_background_ood"
                         )

    print(f"----------Testing same background ----------")
    test_same_background(agent,
                         anchor_dataset=test_dataset,
                         verbose_num=verbose_num,
                         print_details=print_details,
                         plot_name="contrastive_same_background"
                         )


def load_dataset(env_configs, ood_env_configs, train_percentage=0.8, max_samples=50_000):

    train_env_num = len(env_configs) - len(ood_env_configs)
    test_env_num = len(ood_env_configs)
    total_samples = None
    dataset = None
    ood_dataset = None

    dataset_idx = 0
    ood_dataset_idx = 0
    for env_config in env_configs:

        npz = np.load(f"datasets/{env_config}.npz")
        data_array = npz[npz.files[0]]

        if total_samples is None:
            total_samples = min(data_array.shape[0], max_samples)
            dataset = np.zeros(shape=(train_env_num, total_samples, 3, 84, 84), dtype=np.uint8)
            ood_dataset = np.zeros(shape=(test_env_num, total_samples, 3, 84, 84), dtype=np.uint8)

        if env_config in ood_env_configs:

            if data_array.shape[0] > max_samples:
                ood_dataset[ood_dataset_idx] = data_array[:max_samples]
            else:
                ood_dataset[ood_dataset_idx] = data_array

                ood_dataset_idx += 1
            ood_dataset_idx += 1
        elif env_config in env_configs:
            if data_array.shape[0] > max_samples:
                dataset[dataset_idx] = data_array[:max_samples]
            else:
                dataset[dataset_idx] = data_array
            dataset_idx += 1

    train_num = int(total_samples * train_percentage)

    train_dataset = dataset[:, :train_num]
    test_dataset = dataset[:, train_num:]
    ood_train_dataset = ood_dataset[:, :train_num]
    ood_test_dataset = ood_dataset[:, train_num:]

    return train_dataset, test_dataset, ood_train_dataset, ood_test_dataset

def load_agent_model(ppo_agent_ckpt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(f"configs/fully-observable-back/cluster-1-floor.yaml", 0, 0, 0, 0)()
    env.single_action_space = env.action_space
    agent = PPONetwork(env).to(device)
    if ppo_agent_ckpt is not None:
        agent.load_checkpoint(ppo_agent_ckpt)
    return agent

def euclidian_contrastive_loss(temperature, anchor_batch, positive_batch, negative_batch):
    # Calculate the Euclidean distances
    positive_distances = -torch.norm(anchor_batch - positive_batch, dim=1)

    # Broadcast the vectors to have the same shape as the matrices
    anchor_broadcasted = anchor_batch.unsqueeze(1).expand(-1, negative_batch.shape[1], -1)
    negative_distances = -torch.norm(negative_batch - anchor_broadcasted, dim=2)

    # Apply softmax along dimension 1 -- comment: is this correct??
    logits = torch.cat((positive_distances.unsqueeze(1), negative_distances), dim=1)

    # Create labels tensor
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor_batch.device)

    # Calculate CrossEntropyLoss
    output = torch.nn.functional.cross_entropy(logits / temperature, labels)

    return output


if __name__ == "__main__":

    # Making the training process deterministic
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ppo_agent_ckpt = "training/contrastive_cnn_5.ckpt"
    ppo_agent_ckpt = "contrastive_cnn_983_backup.ckpt"

    # Get the environment configurations
    game_name = "butterflies"
    all_env_configs, ood_env_configs = get_background_env_configs(template_name=game_name, train=False)

    # Load the data
    train_dataset, test_dataset, ood_train_dataset, ood_test_dataset = load_dataset(all_env_configs,
                                                                                    [ood_env_configs],
                                                                                    train_percentage=0.9)
    agent = load_agent_model(ppo_agent_ckpt=ppo_agent_ckpt)
    # Hyperparams
    num_epochs = 100
    batch_size = 128
    num_negative = 128
    encoding_size = 512

    optimizer = optim.AdamW(agent.parameters())
    loss_fn = InfoNCE(temperature=0.01, negative_mode='paired')  # cosine similarity
    # loss_fn = euclidian_contrastive_loss  # negative euclidian distance


    # Train the model
    for n in range(1, num_epochs + 1):
        start_time = time.perf_counter()
        loss, logits = train_model(agent,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           batch_size=batch_size,
                           num_negative=num_negative,
                           train_dataset=train_dataset,
                           device=device,
                           )
        end_time = time.perf_counter()
        print(f"Epoch: {n}\tLoss: {np.mean(loss)}, Time: {end_time - start_time:.2f}")

        print(logits)

        # Test the model
        if n % 5 == 1:
            print("-----------TESTING MODEL ----------------")
            test_model(agent, ood_train_dataset[:, ::10], train_dataset[:, ::10], verbose_num=0, print_details=False)
            print("-----------EVALUATING MODEL----------------")
            test_model(agent, ood_test_dataset, test_dataset, verbose_num=0, print_details=False)
            torch.save(agent.state_dict(), f"checkpoints/ppo/training/contrastive_cnn_{n}.ckpt")

    torch.save(agent.state_dict(), f"checkpoints/ppo/training/contrastive_cnn_{num_epochs}.ckpt")
    test_model(agent, ood_test_dataset, test_dataset, verbose_num=4, print_details=False)

