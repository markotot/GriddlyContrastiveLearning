import copy
import time

from info_nce import InfoNCE
from matplotlib import pyplot as plt
from torch import optim

from core.network import PPONetwork
import numpy as np
import torch
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

            axes[i][j].axis('off')
            axes[i][j].imshow(observations[n])
            axes[i][j].set_title(losses[n])
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


def test_same_background(agent, test_dataset, verbose_num, print_details, plot_name="Test"):
    anchor_dataset = copy.deepcopy(test_dataset)
    negative_dataset = copy.deepcopy(test_dataset)

    num_samples = len(list(anchor_dataset.values())[0])
    idx = np.random.permutation(range(num_samples))
    shuffled_negative_dataset = {key: negative_dataset[key][idx] for key in negative_dataset}

    verbose_idx = np.random.permutation(range(num_samples))[:verbose_num]

    background_similarities = {}
    for anchor_key in anchor_dataset.keys():

        plot_observation = []
        plot_similarities = []

        cos_similarities = []
        value_similarities = []
        anchor_encodings = agent.get_latent_encoding(torch.tensor(anchor_dataset[anchor_key]).to(device))
        shuffled_negative_encodings = agent.get_latent_encoding(
            torch.tensor(shuffled_negative_dataset[anchor_key]).to(device))
        for x, y, n in zip(anchor_encodings, shuffled_negative_encodings, range(len(anchor_encodings))):
            cosine_sim = torch.cosine_similarity(x, y, dim=0).cpu().detach().numpy()
            value_sim = torch.mean(torch.abs(x - y)).cpu().detach().numpy()

            cos_similarities.append(cosine_sim)
            value_similarities.append(value_sim)
            if n in verbose_idx:
                plot_observation.append(anchor_dataset[anchor_key][n])
                plot_observation.append(shuffled_negative_dataset[anchor_key][n])
                plot_similarities.append("anchor")
                plot_similarities.append(f"{cosine_sim:.4f}")
        background_similarities[anchor_key] = (cos_similarities, value_similarities)
        if verbose_num > 0:
            better_plot(plot_observation, plot_similarities, f"{plot_name} mean:{np.mean(cos_similarities):.3f}",
                        pairwise=True)

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


def test_out_of_distribution(agent, anchor_dataset, test_dataset, shuffled, verbose_num, print_details,
                             plot_name="Test"):
    np.random.seed(0)
    test_dataset = copy.deepcopy(test_dataset)
    anchor_dataset = copy.deepcopy(anchor_dataset)

    # Shuffle the data
    if shuffled:
        num_samples = len(list(test_dataset.values())[0])
        idx = np.random.permutation(range(num_samples))
        test_dataset = {key: test_dataset[key][idx] for key in test_dataset}

    average_env_similarities = {}
    for anchor_env in anchor_dataset.keys():
        anchor_env_encodings = agent.get_latent_encoding(torch.tensor(anchor_dataset[anchor_env]).to(device))

        for env_config in test_dataset.keys():

            plot_observations = []
            plot_similarities = []

            test_env_encodings = agent.get_latent_encoding(torch.tensor(test_dataset[env_config]).to(device))

            cos_similarities = []
            value_similarities = []
            for x, y, n in zip(anchor_env_encodings, test_env_encodings, range(len(anchor_env_encodings))):

                value_sim = torch.mean(torch.abs(x - y)).cpu().detach().numpy()
                cos_sim = torch.cosine_similarity(x, y, dim=0).cpu().detach().numpy()

                cos_similarities.append(cos_sim)
                value_similarities.append(value_sim)

                if n < verbose_num:
                    plot_observations.append(anchor_dataset[anchor_env][n])
                    plot_observations.append(test_dataset[env_config][n])
                    plot_similarities.append(f"anchor")
                    plot_similarities.append(f"{cos_sim:.4f}")

            if verbose_num > 0:
                better_plot(plot_observations,
                            plot_similarities,
                            plot_name=f"{plot_name} mean:{np.mean(cos_similarities):.3f}",
                            pairwise=True
                            )

            average_env_similarities[f"{anchor_env}_{env_config}"] = (cos_similarities, value_similarities)

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


def train_model(agent, optimizer, loss_fn, batch_size, num_negative, numpy_train_dataset):
    num_envs = numpy_train_dataset.shape[0]
    num_samples = numpy_train_dataset.shape[1]

    losses = []
    shuffled_idxs = np.random.permutation(range(num_samples))

    for n in range(num_samples // batch_size):

        start_batch = time.perf_counter()
        # Select anchor observations
        anchor_envs = [x % num_envs for x in range(n * batch_size, (n + 1) * batch_size)]
        anchor_obs_ind = shuffled_idxs[n * batch_size: (n + 1) * batch_size]
        batched_anchor_obs = numpy_train_dataset[anchor_envs, anchor_obs_ind]

        # Select positive observations
        all_envs = np.arange(num_envs).reshape(1, -1) + np.zeros(shape=(batch_size, num_envs), dtype=np.uint8)

        positive_envs = np.array([np.random.choice(np.delete(env, value)) for env, value in zip(all_envs, anchor_envs)],
                                 dtype=np.uint8)
        batched_positive_obs = numpy_train_dataset[positive_envs, anchor_obs_ind]

        # Select N negative observations
        same_envs_flag = np.random.random(size=num_negative) < 0.5
        negative_envs = [x if flag else np.random.choice(np.delete(y, x)) for x, y, flag in zip(anchor_envs, all_envs, same_envs_flag)]
        all_idx = np.array(
            np.arange(num_samples).reshape(1, -1) + np.zeros(shape=(batch_size, num_samples), dtype=np.int32))
        negative_idx = ([np.random.choice(np.delete(x, value), size=num_negative, replace=False) for x, value in
                         zip(all_idx, anchor_obs_ind)])
        batched_negative_obs = numpy_train_dataset[negative_envs, negative_idx]
        batched_negative_obs = np.reshape(batched_negative_obs, (batch_size * num_negative, 3, 84, 84))

        # Get the latent encodings
        batch_anchor_encodings = agent.get_latent_encoding(torch.from_numpy(batched_anchor_obs).to(device))
        batch_positive_encodings = agent.get_latent_encoding(torch.from_numpy(batched_positive_obs).to(device))
        batch_negative_encodings = agent.get_latent_encoding(torch.from_numpy(batched_negative_obs).to(device))

        # Reshape from (batch_size * num_negative, 512) to (batch_size, num_negative, 512)
        batched_negative_obs = torch.reshape(batch_negative_encodings, (batch_size, num_negative, 512))

        # Update the model
        loss = loss_fn(batch_anchor_encodings, batch_positive_encodings, batched_negative_obs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add the loss for the current batch
        losses.append(loss.cpu().detach().numpy())

    return np.array(losses)


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
                         test_dataset=ood_test_dataset,
                         verbose_num=verbose_num,
                         print_details=print_details,
                         plot_name="contrastive_same_background_ood"
                         )

    print(f"----------Testing same background ----------")
    test_same_background(agent, test_dataset=test_dataset,
                         verbose_num=verbose_num,
                         print_details=print_details,
                         plot_name="contrastive_same_background"
                         )


def load_dataset(env_configs, ood_env_configs, train_percentage=0.8):
    dataset = {}
    ood_dataset = {}
    for env_config in env_configs:

        npz = np.load(f"datasets/{env_config}.npz")
        data_dict = {item: npz[item] for item in npz.files}  # Get all the data from the npz file
        if env_config in ood_env_configs:
            ood_dataset[env_config] = np.array(list(data_dict.values())).squeeze()
        else:
            dataset[env_config] = np.array(list(data_dict.values())).squeeze()

    train_num = int(len(dataset[env_configs[0]]) * train_percentage)

    train_dataset = {key: dataset[key][:train_num] for key in dataset}
    test_dataset = {key: dataset[key][train_num:] for key in dataset}
    ood_train_dataset = {key: ood_dataset[key][:train_num] for key in ood_dataset}
    ood_test_dataset = {key: ood_dataset[key][train_num:] for key in ood_dataset}

    return train_dataset, test_dataset, ood_train_dataset, ood_test_dataset


if __name__ == "__main__":

    # Making the training process deterministic
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(f"configs/cluster-1-floor.yaml", 0, 0, 0, 0)()
    env.single_action_space = env.action_space
    agent = PPONetwork(env).to(device)

    ppo_agent_ckpt = "contrastive_cnn.ckpt"
    agent.load_checkpoint(ppo_agent_ckpt)

    # Setup environments
    all_env_configs = [
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
    train_dataset, test_dataset, ood_train_dataset, ood_test_dataset = load_dataset(all_env_configs, ood_env_configs)
    num_samples = train_dataset[list(train_dataset.keys())[0]].shape[0]

    # Hyperparams
    num_epochs = 100
    batch_size = 128
    num_negative = 64
    encoding_size = 512

    info_nce = InfoNCE(negative_mode='paired')
    optimizer = optim.Adam(agent.parameters())

    # Convert the dataset to numpy
    numpy_train_dataset = np.zeros(shape=(len(train_dataset), num_samples, 3, 84, 84), dtype=np.uint8)
    for idx, env_name in enumerate(train_dataset):
        numpy_train_dataset[idx] = train_dataset[env_name]

    # Train the model
    for n in range(num_epochs):
        loss = train_model(agent,
                           optimizer=optimizer,
                           loss_fn=info_nce,
                           batch_size=batch_size,
                           num_negative=num_negative,
                           numpy_train_dataset=numpy_train_dataset,
                           )
        print(f"Epoch: {n}\tLoss: {np.mean(loss)}")

        # Test the model
        if n % 10 == 0:
            test_model(agent, ood_test_dataset, test_dataset, verbose_num=0, print_details=False)
            torch.save(agent.state_dict(), "checkpoints/ppo/contrastive_cnn.ckpt")
