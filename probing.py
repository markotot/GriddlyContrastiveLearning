import numpy as np
import torch
from contrastive_cnn import load_dataset, load_agent_model
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#TODO: train a PPO agent on all environments, and then do the probing
class ProbingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProbingModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        return self.model(x)

    def get_latent_encoding(self, x):
        return self.model[0](x)

def get_accuracy(probing_model, agent_model, x, y, device):

    accuracy = []
    data_loader = DataLoader(TensorDataset(torch.from_numpy(x).to(device),
                                            torch.from_numpy(y).to(device)),
                                           batch_size=512)

    for x_batch, y_batch in data_loader:
        latent_encoding = agent_model.get_latent_encoding(x_batch).squeeze()
        y_pred = probing_model(latent_encoding)
        batch_acc = (y_pred.argmax(dim=1) == y_batch).float().mean()
        accuracy.append(batch_acc.item())

    return np.mean(accuracy)

def plot_accuracies(training_accuracies, testing_accuracies, x_limit, plot_title):
    x = range(0, len(training_accuracies))

    plt.plot(x, training_accuracies, label="Train")
    plt.plot(x, testing_accuracies, label="Test")
    plt.title(plot_title)
    plt.ylim(0, 1.01)
    plt.xlim(0, x_limit)
    plt.xlabel("Epochs Trained")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # load data
    all_env_configs = [
        "cluster-1-floor.yaml",
        "cluster-2-grass-15.yaml",
        "cluster-9-red.yaml",
    ]

    ood_env_configs = [
        "cluster-2-grass-15.yaml",
    ]

    ppo_agent_ckpt = "contrastive_cnn.ckpt"
    # ppo_agent_ckpt = "ppo-standard.ckpt"
    # ppo_agent_ckpt = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the data
    train_dataset, test_dataset, ood_train_dataset, ood_test_dataset = load_dataset(all_env_configs, ood_env_configs)

    x_train = np.concatenate((train_dataset[0], train_dataset[1]), axis=0)
    y_train = np.zeros(shape=(x_train.shape[0]), dtype=np.uint8)
    y_train[train_dataset.shape[1]:] = 1

    x_test = np.concatenate((test_dataset[0], test_dataset[1]), axis=0)
    y_test = np.zeros(shape=(x_test.shape[0]), dtype=np.uint8)
    y_test[test_dataset.shape[1]:] = 1

    agent_model = load_agent_model(ppo_agent_ckpt)
    dummy_input = torch.from_numpy(train_dataset[0][0]).to(device).unsqueeze(0)
    latent_encoding = agent_model.get_latent_encoding(dummy_input).squeeze()

    probing_model = ProbingModel(input_dim=latent_encoding.shape[0],
                                    hidden_dim=64,
                                    output_dim=train_dataset.shape[0]).to(device)

    probing_train_data_loader = DataLoader(TensorDataset(torch.from_numpy(x_train).to(device),
                                                    torch.from_numpy(y_train).to(device)),
                                    batch_size=32,
                                    shuffle=True)

    probing_test_data_loader = DataLoader(TensorDataset(torch.from_numpy(x_test).to(device),
                                                    torch.from_numpy(y_test).to(device)),
                                    batch_size=32,
                                    shuffle=True)
    # train probing model
    optimizer = torch.optim.Adam(probing_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    training_accuracies = []
    testing_accuracies = []
    accuracy = get_accuracy(probing_model, agent_model, x_train, y_train, device=device)
    training_accuracies.append(accuracy)
    testing_accuracy = get_accuracy(probing_model, agent_model, x_test, y_test, device=device)
    testing_accuracies.append(testing_accuracy)


    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        train_acc = []

        for x_batch, y_batch in probing_train_data_loader:

            optimizer.zero_grad()
            latent_encoding = agent_model.get_latent_encoding(x_batch).squeeze()
            y_pred = probing_model(latent_encoding)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            batch_acc = (y_pred.argmax(dim=1) == y_batch).float().mean()
            train_acc.append(batch_acc.item())

        print(f"Epoch {epoch}\tLoss: {total_loss:.2f}\tAccuracy: {np.mean(train_acc):.4f}")
        if epoch % 10 == 0:
            total_loss = 0
            test_acc = []
            for x_batch, y_batch in probing_test_data_loader:
                latent_encoding = agent_model.get_latent_encoding(x_batch).squeeze()
                y_pred = probing_model(latent_encoding)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()
                batch_acc = (y_pred.argmax(dim=1) == y_batch).float().mean()
                test_acc.append(batch_acc.item())
            print(f"Test: Loss: {total_loss:.2f}\tAccuracy: {np.mean(test_acc):.4f}")

        training_accuracy = get_accuracy(probing_model, agent_model, x_train, y_train, device=device)
        training_accuracies.append(training_accuracy)
        testing_accuracy = get_accuracy(probing_model, agent_model, x_test, y_test, device=device)
        testing_accuracies.append(testing_accuracy)
    plot_accuracies(training_accuracies,
                testing_accuracies,
                x_limit=num_epochs,
                plot_title="FAL-CNN")
    np.savez("FAL-CNN-full-acc.npz", training_accuracies, testing_accuracies)