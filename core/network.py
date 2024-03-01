import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPONetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, 256), std=0.01),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128), std=0.01),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 256), std=1),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 128), std=1),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, 1), std=1),
        )

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None, greedy=False):

        action, log_probs, entropy, value, _ = self.get_action_and_value_and_latent(x, action, greedy)
        return action, log_probs, entropy, value

    def get_action_and_value_and_latent(self, x, action=None, greedy=False):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        if action is None:
            action, log_probs, entropy = self.get_action(logits, greedy)
        else:
            probs = Categorical(logits=logits)
            log_probs = probs.log_prob(action)
            entropy = probs.entropy()

        return action, log_probs, entropy, value, hidden

    def get_action(self, logits, greedy=False):
        probs = Categorical(logits=logits)
        if greedy is False:
            action = probs.sample()
        else:
            action = torch.argmax(logits, dim=1)
        return action, probs.log_prob(action), probs.entropy()


    def get_latent_encoding(self, x):
        return self.network(x / 255.0)

    def freeze_CNN(self):
        for p in self.network.parameters():
            p.requires_grad = False

    def load_checkpoint(self, path="weights.ckpt"):
        self.load_state_dict(torch.load(f"checkpoints/ppo/{path}"))

    def save_checkpoint(self, path="weights.ckpt"):
        torch.save(self.state_dict(), f"checkpoints/ppo/{path}")

class LSTMPPONetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_latent_encoding(self, x):
        return self.network(x / 255.0)

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(f"checkpoints/lstm/{path}"))

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), f"checkpoints/lstm/{path}")