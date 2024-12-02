import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import rlgym
import numpy as np

# Réseau d'acteur-critiques
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Acteur (policy)
        self.policy = nn.Linear(128, action_dim)
        # Critique (value function)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)

    def get_action(self, state):
        logits, _ = self.forward(state)
        mean = torch.tanh(logits)  # Actions bornées entre -1 et 1
        std = torch.ones_like(mean) * 0.1  # Écart-type fixe
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)  # Log_prob pour chaque dimension

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.actor_critic = ActorCritic(input_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        G = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + (1 - done) * self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def train(self, states, actions, log_probs, rewards, dones, next_value):
        # Calculer les retours
        returns = self.compute_returns(rewards, dones, next_value)
        returns = returns.detach()  # Détacher les retours du graphe

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()  # Détacher les anciens log_probs

        for _ in range(self.k_epochs):
            # Recalculer les logits et les valeurs
            logits, values = self.actor_critic(states)
            mean = torch.tanh(logits)
            std = torch.ones_like(mean) * 0.1
            dist = Normal(mean, std)

            # Recalculer les log_probs
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Calcul des ratios
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Calcul des avantages
            advantages = (returns - values.detach().squeeze())

            # PPO Clipping
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()

            # Critic Loss
            loss_value = self.mse_loss(values.squeeze(), returns)

            # Total Loss
            loss = loss_policy + 0.5 * loss_value

            # Optimisation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Retenir le graphe pour les itérations multiples
            self.optimizer.step()


env = rlgym.make()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # Nombre d'actions

agent = PPOAgent(input_dim=obs_dim, action_dim=action_dim)

num_episodes = 1000
max_timesteps = 200

for episode in range(num_episodes):
    state = torch.tensor(env.reset(), dtype=torch.float32)
    log_probs, states, actions, rewards, dones = [], [], [], [], []
    total_reward = 0

    for t in range(max_timesteps):
        action, log_prob = agent.actor_critic.get_action(state)
        next_state, reward, done, _ = env.step(action.detach().numpy())  # Conversion pour RLGym

        # Stocker les transitions
        log_probs.append(log_prob)
        states.append(state)
        actions.append(action)  # Actions multidimensionnelles
        rewards.append(reward)
        dones.append(done)

        state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

        if done:
            break

    # Calcul de la valeur pour la dernière étape
    next_value = agent.actor_critic.forward(state)[1].detach()
    agent.train(states, actions, log_probs, rewards, dones, next_value)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
