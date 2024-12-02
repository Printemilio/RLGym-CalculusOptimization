import rlgym
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.obs_builders import DefaultObs

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 1. Custom Reward Function (Per-Agent)
class CustomRewardFunction(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state):
        # Initialize previous boost amounts and positions for each agent
        # Utilisation de id(player) comme clé pour assurer l'unicité
        self.prev_boost_amount = {id(player): 0.0 for player in initial_state.players}
        self.prev_position = {id(player): np.array(player.car_data.position) for player in initial_state.players}

    def get_reward(self, player, state, previous_action):
        """
        Fournit une récompense basée sur :
        - L'augmentation de la quantité de boost.
        - La distance parcourue.
        - Les actions spécifiques comme les sauts et flips pour encourager l'exploration.
        """
        reward = 0.0

        # player est un objet PlayerData
        # state est également un objet PlayerData
        player_id = id(player)

        # Récompense pour la récupération de boost
        try:
            current_boost = state.boost_amount
            previous_boost = self.prev_boost_amount[player_id]
            if current_boost > previous_boost:
                reward += 1.0  # Récompense pour avoir récupéré du boost
                self.prev_boost_amount[player_id] = current_boost
                print(f"Player {player} picked up boost. Reward +1.0")
        except AttributeError:
            print(f"Warning: 'boost_amount' attribute not found for player {player}")

        # Récompense pour la distance parcourue
        try:
            current_position = np.array(state.car_data.position)
            previous_position = self.prev_position[player_id]
            distance = np.linalg.norm(current_position - previous_position)
            if distance > 0.1:  # Seuil pour éviter des récompenses trop petites
                reward += distance * 0.1  # Récompense proportionnelle à la distance
                self.prev_position[player_id] = current_position
                print(f"Player {player} traveled {distance:.2f} units. Reward +{distance * 0.1:.2f}")
        except AttributeError:
            print(f"Warning: 'position' attribute not found for player {player}")

        # Récompense pour des actions spécifiques (sauts et flips)
        # Supposons que previous_action est une liste ou un tableau où :
        # previous_action[0] = jump
        # previous_action[1] = flip
        if previous_action[0] > 0.1:  # Seuil pour détecter un saut
            reward += 0.5  # Récompense pour avoir sauté
            print(f"Player {player} performed a jump. Reward +0.5")
        if previous_action[1] > 0.1:  # Seuil pour détecter un flip
            reward += 0.5  # Récompense pour avoir effectué un flip
            print(f"Player {player} performed a flip. Reward +0.5")

        return reward

# 2. Feature Extractor with Conv1d
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, input_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * input_dim, 512)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [batch, 64, 89]
        x = F.relu(self.bn2(self.conv2(x)))  # [batch, 128, 89]
        x = F.relu(self.bn3(self.conv3(x)))  # [batch, 256, 89]
        x = self.flatten(x)                   # [batch, 256*89]
        x = F.relu(self.fc(x))                # [batch, 512]
        return x

# 3. Attention Module
class AttentionModule(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)  # [batch, seq_length, feature_dim]
        x = F.relu(self.fc(attn_output))          # [batch, seq_length, feature_dim]
        return x

# 4. Actor-Critic Network with Enhanced Architecture
class ActorCritic(nn.Module):
    def __init__(self, input_channels, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, input_dim)
        self.attention = AttentionModule(feature_dim=512)
        self.policy = nn.Linear(512, action_dim)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        features = self.cnn(x)                  # [batch, 512]
        features = features.unsqueeze(1)        # [batch, 1, 512]
        features = self.attention(features)     # [batch, 1, 512]
        features = features.squeeze(1)          # [batch, 512]
        return self.policy(features), self.value(features)

    def get_action(self, state):
        logits, value = self.forward(state)      # logits: [batch, action_dim], value: [batch, 1]
        mean = torch.tanh(logits)                # Actions normalisées entre -1 et 1
        std = torch.ones_like(mean) * 0.2        # Standard deviation ajustée pour plus d'exploration
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).numpy(), log_prob.item()

# 5. PPO Agent with Enhanced Training
class PPOAgent:
    def __init__(self, input_channels, input_dim, action_dim, lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, entropy_coef=0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.actor_critic = ActorCritic(input_channels, input_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def compute_returns(self, rewards, dones, next_value):
        """
        Calcule les retours cumulés pour une séquence de récompenses et de flags de terminaison.
        """
        returns = []
        G = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + (1 - done) * self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def train_agent(self, states, actions, log_probs, rewards, dones, next_value):
        """
        Entraîne l'agent PPO avec les données collectées.
        """
        returns = self.compute_returns(rewards, dones, next_value)
        returns = returns.detach()

        states = torch.cat(states, dim=0)      # [batch, 1, 89]
        actions = torch.stack(actions, dim=0)  # [batch, 8]
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)  # [batch]

        # Debugging: Print shapes
        print(f"States shape: {states.shape}")          # [batch, 1, 89]
        print(f"Actions shape: {actions.shape}")        # [batch, 8]
        print(f"Old log probs shape: {old_log_probs.shape}")  # [batch]
        print(f"Returns shape: {returns.shape}")        # [batch]

        for epoch in range(self.k_epochs):
            logits, values = self.actor_critic(states)
            mean = torch.tanh(logits)
            std = torch.ones_like(mean) * 0.2
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)    # [batch]
            entropy = dist.entropy().sum(dim=-1)                  # [batch]

            ratios = torch.exp(new_log_probs - old_log_probs)    # [batch]
            advantages = (returns - values.squeeze()).detach()    # [batch]

            # PPO Clipping
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()
            loss_value = self.mse_loss(values.squeeze(), returns)
            loss_entropy = -self.entropy_coef * entropy.mean()
            loss = loss_policy + 0.5 * loss_value + loss_entropy

            # Debugging: Print loss values
            # print(f"Epoch {epoch+1}/{self.k_epochs}, Policy Loss: {loss_policy.item()}, Value Loss: {loss_value.item()}, Entropy Loss: {loss_entropy.item()}, Total Loss: {loss.item()}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

# 6. Create Environment with Self-Play (2v2)
def create_rlgym_env():
    return rlgym.make(
        reward_fn=CustomRewardFunction(),
        terminal_conditions=[],  # Ajouter des conditions de terminaison personnalisées si nécessaire
        state_setter=DefaultState(),
        obs_builder=DefaultObs(),
        team_size=2,  # 2 agents par équipe (total 4 agents)
        tick_skip=8
    )

# 7. Training Function with Self-Play
def train_agent(num_episodes=500, max_timesteps=200, save_path="ppo_optimized_agent.pth"):
    env = create_rlgym_env()

    # Get observation and action dimensions
    obs_space = env.observation_space
    action_space = env.action_space

    # Verify observation shape
    print(f"Observation space shape: {obs_space.shape}")  # Should be (89,)

    # Assume observation is a vector of size 89
    if len(obs_space.shape) == 1:
        input_dim = obs_space.shape[0]
    else:
        # Adjust based on actual observation shape
        input_dim = np.prod(obs_space.shape)

    action_dim = action_space.shape[0]
    input_channels = 1  # Ajustez si l'observation a plusieurs canaux

    agent = PPOAgent(input_channels=input_channels, input_dim=input_dim, action_dim=action_dim)
    rewards_per_episode = []

    for episode in range(num_episodes):
        states = env.reset()
        # Ensure states is a list
        if not isinstance(states, list) and not isinstance(states, tuple):
            states = [states]
        num_agents = len(states)  # Should be 4 for 2v2

        # Reshape states to [batch, channels, width]
        state_tensors = []
        for state in states:
            if isinstance(state, (list, tuple, np.ndarray)):
                state = np.array(state, dtype=np.float32)
                state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)  # [1, 1, 89]
            else:
                state = np.array([state], dtype=np.float32)
                state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)  # [1, 1, 89]
            state_tensors.append(state_tensor)

        # Initialize per-agent data storage
        agent_data = [{'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []} for _ in range(num_agents)]
        total_reward = 0

        for t in range(max_timesteps):
            actions = []
            log_probs = []
            for i in range(num_agents):
                state = state_tensors[i]
                action, log_prob = agent.actor_critic.get_action(state)
                actions.append(action)  # [action_dim] as numpy array
                log_probs.append(log_prob)
                agent_data[i]['states'].append(state)
                agent_data[i]['actions'].append(torch.from_numpy(action).float())
                agent_data[i]['log_probs'].append(log_prob)

            # Ensure actions is a list of 4 [8]-dim arrays
            if len(actions) != num_agents:
                print(f"Expected {num_agents} actions, but got {len(actions)}")
                break

            # Pass actions as list of [action_dim] numpy arrays
            try:
                next_states, rewards, dones, _ = env.step(actions)
            except ValueError as e:
                print(f"Action shape error: {e}")
                break

            # Reshape next states
            if not isinstance(next_states, list) and not isinstance(next_states, tuple):
                next_states = [next_states]

            next_state_tensors = []
            for state in next_states:
                if isinstance(state, (list, tuple, np.ndarray)):
                    state = np.array(state, dtype=np.float32)
                    next_state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)  # [1, 1, 89]
                else:
                    state = np.array([state], dtype=np.float32)
                    next_state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)  # [1, 1, 89]
                next_state_tensors.append(next_state_tensor)

            # Collect rewards and dones
            for i in range(num_agents):
                if isinstance(rewards, (list, tuple)):
                    agent_data[i]['rewards'].append(rewards[i])
                else:
                    agent_data[i]['rewards'].append(rewards)
                if isinstance(dones, (list, tuple)):
                    agent_data[i]['dones'].append(dones[i])
                else:
                    agent_data[i]['dones'].append(dones)
                # Sum rewards
                if isinstance(rewards, (list, tuple)):
                    total_reward += rewards[i]
                else:
                    total_reward += rewards

            state_tensors = next_state_tensors

            # Handle 'dones' being a list or single bool
            if isinstance(dones, (list, tuple)):
                done_flag = any(dones)
            else:
                done_flag = dones

            if done_flag:
                break

        # After episode, compute returns for each agent and collect all data
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_dones = []

        for agent_idx, agent_transition in enumerate(agent_data):
            rewards = agent_transition['rewards']
            dones = agent_transition['dones']
            states = agent_transition['states']
            actions_list = agent_transition['actions']
            log_probs_list = agent_transition['log_probs']

            # Compute returns for this agent
            returns = []
            G = 0.0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                G = reward + (1 - done) * agent.gamma * G
                returns.insert(0, G)

            # Ensure the number of returns matches the number of actions
            if len(returns) != len(actions_list):
                print(f"Mismatch in returns and actions for agent {agent_idx + 1}: {len(returns)} vs {len(actions_list)}")
                continue

            # Append to all data
            all_states.extend(agent_transition['states'])
            all_actions.extend(agent_transition['actions'])
            all_log_probs.extend(agent_transition['log_probs'])
            all_returns.extend(returns)
            all_dones.extend(agent_transition['dones'])

        # Convert to tensors
        all_returns = torch.tensor(all_returns, dtype=torch.float32)

        # Debugging: Check if all_returns matches all_actions
        if len(all_returns) != len(all_actions):
            print(f"Mismatch between returns and actions: {len(all_returns)} vs {len(all_actions)}")
            # Optionally, handle the mismatch, e.g., truncate or pad
            min_len = min(len(all_returns), len(all_actions))
            all_returns = all_returns[:min_len]
            all_actions = all_actions[:min_len]
            all_log_probs = all_log_probs[:min_len]
            all_states = all_states[:min_len]
            all_dones = all_dones[:min_len]
            print(f"Adjusted returns and actions to length {min_len}")

        # Calculate next_value
        with torch.no_grad():
            next_values = []
            for i in range(num_agents):
                _, value = agent.actor_critic(state_tensors[i])
                next_values.append(value.item())
            # If episode ended, next_value is 0
            if done_flag:
                next_value = 0.0
            else:
                next_value = sum(next_values)

        # Train the agent
        agent.train_agent(
            states=all_states,
            actions=all_actions,
            log_probs=all_log_probs,
            rewards=all_returns.tolist(),  # Convert to list
            dones=all_dones,
            next_value=next_value
        )

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    agent.save_model(save_path)

    # Plotting the rewards
    plt.plot(rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("PPO Agent Performance with Self-Play (2v2)")
    plt.show()

    env.close()

# 8. Main Function
def main():
    train_agent(num_episodes=500, max_timesteps=200, save_path="ppo_optimized_agent.pth")

if __name__ == "__main__":
    main()
