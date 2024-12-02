import rlgym
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.obs_builders import DefaultObs

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Custom Reward Function (Only Positive Rewards)
class CustomRewardFunction(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state):
        # Initialize previous touch and goal counts
        self.prev_touch = {player: 0 for player in initial_state.players}
        self.prev_goal = {team: 0 for team in ['Blue', 'Orange']}

    def get_reward(self, state, previous_state, done=False):
        rewards = {player: 0.0 for player in state.players}

        # Encourage touching the ball
        for player in state.players:
            if state.players[player].touch_count > self.prev_touch[player]:
                rewards[player] += 1.0  # Reward for touching the ball
                self.prev_touch[player] = state.players[player].touch_count

        # Encourage scoring goals
        for team in ['Blue', 'Orange']:
            if state.score_info.goals_scored[team] > self.prev_goal[team]:
                for player in state.players:
                    if state.players[player].team == team:
                        rewards[player] += 5.0  # Reward for scoring a goal
                self.prev_goal[team] = state.score_info.goals_scored[team]

        return rewards

# 2. Actor-Critic Network (Simple MLP)
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Define the network architecture
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # Policy head
        self.policy = nn.Linear(256, action_dim)
        # Value head
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.policy(x), self.value(x)

    def get_action(self, state):
        logits, value = self.forward(state)
        mean = torch.tanh(logits)  # Assuming actions are continuous and normalized
        std = torch.ones_like(mean) * 0.1  # Fixed standard deviation
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob.item()

# 3. PPO Agent
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

    def train_agent(self, states, actions, log_probs, rewards, dones, next_value):
        returns = self.compute_returns(rewards, dones, next_value)
        returns = returns.detach()

        states = torch.cat(states, dim=0)  # [batch, input_dim]
        actions = torch.stack(actions, dim=0)  # [batch, action_dim]
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)  # [batch]

        for _ in range(self.k_epochs):
            logits, values = self.actor_critic(states)
            mean = torch.tanh(logits)
            std = torch.ones_like(mean) * 0.1
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = (returns - values.squeeze()).detach()

            # PPO Clipping
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()
            loss_value = self.mse_loss(values.squeeze(), returns)
            loss = loss_policy + 0.5 * loss_value

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

# 4. Create Environment with Self-Play (2v2)
def create_rlgym_env():
    return rlgym.make(
        reward_fn=CustomRewardFunction(),
        terminal_conditions=[],  # Add custom terminal conditions if necessary
        state_setter=DefaultState(),
        obs_builder=DefaultObs(),
        team_size=2,  # 2 agents per team (total 4 agents)
        tick_skip=8
    )

# 5. Training Function with Self-Play
def train_agent(num_episodes=500, max_timesteps=200, save_path="ppo_optimized_agent.pth"):
    env = create_rlgym_env()

    # Get observation and action dimensions
    obs_space = env.observation_space
    action_space = env.action_space

    # Verify observation shape
    print(f"Observation space shape: {obs_space.shape}")

    # Assume observation is a vector of size N
    if len(obs_space.shape) == 1:
        input_dim = obs_space.shape[0]
    else:
        # Adjust based on actual observation shape
        input_dim = np.prod(obs_space.shape)

    action_dim = action_space.shape[0]
    # Assuming continuous action space with dimension 'action_dim'

    agent = PPOAgent(input_dim=input_dim, action_dim=action_dim)
    rewards_per_episode = []

    for episode in range(num_episodes):
        states = env.reset()
        # Ensure states is a list
        if not isinstance(states, list) and not isinstance(states, tuple):
            states = [states]
        num_agents = len(states)
        # Flatten states and convert to tensors
        state_tensors = []
        for state in states:
            state = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state).unsqueeze(0)  # [1, input_dim]
            state_tensors.append(state_tensor)

        log_probs_combined = []
        actions_combined = []
        rewards_combined = []
        dones_combined = []
        total_reward = 0

        for t in range(max_timesteps):
            actions = []
            log_probs = []
            for i in range(num_agents):
                action, log_prob = agent.actor_critic.get_action(state_tensors[i])
                actions.append(action)  # [action_dim] as numpy array
                log_probs.append(log_prob)

            # Pass actions as list of [action_dim] numpy arrays
            next_states, rewards, dones, _ = env.step(actions)

            # Reshape next states
            if not isinstance(next_states, list) and not isinstance(next_states, tuple):
                next_states = [next_states]

            next_state_tensors = []
            for state in next_states:
                state = np.array(state, dtype=np.float32)
                state_tensor = torch.from_numpy(state).unsqueeze(0)  # [1, input_dim]
                next_state_tensors.append(state_tensor)

            # Collect logs and actions
            log_probs_combined.extend(log_probs)
            actions_combined.extend([torch.from_numpy(a).float() for a in actions])
            if isinstance(rewards, (list, tuple)):
                rewards_combined.extend(rewards)
            else:
                rewards_combined.append(rewards)
            if isinstance(dones, (list, tuple)):
                dones_combined.extend(dones)
            else:
                dones_combined.append(dones)

            state_tensors = next_state_tensors
            if isinstance(rewards, (list, tuple)):
                total_reward += sum(rewards)
            else:
                total_reward += rewards

            # Handle 'dones' being a single bool or a list
            if isinstance(dones, (list, tuple)):
                done_flag = any(dones)
            else:
                done_flag = dones

            if done_flag:
                break

        # Calculate next_value
        with torch.no_grad():
            next_values = []
            for i in range(num_agents):
                _, value = agent.actor_critic(state_tensors[i])
                next_values.append(value.item())
            next_value = sum(next_values)

        # Train the agent
        agent.train_agent(
            states=state_tensors,
            actions=actions_combined,
            log_probs=log_probs_combined,
            rewards=rewards_combined,
            dones=dones_combined,
            next_value=next_value
        )

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    agent.save_model(save_path)

    plt.plot(rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("PPO Agent Performance with Self-Play (2v2)")
    plt.show()

    env.close()

# 6. Main Function
def main():
    train_agent(num_episodes=500, max_timesteps=200, save_path="ppo_optimized_agent.pth")

if __name__ == "__main__":
    main()
