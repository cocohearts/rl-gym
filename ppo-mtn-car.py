# %%
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv
import cProfile
import pstats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

# %%


class PolicyModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Tanh()
        ).to(device)
        self.log_std = nn.Parameter(torch.ones(1, device=device) * -1)

    def forward(self, x):
        out = self.model(x)
        return out

    def get_action(self, x):
        out = self(x)
        eps = torch.randn_like(out)
        action = out + eps * torch.exp(self.log_std)
        return action

    def get_probs(self, obs, action):
        out = self(obs)
        probs = torch.distributions.Normal(
            out, torch.exp(self.log_std)).log_prob(action)
        return probs.prod(dim=-1)


class ValueModel(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        ).to(device)

    def forward(self, x):
        out = self.model(x)
        return out

# %%


def discounted_returns(rewards, gamma):
    # rewards: shape [T]
    T = rewards.shape[0]
    device = rewards.device

    indices = torch.arange(T, device=device)
    # Create a T x T grid of indices
    j_mat, i_mat = torch.meshgrid(indices, indices, indexing='ij')

    # Mask for lower-triangular (including diagonal): j >= i
    zero_mask = (j_mat > i_mat)

    # Compute exponents for gamma^(j - i)
    exps = i_mat - j_mat

    # Construct the discount matrix G
    G = gamma ** exps
    G[zero_mask] = 0

    # Compute the discounted returns R = G @ rewards
    return G.to(torch.float32) @ rewards.to(torch.float32)


# %%

# Define a custom dataset for episodes and additional tensors

class EpisodeDataset(Dataset):
    def __init__(self, all_episodes_obs, all_episodes_acts, all_episodes_rews, base_probs, base_advantages, rtgs):
        self.all_episodes_obs = all_episodes_obs
        self.all_episodes_acts = all_episodes_acts
        self.all_episodes_rews = all_episodes_rews
        self.base_probs = base_probs
        self.base_advantages = base_advantages
        self.rtgs = rtgs

    def __len__(self):
        return len(self.all_episodes_obs)

    def __getitem__(self, idx):
        return (self.all_episodes_obs[idx],
                self.all_episodes_acts[idx],
                self.all_episodes_rews[idx],
                self.base_probs[idx],
                self.base_advantages[idx],
                self.rtgs[idx])

    def collate_fn(self, batch):
        return (list([item[0] for item in batch]),
                list([item[1] for item in batch]),
                list([item[2] for item in batch]),
                list([item[3] for item in batch]),
                list([item[4] for item in batch]),
                list([item[5] for item in batch]))


# %%

class Agent:
    def __init__(self, gamma=0.99, gae_lambda=0.95, epsilon=0.2, lr=0.0001, ent_coef=0.01, env_name="MountainCarContinuous-v0", num_envs=8):
        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = AsyncVectorEnv([make_env(env_name)
                                   for _ in range(num_envs)])
        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space
        self.policy_model = PolicyModel(
            input_dim=self.observation_space.shape[0])
        self.value_model = ValueModel(
            input_dim=self.observation_space.shape[0])
        self.policy_optimizer = optim.Adam(
            list(self.policy_model.parameters()), lr=lr)
        self.value_optimizer = optim.Adam(
            list(self.value_model.parameters()), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.ent_coef = ent_coef

    def shaped_reward(self, obs):
        returned = torch.zeros(obs.shape[0], device=device)
        returned[obs[:, 1, 0] < 0.45] = -1000000
        return returned

    def update(self, policy_loss=None, value_loss=None):
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def run_episode(self):
        episode_obs = [torch.tensor([]).to(device)
                       for _ in range(self.num_envs)]
        episode_acts = [torch.tensor([]).to(device)
                        for _ in range(self.num_envs)]
        episode_rews = [torch.tensor([]).to(device)
                        for _ in range(self.num_envs)]

        observations = self.envs.reset()[0]
        obs_output = torch.tensor(observations, dtype=torch.float32).to(device)

        dones = np.zeros(self.num_envs, dtype=bool)

        while not np.all(dones):
            obs_input = obs_output
            with torch.no_grad():
                actions = self.policy_model.get_action(obs_input)

            observations, rewards, terminateds, truncateds, _ = self.envs.step(
                actions.cpu().numpy())
            obs_output = torch.tensor(
                observations, dtype=torch.float32).to(device)

            # Store transitions for each environment
            for env_idx in range(self.num_envs):
                if not dones[env_idx]:
                    episode_obs[env_idx] = torch.cat([
                        episode_obs[env_idx],
                        torch.cat(
                            (obs_input[env_idx:env_idx+1], obs_output[env_idx:env_idx+1]))[None, :].to(device)
                    ])
                    episode_acts[env_idx] = torch.cat([
                        episode_acts[env_idx],
                        actions[env_idx:env_idx+1].to(device)
                    ])
                    episode_rews[env_idx] = torch.cat([
                        episode_rews[env_idx],
                        torch.tensor([rewards[env_idx]]).to(device)
                    ])

            dones = dones | (terminateds | truncateds)

        return episode_obs, episode_acts, episode_rews

    def get_losses(self, states, actions, base_probs, base_advantages, real_rtg, epsilon=0.2):
        mse = nn.MSELoss()
        value_loss = mse(self.value_model(states)[:, 0], real_rtg)

        curr_probs = self.policy_model.get_probs(states, actions)
        clipped_weighted_advantages = base_advantages * \
            torch.clip(curr_probs/base_probs, 1-epsilon, 1+epsilon)
        weighted_advantages = base_advantages * curr_probs/base_probs
        policy_loss = -torch.min(clipped_weighted_advantages,
                                 weighted_advantages).mean()

        log_stds = self.policy_model.log_std
        entropy = torch.distributions.Normal(
            out, torch.exp(self.log_std)).entropy()
        policy_loss -= self.ent_coef * entropy.mean()

        return policy_loss.mean(), value_loss

    def compute_statistics(self, all_episodes_obs, all_episodes_acts, all_episodes_rews):
        base_probs = []
        base_advantages = []
        rtgs = []
        for episode_obs, episode_acts, episode_rews in zip(all_episodes_obs, all_episodes_acts, all_episodes_rews):
            base_probs.append(self.policy_model.get_probs(
                episode_obs[:, 0], episode_acts).detach())

            td_error = episode_rews + self.gamma * \
                self.value_model(episode_obs[:, 1])[
                    :, 0] - self.value_model(episode_obs[:, 0])[:, 0]
            base_advantages.append(discounted_returns(
                td_error, self.gamma).detach())

            rtgs.append(discounted_returns(episode_rews, self.gamma).detach())
        base_probs = torch.cat(base_probs)
        base_probs = base_probs.clip(1e-8, 1-1e-8)
        base_advantages = torch.cat(base_advantages)
        rtgs = torch.cat(rtgs)

        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)
        base_advantages = (base_advantages - base_advantages.mean()
                           ) / (base_advantages.std() + 1e-8)
        return base_probs, base_advantages, rtgs

    def ppo_update(self, all_episodes_obs, all_episodes_acts, all_episodes_rews, steps=4, batch_size=32):
        base_probs, base_advantages, rtgs = self.compute_statistics(
            all_episodes_obs, all_episodes_acts, all_episodes_rews)

        # Create a DataLoader for mini-batching
        dataset = EpisodeDataset(
            torch.cat(all_episodes_obs).tolist(),
            torch.cat(all_episodes_acts).tolist(),
            torch.cat(all_episodes_rews).tolist(),
            base_probs.tolist(),
            base_advantages.tolist(),
            rtgs.tolist()
        )
        # Move dataset to CPU
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

        for _ in range(steps):
            for states, acts, rews, base_prob, base_adv, rtgs in dataloader:
                policy_loss, value_loss = self.get_losses(torch.tensor(states).to(device)[:, 0], torch.tensor(acts).to(
                    device), torch.tensor(base_prob).to(device), torch.tensor(base_adv).to(device), torch.tensor(rtgs).to(device))
                self.update(policy_loss=policy_loss, value_loss=value_loss)

        return policy_loss, value_loss

    def avg_reward(self, episodes_rews):
        return torch.tensor([episode.sum() for episode in episodes_rews]).mean()

    def train(self, num_episodes=100, steps=4, print_loss=True):
        # Collect episodes from all environments
        all_episodes = []
        episodes_per_collection = max(1, num_episodes // self.num_envs)

        for _ in range(episodes_per_collection):
            episode_obs, episode_acts, episode_rews = self.run_episode()
            for env_idx in range(self.num_envs):
                if len(episode_obs[env_idx]) > 0:  # Only add non-empty episodes
                    all_episodes.append((
                        episode_obs[env_idx],
                        episode_acts[env_idx],
                        episode_rews[env_idx]
                    ))

        all_episodes_obs = [episode[0] for episode in all_episodes]
        all_episodes_acts = [episode[1] for episode in all_episodes]
        all_episodes_rews = [self.shaped_reward(
            episode[0]) + episode[2] for episode in all_episodes]
        all_episodes_true_rews = [episode[2] for episode in all_episodes]

        policy_loss, value_loss = self.ppo_update(
            all_episodes_obs, all_episodes_acts, all_episodes_rews, steps=steps)
        total_reward = self.avg_reward(all_episodes_true_rews).item()
        shaped_reward = self.avg_reward(all_episodes_rews).item()

        if print_loss:
            print(f"Policy loss: {policy_loss.item()}")
            print(f"Value loss: {value_loss.item()}")
            print(f"Average total reward: {total_reward}")
            print(f"Average shaped reward: {shaped_reward}")
        return (policy_loss, value_loss, total_reward)

    def demo(self):
        env = gym.make(self.env_name, render_mode="human")
        observation, info = env.reset()
        obs_output = torch.tensor(observation, dtype=torch.float32)[
            None, :].to(device)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = self.policy_model.get_action(obs_output)
            observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy()[
                                                                        0])
            obs_output = torch.tensor(observation, dtype=torch.float32)[
                None, :].to(device)
        env.close()


if __name__ == "__main__":
    # %%
    agent = Agent(num_envs=16, lr=0.001, ent_coef=1)
    policy_losses = []
    value_losses = []
    total_rewards = []

    profiler = cProfile.Profile()
    profiler.enable()
    for i in tqdm(range(500), desc="Training"):
        policy_loss, value_loss, total_reward = agent.train(
            num_episodes=16, steps=8, print_loss=True)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        total_rewards.append(total_reward)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)  # Print top 30 time-consuming operations

    agent.demo()

    # %%

    # Set the style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot total rewards
    sns.lineplot(data=total_rewards, ax=ax1)
    ax1.set_title('Total Rewards over Time')
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Average Total Reward')

    # Plot policy losses
    sns.lineplot(data=[loss.item() for loss in policy_losses], ax=ax2)
    ax2.set_title('Policy Loss over Time')
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Policy Loss')

    # Plot value losses
    sns.lineplot(data=[loss.item() for loss in value_losses], ax=ax3)
    ax3.set_title('Value Loss over Time')
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('Value Loss')

    plt.tight_layout()
    plt.savefig('ppo-mtn-car-loss.png')
