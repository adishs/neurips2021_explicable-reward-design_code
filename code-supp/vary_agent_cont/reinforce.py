import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils
import copy

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x


class Agent:

    def __init__(self, env, discrete_env=None, usereward_env=False, learning_rate=3e-4):
        self.env = env
        self.discrete_env = discrete_env
        self.usereward_env = usereward_env
        self.n_actions = env.n_actions
        self.policy_network = PolicyNetwork(env.dimension, self.n_actions).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy_network.forward(Variable(state))
        highest_prob_action = np.random.choice(self.n_actions, p=np.squeeze(probs.cpu().detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.env.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9).to(device)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def train(self, max_episode=1000, max_step=300, name=""):
        result_dict = {}
        result_dict[name + "_expected_reward"] = []
        for episode in range(max_episode):

            exp_reward = utils.evaluate_policy_given_n_episode(self.env,
                                                               max_episode=5,
                                                               policy=self.get_action)
            result_dict[name + "_expected_reward"].append(exp_reward)

            state = self.env.reset()
            log_probs = []
            rewards = []
            episode_reward = 0

            if episode % 100 == 0:
                print("============{}====================".format(name))
                print("episode " + str(episode) + " / " + str(max_episode))
                print("expected_reward=", exp_reward)

            for steps in range(max_step):
                action, log_prob = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                if self.usereward_env:
                    reward = self.env.reward[self.env.find_nearest(tuple(state)), action]

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if done:
                    self.update_policy(rewards, log_probs)
                    break

                state = copy.deepcopy(new_state)
        return result_dict
    #enndef
#endclass

if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    env = gym.make('LunarLander-v2')
    agent = Agent(env)
    agent.train(3000, 500)

    #render
    observation = env.reset()
    for _ in range(3000):
        env.render()
        action, _ = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
        # env.step(env.action_space.sample()) # take a random action

