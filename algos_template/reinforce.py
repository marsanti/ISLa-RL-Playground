import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
import collections
from utils import TorchModel, ValueModel, init_wandb
import wandb
import random


	
class REINFORCE():
    def __init__(self, params, use_wandb=False):
        self.env = gymnasium.make(params['gym_environment'])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.hidden_layers = params['parameters']['hidden_layers']        # The number of hidden layer of the neural network
        self.nodes_hidden_layers = params['parameters']['nodes_hidden_layers']
        self.lr_opt_policy = params['parameters']['lr_optimizer_pi']

        self.policy = TorchModel(self.state_dim, self.action_dim, self.hidden_layers, self.nodes_hidden_layers)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_opt_policy)
        self.use_baseline = params['parameters']['baseline']

        if self.use_baseline:
            self.lr_opt_vf = params['parameters']['lr_optimizer_vf']
            self.vf = ValueModel()
            self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.lr_opt_vf)
        
        self.gamma =  params['parameters']['gamma']
        self.total_episodes = params['tot_episodes']
        self.use_wandb = use_wandb
	

    def training_loop(self, seed, args_wandb=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if self.use_wandb: init_wandb(args_wandb)

        rewards_list, reward_queue = [], collections.deque(maxlen=100)
        memory_buffer = []
        for ep in range(self.total_episodes):

            # Reset the environment and the episode reward before the episode
            state = self.env.reset(seed=seed)[0]
            ep_reward = 0
            memory_buffer.append([])

            while True:

                # Select the action to perform
                distribution = self.policy(torch.tensor(state.reshape(-1, 4)).type(torch.float)).detach().numpy()[0]
                action = np.random.choice(2, p=distribution)
            
                # Perform the action, store the data in the memory buffer and update the reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                memory_buffer[-1].append([state, action, reward, next_state, done])
                ep_reward += reward

                # Exit condition for the episode
                if done: break
                state = next_state

            # Update the reward list to return
            reward_queue.append(ep_reward)
            rewards_list.append(np.mean(reward_queue))
            print( f"episode {ep:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f})" )
            if self.use_wandb:
                wandb.log({'mean_reward': np.mean(reward_queue)})

      
            # An episode is over,then update
            self.update_policy(memory_buffer)     
            memory_buffer = []

        # Close the enviornment and return the rewards list
        self.env.close()
        wandb.finish()
        return rewards_list if not self.use_wandb else None

    def update_policy(self, memory_buffer):

        for ep in range(len(memory_buffer)):
            # Extraction of the information from the buffer (for the considered episode)
            states = np.array([entry[0] for entry in memory_buffer[ep]]).reshape(-1, 4)
            actions = np.array([entry[1] for entry in memory_buffer[ep]])
            rewards = np.array([entry[2] for entry in memory_buffer[ep]])

        # Iterate over all the trajectories considered
        G = []
        g = 0
        for r in reversed(rewards):  # calculate the return G reversely
            g = self.gamma * g + r
            G.insert(0, g)

        if not self.use_baseline:
            for t in range(len(rewards)):
                state = torch.unsqueeze(torch.tensor(states[t], dtype=torch.float), 0)
                action = actions[t]
                g = G[t]

                a_prob = self.policy(state).flatten()
                policy_loss = -pow(self.gamma, t) * g * torch.log(a_prob[action])
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
        else:
            for t in range(len(rewards)):
                state = torch.unsqueeze(torch.tensor(states[t], dtype=torch.float), 0)
                a = actions[t]
                g = G[t]
                v_s = self.vf(state).flatten()

                # Update policy
                a_prob = self.policy(state).flatten()
                policy_loss = -pow(self.gamma, t) * ((g - v_s).detach()) * torch.log(a_prob[a])
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Update value function
                value_loss = F.mse_loss(v_s, torch.tensor([g], dtype=torch.float))
                self.vf_optimizer.zero_grad()
                value_loss.backward()
                self.vf_optimizer.step()
