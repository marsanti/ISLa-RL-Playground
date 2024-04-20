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
            state = None
            ep_reward = 0
            memory_buffer.append([])

            while True:

                # Select the action to perform
               #TODO
            
                # Perform the action, store the data in the memory buffer and update the reward
               #TODO

                # Exit condition for the episode
                done = True
                if done: break
                state = None

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
            states = []
            actions = []
            rewards = []

        # Iterate over all the trajectories considered
        G = []
        g = 0
        for r in reversed(rewards):  # calculate the return G reversely
            #TODO
            pass

        if not self.use_baseline:
            for t in range(len(rewards)):
               #implement reinforce algorithm
               pass
        else:
            for t in range(len(rewards)):
               #implement reinforce algorithm with baseline
               pass