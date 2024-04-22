import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
import collections
from utils.utils import ActorModel, CriticModel, init_wandb
import wandb
import random
import copy

	
class DDPG():
    def __init__(self, params, use_wandb=False):
        self.env = gymnasium.make(params['gym_environment'])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # DNN configurations
        self.hidden_layers_actor = params['parameters']['hidden_layers_actor']     
        self.hidden_layers_critic = params['parameters']['hidden_layers_actor']
        self.nodes_hidden_layers_actor = params['parameters']['nodes_hidden_layers_actor']
        self.nodes_hidden_layers_critic = params['parameters']['nodes_hidden_layers_critic']
        self.lr_actor = params['parameters']['lr_actor_optimizer']
        self.lr_critic = params['parameters']['lr_critic_optimizer']

        # create actor and critic
        self.actor = ActorModel(self.state_dim, self.action_dim, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic = CriticModel(self.state_dim, self.action_dim, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # create actor and critic target
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
            
        self.gamma =  params['parameters']['gamma']
        self.tau =  params['parameters']['tau'] 
        self.update_freq = params['parameters']['tau'] 
        self.n_updates = params['parameters']['n_updates'] 
        self.total_episodes = params['tot_episodes']
        self.use_wandb = use_wandb
        self.max_action = float(self.env.action_space.high[0])
        self.epsilon = 1.0
        self.epsilon_decay = params['parameters']['eps_decay']
	

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
                if np.random.rand() < self.epsilon:  
                    action = None # your code here
                else:
                    # Add Gaussian noise to actions for exploration
                    # your code here
                    pass

                self.epsilon *= self.epsilon_decay

                # Perform the action, store the data in the memory buffer and update the reward
                # your code here

                # Exit condition for the episode
                done = True
                if done: break
                state = None #next_state

            # Update the reward list to return
            reward_queue.append(ep_reward)
            rewards_list.append(np.mean(reward_queue))
            print( f"episode {ep:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f})" )
            if self.use_wandb:
                wandb.log({'mean_reward': np.mean(reward_queue)})

      
            # Update
            if ep % self.update_freq == 0:
                for _ in range(self.n_updates):
                    self.update_policy(memory_buffer)
            
                memory_buffer = []

        # Close the enviornment and return the rewards list
        self.env.close()
        wandb.finish()
        return rewards_list if not self.use_wandb else None

    def update_policy(self, memory_buffer):

        for ep in range(len(memory_buffer)):
            # your code here
            pass

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
           # yout code here
           pass

        # Compute the critic Q and the target critic loss
        critic_q = None
        critic_loss = None
        
        # Optimize the critic
        # your code here

        # Freeze critic networks to optimize computational effort
        for params in self.critic.parameters():
           pass

        # Compute the actor loss
        # your code here

        # Unfreeze critic networks
        for params in self.critic.parameters():
            # your code here
            pass

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            # your code
            pass

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            # your code
            pass