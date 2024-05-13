import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
import collections
from utils.utils import TorchModel, init_wandb
import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import wandb
import random
import time


	
class REINFORCE():
    def __init__(self, params, use_wandb=False):
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'])
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="macos", worker_id=int(time.time())%10000, time_scale=100, no_graphics=True, max_step=100, action_space_type='discrete')
        
        self.env_name = params['gym_environment']
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.hidden_layers = params['parameters']['hidden_layers']        # The number of hidden layer of the neural network
        self.nodes_hidden_layers = params['parameters']['nodes_hidden_layers']
        self.lr_opt_policy = params['parameters']['lr_optimizer_pi']

        self.policy = TorchModel(self.state_dim, self.action_dim, self.hidden_layers, self.nodes_hidden_layers, last_activation=F.softmax)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_opt_policy)
        self.use_baseline = params['parameters']['baseline']

        if self.use_baseline:
            self.lr_opt_vf = params['parameters']['lr_optimizer_vf']
            self.vf = TorchModel(self.state_dim, 1, 1, 32)
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

        rewards_list, success_list, reward_queue, success_queue = [], [], collections.deque(maxlen=100), collections.deque(maxlen=100)
        memory_buffer = []
        for ep in range(self.total_episodes):

            # Reset the environment and the episode reward before the episode
            state = self.env.reset(seed=seed)[0] if self.env_name != "TB3" else self.env.reset()
            ep_reward = 0
            success = 0
            memory_buffer.append([])

            while True:

                # Select the action to perform
                #TODO: we use try expect to avoid NaN rare case

                try:
                    distribution = None
                    action = None
                except:
                    action = np.random.choice(self.env.action_space.n)
                    
            
                # Perform the action, store the data in the memory buffer and update the reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # TODO
                done = None
                success += int(terminated)

                # reward shaping
                #TODO

                memory_buffer[-1].append([state, action, reward, next_state, done])
                ep_reward += reward

                # Exit condition for the episode
                if done: break
                state = next_state

            # Update the reward list to return
            reward_queue.append(ep_reward)
            success_queue.append(success)
            rewards_list.append(np.mean(reward_queue))
            success_list.append(np.mean(success_queue))
            print( f"episode {ep:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f}) success: {success:3d} (mean success: {success_list[-1]:5.2f})" )
            if self.use_wandb:
                wandb.log({'mean_reward': rewards_list[-1], 'mean_success': success_list[-1]})

      
            # An episode is over,then update
            self.update_policy(memory_buffer)     
            memory_buffer = []

        # Close the enviornment and return the rewards list
        self.env.close()
        wandb.finish()
        return rewards_list if not self.use_wandb else None

    def update_policy(self, memory_buffer):
        pass