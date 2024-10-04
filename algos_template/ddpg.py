import time
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
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'])
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="linux", worker_id=int(time.time())%10000, time_scale=100, no_graphics=True, max_step=100, action_space_type='continuous')
        self.env_name = params['gym_environment']
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.min_action = self.env.action_space.low[0]
        
        # DNN configurations
        self.hidden_layers_actor = params['parameters']['hidden_layers_actor']     
        self.hidden_layers_critic = params['parameters']['hidden_layers_actor']
        self.nodes_hidden_layers_actor = params['parameters']['nodes_hidden_layers_actor']
        self.nodes_hidden_layers_critic = params['parameters']['nodes_hidden_layers_critic']
        self.lr_actor = params['parameters']['lr_actor_optimizer']
        self.lr_critic = params['parameters']['lr_critic_optimizer']

        # create actor and critic
        self.actor = ActorModel(self.state_dim, self.action_dim, self.hidden_layers_actor, self.nodes_hidden_layers_actor, self.max_action)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic = CriticModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # create actor and critic target
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
            
        self.gamma =  params['parameters']['gamma']
        self.tau =  params['parameters']['tau'] 
        self.update_freq = params['parameters']['update_freq'] 
        self.n_updates = params['parameters']['n_updates'] 
        self.total_episodes = params['tot_episodes']
        self.use_wandb = use_wandb
        self.epsilon = 1.0
        self.epsilon_decay = params['parameters']['eps_decay']
        self.batch_size = params['parameters']['batch_size']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            state = self.env.reset(seed=seed)[0] if self.env_name != "TB3" else self.env.reset()
            ep_reward = 0

            while True:
                # Select the action to perform
                if np.random.rand() < self.epsilon:  
                    with torch.no_grad():
                        action = self.env.action_space.sample()
                else:
                    # Add Gaussian noise to actions for exploration
                    with torch.no_grad():
                        action = self.actor(torch.tensor(state)).detach().cpu().numpy()
                        action = (action + np.random.normal(0, 0.1, size=action.shape)).clip(self.min_action, self.max_action)

                self.epsilon *= self.epsilon_decay

                # Perform the action, store the data in the memory buffer and update the reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated
                memory_buffer.append([state, action, reward, next_state, done])

                # Exit condition for the episode
                if done: break
                state = next_state #next_state

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

        # for ep in range(len(memory_buffer)):
        #     # your code here
        #     pass
        
        if(len(memory_buffer) < self.batch_size): return

        # Sample a batch of experiences
        # state, action, reward, next_state, done = zip(*random.sample(memory_buffer, self.batch_size))

        # state = torch.stack(state).type(torch.float).to(self.device)
        # action = torch.tensor(action, dtype=torch.float).to(self.device)
        # reward = torch.tensor(reward, dtype=torch.float).to(self.device).unsqueeze(1)
        # next_state = torch.stack(next_state).type(torch.float).to(self.device)
        # done = torch.tensor(done, dtype=torch.float).to(self.device).unsqueeze(1)

        batch = random.sample(memory_buffer, self.batch_size)

        state = torch.tensor(np.array([s[0] for s in batch])).type(torch.float)
        action = torch.tensor(np.array([s[1] for s in batch])).type(torch.float)
        reward = torch.tensor(np.array([s[2] for s in batch])).type(torch.float).unsqueeze(1)
        next_state = torch.tensor(np.array([s[3] for s in batch])).type(torch.float)
        done = torch.tensor(np.array([s[4] for s in batch])).type(torch.float).unsqueeze(1)

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            actions_next = self.actor_target(next_state)
            Q_target_next = self.critic_target(next_state, actions_next)
            target_Q = reward + (self.gamma * Q_target_next * (1 - done))

        # Compute the critic Q and the target critic loss
        critic_q = self.critic(state, action)
        critic_loss = F.mse_loss(critic_q, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks to optimize computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actions_pred = self.actor(state)
        critic_value = self.critic(state, actions_pred)
        actor_loss = -torch.mean(critic_value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            # target_param.data = target_param.data * (1 - self.tau) + param.data * self.tau
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            # target_param.data = target_param.data * (1 - self.tau) + param.data * self.tau