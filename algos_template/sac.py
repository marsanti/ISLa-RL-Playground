import time
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
import collections
import random
import copy
from utils.utils import ActorSAC, Double_Q_Critic, init_wandb
import wandb


class SAC():
    def __init__(self, params, use_wandb=False):
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'])
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="linux", worker_id=int(time.time())%10000, time_scale=1, no_graphics=False, max_step=100, action_space_type='continuous')
        
        self.env_name = params['gym_environment']
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # DNN configurations
        self.hidden_layers_actor = params['parameters']['hidden_layers_actor']     
        self.hidden_layers_critic = params['parameters']['hidden_layers_critic']
        self.nodes_hidden_layers_actor = params['parameters']['nodes_hidden_layers_actor']
        self.nodes_hidden_layers_critic = params['parameters']['nodes_hidden_layers_critic']
        self.lr_actor = params['parameters']['lr_actor_optimizer']
        self.lr_critic = params['parameters']['lr_critic_optimizer']
        self.adaptive_alpha = params['parameters']['adaptive_alpha']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Hyperparameters
        self.gamma = params['parameters']['gamma']
        self.alpha = params['parameters']['alpha']
        self.tau = params['parameters']['tau']
        self.update_freq = params['parameters']['update_freq']
        self.n_updates = params['parameters']['n_updates']
        self.total_episodes = params['tot_episodes']
        self.use_wandb = use_wandb
        self.epsilon = 1.0
        self.epsilon_decay = params['parameters']['eps_decay']
        self.batch_size = params['parameters']['batch_size']

        # create actor and critic
        self.actor = ActorSAC(self.state_dim, self.action_dim, (self.nodes_hidden_layers_actor, self.nodes_hidden_layers_actor)) 
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critic networks (Q-value estimators)
        self.critic = Double_Q_Critic(self.state_dim, self.action_dim, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Critic target networks
        self.critic_target = copy.deepcopy(self.critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.device)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr_critic)

        

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.device)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def training_loop(self, seed, args_wandb=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if self.use_wandb: init_wandb(args_wandb)

        rewards_list, reward_queue = [], collections.deque(maxlen=100)
        memory_buffer = []
        for ep in range(self.total_episodes):

            # Reset the environment and episode reward
            state = self.env.reset(seed=seed)[0] if self.env_name != "TB3" else self.env.reset()
            ep_reward = 0

            while True:
                # action = self.actor(torch.FloatTensor(state)).detach().numpy()
                action, _ = self.actor(state, False, with_logprob=False)
                action = action.detach().numpy()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                memory_buffer.append([state, action, reward, next_state, done])
                if done:
                    break
                state = next_state

            rewards_list.append(ep_reward)
            reward_queue.append(ep_reward)

            print( f"episode {ep:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f})" )
            if self.use_wandb:
                wandb.log({'mean_reward': np.mean(reward_queue)})

            # Update
            if ep % self.update_freq == 0:
                for _ in range(self.n_updates):
                    self.update_policy(memory_buffer)
            
                # memory_buffer = []
    
    def update_policy(self, memory_buffer: list):  
        
        batch = random.sample(memory_buffer, self.batch_size)

        states = torch.tensor(np.array([s[0] for s in batch])).type(torch.float).to(self.device)
        actions = torch.tensor(np.array([s[1] for s in batch])).type(torch.float).to(self.device)
        rewards = torch.tensor(np.array([s[2] for s in batch])).type(torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array([s[3] for s in batch])).type(torch.float).to(self.device)
        dones = torch.tensor(np.array([s[4] for s in batch])).type(torch.float).unsqueeze(1).to(self.device)

        # Calculate target Q
        # next_actions = self.actor(next_states)
        next_actions, log_prob = self.actor(next_states, deterministic=False, with_logprob=True)
        target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        alpha_log = self.alpha * log_prob
        # print(min_target_Q.shape)
        target_Q = rewards + (self.gamma * (1 - dones) * (target_Q - alpha_log)).detach()

        # Update critic networks
        # Critic 1
        Q1, Q2 = self.critic(states, actions)
        
        q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.critic.parameters(): params.requires_grad = False
        a, log_pi_a = self.actor(states, deterministic=False, with_logprob=True)
        Q1, Q2 = self.critic(states, actions)
        Q = torch.min(Q1, Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.critic.parameters(): params.requires_grad = True

        if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)