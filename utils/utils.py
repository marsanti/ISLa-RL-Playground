import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

CYAN_COL = '\033[96m'
BLUE_COL = '\033[94m'
RED_COL = '\033[91m'
GREEN_COL = '\033[92m'
YELLOW_COL = '\033[93m'
RESET_COL = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def check_parameters(method):
    if method['name'] not in ['PPO', 'REINFORCE', 'DDPG', 'SAC', 'MCTS', 'QLEARNING']: 
        raise ValueError(f"{RED_COL}{method['name']} method is not supported! Select one of the following methods: ['PPO', 'DDPG', 'SAC', 'MCTS', 'QLEARNING'] in the yaml file.{RESET_COL}")
    

class TorchModel(nn.Module):
	"""
	Class that generates a neural network with PyTorch and specific parameters.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	"""
	
	# Initialize the neural network
	def __init__(self, nInputs, nOutputs, nLayer, nNodes, last_activation=F.linear):
		
		super(TorchModel, self).__init__()
		self.nLayer = nLayer
		self.last_activation= last_activation

		# input layer
		self.fc1 = nn.Linear(nInputs, nNodes)

		#hidden layers
		for i in range(nLayer):
			layer_name = f"fc{i+2}"
			self.add_module(layer_name, nn.Linear(nNodes, nNodes))  

		#output
		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		for i in range(2, self.nLayer + 2):
			x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
		x = self.output(x)
		# return x if self.last_activation == F.linear else self.last_activation(x, dim=1)
		return x if self.last_activation == F.linear else self.last_activation(x)
	
class ActorModel(TorchModel):
	def __init__(self, nInputs, nOutputs, nLayer, nNodes):
		super().__init__(nInputs, nOutputs, nLayer, nNodes)	


	def forward(self, x):
		x = F.relu(self.fc1(x))
		for i in range(2, self.nLayer + 2):
			x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
		x = self.output(x)
		return torch.tanh(x)

class CriticModel(TorchModel):
	def __init__(self, nInputs, nOutputs, nLayer, nNodes, last_activation=F.linear):
		super().__init__(nInputs, nOutputs, nLayer, nNodes, last_activation)

	# override forward method
	def forward(self, state, action):
		x = F.relu(self.fc1(torch.cat([state, action], 1)))
		for i in range(2, self.nLayer + 2):
			x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))

		# calculate the q value with the last layer
		q_value = self.output(x)
		
		return q_value

class Double_Q_Critic(nn.Module):
	def __init__(self, stateDim, actionDim, nLayer, nNodes):
		super(Double_Q_Critic, self).__init__()

		self.Q_1 = TorchModel(stateDim + actionDim, 1, nLayer, nNodes)
		self.Q_2 = TorchModel(stateDim + actionDim, 1, nLayer, nNodes)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2

def build_net(layer_shape, hidden_activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

class ActorSAC(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
		super(ActorSAC, self).__init__()
		layers = [state_dim] + list(hid_shape)

		self.a_net = build_net(layers, hidden_activation, output_activation)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def forward(self, state, deterministic, with_logprob):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX) 
		# we learn log_std rather than std, so that exp(log_std) is always > 0
		std = torch.exp(log_std)
		dist = Normal(mu, std)
		if deterministic: u = mu
		else: u = dist.rsample()

		'''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
		a = torch.tanh(u)
		if with_logprob:
			# Get probability density of logp_pi_a from probability density of u:
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a

class GaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(GaussianActor_musigma, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.sigma_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		sigma = F.softplus( self.sigma_head(a) )
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu, _ = self.forward(state)
		return mu

def init_wandb(args):
    wandb.init(
        name=args['run_name'],
        project=args['project'],
        entity=args['entity'],
        mode= "online",
        save_code=False,
        config=args
    )


def plot_results(results):

	# results is a list of dictionaries where we have [{'method': 'name_method', 'mean_rew': [[r1,...,rn] (seed 0), [r1,...,rn], [r1,..., rn]], 'training_episodes': episodes}]

	# we take the training episodes from the first dictionary as if we want to compare two methods they should have performed the same number pof episodes
	t = list(range(0, results[0]['training_episodes']))

	# Plotting
	sns.set_style("darkgrid")
	plt.figure(figsize=(8, 6))  # Set the figure size
 
	for dict in results:
		data = {'Environment Step': [], 'Mean Reward': []}
		for seed, rewards in enumerate(dict['mean_rewards']):
			for step, reward in zip(t, rewards):
				data['Environment Step'].append(step)
				data['Mean Reward'].append(reward)
			df = pd.DataFrame(data)
		
			sns.lineplot(data=df, x='Environment Step', y='Mean Reward', label=[dict['method'], seed], errorbar='se')

	plt.title(f'{dict["env"]}')
	# Add title and labels
	plt.xlabel('Episodes')
	plt.ylabel('Mean Reward')

	# Show legend
	plt.legend()

	# Show plot
	plt.savefig(f'results/{dict["env"]}/plot.pdf', format='pdf')