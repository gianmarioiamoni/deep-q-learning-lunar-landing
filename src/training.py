# training the neural network
import gymnasium as gym
import torch
import random
import numpy as np

# ### SETTING UP THE ENVIRONMENT

# get the Lunarlander-v2 environment
env = gym.make('LunarLander-v2')

# get parameters
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print (f"state_shape: {state_shape}, state_size: {state_size}, number_actions: {number_actions}")

# ### INITIALIZATION OF THE HYPERPARAMETERS

# the size of the memeory of the AI; 
# how many experiences (state, action, reward, next date) in the memory of the agent; 
# used to stabilize and improve the training process
BUFFER_SIZE = int(1e5)  # replay buffer size

# number of observations used in 1 step of the training 
# used to update the model parameters
BATCH_SIZE = 100     # minibatch size

# the present value of future rewards; close to 1
GAMMA = 0.99            # discount factor

# interpolation parameter is used to interpolate the present value of future rewards
TAU = 1e-3              # interpolation parameter for soft update of target parameters

# learning rate is used to update the model parameters 
LR = 5e-4               # learning rate 

# update rating of the network
UPDATE_EVERY = 4        # how often to update the network

# ### IMPLEMENTING EXPERIENCE REPLAY
class ReplayMemory(object):
    def __init__(self, capacity):
        # in case of using GPU for training
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # store memory attribute
        # It is the maximum size of the memory of the agent
        self.capacity = capacity

        # initialize the memory
        # list in which storing the experiences
        # each one containing the sate, the action, the reward, the next state 
        # and wheter we are done or not
        self.memory = []

    # push() method
    # add the experience event into the memory buffer
    # check that we don't exceed the capacity
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            # if capacity is exceeded, delete the less recent event
            del self.memory[0]

    # sample() method
    # sample a batch of experiences from the memory
    def sample(self, batch_size):
        # random sampling
        experiences = random.sample(self.memory, k = batch_size)

        # extract each component of the sample and convert it in a pyTorch tensor
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, next_states, actions, rewards, dones)