import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import gymnasium as gym
import random

###
# CREATE THE NEURAL NETWORK
###

# create a 3 full coneection level neural network 
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed = 42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    # create the forward function to propagate the signal from input layer to output
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

###
# TRAINING
###

# SETTING UP THE ENVIRONMENT

# get the Lunarlander-v2 environment
env = gym.make('LunarLander-v2')

# get parameters
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print (f"state_shape: {state_shape}, state_size: {state_size}, number_actions: {number_actions}")


# INITIALIZATION OF THE HYPERPARAMETERS

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


# IMPLEMENTING EXPERIENCE REPLAY
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

    # Add __len__ method to allow the use of len()
    def __len__(self):
        return len(self.memory)
    
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
    
# IMPLEMENTING THE DQN CLASS

# Agent class
# Define the behaviour of an agent that interacts with our space environment using deep Q-network
#
# The agent maintains 2 Q-networks: local_qnetwork and target_qnetwork
# The local_qnetwork will select the actions
# The target_qnetwork will calculate the target Q-values that will be used
# in the training of the local_qnetwork.
# This double Q-network setup will stabilize the learning process, by using
# the soft_update() method, that update tht target_qnetwork parameters with those in the local_qnetwork
class Agent():
    def __init__(self, state_size, action_size, seed = 42):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # initialize the Q-networks
        self.local_qnetwork = Network(state_size, action_size, seed).to(self.device)
        self.target_qnetwork = Network(state_size, action_size, seed).to(self.device)
        
        # initialize the optimizer
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=LR)

        # initialize the memory
        self.memory = ReplayMemory(BUFFER_SIZE)

        # initialize the step counter
        self.t_step = 0
    
    # step() method
    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.push((state, action, reward, next_state, done))
        
        # every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
    
    # act() method
    # It will help the agent choose an action based on its understanding of the optimal policy.
    # Those actions will be returned from the local_qnetwork, that will forward propagate the state 
    # to return the action values. 
    # Then, following an epsilon-greedy policy, It will retrun the final action.
    # The fact that sometimes we select random actions, allows the agent to explore some more actions
    # which could potentially lead to a better result at the end.
    def act(self, state, eps=0.):
        # convert state to a tensor. Add an extra dimension which correspond to the batch 
        # so that we can feed it into the network
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # set the network in evaluation mode
        self.local_qnetwork.eval()
        # get the action values
        with torch.no_grad(): # no gradient calculations needed
            action_values = self.local_qnetwork(state)
        # set the network back in training mode
        self.local_qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # greedy action: select the action with the highest Q-value
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # random action: select a random action to allow the agent to explore the space
            return random.choice(np.arange(self.action_size))

    # learn() method
    # Updates the agent's Q values based on sampled experiences
    # It uses experiences that are sampled from the replay memory 
    # in order to update the local_qnetwork's Q-values towards
    # the target Q-values
    def learn(self, experiences, gamma):
        # extract each component of the sample and convert it in a pyTorch tensor
        states, next_states, actions, rewards, dones = experiences
        # get the maximum predicted Q-values for the next states from the target network
        Q_targets_next = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        # compute the target Q-values for the current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # get the Q-values for the current states
        Q_expected = self.local_qnetwork(states).gather(1, actions)

        # compute the loss between the expected Q-values and the target Q-values
        loss = F.mse_loss(Q_expected, Q_targets)

        # optimize the model by minimizing the loss
        self.optimizer.zero_grad() # zero the gradient buffers
        # backpropagate the loss through the network to update the weights  
        loss.backward()
        # perform a single optimization step using gradient descent to update the weights
        self.optimizer.step()

        # update the target network with the local network's weights
        self.soft_update(self.local_qnetwork, self.target_qnetwork, TAU)

    # soft_update() method
    # Softly updates the target network with the local network's weights
    # It uses the target network's weights with a factor tau and the local network's weights
    # It is used to update the target network towards the local network in order to stabilize the training
    def soft_update(self, local_model, target_model, tau):
        # loop over the target network's parameters and the local network's parameters
        # and update the target parameters accordingly.
        # The local network's parameters are updated with a factor tau
        # while the target network's parameters are updated with a factor 1-tau
        # this helps to stabilize the training
        # The zip() function returns an iterator that combines multiple iterables into one.
        # It is used to loop over multiple iterables at the same time, so that they can be used in parallel.
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
# INITIALIZING THE DQN AGENT
agent = Agent(state_size=state_size, action_size=number_actions, seed=0)

# TRAINING THE AGENT

# initialise the trainig parameters
episodes = 2000 # max number of episodes over which we want to train the agent
max_t = 1000 # max number of time steps per episode
eps_start = 1.0 # 1st epsilon value in epsilon-greedy startegy
eps_end = 0.01 # final epsilon value in epsilon-greedy startegy
eps_decay = 0.995 # decay rate in epsilon-greedy startegy

# scores tracking
scores = [] # list containing scores from each episode
scores_window = deque(maxlen=100) # last 100 scores
eps = eps_start # initialize epsilon

# training loop
for i in range(episodes + 1):
    # reset the environment to its initial state at each episode
    state, _ = env.reset()
    # reset the score to 0 at each episode
    score = 0

    # loop over the max number of time steps per episode
    for t in range(max_t):
        # select an action using the epsilon-greedy startegy
        action = agent.act(state, eps)
        # execute the action in the environment and receive the next state, reward and other information
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # perform the learning step
        agent.step(state, action, reward, next_state, done)
        # update the state
        state = next_state
        # update the score
        score += reward
        if done:
            break

    # save the most recent score
    scores_window.append(score) 
    scores.append(score) 

    # update epsilon
    eps = max(eps_end, eps_decay*eps) 

    # print information
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
    # if the environment is solved, print the last line with no dynamic override and exit the loop
    if i % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))

    # check if the environment is solved
    if np.mean(scores_window)>=200.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
        # save the parameters of the trained model
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break # exit the training loop


# PLOTTING THE RESULTS
import glob
import io
import base64
import imageio
import imageio_ffmpeg as ffmpeg
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

# show video of model
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frames.append(env.render())
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
    env.close()

    # imageio.mimsave(f'vid/{env_name}.gif', [np.array(frame) for frame in frames], fps=30)
    # imageio.mimsave(f'./{env_name}.gif', [np.array(frame) for frame in frames], fps=30)
    imageio.mimsave('video.mp4', frames, fps=30)
show_video_of_model(agent, 'LunarLander-v2')

# show video
def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()