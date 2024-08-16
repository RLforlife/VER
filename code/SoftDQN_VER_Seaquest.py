#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.signal
import gym, random, pickle, os.path, math, glob
from IPython.core.debugger import set_trace
from gym.wrappers import Monitor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Categorical

import pdb

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output

from atari_wrappers import make_atari, wrap_deepmind
from tensorboardX import SummaryWriter

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.cuda.set_device(2)


# In[2]:


class soft_DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5, REWARD_SCALE = 1):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(soft_DQN, self).__init__()        
        self.REWARD_SCALE = REWARD_SCALE
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
    
    def get_action(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        action_probs = F.softmax(self.fc5(x)/self.REWARD_SCALE,-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)
        return actions


# In[3]:


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    
class Memory_Buffer_PER_IS(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, memory_size=100000, a = 0.6, beta = 0.4, e = 0.0001, beta_increment_per_sampling = 0.4e-6):
        self.tree =  SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.beta = beta
        self.e = e
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) # max to 1
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        # is_weight = np.clip(is_weight, 0, 1)
        return idxs, np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, is_weight
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
        
    def size(self):
        return self.tree.n_entries


# In[4]:


class softDQN_VER_Agent: 
    def __init__(self, in_channels = 1, action_space = [], USE_CUDA = False, memory_size = 10000, lr = 1e-4,reward_scale = 1, prio_a = 0.6, prio_beta = 0.4, prio_e = 0.001, beta_increment_per_sampling = 0.4e-6):
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer_PER_IS(memory_size, a = prio_a, beta = prio_beta, e = prio_e, beta_increment_per_sampling = beta_increment_per_sampling)
        self.alpha = reward_scale
        self.DQN = soft_DQN(in_channels = in_channels, num_actions = action_space.n,REWARD_SCALE = reward_scale)
        self.DQN_target = soft_DQN(in_channels = in_channels, num_actions = action_space.n,REWARD_SCALE = reward_scale)
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        self.optimizer = optim.Adam(self.DQN.parameters(),lr=lr)

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        state =  torch.from_numpy(lazyframe._force().transpose(2,0,1)[None]/255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values
    
    def act(self, state, t=0, explore_time=0):
        """
        random policy first, 
        then sample action according to softmax policy
        """
        if t < explore_time:
            action = self.action_space.sample()
        else:
            action = int(self.DQN.get_action(state).squeeze().cpu().detach().numpy())
        return action
    
    
    def compute_td_loss(self, idxs, states, actions, rewards, next_states, is_done, is_weight, gamma=0.99):
        """ Compute td loss using torch operations only. Use the formula above. """
        actions = torch.tensor(actions).long()    # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype =torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done).type(torch.bool)  # shape: [batch_size]
        is_weight = torch.tensor(is_weight, dtype =torch.float)

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()
            is_weight = is_weight.cuda()

        # get q-values for all actions in current states
        predicted_qvalues = self.DQN(states)
        # get action probs
        action_probs = F.softmax(predicted_qvalues,-1)
        action_prob = action_probs[range(states.shape[0]), actions]
        
        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
          range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.DQN_target(next_states) # YOUR CODE
        # compute V*(next_states) using predicted next q-values
        next_state_values =  self.alpha*torch.logsumexp(predicted_next_qvalues/self.alpha, dim = -1) # YOUR CODE        

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma*next_state_values # YOUR CODE

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # calculate TD error & update priorities
        td_error = predicted_qvalues_for_actions - target_qvalues_for_actions
        self.memory_buffer.update(idxs, (action_prob*td_error).detach().cpu().numpy())

        # mean squared error loss to minimize
        #loss = torch.mean((predicted_qvalues_for_actions -
        #                   target_qvalues_for_actions.detach()) ** 2)
        loss = (F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach(), reduction='none')*is_weight).mean()
        logger.store(QVals = predicted_qvalues.detach().cpu().numpy(), LossQ = loss.item())
        return loss
    
    def sample_from_buffer(self, batch_size):
        self.memory_buffer.beta = np.min([1., self.memory_buffer.beta + self.memory_buffer.beta_increment_per_sampling]) # max to 1
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = []
        segment = self.memory_buffer.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.memory_buffer.tree.get(s)
            
            frame, action, reward, next_frame, done= data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        
        sampling_probabilities = priorities / self.memory_buffer.tree.total()
        is_weight = np.power(self.memory_buffer.tree.n_entries * sampling_probabilities, -self.memory_buffer.beta)
        is_weight /= is_weight.max()
        # is_weight = np.clip(is_weight, 0, 1)
        return idxs, torch.cat(states), actions, rewards, torch.cat(next_states), dones, is_weight
    
    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            idxs, states, actions, rewards, next_states, dones, is_weight = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(idxs, states, actions, rewards, next_states, dones, is_weight)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return(td_loss.item())
        else:
            return(0)


# In[ ]:


from run_utils import setup_logger_kwargs
import itertools
import time
from logx import EpochLogger
import pdb

# Training DQN in PongNoFrameskip-v4 
# pdb.set_trace()
env_id = 'SeaquestNoFrameskip-v4'
experiment_name = "softDQN_VER4_" + env_id
logger_kwargs = setup_logger_kwargs(experiment_name, 0)
logger = EpochLogger(**logger_kwargs)

experiment_dir = os.path.abspath(experiment_name)
monitor_path = os.path.join(experiment_dir, "monitor")
eval_monitor_path = os.path.join(experiment_dir, "eval_monitor")

log_dir = os.path.join(experiment_dir, "log")
model_path = os.path.join(experiment_dir, experiment_name+"_dict.pth.tar")
checkpoint_path = os.path.join(experiment_dir, "check_point")
env = make_atari(env_id)
env = wrap_deepmind(env, scale = False, frame_stack=True , clip_rewards= False, episode_life = True)
env_eval = make_atari(env_id)
env_eval = wrap_deepmind(env_eval, scale = False, frame_stack=True , clip_rewards= False, episode_life = False)

gamma = 0.99
steps_per_epoch = 100000
epochs = 100 # 1000
frames = steps_per_epoch * epochs# 10000000 timestamp/ 
USE_CUDA = True
learning_rate = 1e-4
max_buff = 1000000
prio_a = 0.6
prio_beta = 0.4

update_tar_interval = 10000
batch_size = 32
learning_start = 50000
update_current_step = 4 # update current model every 4 steps
beta_increment_per_sampling = (1-prio_beta)/(frames/update_current_step)
record_video = True
record_video_every = 500 # video every 1000
eval_every = steps_per_epoch
save_freq = 1
num_test_episodes = 10

action_space = env.action_space
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
state_channel = env.observation_space.shape[2]
reward_scale = 0.05
# logger.save_config(locals())
agent = softDQN_VER_Agent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate, memory_size = max_buff, reward_scale = reward_scale,
                         prio_a = prio_a, prio_beta=prio_beta, beta_increment_per_sampling=beta_increment_per_sampling)
# Set up model saving
logger.setup_pytorch_saver(agent.DQN)

def eval_episode(agent, env_eval):
    with torch.no_grad():
        for j in range(num_test_episodes):       
            frame, done, ep_ret, ep_len  = env_eval.reset(), False,0,0
            while not done:
                state_tensor = agent.observe(frame)
                action = agent.act(state_tensor, 0,0)
                frame, reward, done, _ = env_eval.step(action)
                ep_ret += reward
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


if record_video:
    env = Monitor(env,
                 directory=monitor_path,
                 resume=True, mode = "training",
                 video_callable=lambda count: count % record_video_every == 0)
    env_eval = Monitor(env_eval,
                 directory=eval_monitor_path,
                 resume=True,
                 video_callable=lambda count: count % num_test_episodes == 0,
                      mode = "evaluation")

frame, ep_ret, ep_num,ep_len = env.reset(),0, 0,0
loss = 0

# tensorboard
summary_writer = SummaryWriter(log_dir = log_dir, comment= "good_makeatari")


start_time = time.time()
for i in range(frames):
    state_tensor = agent.observe(frame)
    action = agent.act(state_tensor, i, learning_start)
    
    next_frame, reward, done, _ = env.step(action)
    
    ep_ret += reward
    ep_len += 1
    agent.memory_buffer.push(frame, action, np.sign(reward), next_frame, done) # !! Clip reward by its sign
    frame = next_frame
    
    if agent.memory_buffer.size() >= learning_start:
        if i % update_current_step == 0:
            loss = agent.learn_from_experience(batch_size)
         
    if i % update_tar_interval == 0:
        agent.DQN_target.load_state_dict(agent.DQN.state_dict())
    
    if done:
        logger.store(EpRet=ep_ret, EpLen=ep_len)
        frame, ep_ret, ep_len = env.reset(), 0, 0
    
    if (i+1) % steps_per_epoch == 0:  
        epoch = (i+1) // steps_per_epoch

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs):
            logger.save_state({'env': env}, None)

        # Test the performance of the deterministic version of the agent.
        eval_episode(agent, env_eval)
        
        # Write to tensorboard
        summary_writer.add_scalar("Train Reward",logger.get_stats("EpRet")[0], i)
        summary_writer.add_scalar("Test Reward",logger.get_stats("TestEpRet")[0], i)
        summary_writer.add_scalar("Loss Q",logger.get_stats("LossQ")[0], i)
        summary_writer.add_scalar("Train EpLen",logger.get_stats("EpLen")[0], i)
        summary_writer.add_scalar("Test EpLen",logger.get_stats("TestEpLen")[0], i)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TestEpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', i)
        logger.log_tabular('QVals', with_min_and_max=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)


        logger.dump_tabular()
        
summary_writer.close()
torch.save(agent.DQN.state_dict(), model_path)


# In[ ]:




