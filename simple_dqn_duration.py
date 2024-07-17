import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
import os


class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done =  zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim,hidden_dim) 
        self.fc4 = nn.Linear(hidden_dim, output_dim) 
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self,intersection_id, state_dim, action_dim,cfg,phase_list):
        self.intersection_id = intersection_id
        self.phase_list = phase_list
        self.action_dim = action_dim  
        self.device = cfg.device  
        self.gamma = cfg.gamma  
        
        self.frame_idx = 0  
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        
    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            action = self.predict(state)
        else:
            action = random.randrange(self.action_dim)
        return action+5

    def predict(self,state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)[0]
            action = q_values.max(1)[1].item()
        return action

    def remember(self, state, action, reward, next_state,done):
        action -=5
        self.memory.push(state, action, reward, next_state,done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)

        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float).squeeze()
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1) 
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float).squeeze()
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)

        q_values = self.policy_net(state_batch).gather(
            dim=1, index=action_batch)  
        next_q_values = self.target_net(next_state_batch).max(
            1)[0].detach()  
        expected_q_values = reward_batch + \
            self.gamma * next_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1)) 

        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()  

    def save(self, path):
        torch.save(self.target_net.state_dict(), path)

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

class MDQNAgent_duration(object):
    def __init__(self,
                 intersection,
                 state_size,
                 cfg,
                 phase_list,
                 action_dim,
                 ):

        self.intersection = intersection
        self.agents = {}
        self.make_agents(state_size, cfg, phase_list,action_dim)

    def make_agents(self, state_size, cfg, phase_list,action_dim):
        for id_ in self.intersection:
            self.agents[id_] = DQNAgent(intersection_id=id_,
                                        state_dim=state_size,
                                        action_dim=action_dim,
                                        cfg=cfg,
                                        phase_list=phase_list[id_],
                                        )

    def remember(self, state, action, reward, next_state,done):
        for id_ in self.intersection:
            self.agents[id_].remember(state[id_],
                                      action[id_],
                                      reward[id_],
                                      next_state[id_],
                                      done[id_]
                                      )

    def choose_action(self, state):
        action = {}
        for id_ in self.intersection:
            action[id_] = self.agents[id_].choose_action(state[id_])
        return action

    def replay(self):
        for id_ in self.intersection:
            self.agents[id_].update()

    def load(self, name):
        for id_ in self.intersection:
            print(os.path.abspath('.'))
            model_path = name + '.' + id_
            print(model_path)
            assert os.path.exists(model_path), "Wrong checkpoint, file not exists!"
            self.agents[id_].load(name + '.' + id_)
        print("\nloading model successfully!\n")

    def save(self, name):
        for id_ in self.intersection:
            self.agents[id_].save(name + '.' + id_)