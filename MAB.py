# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:46:47 2022

@author: msadi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class MultiArmBanditEnv:
    def __init__(self,n):
        self.arms = n
        self.rewards = np.random.uniform(low=0, high=100, size=(self.arms,))
    
    def get_reward(self,index):
        noise = np.random.normal(0,1,1)
        reward = self.rewards[index]
        reward = reward + noise
        return reward
    
    def get_actual_reward(self):
        return self.rewards
    
    def get_arms(self):
        return self.arms

class Agent:
    def __init__(self,arms,epsilon):
        self.arms = arms
        self.epsilon = epsilon
        self.Q_t = list(np.random.uniform(low=0, high=50, size=(self.arms,)))
        self.N_t = [0]*arms
        self.record = [0]
        self.avg_reward = [0]
        
    
    def select_arm(self):
        num = random.random()
        if num<=(1-self.epsilon):
            # print(1)
            return self.Q_t.index(max(self.Q_t))
        else:
            # print("Error",num,(1-self.epsilon),self.epsilon)
            # x=input()
            max_index = self.Q_t.index(max(self.Q_t))
            new_num = random.random()
            index = int(new_num/(1/(self.arms-1)))
            # print(index,max_index)
            if index>=max_index:
                return index+1
            else:
                return index
    def update(self,index,reward):
        self.record.append( self.record[len(self.record)-1] + reward )
        self.avg_reward.append(self.record[len(self.record)-1]/(len(self.avg_reward)))
        try:
            self.Q_t[index] = self.Q_t[index] + (1/self.N_t[index]) * (reward - self.Q_t[index])
        except:
            self.Q_t[index] = self.Q_t[index] + (reward - self.Q_t[index])
        self.N_t[index] +=1
        
    def plot(self):
        plt.plot(self.record)
        plt.show()
            
    def plot_avg(self):
        plt.plot(self.avg_reward)
        plt.show()
    
    def get_avg(self):
        return self.avg_reward
    
    def get_Q(self):
        return self.Q_t

    def get_N(self):
        return self.N_t
    
env = MultiArmBanditEnv(10)

agent = Agent(env.get_arms(),0.1)
agent2 = Agent(env.get_arms(),0)

for i in range(2000):
    arm = agent.select_arm()
    reward = env.get_reward(arm)
    agent.update(arm,reward)
    # print(reward)
    # x=input()
    arm2 = agent2.select_arm()
    reward2 = env.get_reward(arm2)
    agent2.update(arm2,reward2)
    # print(reward2)
    # x=input()

# agent.plot_avg()
    
reward1= agent.get_avg()
reward2 = agent2.get_avg()
plt.plot(reward1,label="E=0.1")
plt.plot(reward2,label="E=0")
plt.legend()
plt.show()

print(env.get_actual_reward())
print(agent.get_Q())
print(agent2.get_Q())
print("-------------------")
print(agent.get_N())
print(agent2.get_N())



