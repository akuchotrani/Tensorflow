# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:24:55 2017

@author: aakash.chotrani
"""
import gym
#import random
import math

MEMORY_CAPACITY = 100000
LEARNING_RATE = 0.99
MAX_EPSILON = 0.01#max explore factor
MIN_EPSILON = 0.01#min explore factor

LAMBDA = 0.001#speed of decay
class Agent:
    def __init__(self,totalObservationAvailable,totalActionsAvailable):
        self.totalObservationAvailable = totalObservationAvailable
        self.totalActionsAvailable = totalActionsAvailable
        print('totalObservationAvailable',totalObservationAvailable)
        print('totalActionsAvailable',totalActionsAvailable)
    def act(self,s):
        return 0
    
    def observe(self,sample):
        self.memory.add(sample)
    
        #reduce the epsilon based on experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON-MIN_EPSILON)* math.epx(-LAMBDA*self.steps)
        
    
    def replay(self):
        return 0
    
 
        
        
        





class Environment:
    def __init__(self,problem):
        self.problem = problem
        self.env = gym.make(problem)
        
    def run(self):
        for i_episode in range(10):
            state =self. env.reset()
            reward = 0
            for t in range(100):
                self.env.render()
                action = self.env.action_space.sample()
                newState, reward, done, info = self.env.step(action)
                if done:
                    break
                
    def closeEnvironment(self):
        self.env.close()
        
            
def main():
    env = Environment("LunarLander-v2")
    env.run()
    stateCnt  = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n
    agent = Agent(stateCnt, actionCnt)
    env.closeEnvironment()
    
    
    
    
main()