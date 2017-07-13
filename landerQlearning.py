# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:24:55 2017

@author: aakash.chotrani
"""
import gym
import random
import math
import tensorflow as tf
import numpy as np


#-----------------------------BRAIN-----------------------------------
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
BATCH_SIZE = 100

class Brain:
    def __init__(self,observation_count,action_count,data):
        self.observation_count = observation_count
        self.action_count = action_count
        self.model = self.CreateModel(data)
        x = tf.placeholder('float',[None,self.observation_count])
        y = tf.placeholder('float')
        self.n_classes = self.action_count
        
    
    def CreateModel(self,data):
        hidden_1_layer ={'weights': tf.Variable(tf.random_normal([self.observation_count,n_nodes_hl1])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
        hidden_2_layer ={'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
        hidden_3_layer ={'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
        output_layer ={'weights': tf.Variable(tf.random_normal([n_nodes_hl3,self.n_classes])),
                     'biases':tf.Variable(tf.random_normal([self.n_classes]))}
        
        #(input_data*weights) + biases
        l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])#input goes through the sum box
        l1 = tf.nn.relu(l1)#rectified linear is activation function applied to layer 1
        
        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])#input goes through the sum box
        l2 = tf.nn.relu(l1)#rectified linear is activation function applied to layer 2
    
        l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])#input goes through the sum box
        l3 = tf.nn.relu(l3)#rectified linear is activation function applied to layer 3
    
    
        output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
        
        return output
        
    
    def TrainModel(self,train_x,train_y):
        prediction = self.CreateModel(train_x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = train_y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    
        #cycles of feed forward and back propagation
        hm_epochs = 10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            for epoch in range(hm_epochs):
                epoch_loss = 0
            
                i = 0
                while i < len(train_x):
                    start = i
                    end = i+BATCH_SIZE
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                
                    _,c = sess.run([optimizer,cost],feed_dict = {self.x: batch_x,self.y: batch_y})
                    epoch_loss += c
                    i += BATCH_SIZE
                
                print('Epoch',epoch+1,'completed out of', hm_epochs,'loss:',epoch_loss)
            #correct = tf.equal(tf.arg_max(prediction,1),tf.argmax(self.y,1))
        
        #accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        #print('Accuracy: ',accuracy.eval({self.x:test_x,self.y:test_y}))
        
        

#------------------------------MEMORY---------------------------------
class Memory:
    
    samples = []
    def __init__(self,capacity):
        self.capacity = capacity
    
    def add(self,sample):
        self.samples.append(sample)
        
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
    
    def sample(self,n):
        n = min(n,len(self.samples))
        return random.sample(self.samples,n)
        


#--------------AGENT--------------------------------------------------
MEMORY_CAPACITY = 100000
LEARNING_RATE = 0.99
MAX_EPSILON = 0.01#max explore factor
MIN_EPSILON = 0.01#min explore factor

LAMBDA = 0.001#speed of decay
class Agent:
    def __init__(self,totalObservationAvailable,totalActionsAvailable,observation):
        self.totalObservationAvailable = totalObservationAvailable
        self.totalActionsAvailable = totalActionsAvailable
        self.brain = Brain(self.totalObservationAvailable,self.totalObservationAvailable)
        print('totalObservationAvailable',totalObservationAvailable)
        print('totalActionsAvailable',totalActionsAvailable)
        
        
    def act(self,s):
        
        #take a random action
        if random.random() < self.epsilon:
            return random.randint(0,self.totalActionsAvailable-1)
        else:
            return np.argmax()
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
            state = self.env.reset()
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