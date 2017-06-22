# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:46:16 2017

@author: aakash.chotrani
"""

import tensorflow as tf

node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)


#creating a session to evaluate the nodes
sess = tf.Session()
print(sess.run([node1,node2]))

#adding two nodes together
node3 = tf.add(node1,node2)
print("node3: ",node3)
print("sess.run(node3): ",sess.run(node3))

#creating placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node,{a:3,b:4.5}))
print(sess.run(adder_node,{a:[1,3],b:[2,4]}))

add_and_triple = adder_node*3
print(sess.run(add_and_triple,{a:3,b:4.5}))


#creating variables to add trainable parameters to a graph
W = tf.Variable([3],dtype = tf.float32)
b = tf.Variable([-3],dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b


#calling tf method to initialize variable into tensorflow program
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model,{x:[1,2,3,4]}))

#predicting how well is our model. Using squared distance delta method
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss,{x:[1,2,3,4],y:[0,3,6,15]}))


#fixing our model, Using assign method to change the varable value
fixW = tf.assign(W,[-1])
fixb = tf.assign(b,[1])
sess.run([fixW,fixb])
print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]}))
