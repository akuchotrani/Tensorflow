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