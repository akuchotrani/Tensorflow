# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:41:01 2017

@author: aakash.chotrani
"""
import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

#result = x1*x2
result = tf.multiply(x1,x2)

print(result)

sess = tf.Session()
print(sess.run(result))
sess.close()

#with tf.Session() as sess:
#    print(sess.run(result))


