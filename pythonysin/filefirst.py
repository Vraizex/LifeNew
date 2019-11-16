from math import pi
import tensorflow as tf
import numpy as np
##This is rows  for CPU not GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##input data
mean = 0.0
sigma = 1.0
k = 1

##Gauss распределение
m5 = (tf.exp(tf.negative(tf.pow(k - mean, 2.0) /
                    (2.0 * tf.pow(sigma, 2.0) ))) *
 (1.0 / (sigma * tf.sqrt(2.0 * pi))))

##example input data
m1 = [[1.0, 2.0],
      [3.0, 4.0]]
m2 = np.array([[1.0, 2.0],
      [3.0, 4.0]], dtype=np.float)
m3 = tf.constant([[1.0, 2.0],
      [3.0, 4.0]])

print(type(m1))
print(type(m2))
print(type(m3))

##example for array , convert to Tensorflow
t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m1, dtype=tf.float32)
t3 = tf.convert_to_tensor(m1, dtype=tf.float32)
t4 = tf.ones([500,500])*0.5

##output data
print(type(t1))
print(type(t2))
print(type(t3))
print(type(t4))
print(type(m5))


##used session for processing
##sess = tf.InnteractiveSession()
##x = tf.constant([[1., 2.]])
##negMatrix = tf.negative(x)
##result = negMatrix.eval
##print(result)
##sess.close()


x = tf.constant([[1., 2.]])
negMatrix = tf.negative(x)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      result = sess.run(negMatrix)

print(result)


sess = tf.IneractiveSession()
