#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

import time

from json import *
import json

import tensorflow as tf
from tensorflow import feature_column as fc

import random as rnd    

import scipy.stats

print tf.__version__


# reference to:
# https://blog.csdn.net/qq_25737169/article/details/76695935

# MAX_GRAD_NORM = 1e-1
epoch = 1000
embedding_dim = 3
lr = 1e-3
batch_size = 2000
LAMBDA=1e1
act = tf.nn.selu
MAX_NORM=1.

# optimizer = tf.train.GradientDescentOptimizer(lr)
# optimizer = tf.train.AdadeltaOptimizer(lr)
# optimizer = tf.train.AdagradOptimizer(lr)
# optimizer = tf.train.AdaMaxOptimizer(lr)
# optimizer = tf.train.AdamOptimizer(lr)
optimizer = tf.train.RMSPropOptimizer(lr)
# optimizer = tf.train.FtrlOptimizer(lr)

x = np.array(range(3))
prob0 = np.array([0.7, 0.1, 0.2])
prob1 = np.array([0.2, 0.1, 0.7])
# prob1 = np.array([0.3, 0.25, 0.45])

print scipy.stats.entropy(prob0, prob1)
print scipy.stats.entropy(prob1, prob0)

# clip model weights to a given hypercube

class ClipConstraint(tf.keras.constraints.Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
const = ClipConstraint(0.1)


embedding_x = fc.categorical_column_with_vocabulary_list(
    'x', [d for d in range(0, 100)], dtype=tf.int32, num_oov_buckets=1)

embedding_y = fc.categorical_column_with_vocabulary_list(
    'y', [d for d in range(0, 100)], dtype=tf.int32, num_oov_buckets=1)

embedding = tf.feature_column.shared_embedding_columns([embedding_x, embedding_y], dimension=embedding_dim)

def gen_train_data():
    while 1:
      yield (np.random.choice(a=x, size=1, replace=True, p=prob0), 
        np.random.choice(a=x, size=1, replace=True, p=prob1))

dataset = tf.data.Dataset.from_generator(gen_train_data, (tf.int32, tf.int32))
data = dataset.batch(batch_size)
it = data.make_one_shot_iterator()
train_data = it.get_next()



with tf.name_scope('embedding_columns'):
    embedding = fc.input_layer({'x':train_data[0], 'y':train_data[1]}, embedding)
    # embedding_y = fc.input_layer({}, embedding)

def net(name, _input, layer_units, act=act, out_units=1):
  # with tf.name_scope(name): # with tf.variable_scope(name):
  mu, stddev = 0.0, 0.01

  out = _input
  for ix, units in enumerate(layer_units):
    out = tf.layers.dense(out, units=units, activation=act, name="{0}_L{1}".format(name, ix),
      # kernel_initializer=tf.truncated_normal_initializer(mu, stddev)
      # ,kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
      # ,bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
      # kernel_constraint=tf.keras.constraints.max_norm(MAX_NORM)
      # ,bias_constraint=tf.keras.constraints.max_norm(MAX_NORM)
      kernel_constraint=const
      ,bias_constraint=const
      ,reuse=tf.AUTO_REUSE
      )
  
  out = tf.layers.dense(out, units=out_units, name="{0}".format(name),
      # kernel_initializer=tf.truncated_normal_initializer(mu, stddev)
      # ,kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
      # ,bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
      kernel_constraint=const
      ,bias_constraint=const
      # kernel_constraint=tf.keras.constraints.max_norm(MAX_NORM)
      # ,bias_constraint=tf.keras.constraints.max_norm(MAX_NORM)
      ,reuse=tf.AUTO_REUSE
      )
  return out

logits0 = net('NN', embedding[:,:embedding_dim], [128,128,128,10])
logits1 = net('NN', embedding[:,embedding_dim:], [128,128,128,10])

# test = tf.placeholder(tf.int32)
# test_inp = fc.input_layer({'x':test, 'y':test}, embedding)
# test_out = net('NN', test_inp, [128,128,128,10])

loss = tf.reduce_mean(logits1) - tf.reduce_mean(logits0)
# loss = tf.reduce_mean(tf.exp(logits0-1.)) - tf.reduce_mean(logits1)
# loss = tf.reduce_mean(tf.exp(logits1-1.)) - tf.reduce_mean(logits0)

alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
_x = (1 - alpha) * embedding[:,:embedding_dim] + alpha * embedding[:,embedding_dim:]

gradients = tf.gradients(net('NN', _x, [128,128,128,10]), [_x])[0]
# slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients)/tf.reshape(tf.norm(gradients, axis=1), (-1,1)), reduction_indices=[1]))
# slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]) + 1.0e-12)
slopes = tf.norm(gradients, axis=1)
print(gradients.shape)
# print(slopes.shape)
gradient_penalty = tf.reduce_mean((slopes-1.) ** 2)
# gradient_penalty = tf.reduce_mean(tf.maximum(slopes-1., 0.) ** 2)

with tf.name_scope('optimizer'):
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # trainable_variables = tf.trainable_variables() # 获取到模型中所有需要训练的变量
  # grads = tf.gradients(loss, trainable_variables)
  # grads = [tf.clip_by_value(g, -MAX_GRAD_NORM, MAX_GRAD_NORM) for g in grads]
  # train_op = optimizer.apply_gradients(
  #             zip(grads, trainable_variables), global_step=global_step
  #         )

  grads_and_vars = optimizer.compute_gradients(loss + LAMBDA * gradient_penalty)
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  

with tf.Session() as sess:
  sess.run([
      tf.tables_initializer(),
      tf.local_variables_initializer(),
      tf.global_variables_initializer(),
    ])

  # print sess.run(embedding_x)
  # print sess.run([embedding[:,:3], embedding[:,3:]])


  # print sess.run([logits0, logits1, loss])
  # print sess.run(tf.random_uniform((batch_size,1)))

  for _ in range(epoch):
    _, _step, _loss, _data, _logits = sess.run([train_op, global_step, loss, train_data, logits1])

    # if _step % 1:
    if _step % 1 == 0:
      print _step, _loss, {ix.tolist()[0]:v.tolist()[0] for ix, v in zip(_data[1], _logits)}
    else:
      print _step, _loss

  # print sess.run(test, feed_dict={test:[0,1,2]})
  # print sess.run(test, feed_dict={test:np.array([[0],[1],[2]])})



  # print sess.run(it.get_next()[0])
    
