#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse

import numpy as np
import pandas as pd
from tensorflow.python.ops.clip_ops import clip_by_value

from tqdm import tqdm

import time

from json import *
import json

import tensorflow as tf

from tensorflow.keras import backend as K, layers, utils, Model, Input, optimizers as opt
from tensorflow.keras.activations import tanh, softmax
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Lambda

from tensorflow.keras.constraints import max_norm,MinMaxNorm

import random as rnd    

import scipy.stats

print tf.__version__
# zhihu: https://zhuanlan.zhihu.com/p/245566551

batch = 200
epoch = 1000
steps_per_epoch = 1000
# act = 'selu'
act = 'elu'
# act = 'relu'

max_value = 0.03

dim_size = 8
lr = 0.01

hidden_size = 32
cate_size = 20
KL = True

epsilon = 0.5

# opt = tf.train.GradientDescentOptimizer(lr)
# opt = tf.train.RMSPropOptimizer(lr)
opt = tf.train.AdagradOptimizer(lr)
# opt = tf.keras.optimizers.SGD(lr, clipnorm=0.01)
# opt = tf.keras.optimizers.SGD(lr)

print 'KL divergence' if KL else 'reverse KL divergence'


# x = np.array(range(8))
# prob0 = np.array([0.02, 0.2, 0.2, 0.05, 0.03, 0.4, 0.06, 0.04])
# prob1 = np.array([0.2, 0.2, 0.2, 0.15, 0.13, 0.02, 0.04, 0.06])


x = np.array(range(3))
prob0 = np.array([0.5, 0.3, 0.2])
prob1 = np.array([0.3, 0.25, 0.45])


print scipy.stats.entropy(prob0, prob1)
print scipy.stats.entropy(prob1, prob0)

class LayerLoss(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerLoss, self).__init__(**kwargs)

    def call(self, inputs):
      logits0, logits1 = inputs
      # return tf.reduce_mean(logits1) - tf.reduce_mean(logits0)
      return tf.reduce_mean(tf.exp(logits1-1.)) - tf.reduce_mean(logits0)
      # return  tf.reduce_mean(logits0) - tf.reduce_mean(tf.exp(logits1-1))


class LayerReg1(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerReg1, self).__init__(**kwargs)

    def call(self, inputs):
      inp0, inp1 = inputs
      x = epsilon * inp0 + (1-epsilon) * inp1

      return x
      x = layer_4(layer_3(layer_2(layer_1(layer_0(x)))))#, perm=[1,0])


class LayerReg2(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerReg2, self).__init__(**kwargs)

    def call(self, inputs):
      x = inputs
      return 0.0
      # return tf.reduce_mean(tf.square(x-tf.reshape(x, (1, -1)))-1)


class LayerMixLoss(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerMixLoss, self).__init__(**kwargs)

    def call(self, inputs):
      x, y = inputs
      return x+y

def kl(dist1, dist2):
  _1, _2 = np.unique(dist1, return_counts=True)
  d1 = dict(zip(_1, _2))

  _1, _2 = np.unique(dist2, return_counts=True)
  d2 = dict(zip(_1, _2))

  d = {}
  for k in set(d1.keys() + d2.keys()):
    d[k] = [d1[k] * 1. / batch if k in d1 else 0.0, d2[k] * 1. / batch if k in d2 else 0.0]

  prob0, prob1 = [v[0] for v in d.values()], [v[1] for v in d.values()]

  return np.sum(np.array(prob0) * np.log(np.array(prob0) / np.array(prob1))), np.sum(np.array(prob1) * np.log(np.array(prob1) / np.array(prob0)))


def create_model():
  # Layers ...
  inp0 = Input(shape=(1,), dtype=tf.int32, name='input0')
  inp1 = Input(shape=(1,), dtype=tf.int32, name='input1')

  layer_embedding0 = layers.Embedding(cate_size, dim_size, name='embedding0', trainable=True)

  layer_0 = layers.Dense(hidden_size, name='layer_0', activation=act)
  layer_1 = layers.Dense(hidden_size, name='layer_1', activation=act)
  layer_2 = layers.Dense(hidden_size, name='layer_2', activation=act)
  layer_3 = layers.Dense(hidden_size, name='layer_3')
  layer_4 = layers.Dense(1, name='layer_4')

  layerloss = LayerLoss()

  # Graph ...
  feat0 = layer_embedding0(inp0)
  feat1 = layer_embedding0(inp1)

  logits0 = layer_4(layer_3(layer_2(layer_1(layer_0(feat0)))))
  logits1 = layer_4(layer_3(layer_2(layer_1(layer_0(feat1)))))

  loss = layerloss([logits0, logits1]) if KL else layerloss([logits1, logits0])

  # grad = K.gradients(logits0, feat0)[0]
  # print (grad.shape)

  # reg - gp

  # regloss1, regloss2 = LayerReg1(), LayerReg2()

  # x = regloss1([feat0, feat1])
  # d = layer_4(layer_3(layer_2(layer_1(layer_0(x)))))

  # regloss = regloss2([d])

  # mixloss = LayerMixLoss()
  # loss = mixloss([loss, regloss])
  
  return Model([inp0, inp1], [loss])

def loss(_, v):
  return v

def div(_, v):
  return -v

def loss2(_, out1):
  logits0, logits1 = out1[0], out1[1]
  return tf.reduce_mean(tf.exp(logits1-1)) - tf.reduce_mean(logits0)

def gen_train_data(x):
    while 1:
      yield [np.random.choice(a=x, size=batch, replace=True, p=prob0), 
        np.random.choice(a=x, size=batch, replace=True, p=prob1)], [None]

model = create_model()
model.compile(optimizer=opt, loss=loss, metrics=[div]) 

# print model.summary()

train_data = gen_train_data(x)
test_data = gen_train_data(x)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./model', # log 目录
  histogram_freq=1, # 按照何等频率（epoch）来计算直方图，0为不计算
  batch_size=32, # 用多大量的数据计算直方图
  # write_graph=True, # 是否存储网络结构图
  write_grads=True, # 是否可视化梯度直方图
  write_images=True, # 是否可视化参数
  # embeddings_freq=0,
  # embeddings_layer_names=None,
  # embeddings_metadata=None
  )

model.fit_generator(train_data, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=epoch,
                    verbose=1,
                    callbacks=[tb_callback], 
                    validation_data=test_data, 
                    validation_steps=100
)


# print model.predict([np.array([[0],[1],[2]]), np.array([[0],[1],[2]]), ])

