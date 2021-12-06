#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging

logging.basicConfig(
  level=logging.INFO,
  format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',# format='%(asctime)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S')
# logging.getLogger("tensorflow").setLevel(logging.INFO)

import os, sys

import numpy as np
from sklearn import metrics
import sklearn

import json
from json import *

import tensorflow as tf

from tensorflow import feature_column as fc

tf.set_random_seed(2019)

def setup_logger(log_name, log_level, out_file=None):
  logger = logging.getLogger(log_name)
  # logger.handlers = []
  logger.setLevel(log_level)

  # fh = logging.FileHandler(out_file)
  fh = logging.StreamHandler(out_file)
  fh.setLevel(log_level)

  # fmt = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)s %(funcName)s] %(message)s'
  fmt = '[%(levelname)s] [%(asctime)s %(filename)s:%(lineno)s] %(message)s'
  datefmt='%Y-%m-%d %H:%M:%S'
  fh.setFormatter(logging.Formatter(fmt, datefmt))
  # logger.addHandler(fh)
  return logger

# logger = setup_logger('tensorflow', logging.INFO, sys.stderr)

label_columns = ['assignd', 'canceld', 'timeoutd', 'assignTime', 'cancelTime', 'timeoutTime']
# label_columns = ['canceld', 'assignd', 'timeoutd']

# int64_columns = ['null_avg_submit_num_10min','null_avg_submit_num_5min','null_avg_submit_num_15min','null_avg_submit_num_30min','null_avg_schedule_time_5min','null_avg_schedule_time_10min','null_avg_schedule_time_15min','null_avg_schedule_time_30min','null_avg_result_time_5min','null_avg_result_time_10min','null_avg_result_time_15min','null_avg_result_time_30min','null_avg_schedule_num_5min','null_avg_schedule_num_10min','null_avg_schedule_num_15min','null_avg_schedule_num_30min','null_avg_cancel_num_5min','null_avg_cancel_num_10min','null_avg_cancel_num_15min','null_avg_cancel_num_30min','null_avg_disp_timeout_cancel_num_5min','null_avg_disp_timeout_cancel_num_10min','null_avg_disp_timeout_cancel_num_15min','null_avg_disp_timeout_cancel_num_30min','null_avg_submit_num_5min_pre1','null_avg_submit_num_10min_pre1','null_avg_submit_num_15min_pre1','null_avg_submit_num_30min_pre1','null_avg_schedule_time_5min_pre1','null_avg_schedule_time_10min_pre1','null_avg_schedule_time_15min_pre1','null_avg_schedule_time_30min_pre1','null_avg_result_time_5min_pre1','null_avg_result_time_10min_pre1','null_avg_result_time_15min_pre1','null_avg_result_time_30min_pre1','null_avg_schedule_num_5min_pre1','null_avg_schedule_num_10min_pre1','null_avg_schedule_num_15min_pre1','null_avg_schedule_num_30min_pre1','null_avg_cancel_num_5min_pre1','null_avg_cancel_num_10min_pre1','null_avg_cancel_num_15min_pre1','null_avg_cancel_num_30min_pre1','null_avg_disp_timeout_cancel_num_5min_pre1','null_avg_disp_timeout_cancel_num_10min_pre1','null_avg_disp_timeout_cancel_num_15min_pre1','null_avg_disp_timeout_cancel_num_30min_pre1','null_avg_submit_num_5min_day_ago_next','null_avg_submit_num_10min_day_ago_next','null_avg_submit_num_15min_day_ago_next','null_avg_submit_num_30min_day_ago_next','null_avg_schedule_time_5min_day_ago_next','null_avg_schedule_time_10min_day_ago_next','null_avg_schedule_time_15min_day_ago_next','null_avg_schedule_time_30min_day_ago_next','null_avg_result_time_5min_day_ago_next','null_avg_result_time_10min_day_ago_next','null_avg_result_time_15min_day_ago_next','null_avg_result_time_30min_day_ago_next','null_avg_schedule_num_5min_day_ago_next','null_avg_schedule_num_10min_day_ago_next','null_avg_schedule_num_15min_day_ago_next','null_avg_schedule_num_30min_day_ago_next','null_avg_cancel_num_5min_day_ago_next','null_avg_cancel_num_10min_day_ago_next','null_avg_cancel_num_15min_day_ago_next','null_avg_cancel_num_30min_day_ago_next','null_avg_disp_timeout_cancel_num_5min_day_ago_next','null_avg_disp_timeout_cancel_num_10min_day_ago_next','null_avg_disp_timeout_cancel_num_15min_day_ago_next','null_avg_disp_timeout_cancel_num_30min_day_ago_next','null_avg_distance','null_min_distance','null_nearByCarNum3000','null_avg_distance_3000','null_online_avgCancelNumber5min','null_online_avgCancelNumber10min','null_online_avgCancelNumber15min','null_online_avgCancelNumber30min','null_online_avgScheduleNumber5min','null_online_avgScheduleNumber10min','null_online_avgScheduleNumber15min','null_online_avgScheduleNumber30min','null_online_avgScheduleTime5min','null_online_avgScheduleTime10min','null_online_avgScheduleTime15min','null_online_avgScheduleTime30min','null_online_avgSubmitNumber5min','null_online_avgSubmitNumber10min','null_online_avgSubmitNumber15min','null_online_avgSubmitNumber30min','null_online_avgResponseRate5min','null_online_avgResponseRate10min','null_online_avgResponseRate15min','null_online_avgResponseRate30min','null_online_avgSubmitNumber5minType','null_online_avgSubmitNumber10minType','null_online_avgSubmitNumber15minType','null_online_avgSubmitNumber30minType','null_online_avgScheduleTime5minType','null_online_avgScheduleTime10minType','null_online_avgScheduleTime15minType','null_online_avgScheduleTime30minType','null_online_avgSubmitNumberEnd5min','null_online_avgSubmitNumberEnd10min','null_online_avgSubmitNumberEnd15min','null_online_avgSubmitNumberEnd30min','null_online_avgResponseRateType5min','null_online_avgResponseRateType10min','null_online_avgResponseRateType15min','null_online_avgResponseRateType30min']
# float32_columns = ['avg_submit_num_5min','avg_submit_num_10min','avg_submit_num_15min','avg_submit_num_30min','avg_schedule_time_5min','avg_schedule_time_10min','avg_schedule_time_15min','avg_schedule_time_30min','avg_result_time_5min','avg_result_time_10min','avg_result_time_15min','avg_result_time_30min','avg_schedule_num_5min','avg_schedule_num_10min','avg_schedule_num_15min','avg_schedule_num_30min','avg_cancel_num_5min','avg_cancel_num_10min','avg_cancel_num_15min','avg_cancel_num_30min','avg_disp_timeout_cancel_num_5min','avg_disp_timeout_cancel_num_10min','avg_disp_timeout_cancel_num_15min','avg_disp_timeout_cancel_num_30min','avg_submit_num_5min_pre1','avg_submit_num_10min_pre1','avg_submit_num_15min_pre1','avg_submit_num_30min_pre1','avg_schedule_time_5min_pre1','avg_schedule_time_10min_pre1','avg_schedule_time_15min_pre1','avg_schedule_time_30min_pre1','avg_result_time_5min_pre1','avg_result_time_10min_pre1','avg_result_time_15min_pre1','avg_result_time_30min_pre1','avg_schedule_num_5min_pre1','avg_schedule_num_10min_pre1','avg_schedule_num_15min_pre1','avg_schedule_num_30min_pre1','avg_cancel_num_5min_pre1','avg_cancel_num_10min_pre1','avg_cancel_num_15min_pre1','avg_cancel_num_30min_pre1','avg_disp_timeout_cancel_num_5min_pre1','avg_disp_timeout_cancel_num_10min_pre1','avg_disp_timeout_cancel_num_15min_pre1','avg_disp_timeout_cancel_num_30min_pre1','avg_submit_num_5min_day_ago_next','avg_submit_num_10min_day_ago_next','avg_submit_num_15min_day_ago_next','avg_submit_num_30min_day_ago_next','avg_schedule_time_5min_day_ago_next','avg_schedule_time_10min_day_ago_next','avg_schedule_time_15min_day_ago_next','avg_schedule_time_30min_day_ago_next','avg_result_time_5min_day_ago_next','avg_result_time_10min_day_ago_next','avg_result_time_15min_day_ago_next','avg_result_time_30min_day_ago_next','avg_schedule_num_5min_day_ago_next','avg_schedule_num_10min_day_ago_next','avg_schedule_num_15min_day_ago_next','avg_schedule_num_30min_day_ago_next','avg_cancel_num_5min_day_ago_next','avg_cancel_num_10min_day_ago_next','avg_cancel_num_15min_day_ago_next','avg_cancel_num_30min_day_ago_next','avg_disp_timeout_cancel_num_5min_day_ago_next','avg_disp_timeout_cancel_num_10min_day_ago_next','avg_disp_timeout_cancel_num_15min_day_ago_next','avg_disp_timeout_cancel_num_30min_day_ago_next','avg_distance','min_distance','nearByCarNum3000','avg_distance_3000','online_avgCancelNumber5min','online_avgCancelNumber10min','online_avgCancelNumber15min','online_avgCancelNumber30min','online_avgScheduleNumber5min','online_avgScheduleNumber10min','online_avgScheduleNumber15min','online_avgScheduleNumber30min','online_avgScheduleTime5min','online_avgScheduleTime10min','online_avgScheduleTime15min','online_avgScheduleTime30min','online_avgSubmitNumber5min','online_avgSubmitNumber10min','online_avgSubmitNumber15min','online_avgSubmitNumber30min','online_avgResponseRate5min','online_avgResponseRate10min','online_avgResponseRate15min','online_avgResponseRate30min','online_avgSubmitNumber5minType','online_avgSubmitNumber10minType','online_avgSubmitNumber15minType','online_avgSubmitNumber30minType','online_avgScheduleTime5minType','online_avgScheduleTime10minType','online_avgScheduleTime15minType','online_avgScheduleTime30minType','online_avgSubmitNumberEnd5min','online_avgSubmitNumberEnd10min','online_avgSubmitNumberEnd15min','online_avgSubmitNumberEnd30min','online_avgResponseRateType5min','online_avgResponseRateType10min','online_avgResponseRateType15min','online_avgResponseRateType30min']

int64_columns = [
'null_online_avgCancelNumber5min',
'null_online_avgCancelNumber10min',
'null_online_avgCancelNumber15min',
'null_online_avgCancelNumber30min',
'null_online_avgScheduleNumber5min',
'null_online_avgScheduleNumber10min',
'null_online_avgScheduleNumber15min',
'null_online_avgScheduleNumber30min',
'null_online_avgScheduleTime5min',
'null_online_avgScheduleTime10min',
'null_online_avgScheduleTime15min',
'null_online_avgScheduleTime30min',
'null_online_avgSubmitNumber5min',
'null_online_avgSubmitNumber10min',
'null_online_avgSubmitNumber15min',
'null_online_avgSubmitNumber30min',
'null_online_avgSubmitNumberEnd5min',
'null_online_avgSubmitNumberEnd10min',
'null_online_avgSubmitNumberEnd15min',
'null_online_avgSubmitNumberEnd30min',
]

float32_columns = [
'online_avgCancelNumber5min',
'online_avgCancelNumber10min',
'online_avgCancelNumber15min',
'online_avgCancelNumber30min',
'online_avgScheduleNumber5min',
'online_avgScheduleNumber10min',
'online_avgScheduleNumber15min',
'online_avgScheduleNumber30min',
'online_avgScheduleTime5min',
'online_avgScheduleTime10min',
'online_avgScheduleTime15min',
'online_avgScheduleTime30min',
'online_avgSubmitNumber5min',
'online_avgSubmitNumber10min',
'online_avgSubmitNumber15min',
'online_avgSubmitNumber30min',
'online_avgSubmitNumberEnd5min',
'online_avgSubmitNumberEnd10min',
'online_avgSubmitNumberEnd15min',
'online_avgSubmitNumberEnd30min',
]

user_null_columns = ['null_ord_num', 'null_ord_cancel_num', 'null_min_cancel_xt', 'null_avg_cancel_xt', 'null_max_cancel_xt', 'null_ord_assign_num', 'null_min_assign_xt', 'null_avg_assign_xt', 'null_max_assign_xt']
user_float32_columns = ['ord_num', 'ord_cancel_num', 'min_cancel_xt', 'avg_cancel_xt', 'max_cancel_xt', 'ord_assign_num', 'min_assign_xt', 'avg_assign_xt', 'max_assign_xt']

other_columns = {
  'dt': tf.FixedLenFeature([], tf.string),
  'user_id' : tf.FixedLenFeature([], tf.string),
  'oid' : tf.FixedLenFeature([], tf.string),
  'sHexID' : tf.FixedLenFeature([], tf.string),
  'eHexID' : tf.FixedLenFeature([], tf.string),
  'timeSlice' : tf.FixedLenFeature([], tf.string),  
  'dayofweek': tf.FixedLenFeature([], tf.string),
  "reserve_partner_car_type_id": tf.VarLenFeature(tf.string),
  'dist' : tf.FixedLenFeature([], tf.float32),
}

def parse_tfrecords_function(example_proto):
  features_conf = [
    {'Xt': tf.FixedLenFeature([1], tf.int64, default_value=0)},
    {c: tf.FixedLenFeature([], tf.int64, default_value=0) for c in label_columns},
    {c: tf.FixedLenFeature([], tf.int64, default_value=0) for c in int64_columns},
    {c: tf.FixedLenFeature([], tf.float32, default_value=0) for c in float32_columns},
    {c: tf.FixedLenFeature([], tf.int64, default_value=1) for c in user_null_columns},
    {c: tf.FixedLenFeature([], tf.float32, default_value=0) for c in user_float32_columns},
    other_columns
  ]

  features = {k:v for dic in features_conf for k, v in dic.items()}

  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features, [parsed_features[c] for c in ['assignd', 'canceld', 'timeoutd']], tf.cast(parsed_features['Xt'], tf.float32)

def create_placeholder():
  features_conf = [
    {c: tf.placeholder(tf.int64, [1, 1], name=c) for c in int64_columns},
    {c: tf.placeholder(tf.float32, [1, 1], name=c) for c in float32_columns},
    {
      'user_id' :   tf.placeholder(tf.string, [1, 1], name='user_id'),
      'sHexID' :    tf.placeholder(tf.string, [1, 1], name='sHexID'),
      'eHexID' :    tf.placeholder(tf.string, [1, 1], name='eHexID'),
      'timeSlice' : tf.placeholder(tf.string, [1, 1], name='timeSlice'),
      'dayofweek':  tf.placeholder(tf.string, [1, 1], name='dayofweek'),
      "reserve_partner_car_type_id": tf.placeholder(tf.string, [1, None], name='reserve_partner_car_type_id'),
      'dist' : tf.placeholder(tf.float32, [1, 1], name='dist'),
    }
  ]

  return {k:v for dic in features_conf for k, v in dic.items()}

def normalizer_fn(v, min_v, max_v):
  return (v - min_v) * 1. / (max_v - min_v)


def make_columns_with_normalizer():
  with open('summary.json') as fp:
    import pandas as pd
    summary = pd.DataFrame(json.load(fp)).T

  # categorical_column_with_vocabulary_list
  with open('user_id') as fp:
    user_id = fc.embedding_column(fc.categorical_column_with_vocabulary_list(
      'user_id', fp.read().splitlines(), dtype=tf.string, num_oov_buckets=4), dimension=6)

  with open('partner_car_type_id') as fp:
    partner_ids = fc.categorical_column_with_vocabulary_list(
      'reserve_partner_car_type_id', fp.read().splitlines(), dtype=tf.string, num_oov_buckets=1)
    partner_ids_embedding = fc.embedding_column(partner_ids, dimension=3)
    # partner_ids_embedding = fc.indicator_column(partner_ids)

  dayofweek = fc.embedding_column(fc.categorical_column_with_vocabulary_list(
    'dayofweek', [str(d) for d in range(0, 8)], dtype=tf.string, num_oov_buckets=1), dimension=3)

  with open('timeSlice') as fp:
    timeSlice = fc.embedding_column(fc.categorical_column_with_vocabulary_list(
      'timeSlice', fp.read().splitlines(), dtype=tf.string, num_oov_buckets=1), dimension=3)
  with open('sHexID') as fp:
    sHexID = fc.embedding_column(fc.categorical_column_with_vocabulary_list(
      'sHexID', fp.read().splitlines(), num_oov_buckets=1), dimension=6)
  with open('eHexID') as fp:
    eHexID = fc.embedding_column(fc.categorical_column_with_vocabulary_list(
      'eHexID', fp.read().splitlines(), num_oov_buckets=1), dimension=6)
  order_columns = [
    fc.numeric_column('dist', normalizer_fn=lambda v: normalizer_fn(v, float(summary['dist']['min']), float(summary['dist']['max'])))
    ]

  user_columns = [fc.numeric_column(c) for c in user_null_columns + user_float32_columns]
  # TODO: summary info

  spacetime_columns = [fc.numeric_column(c) for c in int64_columns]
  spacetime_columns += [fc.numeric_column(c, normalizer_fn=lambda v: normalizer_fn(v, float(summary[c]['min']), float(summary[c]['max']))) for c in float32_columns]

  embedding_columns = [user_id, partner_ids_embedding, dayofweek, timeSlice, sHexID]
  # embedding_columns = [dayofweek]
  # return embedding_columns + order_columns, embedding_columns + order_columns + spacetime_columns
  return embedding_columns, order_columns, spacetime_columns, user_columns

def make_columns_with_normalizer_with_file():
  with open('summary.json') as fp:
    import pandas as pd
    summary = pd.DataFrame(json.load(fp)).T

  # categorical_column_with_vocabulary_list
  user_id = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'user_id', vocabulary_file='user_id', dtype=tf.string, num_oov_buckets=4), dimension=6)

  partner_ids = fc.categorical_column_with_vocabulary_file(
    'reserve_partner_car_type_id', vocabulary_file='partner_car_type_id', dtype=tf.string, num_oov_buckets=1)
  partner_ids_embedding = fc.embedding_column(partner_ids, dimension=3)
  # partner_ids_embedding = fc.indicator_column(partner_ids)

  dayofweek = fc.embedding_column(fc.categorical_column_with_vocabulary_list(
    'dayofweek', [str(d) for d in range(0, 8)], dtype=tf.string, num_oov_buckets=1), dimension=3)
  timeSlice = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'timeSlice', vocabulary_file='timeSlice', dtype=tf.string, num_oov_buckets=1), dimension=3)

  sHexID = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'sHexID', vocabulary_file='sHexID', num_oov_buckets=1), dimension=6)
  eHexID = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'eHexID', vocabulary_file='eHexID', num_oov_buckets=1), dimension=6)
  order_columns = [
    fc.numeric_column('dist', normalizer_fn=lambda v: normalizer_fn(v, float(summary['dist']['min']), float(summary['dist']['max'])))
    ]

  user_columns = [fc.numeric_column(c) for c in user_null_columns + user_float32_columns]
  # TODO: summary info

  spacetime_columns = [fc.numeric_column(c) for c in int64_columns]
  spacetime_columns += [fc.numeric_column(c, normalizer_fn=lambda v: normalizer_fn(v, float(summary[c]['min']), float(summary[c]['max']))) for c in float32_columns]

  embedding_columns = [user_id, partner_ids_embedding, dayofweek, timeSlice, sHexID]
  # embedding_columns = [dayofweek]
  # return embedding_columns + order_columns, embedding_columns + order_columns + spacetime_columns
  return embedding_columns, order_columns, spacetime_columns, user_columns

def make_columns():
  user_id = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'user_id', vocabulary_file='user_id', dtype=tf.string, num_oov_buckets=4), dimension=6)

  partner_ids = fc.categorical_column_with_vocabulary_file(
    'reserve_partner_car_type_id', vocabulary_file='partner_car_type_id', dtype=tf.string, num_oov_buckets=1)
  partner_ids_embedding = fc.embedding_column(partner_ids, dimension=3)
  # partner_ids_embedding = fc.indicator_column(partner_ids)

  dayofweek = fc.embedding_column(fc.categorical_column_with_vocabulary_list(
    'dayofweek', [str(d) for d in range(0, 8)], dtype=tf.string, num_oov_buckets=1), dimension=3)
  timeSlice = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'timeSlice', vocabulary_file='timeSlice', dtype=tf.string, num_oov_buckets=1), dimension=3)

  sHexID = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'sHexID', vocabulary_file='sHexID', num_oov_buckets=1), dimension=6)
  eHexID = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    'eHexID', vocabulary_file='eHexID', num_oov_buckets=1), dimension=6)
  order_columns = [
    fc.numeric_column('dist')
    ]

  user_columns = [fc.numeric_column(c) for c in user_null_columns + user_float32_columns]
  spacetime_columns = [fc.numeric_column(c) for c in int64_columns + float32_columns]

  embedding_columns = [user_id, partner_ids_embedding, dayofweek, timeSlice, sHexID]
  # embedding_columns = [dayofweek]
  # return embedding_columns + order_columns, embedding_columns + order_columns + spacetime_columns
  return embedding_columns, order_columns, spacetime_columns, user_columns

def net(name, _input, layer_units, out_units=1):
  # with tf.name_scope(name): # with tf.variable_scope(name):
  mu, stddev = 0.0, 0.01
  # act = tf.nn.relu
  # act = tf.nn.sigmoid
  # act = None

  out = _input
  for ix, units in enumerate(layer_units):
    out = tf.layers.dense(out, units=units, activation=tf.nn.relu, name="{0}_L{1}".format(name, ix),
      kernel_initializer=tf.truncated_normal_initializer(mu, stddev)
      # kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-12)
      )
  
  out = tf.layers.dense(out, units=out_units, name="{0}".format(name),
      kernel_initializer=tf.truncated_normal_initializer(mu, stddev)
      # kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-12)
      )
  return out

def loglossOp(labels, orig_predictions):
  # auc = sklearn.metrics.roc_auc_score(y_true=labels, y_score=orig_predictions)
  # mae = sklearn.metrics.mean_absolute_error(y_true=labels, y_pred=orig_predictions)
  logloss = sklearn.metrics.log_loss(y_true=labels, y_pred=orig_predictions)

  # return np.array([auc, logloss, mae], dtype=np.float32)
  return np.array(logloss, dtype=np.float32)

def aucOp(labels, orig_predictions):
  auc = sklearn.metrics.roc_auc_score(y_true=labels, y_score=orig_predictions)
  # mae = sklearn.metrics.mean_absolute_error(y_true=labels, y_pred=orig_predictions)
  # logloss = sklearn.metrics.log_loss(y_true=labels, y_pred=orig_predictions)

  # return np.array([auc, logloss, mae], dtype=np.float32)
  return np.array(auc, dtype=np.float32)

class Model():
  def __init__(self, d_k=5, c_k=5, layer_units=[64, 64, 64]):
    self.d_k, self.c_k = d_k, c_k
    self.layer_units = layer_units
  
  def restore(self, log_dir, is_train=False):
    checkpoint_path = log_dir
    checkpoint_file_path = checkpoint_path + "/checkpoint.ckpt"
    latest_checkpoint_file_path = tf.train.latest_checkpoint(checkpoint_path)
    
    sess = tf.Session()
    sess.run([
      tf.tables_initializer(),
      tf.local_variables_initializer(),
      tf.global_variables_initializer(),
    ])
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.1)

    if latest_checkpoint_file_path:
      logging.info(
        "Restore session from checkpoint: {}".format(checkpoint_file_path))
      saver.restore(sess, latest_checkpoint_file_path)
    else:
      logging.error("Checkpoint not found: {}".format(checkpoint_file_path))
      if not is_train:
        sys.exit(0)

    return sess

  def inference(self, feats):
    # embedding_columns, order_columns, spacetime_columns, user_columns = make_columns()
    embedding_columns, order_columns, spacetime_columns, user_columns = make_columns_with_normalizer()

    # TODO: feats 按columns拆分
    with tf.name_scope('embedding_columns'):
      embedding_tensor = fc.input_layer(feats, embedding_columns)
    with tf.name_scope('order_columns'):
      order_tensor = fc.input_layer(feats, order_columns)
    with tf.name_scope('spacetime_columns'):
      spacetime_tensor = fc.input_layer(feats, spacetime_columns)

    input_tensor = tf.concat([embedding_tensor, order_tensor, spacetime_tensor], axis=1, name='input_concat')

    eta_d = tf.concat([net('eta_d_%d'%ix, input_tensor, self.layer_units) for ix in range(self.d_k)], axis=1)
    eta_c = tf.concat([net('eta_c_%d'%ix, input_tensor, self.layer_units) for ix in range(self.c_k)], axis=1)

    d_softmax_logits = net('logits_d', input_tensor, self.layer_units, self.d_k)
    c_softmax_logits = net('logits_c', input_tensor, self.layer_units, self.c_k)

    return eta_d, eta_c, d_softmax_logits, c_softmax_logits

  def train(self, train_data, test_data, log_dir, epochs=20):
    with tf.name_scope('train_input'):
      data = tf.data.TFRecordDataset(train_data)
      dataset = data.map(parse_tfrecords_function).prefetch(300000).batch(1024)
      it = dataset.make_initializable_iterator()

    with tf.name_scope('eval_input'):
      test_data = tf.data.TFRecordDataset(test_data)
      eval_dataset = test_data.map(parse_tfrecords_function).prefetch(300000).batch(1024*8) # .repeat(100)
      eval_it = eval_dataset.make_initializable_iterator()

    with tf.name_scope('input'):
      train_input = lambda: it.get_next()
      eval_input = lambda: eval_it.get_next()
      trainSwitch = tf.placeholder(tf.int32)
      feats, labels, Xts = tf.cond(tf.equal(trainSwitch, tf.constant(1)), train_input, eval_input, name='input_cond')

    eta_d, eta_c, d_softmax_logits, c_softmax_logits = self.inference(feats)
    lambda_d = tf.math.exp(eta_d, name='exp_d')
    lambda_c = tf.math.exp(eta_c, name='exp_c')

    with tf.name_scope('out'):
      out0 = -tf.reduce_logsumexp(eta_d - lambda_d * Xts + tf.nn.log_softmax(d_softmax_logits), axis=1) - tf.reduce_logsumexp(-lambda_c * Xts + tf.nn.log_softmax(c_softmax_logits), axis=1) # TODO: + tf.nn.log_softmax() 
      out1 = -tf.reduce_logsumexp(eta_c - lambda_c * Xts + tf.nn.log_softmax(c_softmax_logits), axis=1) - tf.reduce_logsumexp(-lambda_d * Xts + tf.nn.log_softmax(d_softmax_logits), axis=1) # TODO: + tf.nn.log_softmax() 
      out2 = -tf.reduce_logsumexp(-lambda_d * Xts + tf.nn.log_softmax(d_softmax_logits), axis=1) - tf.reduce_logsumexp(-lambda_c * Xts + tf.nn.log_softmax(c_softmax_logits), axis=1) # TODO: + tf.nn.log_softmax() 

      out0, out1, out2 = tf.expand_dims(out0, 1), tf.expand_dims(out1, 1), tf.expand_dims(out2, 1)
      preds = tf.concat([out0, out1, out2], axis=1, name='preds')

    with tf.name_scope('loss'):
      out = tf.reduce_sum(preds * tf.cast(labels, tf.float32), 1, name='out')
      loss = tf.reduce_mean(out, name='loss')

    with tf.name_scope('optimizer'):
      lr = 1e-2
      # optimizer = tf.train.GradientDescentOptimizer(lr)
      # optimizer = tf.train.AdadeltaOptimizer(lr)
      # optimizer = tf.train.AdagradOptimizer(lr)
      # optimizer = tf.train.AdaMaxOptimizer(lr)
      optimizer = tf.train.AdamOptimizer(1e-3)
      # optimizer = tf.train.RMSPropOptimizer(lr)
      # optimizer = tf.train.FtrlOptimizer(lr)

      global_step = tf.Variable(0, name="global_step", trainable=False)

      grads_and_vars = optimizer.compute_gradients(loss)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

      # train_op = optimizer.minimize(loss)

    # with tf.name_scope('eval'):
    #   eval_auc = tf.py_func(aucOp, inp=[labels[:,1], probs], Tout=tf.float32, name='auc') # labels[:,1] 第二列
    #   eval_fixed_auc = tf.py_func(aucOp, inp=[labels[:,1], probs_with_notimeout], Tout=tf.float32, name='fixed_auc') # labels[:,1] 第二列
    #   eval_logloss = tf.py_func(loglossOp, inp=[labels[:,1], probs], Tout=tf.float32, name='logloss')

    with tf.name_scope('summary'):
      # Initialize saver and summary
      saver = tf.train.Saver()

      train_summary_op = tf.summary.merge([
        tf.summary.scalar("train_obj_loss", loss)
        # , tf.summary.scalar("d_softmax_logits", d_softmax_logits)
        # , tf.summary.scalar("c_softmax_logits", c_softmax_logits)
        # , tf.summary.scalar("train_auc", eval_auc)
        # , tf.summary.scalar("train_fixed_auc", eval_fixed_auc)
        # , tf.summary.scalar("train_logloss", eval_logloss)
        
        # , tf.summary.histogram("outd", out0)
        # , tf.summary.histogram("outc", out1)
        # , tf.summary.histogram("outtt", out2)
        
        # , tf.summary.histogram("pred_c_time", pred_c_time)
        # , tf.summary.histogram("pred_d_time", pred_d_time)
        
      ])

      eval_summary_op = tf.summary.merge([
        tf.summary.scalar("eval_obj_loss", loss)
        # , tf.summary.scalar("eval_auc", eval_auc)
        # , tf.summary.scalar("eval_fixed_auc", eval_fixed_auc)
        # , tf.summary.scalar("eval_logloss", eval_logloss)
      ])

      with self.restore(log_dir, is_train=True) as sess:
        def evaluate():
          try:
            _summary, _loss = sess.run([eval_summary_op, loss], feed_dict={trainSwitch: 0})
            # _summary = sess.run([eval_summary_op], feed_dict={trainSwitch: 0})

            writer.add_summary(_summary, _step)
            logging.info("eval {0} epoch, loss:{1}".format(epoch, _loss))

          except tf.errors.OutOfRangeError:
            logging.info('eval {0} complete.'.format(epoch))
            sess.run([eval_it.initializer])
            
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run([it.initializer, eval_it.initializer])
        
        for epoch in range(epochs):
          while 1:
            try:
              _, _summary, _step, _loss = sess.run([train_op, train_summary_op, global_step, loss], feed_dict={trainSwitch: 1})
              # _summary, _step, _loss, _grads_and_vars = sess.run([train_summary_op, global_step, loss, grads_and_vars], feed_dict={trainSwitch: 1})
              logging.info("step:[{0}] loss:{1}".format(_step, _loss))
              
              if _step % 100 == 0:
                writer.add_summary(_summary, _step)

              if _step % 100 == 0:
                saver.save(sess, log_dir + "/checkpoint.ckpt", global_step=_step)
                evaluate()

            except tf.errors.OutOfRangeError:
              logging.info('epoch {0} complete.'.format(epoch))

              # 重新初始化数据
              sess.run([it.initializer])
              break
     
  def evaluate(self, log_dir, testdata):
    test_data = tf.data.TFRecordDataset(testdata)
    eval_dataset = test_data.map(parse_tfrecords_function).prefetch(100).batch(1) # .repeat(100)
    eval_it = eval_dataset.make_initializable_iterator()
    feats, labels, Xts = eval_it.get_next()

    # _eta_d, _eta_c, _d_softmax_logits, _c_softmax_logits = self.inference(feats)
    eta_d, eta_c, d_softmax_logits, c_softmax_logits = self.inference(feats)
    out = tf.concat([
      tf.math.exp(eta_d), tf.math.exp(eta_c),
      tf.nn.softmax(d_softmax_logits), tf.nn.softmax(c_softmax_logits)
      ], axis=0, name='out')
      
    # TODO: evaluate data.test
    with self.restore(log_dir) as sess:
      sess.run([eval_it.initializer])

      ret = sess.run([out])
      print (ret)

  def evaluate_test2(self, log_dir, testdata):
    test_data = tf.data.TFRecordDataset(testdata)
    eval_dataset = test_data.map(parse_tfrecords_function).prefetch(10**6).batch(10**4) # .repeat(100)
    eval_it = eval_dataset.make_initializable_iterator()
    feats, labels, Xts = eval_it.get_next()

    # _eta_d, _eta_c, _d_softmax_logits, _c_softmax_logits = self.inference(feats)
    eta_d, eta_c, d_softmax_logits, c_softmax_logits = self.inference(feats)

    feature_names = [name for name, _ in create_placeholder().items() if name != 'reserve_partner_car_type_id']

    # TODO: convert data.test to json
    data, reserve_partner_car_type_id = [], []
    with self.restore(log_dir) as sess:
      sess.run(eval_it.initializer)

      while 1:
        try:
          # eval_tensor = [probs, pred_d_time, pred_c_time] + [feats[c] for c in ['dt', 'oid', 'Xt', 'assignd', 'canceld', 'timeoutd']]
          # eval_tensor = [assigned_probs_with_notimeout, probs_with_notimeout, timeout_probs, pred_d_time, pred_c_time] + [feats[c] for c in ['dt', 'oid', 'Xt', 'assignd', 'canceld', 'timeoutd']]
          eval_tensor = [
            tf.math.exp(eta_d), tf.math.exp(eta_c),
            tf.nn.softmax(d_softmax_logits), tf.nn.softmax(c_softmax_logits),
            ] + [feats[c] for c in ['dt', 'oid', 'Xt', 'assignd', 'canceld', 'timeoutd']]

          ret = sess.run(eval_tensor)

          reserve_partner_car_type_id.append(ret[0])
          data.append(ret[1:])
        except tf.errors.OutOfRangeError:
          break
    
    import pandas as pd
    df = pd.concat([pd.DataFrame(np.concatenate(fs, axis=0)) for fs in zip(*data)], axis=1, ignore_index=True)
    df.to_excel('eval_test_v6.xlsx', header=False, index=False)

  def evaluate_test(self, log_dir, testdata, sz=1000):
    test_data = tf.data.TFRecordDataset(testdata)
    eval_dataset = test_data.map(parse_tfrecords_function).prefetch(10**6).batch(10**3).repeat(1)
    eval_it = eval_dataset.make_initializable_iterator()
    feats, labels, Xts = eval_it.get_next()

    # _eta_d, _eta_c, _d_softmax_logits, _c_softmax_logits = self.inference(feats)
    eta_d, eta_c, d_softmax_logits, c_softmax_logits = self.inference(feats)

    # feature_names = [name for name, _ in create_placeholder().items() if name != 'reserve_partner_car_type_id']
    feature_names = create_placeholder().keys()

    def filterEmpty(array2d):
      # return np.array([[e for e in lst if e != b''] for lst in array2d])
      return pd.Series([[e for e in lst if e != b''] for lst in array2d])

    # TODO: convert data.test to json
    import pandas as pd
    dataFrames = []
    with self.restore(log_dir) as sess:
      sess.run(eval_it.initializer)

      while 1:
        try:
          eval_tensor = [{c: feats[c] if c != 'reserve_partner_car_type_id' 
              else tf.sparse_tensor_to_dense(feats['reserve_partner_car_type_id'], default_value='') 
              for c in feature_names
            },
            {c: t for c, t in zip(['dispLambda', 'cancLambda', 'dispSoftmax', 'cancSoftmax'], [
              tf.math.exp(eta_d), tf.math.exp(eta_c),
              tf.nn.softmax(d_softmax_logits), 
              tf.nn.softmax(c_softmax_logits)])},
            {c: feats[c] for c in ['dt', 'oid', 'Xt', 'assignd', 'canceld', 'timeoutd']}
          ]
          eval_tensor = {k:v for dic in eval_tensor for k, v in dic.items()}

          ret = sess.run(eval_tensor)
          ret['reserve_partner_car_type_id'] = filterEmpty(ret['reserve_partner_car_type_id'].tolist())
          ret['dispLambda'] = pd.Series(ret['dispLambda'].tolist())
          ret['cancLambda'] = pd.Series(ret['cancLambda'].tolist())
          ret['dispSoftmax'] = pd.Series(ret['dispSoftmax'].tolist())
          ret['cancSoftmax'] = pd.Series(ret['cancSoftmax'].tolist())
          ret['Xt'] = ret['Xt'].squeeze()

          dataFrames.append(pd.DataFrame(ret))
          # break
        except tf.errors.OutOfRangeError:
          break
    
    df = pd.concat(dataFrames, axis=0)
    with open('test_cases_%d.json'%sz, 'w') as fp:
      fp.write(df[0:sz].to_json(orient='records'))
    df.to_excel('test_prediction_result.xlsx')

  def export_model(self, log_dir, export_dir='./pbModel'):
    inf_feat = create_placeholder()
    features = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in inf_feat.items()}

    eta_d, eta_c, d_softmax_logits, c_softmax_logits = self.inference(inf_feat)
    out = tf.concat([
      tf.math.exp(eta_d), tf.math.exp(eta_c),
      tf.nn.softmax(d_softmax_logits), tf.nn.softmax(c_softmax_logits)
      ], axis=0, name='out')

    with self.restore(log_dir) as sess:
      # https://www.tensorflow.org/tfx/serving/saved_model_warmup
      # https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/saved_model_warmup.md
      
      prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
          inputs=features,
          # outputs={
          #     '_c_softmax_logits': tf.saved_model.utils.build_tensor_info(_c_softmax_logits),
          #     '_d_softmax_logits': tf.saved_model.utils.build_tensor_info(_d_softmax_logits),
          #     '_eta_c': tf.saved_model.utils.build_tensor_info(_eta_c),
          #     '_eta_d': tf.saved_model.utils.build_tensor_info(_eta_d),
          # },
          outputs={
              'out': tf.saved_model.utils.build_tensor_info(out)
          },
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

      legacy_init_op = tf.group(
          tf.tables_initializer(), name='legacy_init_op')

      builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'inf':
                  prediction_signature,
          },
          clear_devices=True,
          legacy_init_op=legacy_init_op)

      builder.save(True)
      # builder.save()
    with open('{0}/feature_names'.format(export_dir), 'w') as fp:
      fp.write('\n'.join(inf_feat.keys()))

def view_tfrecord(testdata_file):
  test_data = tf.data.TFRecordDataset(testdata_file)
  eval_dataset = test_data.map(parse_tfrecords_function).prefetch(100).batch(2) # .repeat(100)
  eval_it = eval_dataset.make_initializable_iterator()
  feats, labels, Xts = eval_it.get_next()

  with tf.Session() as sess:
    sess.run([eval_it.initializer])

    record = sess.run(feats)
    for k, v in record.items():
      print (k, v)

    with open('test_cases', 'w') as fp:
      fp.write('\n'.join(["{0}\t{1}".format(k, v) for k, v in record.items()]))

    # tf.sparse_tensor_to_dense
    ret = sess.run([tf.sparse_tensor_to_dense(feats['reserve_partner_car_type_id'], default_value='')])
    print (ret)

if __name__ == "__main__":
  task, traindata_file, testdata_file, log_dir = sys.argv[1:]  
  
  # self.layer_units = [128, 64, 32]
  model = Model(d_k=5, c_k=5)
  # model = Model(d_k=5, c_k=5, layer_units=[128, 64, 32])
  # model = Model(d_k=2, c_k=2)

  if task == 'train':
    model.train(traindata_file, testdata_file, log_dir, epochs=20)

  if task == 'export_model':
    model.export_model(log_dir)

  if task == 'evaluate':
    model.evaluate(log_dir, testdata_file)

  if task == 'evaluate_test':
    model.evaluate_test(log_dir, testdata_file, 1000)

  if task == 'view_tfrecord':
    view_tfrecord(testdata_file)
