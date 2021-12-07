#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse

import tensorflow as tf

import numpy as np
import pandas as pd

from tqdm import tqdm

import time

from json import *
import json

from tensorflow.keras import backend as K, layers, utils, Model, Input, optimizers as opt
from tensorflow.keras.activations import tanh, softmax

import random as rnd    

# https://www.tensorflow.org/tutorials/keras/save_and_load
# https://www.tensorflow.org/guide/keras/save_and_serialize?hl=zh-cn


import pickle

learning_rate = 0.001
batch_size = 256
timestep_size = 59

lstm_layers = 1
poi_embedding_size = 16
dept_embedding_size = 16
driver_embedding_size = 16
grid5_embedding_size = 5
grid6_embedding_size = 8
grid7_embedding_size = 16
# embedding_size = 16
hidden_size = 64
pointer_w_size = 64

MISS_ID = 'Non'

def calc_fp_line_no(filename):
    with open(filename) as fp:
        no = 0
        for _ in fp:
            no += 1
        return no

def get_fp_fields(filename):
    data = []
    with open(filename) as fp:
        for line in fp:
            try:
                d = JSONDecoder().decode(line)
            except Exception as e:
                raise Exception([line, e])
            
            data.append(len(d['Y']))
    return np.array(data)

def miss_ana(filename, poi2idx, dept2idx, grid5idx, grid6idx):
    with open(filename) as fp, open('seq.data.part0', 'w') as wfp0, open('seq.data.part1', 'w') as wfp1:
        miss_cnt, miss_sum_cnt = np.zeros(4), np.zeros(4)
        for line in fp:
            try:
                d = JSONDecoder().decode(line)
            except Exception as e:
                raise Exception([line, e])

            dept_id = d['grid_station_id']
            inp_seq = d['poi_id_sequence_app']
            grid5_sequence_app = d['grid5_sequence_app']
            grid6_sequence_app = d['grid6_sequence_app']
            grid7_sequence_app = d['grid7_sequence_app']
            pcs_sequence_app = d['pcs_sequence_app']
            driver_code = d['driver_code']
            
            # data_x.append([poi2idx[p] if p in poi2idx else poi2idx[MISS_ID] for p in inp_seq])
            # data_dept.append([dept2idx[dept_id] if dept_id in dept2idx else dept2idx[MISS_ID]])
            # data_grid5.append([grid5idx[p] if p in grid5idx else grid5idx[MISS_ID] for p in grid5_sequence_app])
            # data_grid6.append([grid6idx[p] if p in grid6idx else grid6idx[MISS_ID] for p in grid6_sequence_app])

            _cnt = np.zeros(4)
            if not dept_id in dept2idx:
                _cnt[0] += 1

            for _ in inp_seq:
                if not _ in poi2idx:
                    _cnt[1] += 1

            for _ in grid5_sequence_app:
                if not _ in grid5idx:
                    _cnt[2] += 1

            for _ in grid6_sequence_app:
                if not _ in grid6idx:
                    _cnt[3] += 1
            
            miss_sum_cnt += _cnt
            miss_cnt += (_cnt > 0).astype('int')

            if all(_cnt == 0):
                wfp0.write(line)
            else:
                wfp1.write(line)

    print miss_cnt, miss_sum_cnt


def gen_train_data2(filename, poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx, timestep_size, bs):
    def process_x(_x):
        return tf.keras.preprocessing.sequence.pad_sequences(_x, padding='post', maxlen=timestep_size)
    def process_y(_y):
        y = tf.keras.preprocessing.sequence.pad_sequences(_y, padding='post', maxlen=timestep_size, value=-1)
        y = utils.to_categorical(y, num_classes=timestep_size+1)
        y = np.delete(y, -1, axis=-1)
        return y
    def process_a(_ll):
        return np.array(_ll)

    def process_pcs(_x):
        _p = tf.keras.preprocessing.sequence.pad_sequences(_x, padding='post', maxlen=timestep_size)
        p = np.expand_dims(_p, axis=-1)
        return np.log(1 + p)

    while 1:
        with open(filename) as fp:
            # print 'open %s ...' %filename

            data_x, data_y, data_dept, data_grid5, data_grid6, data_grid7, data_driver, data_pcs, data_dist = [], [], [], [], [], [], [], [], []

            for line in fp:
                try:
                    d = JSONDecoder().decode(line)
                except Exception as e:
                    raise Exception([line, e])

                dept_id = d['grid_station_id']
                inp_seq = d['poi_id_sequence_app']
                grid5_sequence_app = d['grid5_sequence_app']
                grid6_sequence_app = d['grid6_sequence_app']
                grid7_sequence_app = d['grid7_sequence_app']
                pcs_sequence_app = d['pcs_sequence_app']
                driver_code = d['driver_code']

                if len(data_x) >= bs:
                    yield [process_x(data_x), process_a(data_dept), process_x(data_grid5), process_x(data_grid6), process_x(data_grid7), process_a(data_driver), process_pcs(data_pcs), process_a(data_dist)], process_y(data_y)
                    data_x, data_y, data_dept, data_grid5, data_grid6, data_grid7, data_driver, data_pcs, data_dist = [], [], [], [], [], [], [], [], []
                
                data_x.append([poi2idx[p] if p in poi2idx else poi2idx[MISS_ID] for p in inp_seq])
                data_dept.append([dept2idx[dept_id] if dept_id in dept2idx else dept2idx[MISS_ID]])
                data_grid5.append([grid5idx[p] if p in grid5idx else grid5idx[MISS_ID] for p in grid5_sequence_app])
                data_grid6.append([grid6idx[p] if p in grid6idx else grid6idx[MISS_ID] for p in grid6_sequence_app])
                data_grid7.append([grid7idx[p] if p in grid7idx else grid7idx[MISS_ID] for p in grid7_sequence_app])
                data_driver.append([driver2idx[driver_code] if driver_code in driver2idx else driver2idx[MISS_ID]])
                data_pcs.append(pcs_sequence_app)
                
                data_dist.append(np.pad(d['earth_dist'], ((0, timestep_size-len(inp_seq)), (0, timestep_size-len(inp_seq))), mode='constant'))
                data_y.append(d['Y'])

            yield [process_x(data_x), process_a(data_dept), process_x(data_grid5), process_x(data_grid6), process_x(data_grid7), process_a(data_driver), process_pcs(data_pcs), process_a(data_dist)], process_y(data_y)

def load_stat_data():
    with open('poi2idx', 'rb') as fp:
        poi2idx = pickle.load(fp)
    with open('dept2idx', 'rb') as fp:
        dept2idx = pickle.load(fp)
    with open('grid5idx', 'rb') as fp:
        grid5idx = pickle.load(fp)
    with open('grid6idx', 'rb') as fp:
        grid6idx = pickle.load(fp)
    with open('grid7idx', 'rb') as fp:
        grid7idx = pickle.load(fp)
    with open('driver2idx', 'rb') as fp:
        driver2idx = pickle.load(fp)
    return poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx

def save_stat_data(data_file):
    with open(data_file) as fp:
        timestep_size, poi_ids, dept_ids, driver_ids = 0, set(), set(), set()
        grid5_ids, grid6_ids, grid7_ids = set(), set(), set()

        cnt = [0] * 6
        pois = []
        print 'gen_stat_data ...', data_file
        for line in fp:
            _ = JSONDecoder().decode(line)

            if len(_['poi_id_sequence_app']) != len(_['grid5_sequence_app']):
                print line
                break
                cnt[0] += 1
                # pass
            if len(_['poi_id_sequence_app']) != len(_['grid6_sequence_app']):
                cnt[1] += 1
                # pass
            if len(_['poi_id_sequence_app']) != len(_['grid7_sequence_app']):
                cnt[2] += 1
                # pass
            if len(_['poi_id_sequence_app']) != len(_['pcs_sequence_app']):
                cnt[3] += 1
                # pass    
            if len(_['poi_id_sequence_app']) != len(_['reach_minute_sequence_real']):
                cnt[4] += 1
                # pass        

            if not 'grid_station_id' in _:
                # print _
                cnt[5] += 1
                continue

            pois += _['poi_id_sequence_app']

            poi_ids |= set(_['poi_id_sequence_app'])
            dept_ids |= set([_['grid_station_id']])
            grid5_ids |= set(_['grid5_sequence_app'])
            grid6_ids |= set(_['grid6_sequence_app'])
            grid7_ids |= set(_['grid7_sequence_app'])
            driver_ids |= set([_['driver_code']])
            
            timestep_size = len(_['poi_id_sequence_app']) if timestep_size < len(_['poi_id_sequence_app']) else timestep_size
    
    print 'error cnt: ', cnt
    poi2idx = {_2:_1 for _1, _2 in enumerate(poi_ids | {MISS_ID}, 1)}
    dept2idx = {_2:_1 for _1, _2 in enumerate(dept_ids | {MISS_ID}, 1)}
    grid5idx = {_2:_1 for _1, _2 in enumerate(grid5_ids | {MISS_ID}, 1)}
    grid6idx = {_2:_1 for _1, _2 in enumerate(grid6_ids | {MISS_ID}, 1)}
    grid7idx = {_2:_1 for _1, _2 in enumerate(grid7_ids | {MISS_ID}, 1)}
    driver2idx = {_2:_1 for _1, _2 in enumerate(driver_ids | {MISS_ID}, 1)}

    pd.Series(pois).value_counts().to_csv('pois', header=False)

    with open('poi2idx', 'wb') as fp:
        pickle.dump(poi2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('dept2idx', 'wb') as fp:
        pickle.dump(dept2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('grid5idx', 'wb') as fp:
        pickle.dump(grid5idx, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('grid6idx', 'wb') as fp:
        pickle.dump(grid6idx, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('grid7idx', 'wb') as fp:
        pickle.dump(grid7idx, fp, protocol=pickle.HIGHEST_PROTOCOL)    
    with open('driver2idx', 'wb') as fp:
        pickle.dump(driver2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)    

    return timestep_size

class LayerW(layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(LayerW, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight('w',
            shape=(self.input_dim, self.output_dim),
            initializer="random_normal",
            dtype="float32",
        )

    def call(self, inputs, mask=None):
        # return tf.matmul(inputs, self.W)
        return tf.einsum('bij,jk->bik', inputs, self.W)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        }
        base_config = super(LayerW, self).get_config()
        return dict(base_config, **config)

class LayerV(layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(LayerV, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight('w',
            shape=(self.input_dim,),
            initializer="random_normal",
            dtype="float32",
        )

    def call(self, inputs, mask=None):
        # return tf.matmul(inputs, self.W)
        return tf.einsum('bijk,k->bij', inputs, self.W)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
        }
        base_config = super(LayerV, self).get_config()
        return dict(base_config, **config)

class LayerPointer(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerPointer, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        e, d = inputs
        return tanh(tf.expand_dims(e, -2) + tf.expand_dims(d, -3))



class LayerSqueeze(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerSqueeze, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, x, mask=None):
        return tf.squeeze(x, -1)

class LayerDot(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerDot, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, inputs, normalize=True):
        e, d = inputs
        return layers.dot([e, d], axes=-1, normalize=normalize)

class LayerRepeat(layers.Layer):
    def __init__(self, timestep_size, **kwargs):
        super(LayerRepeat, self).__init__(**kwargs)
        self.timestep_size = timestep_size
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return K.repeat(tf.squeeze(inputs, -2), self.timestep_size)

    def get_config(self):
        config = {
            'timestep_size': self.timestep_size,
        }
        base_config = super(LayerRepeat, self).get_config()
        return dict(base_config, **config)

class LayerSoftmax(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerSoftmax, self).__init__(**kwargs)
        self.supports_masking = True
        self._small_negative = -1e18
        
    def call(self, logits, mask=None):
        if mask is not None:
            # print mask
            m = tf.cast(mask[0], logits.dtype)
            # m = tf.cast(mask[0][0], logits.dtype)
            mm = tf.expand_dims(m, -2) * tf.expand_dims(m, -1)
            
            adder = (1.0 - mm) * self._small_negative
            return logits + adder, softmax(logits + adder)
            
        return logits, softmax(logits)

class LayerAdd(layers.Layer):
    def __init__(self, **kwargs):
        super(LayerAdd, self).__init__(**kwargs)
        self.supports_masking = True
    def call(self, inputs, mask=None):
        return inputs[0] + inputs[1]

def tsp_loss(y_true, logits):
    # TODO: loss func
    return -tf.reduce_sum(y_true * tf.nn.log_softmax(logits, axis=-1))
    # return -tf.reduce_mean(y_true * tf.nn.log_softmax(logits, axis=-1))

    # return -tf.reduce_sum(y_true * tf.nn.log_softmax(logits, axis=-1)) + -tf.reduce_sum(y_true * tf.nn.log_softmax(logits, axis=-2)) + tf.reduce_sum(
    #     y_true * tf.abs(tf.reduce_logsumexp(logits, axis=-2, keepdims=True) - tf.reduce_logsumexp(logits, axis=-1, keepdims=True))
    # )

def tsp_metric(y_true, logits):
    # print y_true, logits
    return 0.0

def create_model(poi_vocab_size, dept_vocab_size, grid5_size, grid6_size, grid7_size, driver_size, timestep_size):
    # input other feature: dept_id ...
    dept_id = Input(shape=(1,), dtype=tf.int32, sparse=False, name='dept_id')
    driver_code = Input(shape=(1,), dtype=tf.int32, sparse=False, name='driver_code')
    dist = Input(shape=(timestep_size, timestep_size + 1), dtype=tf.float32, name='dist')
    grid7_sequence_app = Input(shape=(timestep_size,), dtype=tf.int32, sparse=False, name='grid7_sequence_app')
    grid6_sequence_app = Input(shape=(timestep_size,), dtype=tf.int32, sparse=False, name='grid6_sequence_app')
    grid5_sequence_app = Input(shape=(timestep_size,), dtype=tf.int32, sparse=False, name='grid5_sequence_app')

    pcs_sequence_app = Input(shape=(timestep_size,1), dtype=tf.float32, sparse=False, name='pcs_sequence_app')

    seq = Input(shape=(timestep_size,), dtype=tf.int32, sparse=False, name='seq')
    layer_embedding = layers.Embedding(poi_vocab_size, poi_embedding_size, mask_zero=True, name='poi_embedding')
    dept_embedding = layers.Embedding(dept_vocab_size, dept_embedding_size, mask_zero=False, name='dept_embedding')
    driver_embedding = layers.Embedding(driver_size, driver_embedding_size, mask_zero=False, name='driver_embedding')
    

    grid7_embedding = layers.Embedding(grid7_size, grid7_embedding_size, mask_zero=True, name='grid7_embedding')
    grid6_embedding = layers.Embedding(grid6_size, grid6_embedding_size, mask_zero=True, name='grid6_embedding')
    grid5_embedding = layers.Embedding(grid5_size, grid5_embedding_size, mask_zero=True, name='grid5_embedding')
    
    inp = layers.Concatenate(name='concat')([
        # layer_embedding(seq), 
        layers.Add(name='Add')([layer_embedding(seq), dept_embedding(dept_id), driver_embedding(driver_code)])
        , grid7_embedding(grid7_sequence_app)
        , grid6_embedding(grid6_sequence_app)
        , grid5_embedding(grid5_sequence_app)
        , pcs_sequence_app
        , dist
        ])

    # inp = layers.Concatenate(name='concat')([
    #     layer_embedding(seq)
    #     , LayerRepeat(timestep_size, name='dept_repeater')(dept_embedding(dept_id))
    #     , LayerRepeat(timestep_size, name='driver_repeater')(driver_embedding(driver_code))
    #     , grid7_embedding(grid7_sequence_app)
    #     , grid6_embedding(grid6_sequence_app)
    #     , grid5_embedding(grid5_sequence_app)
    #     , pcs_sequence_app
    #     , dist
    #     ])
    # encoder = layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='encoder')
    # decoder = layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder')
    # encode, h, c = encoder(inp)
    # decode, _, _ = decoder(encode, initial_state=[h,c])

    encoders = [layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='encoder%d'%i) for i in range(lstm_layers)]
    decoders = [layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder%d'%i) for i in range(lstm_layers)]

    encode, h, c = encoders[0](inp)
    for e in encoders[1:]:
        encode, h, c = e([encode, h, c])
    
    decode, h, c = decoders[0](encode, initial_state=[h,c])
    for d in decoders[1:]:
        decode, h, c = d([decode, h, c])

    # encode, h, c = encoders[2](encoders[1](encoders[0](inp)))
    # decode, _, _ = decoders[2](decoders[1](decoders[0](encode, initial_state=[h,c])))

    # repeater = LayerRepeat(timestep_size, name='repeater')
    # feat = dept_embedding(dept_id)
    # dec_inp = repeater(feat, mask=mask)

    # mask = layer_embedding.compute_mask(seq)
    # decode, _, _ = decoder(encode, initial_state=[h,c], mask=mask)


    layer_w_e = LayerW(hidden_size, pointer_w_size, name='W_e')
    layer_w_d = LayerW(hidden_size, pointer_w_size, name='W_d')
    layer_v = LayerV(pointer_w_size, name='V')
    layer_pointer = LayerPointer(name='pointer')
    layer_squeeze = LayerSqueeze(name='squeeze')

    u_ij = layer_v(layer_pointer([layer_w_e(encode), layer_w_d(decode)]))
    # logits = layer_squeeze(u_ij)

    # logits = u_ij
    # layer_dot = LayerDot(name='dot')
    # logits = layer_dot([encode, decode])
        
    # y_pred = softmax(logits, -1)

    # u_ij = LayerAdd(name='adder')([u_ij, dist])

    # y_pred = layers.Softmax(axis=-1, name='layer_softmax')(logits)
    logits, y_pred = LayerSoftmax(name='softmax')(u_ij)
    # logits = LayerSoftmax(name='softmax')(logits)
    # y_pred = logits
    # return Model([seq], logits)

    return Model([seq, dept_id, grid5_sequence_app, grid6_sequence_app, grid7_sequence_app, driver_code, pcs_sequence_app, dist], logits)
    # return Model([seq, inp, dist], logits)


def train(args):
    poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx = load_stat_data()

    dept_vocab_size = len(dept2idx) + 1
    poi_vocab_size = len(poi2idx) + 1
    grid5_size = len(grid5idx) + 1
    grid6_size = len(grid6idx) + 1
    grid7_size = len(grid7idx) + 1
    driver_size = len(driver2idx) + 1
    # timestep_size = 59
    print '# ' * 20
    # print poi_vocab_size, dept_vocab_size, timestep_size, len(grid5idx), len(grid6idx), len(grid7idx)

    batch_train = gen_train_data2(args.train_data, poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx, timestep_size, batch_size)
    batch_test = gen_train_data2(args.test_data, poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx, timestep_size, batch_size)
    
    # samples = next(batch_train)
    # print [_.shape for _ in samples[0]], samples[1].shape

    model = create_model(poi_vocab_size, dept_vocab_size, grid5_size, grid6_size, grid7_size, driver_size, timestep_size)

    # model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=tsp_loss)

    from tensorflow.keras.callbacks import LearningRateScheduler
    
    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的 0.96
        decay = 0.96
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * decay)
        return K.get_value(model.optimizer.lr)

    lr_schedule = LearningRateScheduler(scheduler, verbose=1)
    
    callbacks_list = [tf.keras.callbacks.TensorBoard(log_dir='./model/', write_graph=True, write_images=False)]
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # tf.keras.optimizers.Adam(learning_rate),
    # tf.train.AdamOptimizer(learning_rate),
    
    def acc(y_true, y_pred):
        return tf.keras.metrics.binary_accuracy(y_true, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
        #   loss='categorical_crossentropy',
        # loss= lambda _y, _p: tf.keras.losses.categorical_crossentropy,
        # loss= tf.keras.losses.categorical_crossentropy,
        # loss= tf.losses.softmax_cross_entropy,

        # loss = lambda _y, _logits: tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=_logits),
            loss=tsp_loss,
            metrics=['accuracy']
            )

    if args.reload:
        model.load_weights(args.log_dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.log_dir,
                                                save_weights_only=False,
                                                verbose=0)

    # model.load_weights(args.log_dir)
    # model.compile(optimizer=opt.RMSprop(learning_rate=0.005), loss=tsp_loss)
    model.summary(line_length=150, positions=[0.2, 0.6, 0.8, 1.0])
    # print(model.get_config())

    train_no = calc_fp_line_no(args.train_data)
    eval_no = calc_fp_line_no(args.test_data)
    train_num, test_num = train_no, eval_no
    # train_num, test_num = 1024, 1024
    steps_per_epoch = train_num // batch_size + 1 if train_num % batch_size else 0
    steps_per_eval = test_num // batch_size + 1 if test_num % batch_size else 0
    print steps_per_epoch, steps_per_eval
    model.fit_generator(batch_train, steps_per_epoch=steps_per_epoch, epochs=args.epochs, 
        max_queue_size=10, # use_multiprocessing=True, # workers=12, 
        validation_data=batch_test, validation_steps=steps_per_eval, 
        # callbacks=[cp_callback] # + callbacks_list
        )
    model.save(args.log_dir)

    # from tensorflow.python.client import timeline
    # tl = timeline.Timeline(run_metadata.step_stats)
    # ctf = tl.generate_chrome_trace_format()
    # with open('timeline.json', 'w') as f:
    #     f.write(ctf)
    # print('timeline.json has been saved!')
    

def evaluate(args):
    poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx = load_stat_data()

    dept_vocab_size = len(dept2idx) + 1
    poi_vocab_size = len(poi2idx) + 1
    grid5_size = len(grid5idx) + 1
    grid6_size = len(grid6idx) + 1
    grid7_size = len(grid7idx) + 1
    driver_size = len(driver2idx) + 1

    # timestep_size = 59
    print '# ' * 20
    # print poi_vocab_size, dept_vocab_size, timestep_size, len(grid5idx), len(grid6idx), len(grid7idx)
    # return

    # batch_train = gen_train_data2(args.train_data, poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx, timestep_size, batch_size)
    batch_test = gen_train_data2(args.test_data, poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx, timestep_size, batch_size)
    
    # samples = next(batch_train)
    # print [_.shape for _ in samples[0]], samples[1].shape
    model = create_model(poi_vocab_size, dept_vocab_size, grid5_size, grid6_size, grid7_size, driver_size, timestep_size)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss=tsp_loss,
            metrics=['accuracy', tsp_metric]
            )
    model.load_weights(args.log_dir)

    eval_no = calc_fp_line_no(args.test_data)

    train_num, test_num = 0, eval_no # 24954, 13063
    # steps_per_epoch = train_num // batch_size + 1 if train_num % batch_size else 0
    steps_per_eval = test_num // batch_size + 1 if test_num % batch_size else 0
    print steps_per_eval, eval_no

    pred = model.predict_generator(batch_test, steps=steps_per_eval, verbose=1)
    np.savez('y_pred_eval', timesteps=get_fp_fields(args.test_data), logits=pred)

    loss, acc, o = model.evaluate_generator(batch_test, steps=steps_per_eval)
    print loss, acc, o
        

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-x",
        "--train_data",
        default='seq.data.train',
        type=str,
        help="view")
    parser.add_argument(
        "-y",
        "--test_data",
        default='seq.data.test',
        type=str,
        help="view")
    parser.add_argument(
        "-v",
        "--view",
        action="store_true",
        help="view")
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="train")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="eval")    
    parser.add_argument(
        "-e",
        "--epochs",
        default=1,
        type=int,
        help="epochs")
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help="stat")    
    parser.add_argument(
        "-r",
        "--reload",
        action="store_true",
        help="reload model")        
    parser.add_argument(
        "-l",
        "--log_dir",
        default='model/pointer_network',
        type=str,
        help="tf_logs")

    args = parser.parse_args()
    print(args)

    if args.stat:
        timesteps = save_stat_data(args.train_data)
        print 'timesteps: ', timesteps

    if args.view:
        # with open('poi2idx', 'rb') as fp:
        #     poi2idx = pickle.load(fp)

        # with open('dept2idx', 'rb') as fp:
        #     dept2idx = pickle.load(fp)
        # with open('grid5idx', 'rb') as fp:
        #     grid5idx = pickle.load(fp)
        # with open('grid6idx', 'rb') as fp:
        #     grid6idx = pickle.load(fp)
        # with open('grid7idx', 'rb') as fp:
        #     grid7idx = pickle.load(fp)
        # with open('driver2idx', 'rb') as fp:
        #     driver2idx = pickle.load(fp)


        poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx = load_stat_data()
        
        dept_vocab_size = len(dept2idx) + 1
        poi_vocab_size = len(poi2idx) + 1
        grid6_size = len(grid6idx) + 1
        driver_size = len(driver2idx) + 1

        # timestep_size = 59
        print '# ' * 20
        print poi_vocab_size, dept_vocab_size, timestep_size, len(grid5idx), len(grid6idx), len(grid7idx), driver_size
        # return
            
        bs = 2
        batch_train = gen_train_data2(args.test_data, poi2idx, dept2idx, grid5idx, grid6idx, grid7idx, driver2idx, timestep_size, bs)
        print next(batch_train)


        # miss_ana(args.test_data, poi2idx, dept2idx, grid5idx, grid6idx)

    if args.train:
        train(args)

    if args.eval:
        evaluate(args)

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main)
    main()
