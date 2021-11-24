
import collections
import math
import pdb
import random as rnd
import time

from ipywidgets import *
import ipywidgets as widgets

from json import *

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from pyspark.sql import Row

import pyspark.sql.functions as functions

from pyspark.sql.types import *



# class udf(object):
#     def __init__(self, func):
#         attr = func.__dict__
        
#         ret_type = attr['ret_type'] if 'ret_type' in attr else DoubleType()
#         func = attr['func'] if 'func' in attr else func
        
#         spark.sql('''DROP TEMPORARY FUNCTION IF EXISTS {0}'''.format(func.__name__))
#         spark.udf.register(func.__name__, func, ret_type)
        
#         self.func = func
#     def __call__(self, *args, **kwargs):
# #         print '__call__ from udf'
#         return self.func(*args, **kwargs)
    
# def udf_ret_type(ret_type):
#     def decorator(func):
#         def wrapped(*args, **kwargs):
#             return func(*args, **kwargs)
#         wrapped.__dict__['ret_type'] = ret_type
# #         print func.__name__
#         wrapped.__dict__['func'] = func
#         return wrapped

#     return decorator

def udf(ret_type):
    def decorator(func):
        spark.sql('''DROP TEMPORARY FUNCTION IF EXISTS {0}'''.format(func.__name__))
        spark.udf.register(func.__name__, func, ret_type)

        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    return decorator

def udf_register(func, ret_type=None):
    spark.sql('''DROP TEMPORARY FUNCTION IF EXISTS {0}'''.format(func.__name__))
    spark.udf.register(func.__name__, func, ret_type)
        
def getDataFrame(sqlStat):
    print(sqlStat)
    return spark.sql(sqlStat).toPandas()

def func(name, f):
    f.__name__ = name
    return f

def getRDDBySQL(spark, sqlStat):
    return spark.sql(sqlStat).rdd.repartition(100)

def getRDD(spark, tableName, day):
    sqlStat = """SELECT * FROM {tableName} WHERE dt = {day}""".format(tableName = tableName, day = day)
    return spark.sql(sqlStat).rdd.repartition(100)

def FlatTuple(row):
    if isinstance(row[0], tuple):
        return row[0] + tuple([row[1]])
    else:
        return row

def Value(row):
    if isinstance(row, tuple) and not isinstance(row, Row) and len(row) == 2:
        return row[1]
    else:
        raise Exception('isinstance(row, tuple) and not isinstance(row, Row) and len(row) == 2', row)

def depthFlatTuple(row):
    ret = tuple()
    for ele in row:
        if isinstance(ele, Row):
            ret += tuple([ele])
        elif isinstance(ele, tuple):
            ret += depthFlatTuple(ele)
        else:
            ret += tuple([ele])
    return ret

def viewRow(row):
    for k, v in row.asDict().iteritems():
        print ("%50s = %s : %s" %(k, v, type(v)))
    print ('-' * 60)

def sample(data, k):
    pool = [None for _ in range(min(len(data), k))]

    for ix, ele in enumerate(data):
        if ix < k:
            pool[ix] = ele
            continue
        choice_ix = rnd.randint(0, ix)

        if choice_ix < k:
            pool[choice_ix] = ele

    return pool

from contextlib import contextmanager

@contextmanager
def Timer(name):
    start = time.time()
    yield
    print ('[{0}] done in {1} s'.format(name, time.time() - start))


# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

# import sys
# stdo = sys.stdout
# reload(sys)
# sys.setdefaultencoding('utf-8')
# sys.stdout= stdo

def Raise(e):
    raise Exception(e)

# pd.set_option('max_columns', 200)
# pd.set_option('max_colwidth', 200)

from pyspark.sql import SparkSession, Row
from pyspark.sql import HiveContext

spark = SparkSession.builder.\
    config("spark.app.name", "pyspark_pyspark"). \
    config("spark.executor.memory", "10G"). \
    config("spark.driver.memory", "6G"). \
    config("spark.sql.execution.arrow.enabled", "true"). \
    config("spark.dynamicAllocation.enabled", "true"). \
    config("spark.dynamicAllocation.minExecutors", "100"). \
    config("spark.dynamicAllocation.maxExecutors", "1000"). \
    config("hive.exec.max.dynamic.partitions", "10000"). \
    config("mapred.map.child.java.opts", "-Xmx10240m"). \
    config("mapreduce.map.memory.mb", "10240"). \
    config("mapred.reduce.child.java.opts", "-Xmx10240m"). \
    config("mapreduce.reduce.memory.mb", "10240"). \
    config("spark.local.dir", "/opt/xxxx/tmp/"). \
    config("spark.driver.maxResultSize", "12g"). \
    config("spark.hadoop.validateOutputSpecs", "false"). \
    config("hive.exec.dynamic.partition", "true"). \
    config("hive.exec.parallel", "true"). \
    config("hive.exec.dynamic.partition.mode", "nonstrict"). \
    config("mapred.max.split.size", "32000000"). \
    config("mapred.min.split.size.per.node", "32000000"). \
    config("mapred.min.split.size.per.rack", "32000000"). \
    config("hive.exec.reducers.bytes.per.reducer", "128000000"). \
    enableHiveSupport().getOrCreate()
    
spark.sql("CREATE TEMPORARY FUNCTION projHex AS 'com.xxxxxxxx.hive.udf.proj2Hex'")
spark.sql("CREATE TEMPORARY FUNCTION JSONARRAY as 'com.xxxxxxxx.hive.udf.JSONArray'")

from IPython.core.magic import (register_line_magic,
                                register_cell_magic, register_line_cell_magic)

def _hive(line, cell, debug=False):
    param = eval(line) if line else {}
    stat = cell.format(**param)
    if debug:
        print (stat)
    return stat, spark.sql(stat)

@register_line_cell_magic
def hive(line, cell):
    stat, df = _hive(line, cell)
    return df

@register_line_cell_magic
def hive_debug(line, cell):
    stat, df = _hive(line, cell, debug=True)
    return df

def _hive_batch(line, cell, debug=False):
    param = eval(line) if line else {}
    batch = param.pop('batch', None)
    if not batch:
        print('batch is None')
        return
    
    # batch parameter format convert
    param_batch = [dict(zip(batch.keys(), r), **param) for r in zip(*batch.values())]
    
    stat_batch, df_batch = [], []
    for param in param_batch:
        stat = cell.format(**param)
        if debug:
            print (stat)
        stat_batch.append(stat)
        df_batch.append(spark.sql(stat))
    return stat_batch, reduce(lambda a, b: a.unionAll(b), df_batch)

@register_line_cell_magic
def hive_batch(line, cell):
    stat, df = _hive_batch(line, cell)
    return df

@register_line_cell_magic
def hive_batch_debug(line, cell):
    stat, df = _hive_batch(line, cell, debug=True)
    return df

def view(_df, _columns=None, step=10):
    @interact
    def _plot(i=widgets.IntSlider(0,0,len(_df),step)):
        if _columns:
            return _df[i:i+step][_columns].T
        return _df[i:i+step].T

from operator import itemgetter
from itertools import groupby

def native_group_by(dict_lst, key):
    dict_lst.sort(key=key)
    grouped = groupby(dict_lst, key)
    return grouped

pd.options.display.float_format = '{:,.5f}'.format
