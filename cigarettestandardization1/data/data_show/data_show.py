#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   data_show.py
@Time    :   2022/12/28 15:37:35
@Author  :   TongYao 
@Version :   1.0
@Site    :   Lenovo_SSG
@Desc    :   None
'''
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform


def get_mode(num):
    '''
    众数
    '''
    return stats.mode(num)[0][0]

def get_median(num):
    '''
    中位数
    '''
    return np.median(num)

def get_mean(num):
    '''
    平均数
    '''
    return np.mean(num)

def get_variation_ratio(num):
    '''
    异众比率
    '''
    return 1 - stats.mode(num)[1][0] / len(num)

def get_quartile_deviation(num):
    '''
    四分位差
    '''
    return np.percentile(num, 75) - np.percentile(num, 25)

def get_range(num):
    '''
    极差
    '''
    return np.max(num) - np.min(num)

def get_standard_deviation(num):
    '''
    标准差
    '''
    return np.std(num)

def get_variance(num):
    '''
    方差
    '''
    return np.var(num)

def get_coefficient_of_variation(num):
    '''
    离散系数
    '''
    return np.std(num) / np.mean(num)

def get_skewness(num):
    '''
    偏度
    '''
    return stats.skew(num)

def get_kurtosis(num):
    '''
    峰度
    '''
    return stats.kurtosis(num)


def get_variable_statistics(df, col):
    '''
    单变量的简单描述性统计
    '''
    descriptive_statistics = {}

    descriptive_statistics['众数'] = get_mode(df[col])
    descriptive_statistics['中位数'] = get_median(df[col])
    descriptive_statistics['平均数'] = get_mean(df[col])
    descriptive_statistics['异众比率'] = get_variation_ratio(df[col])
    descriptive_statistics['四分位差'] = get_quartile_deviation(df[col])
    descriptive_statistics['极差'] = get_range(df[col])
    descriptive_statistics['方差'] = get_variance(df[col])
    descriptive_statistics['标准差'] = get_standard_deviation(df[col])
    descriptive_statistics['离散系数'] = get_coefficient_of_variation(df[col])
    descriptive_statistics['偏态系数'] = get_skewness(df[col])
    descriptive_statistics['峰态系数'] = get_kurtosis(df[col])

    return pd.DataFrame([descriptive_statistics])


def get_data_statistics(data):
    '''
    多变量的简单描述性统计
    '''
    return data.describe()


def percent_missing(data):
    '''
    数据缺失情况统计
    '''
    df = pd.DataFrame(data)
    df_cols = list(df.columns)
    miss = {}
    for i in range(0, len(df_cols)):
        miss.update({df_cols[i]: round(df[df_cols[i]].isnull().mean()*100,2)})   
    data_miss = sorted(miss.items(), key=lambda x: x[1], reverse=True)
    return data_miss


def distcorr(X, Y):
    '''
    距离相关系数
    '''
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor