#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   data_clean.py
@Time    :   2022/12/28 16:17:23
@Author  :   TongYao 
@Version :   1.0
@Site    :   Lenovo_SSG
@Desc    :   None
'''
import pandas as pd
import numpy as np
import re
from config.feature_config import category_cols, numeric_dtypes

def del_dup_samples(data, col=[]):
    '''
    删除重复样本
    '''
    if col == []:
        data.drop_duplicates(keep='first', inplace=True)
    else:
        data.drop_duplicates(subset=col, keep='first', inplace=True)
    return data


def del_dup_columns(data):
    '''
    删除重复字段
    '''
    data = data.loc[:,~data.columns.duplicated()]
    return data


def get_abnormal_data_by_IQR(data,clean_column,up_point_cc=1.5,low_point_cc=1.5):
    '''
    四分位法寻找极端值
    '''
    base_data = data

    up_quartile = base_data[clean_column].quantile(0.75)
    low_quartile = base_data[clean_column].quantile(0.25)
    iqr = up_quartile - low_quartile

    up_point = up_quartile + up_point_cc * iqr
    low_point = low_quartile - low_point_cc * iqr
    if low_point< 0:
        low_point = 0

    abnormal_data = base_data[(base_data[clean_column] > up_point) | (base_data[clean_column] < low_point) ]
    normal_data = base_data[(base_data[clean_column] <= up_point) & (base_data[clean_column] >= low_point) ]
    return abnormal_data,normal_data


def clean_extreme_data(data, feature):
    '''
    删除极端值
    '''
    df = data.reset_index(drop=True).copy()

    if feature != []:
        cols = feature
    else:
        cols = [i for i in list(df.columns) if i not in category_cols] 
    for col in cols:
        if df[col].dtypes not in numeric_dtypes:
            df[col] = df[col].astype(float)
        abnormal_data, normal_data = get_abnormal_data_by_IQR(df,clean_column=col)
        abnormal_index = []
        abnormal_index.extend(abnormal_data.index)    
        df.loc[df.index.isin(abnormal_index), col] = np.nan
    return df


def find_numeric_features(data,not_list=[]):
    '''
    查找数值型字段
    '''
    numeric = []
    for i in data.columns:
        if data[i].dtype in numeric_dtypes:
            if i in not_list:
                pass
            else:
                numeric.append(i)
    return numeric


def fine_object_features(data):
    '''
    查找字符型字段
    '''
    object_columns = data.select_dtypes(include='object').columns.to_list()
    return object_columns


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    return num if pattern.match(str(num)) else np.nan


def clean_not_number(data, feature):
    '''
    删除字符型数据
    '''
    df = data.copy()
    if feature != []:
        cols = feature
    else:
        cols = [i for i in list(df.columns) if i not in category_cols] 

    for col in cols:
        df[col] = df[col].apply(lambda x:is_number(x))
    return df


def trans_float_data(data):
    
    df = data.copy()
    cols = [i for i in list(df.columns) if i not in category_cols]

    for col in cols:
        if df[col].dtypes not in numeric_dtypes:
            df[col] = df[col].astype(float)
    return df

    