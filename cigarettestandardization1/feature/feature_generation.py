#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   feature_generation.py
@Time    :   2023/01/05 11:33:46
@Author  :   TongYao 
@Version :   1.0
@Site    :   Lenovo_SSG
@Desc    :   None
'''

import pandas as pd
import numpy as np
from config.feature_config import date_col


def time_feature(data, n, feature):
    df = data.sort_values(date_col).reset_index(drop=True)

    res = pd.DataFrame()

    for i in range(n, len(df)):

        tmp = df[df.index.isin(list(range(i-n,i)))].sort_values(date_col)
        tmp = tmp[[date_col,feature]].set_index(date_col).T

        tmp.columns = [str(i)+'_'+feature for i in range(1,n+1)]

        tmp[date_col] = df[df.index==i][date_col].values
        tmp = tmp.reset_index(drop=True)

        res = pd.concat([res,tmp])

    return res.reset_index(drop=True)


def diff_feature(data, feature):
    df = data.sort_values(date_col).reset_index(drop=True)

    df[feature+'_变化率'] = df[feature].diff()
    return df
