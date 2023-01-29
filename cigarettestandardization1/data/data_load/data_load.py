#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   data_load.py
@Time    :   2022/12/28 11:50:22
@Author  :   TongYao 
@Version :   1.0
@Site    :   Lenovo_SSG
@Desc    :   None
'''

import traceback
import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

def get_csv_data(path, table_name):
    '''
    获取csv文件
    '''
    path = os.path.join(path, table_name)
    try:
        df = pd.read_csv(path)
        return df
    except Exception as ex:
        traceback.print_exc()
        print("exception in get {0} {1} data, ex is {2}".format(path, table_name, ex))



def get_excel_data(path, table_name):
    '''
    同一个excel文件, 不同sheet的表名和数据格式要相同
    
    '''
    path = os.path.join(path, table_name)
    try:
        data = pd.read_excel(path, sheet_name=None)
        df = pd.DataFrame()
        for sheet_name in data:
            sheet_df = data[sheet_name]
            df = pd.concat([df,sheet_df],ignore_index=True)
        return df
    except Exception as ex:
        traceback.print_exc()
        print("exception in get {0} {1} data, ex is {2}".format(path, table_name, ex))


def concat_data(path, data_type, concat_type):
    
    os.chdir(path)    
    file_chdir = os.getcwd()

    data_list = []
    df = pd.DataFrame()
    for root, dirs, files in os.walk(file_chdir):
        for file in files:
            if os.path.splitext(file)[1] == data_type:
                data_list.append(file)

                if data_type in '.csv':
                    data = get_csv_data(root, file)

                if data_type in ['.xls',  '.xlsx']:
                    data = get_excel_data(root, file)

                if concat_type == 'append':
                    df = df.append(data)
    return df