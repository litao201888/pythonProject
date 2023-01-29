#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   make_sample_main.py
@Time    :   2023/01/11 17:21:36
@Author  :   wk
@Version :   1.0
@Site    :   Lenovo_SSG
@Desc    :   None
'''

import os
import glob
import warnings
from feature.api import GetSample
from config.setting import root_path, output_data_path
from config.function_setting import is_train

warnings.filterwarnings("ignore")

def run():
    #1.信息采集模块
    get_data = GetSample(is_train)
    data = get_data.data
    print("success get data")
    #2.数据探索模块
    #print('Step 2 end.')

    #3.数据清洗模块
    # print('Step 3 end.')

    #4.特征工程模块
    # print('Step 4 end.')

    #5.时滞模型
    # print('Step 5 end.')

    #6.状态识别模型
    # print('Step 6 end.')

    #7.BaseModel
    # print('Step 7 end.')

    #8.预测仿真模型
    # print('Step 8 end.')

    #9.智能控制模型
    # print('Step 9 end.')

    #the end.
    path = os.path.join(root_path, output_data_path)
    data.to_csv(path+'/data.csv', index=False, encoding='utf-8_sig')

if __name__ == "__main__":
    run()
