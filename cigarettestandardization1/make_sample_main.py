#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   make_sample_main.py
@Time    :   2023/01/11 17:21:36
@Author  :   TongYao 
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
    get_data = GetSample(is_train)
    data = get_data.data
    print("success get data")

    path = os.path.join(root_path, output_data_path)
    data.to_csv(path+'/data.csv', index=False, encoding='utf-8_sig')

if __name__ == "__main__":
    run()
