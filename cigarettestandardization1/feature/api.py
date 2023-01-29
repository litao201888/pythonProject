#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   api.py
@Time    :   2023/01/09 13:51:37
@Author  :   TongYao 
@Version :   1.0
@Site    :   Lenovo_SSG
@Desc    :   None
'''

import os
import glob
import pandas as pd
import numpy as np
import json
from config.feature_config import *
from config.function_setting import *
from config.setting import root_path, train_data_path, predict_data_path, output_data_path
from data.data_load.data_load import concat_data
from data.data_clean.data_clean import del_dup_samples, del_dup_columns, clean_not_number, clean_extreme_data
from feature.feature_select.feature_selector import FeatureSelector
from feature.feature_generation import time_feature, diff_feature


class GetSample(object):

    def __init__(self, is_train=True):

        self.is_train = is_train


        self._get_original_data()

        if del_dup_data:
            self._del_dup_data()

        if del_invalid_sample:
            self._del_invalid_sample()

        if del_str_data:
            self._del_str_data()

        if del_extreme_data:
            self._del_extreme_data()

        if explore_data:
            self._explore_data()
        
        if time_features:
            self._time_features()

        if statistics_features:
            self._statistics_features()

        if fillna_missing:
            self._fillna_missing()

        if remove_features:
            self._remove_features()

        

    def _get_original_data(self):
        if self.is_train:
            path = os.path.join(root_path, train_data_path)
        else:
            path = os.path.join(root_path, predict_data_path)

        self.data = concat_data(path, data_type, concat_type)
        self.original_data = self.data.copy()
        print(self.data.shape)

    def _del_dup_data(self):

        self.data = del_dup_columns(del_dup_samples(self.data))
  
    
    def _del_invalid_sample(self):

        self.data = self.data[self.data[state_col].isin(state_values)]


    def _del_str_data(self):

        self.data = clean_not_number(self.data, str_col)


    def _del_extreme_data(self):

        self.data = clean_extreme_data(self.data, extreme_col)


    def _explore_data(self):
        if self.is_train:
            fs = FeatureSelector(data = self.data.drop(columns=[target_col]))

            fs.identify_missing(missing_threshold = missing_threshold)
            fs.identify_collinear(correlation_threshold = correlation_threshold)
            self.missing_features = fs.ops['missing']
            self.collinear_features = fs.ops['collinear']

            with open(os.path.join(root_path, output_data_path)+'/encoding_map.json','w') as f:
                json.dump(fs.ops,f)
        else:
            with open(os.path.join(root_path, output_data_path)+ '/encoding_map.json', 'r') as f:
                self.encoding_map = json.load(f)
                self.missing_features = self.encoding_map['missing']
                self.collinear_features = self.encoding_map['collinear']  

    
    def _time_features(self):

        if time_feature_cols == []:

            cols = [i for i in list(self.data.columns) if i not in category_cols]
            for col in cols:
                if self.data[col].dtypes in numeric_dtypes:
                    self.data = self.data.merge(time_feature(self.data, n_shift, col), on=[date_col], how='left')
        
        else:

            for col in time_feature_cols:
                if self.data[col].dtypes in numeric_dtypes:
                    self.data = self.data.merge(time_feature(self.data, n_shift, col), on=[date_col], how='left')     

    
    def _statistics_features(self):

        if diff_feature_cols == []:
            cols = [i for i in list(self.data.columns) if i not in category_cols]
            for col in cols:
                if self.data[col].dtypes in numeric_dtypes:
                    self.data = diff_feature(self.data, col)
        
        else:
            for col in diff_feature_cols:
                if self.data[col].dtypes in numeric_dtypes:
                    self.data = diff_feature(self.data, col)  


    def _fillna_missing(self):

        for key in fillna_dic:
            if key in list(self.data.columns):
                self.data[key] = self.data[key].fillna(fillna_dic[key])
        
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')

    
    def _remove_features(self):

        invalid_features = list(set(self.missing_features + self.collinear_features + ignore_cols))
        del_features = [col for col in invalid_features if col not in retain_cols]
        
        self.data = self.data.drop(columns=del_features)