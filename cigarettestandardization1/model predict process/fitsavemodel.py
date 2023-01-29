#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   fitsavemodel.py
@Time    :   2023/01/28
@Author  :   litao
@Version :   1.0
@Site    :   Lenovo_SSG
@Desc    :   None
'''

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# from cigarettestandardization1.config import e1fitsavemodel_setting as e1

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import e1fitsavemodel_setting as e1

def get_evaluating_indicator(y_real, y_pred, method='passrate', prrange=1):
    '''
    评价指标
    :param y_real:真实值或者设定值，dtype为列表
    :param y_pred:预测值或者实际值，dtype为列表
    :param method:评判方法，参数[passrate，accuracy，deviation ],passrate:合格率，accuracy:准确率，deviation：标准偏差，dtype为str
    :param prrange: 上下限范围，选择passrate参数时，需要指定改参数。dtype为int
    :return: 指标结果，dtype为浮点型
    '''
    import numpy as np

    if method == 'passrate':  # 合格率
        score = [1 if abs(i - j) <= prrange else 0 for i, j in zip(y_real, y_pred)]
        return round(sum(score) / len(score), 2)

    elif method == 'accuracy':  # 准确率
        score = [1 - abs(i - j) / i for i, j in zip(y_real, y_pred)]
        return round(np.average([i if i >= 0 else 0 for i in score]), 2)
    elif method == 'deviation':  # 标偏
        score = [abs(i - j) for i, j in zip(y_real, y_pred)]
        return round(np.average(score), 2)


def xgb_search_n_estimators(X_train, y_train,cv=2, n_jobs=-1, scoring='neg_mean_absolute_error'):
    '''寻找最佳的 n_estimators'''
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    xgbr = xgb.XGBRegressor()
    min_value, max_value = 10, 911
    step = 100

    while True:
        if (max_value - min_value < 10) or step == 1:
            break

        param_grid = {'n_estimators': np.arange(min_value, max_value, step=step, dtype=int)}
        gscv = GridSearchCV(xgbr, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
        gscv.fit(X_train, y_train)
        best_params_ = gscv.best_params_['n_estimators']

        if best_params_ > (max_value - step):
            max_value = max_value + 10 * step
            min_value = max(2, max_value - 1 * step)

        else:
            max_value = best_params_ + 1 * step
            min_value = max(2, best_params_ - 1 * step)
            step = max(1, int(step / 2))

    return best_params_


#  max_depth: 控制树结构的深度
def xgb_search_max_depth(X_train, y_train,n_estimators_, cv=2, n_jobs=-1, scoring='neg_mean_absolute_error'):
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    xgbr = xgb.XGBRegressor(
        n_estimators=n_estimators_
    )
    min_value, max_value = 2, 53
    step = 5

    while True:
        if (max_value - min_value < 3) or step == 1:
            break

        param_grid = {'max_depth': np.arange(min_value, max_value, step=step, dtype=int)}
        gscv = GridSearchCV(xgbr, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
        gscv.fit(X_train, y_train)
        best_params_ = gscv.best_params_['max_depth']

        if best_params_ > (max_value - step):
            max_value = max_value + 10 * step
            min_value = max(2, max_value - 1 * step)

        else:
            max_value = best_params_ + 1 * step
            min_value = max(2, best_params_ - 1 * step)
            step = max(1, int(step / 2))

    return best_params_


def xgb_search_gamma(X_train, y_train,n_estimators_, max_depth_, cv=2, n_jobs=-1, scoring='neg_mean_absolute_error'):
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    xgbr = xgb.XGBRegressor(
        n_estimators=n_estimators_,
        max_depth=max_depth_
    )
    min_value, max_value = 0.5, 1.01
    step = 0.1

    param_grid = {'gamma': np.arange(min_value, max_value, step=step, )}
    gscv = GridSearchCV(xgbr, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    gscv.fit(X_train, y_train)
    best_params_ = gscv.best_params_['gamma']

    return best_params_


def xgb_search_reg_alpha(X_train, y_train,n_estimators_, max_depth_, gamma_, cv=2, n_jobs=-1, scoring='neg_mean_absolute_error'):
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    xgbr = xgb.XGBRegressor(
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        gamma=gamma_
    )
    min_value, max_value = 0, 0.6
    step = 0.1

    param_grid = {'reg_alpha': np.arange(min_value, max_value, step=step, )}
    gscv = GridSearchCV(xgbr, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    gscv.fit(X_train, y_train)
    best_params_ = gscv.best_params_['reg_alpha']

    return best_params_


def xgb_search_reg_lambda(X_train, y_train,n_estimators_, max_depth_, gamma_, reg_alpha_, cv=2, n_jobs=-1,
                          scoring='neg_mean_absolute_error'):
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    xgbr = xgb.XGBRegressor(
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        gamma=gamma_,
        reg_alpha=reg_alpha_
    )
    min_value, max_value = 0, 0.6
    step = 0.1

    param_grid = {'reg_lambda': np.arange(min_value, max_value, step=step, )}
    gscv = GridSearchCV(xgbr, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    gscv.fit(X_train, y_train)
    best_params_ = gscv.best_params_['reg_lambda']

    return best_params_


def xgb_search_learning_rate(X_train, y_train,n_estimators_, max_depth_, gamma_, reg_alpha_, reg_lambda_, cv=2, n_jobs=-1,
                             scoring='neg_mean_absolute_error'):
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    xgbr = xgb.XGBRegressor(
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        gamma=gamma_,
        reg_alpha=reg_alpha_,
        reg_lambda=reg_lambda_
    )
    min_value, max_value = 0.1, 0.5
    step = 0.1

    param_grid = {'learning_rate': np.arange(min_value, max_value, step=step, )}
    gscv = GridSearchCV(xgbr, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    gscv.fit(X_train, y_train)
    best_params_ = gscv.best_params_['learning_rate']

    if best_params_ == 0.1:
        min_value = 0.01
        max_value = 0.11
    min_value = best_params_ - step
    max_value = best_params_ + step
    step = 0.01
    param_grid = {'learning_rate': np.arange(min_value, max_value, step=step, )}
    gscv = GridSearchCV(xgbr, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    gscv.fit(X_train, y_train)
    best_params_ = gscv.best_params_['learning_rate']

    return best_params_


def fit_model_xgboost(X_train, y_train, cv=2, n_jobs=-1, scoring='neg_mean_absolute_error', default_para=False):
    '''
    自动化训练xgb模型
    '''
    import xgboost as xgb

    if default_para:
        return xgb.XGBRegressor(n_jobs=-1).fit(X_train, y_train)

    else:
        n_estimators_ = xgb_search_n_estimators(X_train, y_train,cv=cv, n_jobs=n_jobs, scoring=scoring)
        max_depth_ = xgb_search_max_depth(X_train, y_train,n_estimators_, cv=cv, n_jobs=n_jobs, scoring=scoring)
        gamma_ = xgb_search_gamma(X_train, y_train,n_estimators_, max_depth_, cv=cv, n_jobs=n_jobs, scoring=scoring)
        reg_alpha_ = xgb_search_reg_alpha(X_train, y_train,n_estimators_, max_depth_, gamma_, cv=cv, n_jobs=n_jobs, scoring=scoring)
        reg_lambda_ = xgb_search_reg_lambda(X_train, y_train,n_estimators_, max_depth_, gamma_, reg_alpha_, cv=cv, n_jobs=n_jobs,
                                            scoring=scoring)
        learning_rate_ = xgb_search_learning_rate(X_train, y_train,n_estimators_, max_depth_, gamma_, reg_alpha_, reg_lambda_, cv=cv,
                                                  n_jobs=n_jobs, scoring=scoring)
        xgbr = xgb.XGBRegressor(
            n_estimators=n_estimators_,
            max_depth=max_depth_,
            gamma=gamma_,
            reg_alpha=reg_alpha_,
            reg_lambda=reg_lambda_,
            n_jobs=-1
        )
        return xgbr.fit(X_train, y_train)


def fit_model(X_train,y_train,cv=2, n_jobs=-1,scoring='neg_mean_absolute_error', default_para=False, modelname='XGBR'):
    '''
    训练模型
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param cv: k折交叉验证
    :param n_jobs: 进程数目
    :param scoring: 调参的评价指标
    :param default_para: 是否使用默认参数，如果True。则使用默认参数，不进行调参
    :param modelname: 模型名称,[XGBR, ]
    :return: 训练好的模型
    '''
    if modelname == 'XGBR':
        return fit_model_xgboost(X_train,y_train,cv=cv,n_jobs=n_jobs,scoring=scoring,default_para=default_para)


def save_model(model, name, path='./model/'):
    '''
    保存模型
    :param model: 训练好的模型
    :param name: 保存模型的名称
    :param path: 保存模型的路径
    :return: 模型保存地址：name+path+'.pkl'
    '''

    import os
    import joblib
    if not os.path.exists(path):
        os.makedirs(path)

    joblib.dump(model, '{}{}.pkl'.format(path, name))

    return '{}{}.pkl'.format(path, name)


def fit_save_model(X_train,y_train, cv=2, n_jobs=-1,scoring='neg_mean_absolute_error',default_para=True,modelname='XGBR',name='xgbr',
                   path='./model/'):
    '''
    训练模型并保存，返回模型保存地址
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param cv: 训练时候的交叉验证数
    :param n_jobs: 训练时候的进程数
    :param scoring: 训练时候的模型评价
    :param default_para: 是否使用默认参数，如果True。则使用默认参数，不进行调参
    :param modelname: 模型名称,[XGBR, ]
    :param name: 模型保存的名称
    :param path: 模型保存的路径
    :return: 型保存地址
    '''
    # 训练模型
    xgbr_model = fit_model(X_train, y_train, cv=cv, n_jobs=n_jobs, scoring=scoring,default_para=default_para,modelname=modelname)

    # 保存模型
    modelpath = save_model(xgbr_model, name=name, path=path)

    return modelpath

def my_train_test_split(x, y, test_size=0.2):
    '''
    跨期切分数据
    :param x: 切分的特征字段，dtype=DateFrame
    :param y: 切分的标签字段， dtype=DateFrame或者array
    :param test_size: 测试集的占比， dtype=float
    :return: 训练集-x，测试集-x，训练集-y，测试集-y
    '''
    test_size = int(test_size * len(y))
    return x.iloc[:-test_size], x.iloc[-test_size:], y.iloc[:-test_size], y.iloc[-test_size:]


if __name__ == '__main__':
    # ——————————————————提供批量——————————————
    usecols = [
        '533_Para2_Up', '533_Para119_Up', '533_Para19_Up', '533_Para32_Up', '533_Para134_Up',
        '533_Para101_Up', '533_Para103_Up', '533_Para129_Up', '533_Para117_Up',
        '533_Para181_Up', '533_Para114_Up', '533_Para132_Up', '533_Para120_Up',
    ]
    path = 'D:\\03project\\jupyter\\企业项目\\中国烟草\\烟草产品化\\code\\sampledata.csv'
    data = pd.read_csv(path)
    print(data.shape)
    data = data[usecols]
    print(data.shape)
    X_train, X_test, y_train, y_test = my_train_test_split(data.iloc[:, :10], data.iloc[:, 11])

    # ——————————————————训练模型的调用函数——————————————
    modelpath = fit_save_model(X_train, y_train, cv=e1.cv, n_jobs=e1.n_jobs, scoring='neg_mean_absolute_error', default_para=e1.default_para,
                               modelname='XGBR', name=e1.name,path=e1.path)
    print(modelpath)   # 返回模型保存的地址,path+name。例如：./model/xgbr.pkl