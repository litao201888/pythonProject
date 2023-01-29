# 训练还是预测
is_train = True


# 删除重复数据
del_dup_data = True


# 删除无效数样本
del_invalid_sample = False
state_col = '生产状态' # 状态变量
state_values = [1,2]  # 保留的状态变量


# 删除数据中存在非数值型数据
del_str_data = True
str_col = [] # 需要检查字符数据的变量(为空的时候，则会检查全部变量)


# 删除极端数据
del_extreme_data = True
extreme_col = [] # 需要清洗异常值的变量(为空的时候，则会清洗全部变量)


# 数据缺失和相关性的探索
explore_data = True
missing_threshold = 0.6 # 空值的阈值
correlation_threshold = 0.9 # 相关的阈值


# 空值填充
fillna_missing = True
fillna_dic = {'入口物料含水率%_实际值':1}  # 空值填充(业务填充，为空的时候，则会使用前后值进行填充)


# 时间特征
time_features = True
n_shift = 5
time_feature_cols = ['入口物料含水率%_实际值']


# 统计特征
statistics_features = True
diff_feature_cols = ['入口物料含水率%_实际值']


# 删除无用特征（高缺失，高相关，业务不需要的）
remove_features = True
