# 特征数据类型
data_type = '.csv'
# 数据合并方式
concat_type = 'append'

# 时间变量
date_col = '时间'
# 目标值
target_col = '出口物料含水率%_实际值'

# 非数值型变量
category_cols = ['时间','批次id']
# 数值变量
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# 业务忽略的变量
ignore_cols = ['设备运行阶段']

# 必须保留的变量
retain_cols = ['入口物料含水率%_实际值']

