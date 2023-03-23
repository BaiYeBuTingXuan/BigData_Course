import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
import os
from math import sqrt
# 1 加载数据集
data = pd.read_csv('./1.csv')  # 附件1
print(data.info(null_counts=True))  # 观察缺漏情况
user_id_list = list(set(data['user_id'].tolist()))  # 获得全部的用户id和全部的指标类型
metrics_list = list(set(data['metrics'].tolist()))
data = data.sort_index()  # 整理索引 防止索引乱序

# 对于单位是非%的
# max<avg或min>avg的情况数据在总数据中具有10.5% 的占比，不宜删除。
# 对于单位是%的
for i in range(0, data.shape[0]):
    if ((data['unit'][i] == '%') & (data['value_avg'][i] > 100)) | (data['value_avg'][i] < 0):
        print(i)
        if data['metrics'][i] == data['metrics'][i-1]:
            if data['metrics'][i] == data['metrics'][i-2]:
                data['value_avg'][i] = 2*data['value_avg'][i-1] - \
                    data['value_avg'][i-2]  # 前向插值
            else:
                data['value_avg'][i] = data['value_avg'][i-1]
        else:
            data['value_avg'][i] = 2*data['value_avg'][i+1] - \
                data['value_avg'][i+2]  # 迫不得已就后向
        if(data['value_avg'][i] < 0):  # 如果插值不行
            data['value_avg'][i] = data['value_avg'][i+1]  # 直接替换
        print(data['value_avg'][i])

# 滤波函数
def np_move_avg(a, n, mode="same"):
    if n < 3:  # 长度为3的滑窗
        return(np.convolve(a, np.ones((n,))/n, mode=mode))
    else:  # 若序列长度小于3，则以其自身长度进行滑窗
        return(np.convolve(a, np.ones((3,))/3, mode=mode))

df = pd.DataFrame(columns=['user_id', 'metrics', 'unit', 'ds', 'value_avg'])
total_data = pd.DataFrame(
    columns=['user_id', 'metrics', 'unit', 'ds', 'value_avg'])
user_id_list = list(set(data['user_id'].tolist()))  # 获得全部的用户id和全部的指标类型
metrics_list = list(set(data['metrics'].tolist()))
for (user_id, metric) in itertools.product(user_id_list, metrics_list):  # 遍历数据
    try:
        d1 = np.array(data.loc[(data['user_id'] == user_id) & (
            data['metrics'] == metric), 'value_avg'])
        if d1.size != 0:  # 数据处理
            # 利用分位数的办法剔除极大值/极小值
            ql = np.percentile(d1, 25)
            qu = np.percentile(d1, 75)
            iqr = qu-ql
            len = d1.shape[0]
            mask = np.squeeze(np.argwhere(np.array(
                d1[1:-1] < ql-1.5*iqr) | np.array(d1[1:-1] > qu+1.5*iqr)))  # 剔除掉第一个值和最后一个值的超界数值的索引
            d1[mask+1] = (d1[mask]+d1[mask+2]/2)
            # 处理第一个和最后一个
            if (d1[0] < ql-1.5*iqr) | (d1[0] > qu+1.5*iqr):
                d1[0] = d1[1]
            if (d1[len-1] < ql-1.5*iqr) | (d1[len-1] > qu+1.5*iqr):
                d1[len-1] = d1[len-2]
            ds = list(data[data.user_id == user_id][data.metrics == metric].ds)
            unit = list(data[data.user_id == user_id]
                        [data.metrics == metric].unit)
            d1 = np_move_avg(d1, len)  # 滤波
            # 保存数据
            df['value_avg'] = d1
            df['user_id'] = user_id
            df['metrics'] = metric
            df['ds'] = ds
            df['unit'] = unit
            total_data = total_data.append(df)
            df.drop(df.index, inplace=True)
    except KeyError:
        print("error")
total_data.to_csv('./1_pred.csv')  # 存储数据
