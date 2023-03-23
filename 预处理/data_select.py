from hmac import new
import itertools
from msilib import PID_TITLE    
import os
from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def eval_metrics_per_user(data,user_id_list:list,metrics_list:list,evaluation:str = 'len')->dict:
    """ 
    度量对每个用户每个指标的质量，度量方式由evaluation决定
    Input：
        data(pandas.DataFrame)：数据表，索引为['user_id','metrics']
        user_id_list,metrics_list(List)：索引列表
        evaluation(str)：评价指标,可选len,mean,var,median;default = len
    Output: 
        eval_dict(dict):每个用户每个指标的质量（除极差ptp确保无量纲）
    """
    eval_dict = {} # 评价字典
    for (user_id,metric) in itertools.product(user_id_list,metrics_list):
        try:
            # TODO:尝试divide by np.divide()
            l = list(data.loc[user_id,metric]['value_avg']) # 分析的数据：确定user和metric的时间序列数据
            if np.ptp(l) == 0 or l == []:
                # print("无效指标:"+user_id+' '+metric)
                eval_dict[(user_id,metric)] = 0
            else:
                if evaluation == 'len': # 评价函数为数据量
                    eval_dict[(user_id,metric)] = len(l)
                elif evaluation == 'mean': # 评价函数为平均值
                    eval_dict[(user_id,metric)] = np.mean(l)/np.ptp(l)
                elif evaluation == 'var': # 评价函数为标准差
                    eval_dict[(user_id,metric)] = sqrt(np.var(l))/np.ptp(l)
                elif evaluation == 'median': # 评价函数为中位数
                    eval_dict[(user_id,metric)] = np.median(l)/np.ptp(l)
                elif evaluation == 'highquarter': # 评价函数为上四分卫位 尚未调试
                    eval_dict[(user_id,metric)] = np.quantile(l,0.75,interpolation='higher')/np.ptp(l)
                elif evaluation == 'lowquarter': # 评价函数为下四分卫位 尚未调试
                    eval_dict[(user_id,metric)] = np.quantile(l,0.25,interpolation='lower')/np.ptp(l)
                else:
                    print('Unknown evaluation:',evaluation)
                    return {}
        except (KeyError): # 没有这样的索引，说明这个用户的这个指标的一切都是0
            eval_dict[(user_id,metric)] = 0 
    return eval_dict.copy()


def eval_metrics(metric_eval_dict:dict,user_id_list:list,metrics_list:list,evaluation:str = 'mean')->list:
    """
    评价并且筛选指标，评价方式由evaluation决定，筛选门槛由threshold_method
    Input：
        metric_eval_dict(dict)：数据字典，索引为(user_id,metric)，值为每个用户每个指标的质量
        user_id_list,metrics_list(List)：索引列表
        evaluation(str)：评价方式,可选mean,median;default = len
        threshold_method(str)：入围门槛,可选mean,median;default = mean
        not_less_zero(bool):是否排除0 计算各种指标时，是否考虑零和负数的干扰？
    Output: 
        valuable_metrics(list):筛选后入围的指标列表
    Attention:
        1. 必须确保字典metric_eval_dict对每个索引(user_id,metric)都有值，或者要先运行函数eval_metrics_per_user()
        2. 当evaluation:str = 'mean'时，metric_eval_dict对每个指标每个用户的评估是数据长度
    """
    metric_eval = {}
    for metric in metrics_list:
        l = [metric_eval_dict[(user_id,metric)] for user_id in user_id_list]
        if(len(l)==0):
            metric_eval[metric] = 0
        else:
            if evaluation == 'mean': # 评价函数为数据长度的均值
                metric_eval[metric] = np.mean(l)
            elif evaluation == 'median': # 评价函数为数据长度的中位数
                metric_eval[metric] = np.median(l)
            elif evaluation == 'num': # 评价函数为指标覆盖用户数
                metric_eval[metric] = len(l) # np.sum([int(metric_eval_dict[(user_id,metric)]>0) for user_id in user_id_list])
            elif evaluation == 'highquarter': # 评价函数为上四分位数
                metric_eval[metric] = np.quantile(l,0.75,interpolation='higher')
            elif evaluation == 'lowquarter': # 评价函数为下四分位数
                metric_eval[metric] = np.quantile(l,0.25,interpolation='lower')   
            else:
                print('Unknown evaluation:',evaluation)
                return []
    return metric_eval.copy()
                
def select_metric(metric_eval:dict, threshold_method:str = 'mean',not_less_zero:bool = False,larger:bool = True):
    l = list(metric_eval.values())
    if not_less_zero:
        l = [item for item in l if item > 0]
    if threshold_method == 'mean': # 门槛确定方式为均值
        threshold_value = np.mean(l)
    elif threshold_method == 'median': # # 门槛确定方式为中位数
        threshold_value = np.median(l)
    elif threshold_method == 'highquarter': # # 门槛确定方式为上四分位数
        threshold_value = np.quantile(l,0.75,interpolation='higher')
    elif threshold_method == 'lowquarter': # # 门槛确定方式为下四分位数
        threshold_value = np.quantile(l,0.25,interpolation='lower')
    else:
        print('Unknown threshold_method:',threshold_method)
        return set()
    if larger:
        valuable_metrics = {metric for metric in metrics_list if metric_eval[metric] >= threshold_value}
    else:
        valuable_metrics = {metric for metric in metrics_list if metric_eval[metric] <= threshold_value}
    # print(valuable_metrics)
    return valuable_metrics.copy()

if __name__ == '__main__':
    # 1 加载数据集
    df1 = pd.read_csv("1_pred.csv")
    df2 = pd.read_csv("3_pred.csv")
    data=pd.concat([df1,df2])
    user_id_list = list(set(data['user_id'].tolist())) # 获得全部的用户id和全部的指标类型
    metrics_list = list(set(data['metrics'].tolist()))
    print("加载完毕 \n 有用户{0}名，指标共{1}类".format(len(user_id_list),len(metrics_list)))
    # print(metrics_list)
    for col in ['unit']:
        del data[col] 
    data = data.set_index(['user_id','metrics']) # 双索引模式
    data = data.sort_index() # 整理索引 防止索引乱序

    # 这一段跑很慢 所以我跑完就存到文件里，后面注释掉
    print("分别计算{1}类指标在{0}名用户的分布情况...".format(len(user_id_list),len(metrics_list)))
    data_len_dict = eval_metrics_per_user(data=data,user_id_list=user_id_list,metrics_list=metrics_list,evaluation='len')
    data_var_dict = eval_metrics_per_user(data=data,user_id_list=user_id_list,metrics_list=metrics_list,evaluation='var')
   
    with open( './data_len_dict.txt', 'w',encoding = 'utf-8' ) as f1,open( './data_var_dict.txt', 'w',encoding = 'utf-8' ) as f2:
        print(data_len_dict,file=f1)       
        print(data_var_dict,file=f2)
    print("计算完成")

    #2 找出 valuable_metrics
    with open( './data_len_dict.txt','r',encoding = 'utf-8' ) as data_len_file,open( './data_var_dict.txt', 'r',encoding = 'utf-8' ) as data_var_file:
        data_len_dict,data_var_dict = eval(data_len_file.read()),eval(data_var_file.read())
        # 三个集合：时间方向上长度够长的、方差够大的、覆盖用户够广的
        # TODO:最小值最大值说不定可以使用
        print("计算{0}类指标的统计价值...".format(len(metrics_list)))
        eval_metrics_from_len = eval_metrics(metric_eval_dict=data_len_dict,user_id_list=user_id_list,metrics_list=metrics_list,evaluation = 'mean')
        eval_metrics_from_var = eval_metrics(metric_eval_dict=data_var_dict,user_id_list=user_id_list,metrics_list=metrics_list,evaluation = 'mean')
        eval_metrics_from_num = eval_metrics(metric_eval_dict=data_len_dict,user_id_list=user_id_list,metrics_list=metrics_list,evaluation = 'num')
        print("筛选指标...")
        valuable_metrics_from_len = select_metric(metric_eval=eval_metrics_from_len,threshold_method = 'mean',not_less_zero = False,larger = True)
        valuable_metrics_from_var = select_metric(metric_eval=eval_metrics_from_var,threshold_method = 'mean',not_less_zero = False,larger = True)
        valuable_metrics_from_num = select_metric(metric_eval=eval_metrics_from_var,threshold_method = 'mean',not_less_zero = False,larger = True)
        
        # 三个集合取交集
        valuable_metrics = valuable_metrics_from_len & valuable_metrics_from_var & valuable_metrics_from_num
        print("筛选得到{0}类指标:".format(len(valuable_metrics)))
    
    # 3 找出 valuable_user
    valuable_metrics_len = {} # 先确定每个用户的valuable_metrics长度
    for (user_id,metric) in itertools.product(user_id_list,valuable_metrics):
        valuable_metrics_len[(user_id,metric)] = data_len_dict[(user_id,metric)]
    valuable_metrics_len_avg = {} # 再确定每个valuable_metrics在用户上的平均长度（或者中位数长度）
    for metric in valuable_metrics:
       valuable_metrics_len_avg[metric] = np.median([data_len_dict[(user_id,metric)] for user_id in user_id_list])

    valuable_user = [] # # 最后，如果某用户的valuable_metrics不小于用户在这个valuable_metrics上的平均（中位）长度，这个用户就是有价值的
    for (user_id,metric) in itertools.product(user_id_list,valuable_metrics):
        if valuable_metrics_len[(user_id,metric)] >= valuable_metrics_len_avg[metric]:
            valuable_user.append(user_id)
    valuable_user = set(valuable_user)

    # 4 记录
    print('valuable metrics:',valuable_metrics)
    print(str(len(valuable_metrics))+' in total')
    print('valuable users:',valuable_user)
    print(str(len(valuable_user))+' in total')
    with open( "./valuable.txt", 'w',encoding = 'utf-8' ) as f:
        print(valuable_metrics,",",valuable_user,file=f)
    
    new_data = data.copy()
    for (user_id,metric) in itertools.product(user_id_list,metrics_list):
        if metric not in valuable_metrics:
            new_data.drop([user_id,metric],axis=0)
    new_data.to_csv("./Merge.csv")
    # load:

    # with open( "./valuable.txt", 'r',encoding = 'utf-8' ) as f:
    #     pvaluable_metrics,valuable_user=eval(f.read())

    # 5 画图 对每个valuable_user，每个valuable_metrics，画图
    print("作图...")
    img_path = "./img/"
    for (user_id,metric) in itertools.product(user_id_list,valuable_metrics):
        try:
            df = data.loc[user_id,metric] # 可能引发没有键值错误：KeyError
            df.plot(x='ds',y='value_avg') # 画图
            plt.title(metric+" of "+user_id+",amount:"+str(len(df['value_avg'])))
            plt.xticks(rotation=20)
            plt.xlabel("date("+str(len(df['value_avg']))+")")
            plt.ylabel("value of "+metric)
            plt.savefig(img_path+metric+"/"+user_id+'.png') # 可能引发没有路径错误：FileNotFoundError
            plt.close() # 不关画图的话，图一直在内存里，容易移除
        except FileNotFoundError:
            os.makedirs(img_path+metric)
            plt.savefig(img_path+metric+"/"+user_id+'.png') # 建完路径后，重新保存图片
            plt.close() # 不关画图的话，图一直在内存里，会内存溢出
            print("Create the direction:"+img_path+metric+'/')
        except KeyError:
            with open( img_path+"log.txt", 'w+',encoding = 'utf-8' ) as f:
                print(user_id+" DONT have the metric of "+metric,file = f)
    print("程序执行完毕")

