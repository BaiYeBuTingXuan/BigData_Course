import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
# 读取.npy文件
data = np.load('data.npy', allow_pickle=True)
print(data.shape)
flag=0
for user in range(data.shape[0]):
    temp = []
    for i in range(data[user].shape[1]):
        temp.append(data[user][:,i])    
    temp = np.array(temp).reshape(1,-1)
    if flag==0:
        total_data = temp
        flag=1
    else:
        total_data = np.vstack((total_data,temp))
# total_data: 432(用户)x1024(8x128,相邻8个处于同一时间点)

#动态PCA
pca = PCA(n_components=4)  # 先以特征降维
newX = pca.fit_transform(total_data)  # 归一化与变换
print(pca.explained_variance_ratio_)  # 观察各特征方差
pca_data = newX[:,0:3].reshape(-1,3)#决定用三个作为主成分

#存储数据
df = pd.DataFrame(pca_data) #构造dataframe表格
df.to_csv('./pca.csv')
