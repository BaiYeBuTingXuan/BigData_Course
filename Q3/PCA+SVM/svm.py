import numpy as np
from sklearn.datasets._base import Bunch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#导入数据
data = np.load('pca.npy', allow_pickle=True)
label = np.load('label.npy', allow_pickle=True)
label = np.reshape(label,(-1,1))

#划分训练集和测试集
train_data, test_data, train_label, test_label = \
    train_test_split(np.array(data), np.array(label),
                     random_state=1, train_size=0.75, test_size=0.25)

# 训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=0.001,
                     decision_function_shape='ovr')  # ovr:一对多策略
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

# 使用题目的评价方法评估模型性能
model_res = classifier.predict(data) 
print('准确率为：', 
accuracy_score(label,model_res))      
print('精确率为：',
      precision_score(label,model_res))
print('召回率为：',
      recall_score(label,model_res))
print('F1值为：',
      f1_score(label,model_res))