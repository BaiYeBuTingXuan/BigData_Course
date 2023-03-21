1. 文件说明
- data.npy,label.npy:数据集
- model.py:模型框架
  - LSTMCell:封装的LSTM的类
    - x = LayerNorm(x),层归一化
    - x = LSTM(x),通过LSTM，num_layers = 4
    - x = tanh(x),感知
  - BiClassifier:CONV+MLP二分类网络
    - Conv+LeakyReLU *4
    - Linear1 融合特征
    - Sigmoid 
    - Linear2+Softmax输出
  - LSTMBiClassifier:LSTMCell + BiClassifier(淘汰)
  - LSTMResBiClassifier:LSTMCell + ResNet + BiClassifier(最终模型)
    - LSTM提取时序长期变化的特征+x原始特征
    - Conv将时序维度信息转化为特征维度的信息
    - MLP融合特征，实验证明两成最好，Sigmoid可以弱化ReLU的梯度消失
  - TransformerBiClassifier:好像不咋地
- Dataset.npy:数据集调用
- train.py:训练相关函数
- test.py:测试 model.pth，输出相关指标
- model.pth:训练好的模型
- eventxxx.0:tensorboard 记录
2. LSTMResBiClassifier模型结构
   x(B * 8 * 128) --> x + LSTMCell(x)(B * 8 * 128)--Conv1+leakReLU-->(B * 256 * 64)--Conv2+leakReLU-->(B * 512 * 32)--Conv3+leakReLU-->(B * 512 * 16)--Conv4+leakReLU-->(B * 64 * 8)--Reshape-->(B * 512)-Linear->(B * 64)-Sigmoid->--Linear-->(B * 2)--Softmax(dim=1)-->Out = (B * 2)
3. 操作流程
   1. 训练模型
    ```bash
      python train.py
    ```
    2. 检测模型成果
    ```bash
      python test.py
    ```
    若检测的不是./model.py，则需要修改文件路径