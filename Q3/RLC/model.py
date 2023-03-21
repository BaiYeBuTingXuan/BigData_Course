from turtle import forward
from typing import Tuple
from xml.dom import INDEX_SIZE_ERR
from xmlrpc.server import MultiPathXMLRPCServer
import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


class LSTMCell(nn.Module):
    """
    Input:3D Tensor: batch_size*sequence_length*feature_num (250*128*9)
    Output:3D Tensor: batch_size*sequence_length*feature_num (250*128*9)
    """

    def __init__(self, input_feature=8, hidden_feature=8, num_layers=1, dropout=0.5) -> None:
        super(LSTMCell, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=input_feature)
        self.LSTM = nn.LSTM(input_size=input_feature, hidden_size=hidden_feature, num_layers=num_layers, bias=True,
                            dropout=dropout, batch_first=True)
        
    def forward(self, x):
        x = x.float()
        x = self.norm(x)
        x, (_, _) = self.LSTM(x)
        x = F.tanh(x)
        return x


class TransformerCell(nn.Module):
    def __init__(self, out_channel=1, sequence_length=128) -> None:
        super(TransformerCell, self).__init__()
        self.out_channel = out_channel
        self.transformer_model = nn.Transformer(batch_first=True, d_model=sequence_length)

    def forward(self, x):
        batch_size, features, time_steps = x.size()
        target = torch.zeros([batch_size, self.out_channel, time_steps])
        x = self.transformer_model(x, target)
        return x


class BiClassifier(nn.Module):
    """
    Input:3D Tensor: batch_size*feature_num*sequence_length (250*8*128)
    Output:3D Tensor: batch_size*feature_num*sequence_length (250*1*2)
    """
    def __init__(self, in_channel=8, out_channel=2, dropout=0.5) -> None:
        super(BiClassifier, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=256, padding=1, stride=2, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, padding=1, stride=2, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, padding=1, stride=2, kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=64, padding=1, stride=2, kernel_size=4)
        
        self.norm = nn.LayerNorm(normalized_shape=8*64)

        self.mlp = nn.Sequential(
                            nn.Linear(in_features=8*64, out_features=64),
                            nn.Sigmoid(),
                            nn.Linear(in_features=64, out_features=out_channel),
                            nn.Softmax(dim=1)
                    )

    def forward(self, x):

        x = F.dropout(x, p=self.dropout)

        x = x.float()
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = F.leaky_relu(x)

        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1)

        x = self.norm(x)
        x = self.mlp(x)
        
        return x


class LSTMBiClassifier(nn.Module):
    def __init__(self, in_channel=8) -> None:
        super(LSTMBiClassifier, self).__init__()
        self.LSTM = LSTMCell(input_feature=in_channel, hidden_feature=in_channel, num_layers=16, dropout=0.5)
        self.classifier = BiClassifier(in_channel=in_channel)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.LSTM(x)
        x = x.transpose(1, 2)
        x = self.classifier(x)
        return x


class TransformerBiClassifier(nn.Module):
    def __init__(self, mid_channel=1, sequence_length=128) -> None:
        super(TransformerBiClassifier, self).__init__()
        self.Transformer = TransformerCell(out_channel=mid_channel, sequence_length=sequence_length)
        self.classifier = BiClassifier(in_channel=mid_channel)

    def forward(self, x):
        x = self.Transformer(x)
        # x = x.transpose(1, 2)
        x = self.classifier(x)
        return x


class LSTMResBiClassifier(nn.Module):
    def __init__(self, in_channel=8) -> None:
        super(LSTMResBiClassifier, self).__init__()
        self.LSTM = LSTMCell(input_feature=in_channel, hidden_feature=in_channel, num_layers=16, dropout=0.5)
        self.classifier = BiClassifier(in_channel=in_channel)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x + self.LSTM(x)

        x = x.transpose(1, 2)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = LSTMResBiClassifier()
    r = torch.rand([5, 8, 128])
    print(r.size())
    u = model(r)
    print(u.size())
    print(u)
