import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy
from einops import repeat

from layer import mytransformerencoder
from layer import GRNlayer

def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


def global_max_pool(x):
    a = x.max(dim=2)

    return a


def global_avg_pool(x):
    a = x.sum(dim=2) / 100

    return a


class AR(nn.Module):
    def __init__(self, window, n_multiv):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)  # （120,1）
        self.out_linear = nn.Linear(n_multiv, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        x = self.out_linear(x)
        x = torch.squeeze(x, 2)
        return x


class ModelB(nn.Module):
    def __init__(self, args):
        """回归+分类模型结构

        Arguments:
        ----------
            input_dim {int} -- 输入特征的维度
            hidden_dim {int} -- 隐藏层单元数

        Keyword Arguments:
        ----------
            num_classes {int} -- 分类类别数 (default: {2})
        """
        super(ModelB, self).__init__()
        self.args = args
        self.b = args.batch
        self.num_leafs = args.num_groups

        self.ar = AR(window=args.input_length, n_multiv=args.input_dim)

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2)
        self.transformer = mytransformerencoder.Encoder(args.input_length)
        self.GRN3 = GRNlayer.VariableSelectionNetwork(32, 26, 26 * 16, 0.2)
        self.relu = nn.ReLU()  # 模块的激活函数
        
        self.cls_linear = nn.Linear(416, args.hidden_dim)
        self.TCPConfidenceLayer = nn.Linear(args.hidden_dim, 1)
        self.TCPClassifierLayer = nn.Linear(args.hidden_dim, self.num_leafs)

        self.final_cls = nn.Linear(args.hidden_dim, self.num_leafs)
        self.final_reg = nn.Linear(416, self.num_leafs)

    def forward(self, x, group=None, infer=False):
        x = x.float().cuda()
        x_raw = torch.transpose(x.reshape(self.b, self.args.input_length, -1), 2, 1)    # Input
        y = self.conv1d(x_raw)
        y = torch.transpose(y, 2, 1)
        trans = self.transformer(y, y.T)
        grn, rr3 = self.GRN3(trans[0].reshape(self.b, -1))
        
        feature_cls = self.cls_linear(grn)
        TCPLogit = self.TCPClassifierLayer(feature_cls)
        TCPConfidence = self.TCPConfidenceLayer(feature_cls)
        feature_cls = TCPConfidence * feature_cls
        classification = self.final_cls(feature_cls)

        cls_regression = self.final_reg(grn)
        cls_regression = self.relu(cls_regression)

        return cls_regression, TCPConfidence, TCPLogit, classification  # delta，分类

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit
