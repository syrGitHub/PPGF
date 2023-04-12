import os
import urllib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import math
import scipy.sparse as sp
from zipfile import ZipFile
from einops import rearrange, repeat
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch_scatter
import torch.optim as optim
import datetime
import random
import mytransformerencoder
class ModelB(nn.Module):
    def __init__(self, Timewindow):
        """一维CNN+Transformerencoder模型结构

        Arguments:
        ----------
            input_dim {int} -- 输入特征的维度
            hidden_dim {int} -- 隐藏层单元数
            Timewindow {int} -- 序列长度


        """
        super(ModelB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2)

        # self.cls_token_cas = nn.Parameter(torch.ones(1, 1, 32))
        # self.cls_token_ipt = nn.Parameter(torch.ones(1, 1, 32))

        self.trans_ipt = mytransformerencoder.Encoder(Timewindow)


    def forward(self,ipt):



        #ipt是输入序列
        ipt=torch.transpose(ipt.unsqueeze(dim=2).float().cuda(),2,1)
        ipt_con1d_out=self.conv1(ipt)
        ipt=torch.transpose(ipt_con1d_out,2,1)

        ipt_output=self.trans_ipt(ipt,ipt.sum(dim=2).T)
        #Tranformerencoder输出两部分 ipt_output[0]是encoder结果 ipt_output[1]是多头注意力权重值


        return ipt_output[0],ipt_output[1]
