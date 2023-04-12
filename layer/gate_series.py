
import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
seed = random.randint(1, 200)
seed = 1990
np.random.seed(seed)
torch.manual_seed(seed)

class Gate3(nn.Module):
    def __init__(
        self,
        input_size: int,

        dropout: float = 0.2,

    ):
        super().__init__()
        self.input_size = input_size
        self.liner=nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(input_size):
            self.liner.append(nn.Linear(32, 1))

        self.norm = nn.LayerNorm(32)

        self.liner2 = nn.Linear(32, 32)








    def forward(self, x1):
        list1=[]
        list2=[]
        list3=[]
        res=[]
        res2=[]
        softres=[]
        for i in range(x1.size()[1] ):

            tmp1=F.sigmoid(self.dropout(self.liner[i](x1[:,i])))
            res.append(tmp1*(x1[:,i]))

            res2.append(tmp1)

        # sfm=F.softmax(torch.stack(softres,dim=1).squeeze())
        rr=torch.stack(res,dim=1)
        rr2=torch.stack(res2,dim=1)
        return rr,rr2

