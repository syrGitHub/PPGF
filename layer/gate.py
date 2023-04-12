import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
seed = random.randint(1, 200)
seed = 1990
np.random.seed(seed)
torch.manual_seed(seed)
def expand2(a):
    tmp = []
    for i in range(a.size()[1]):
        tmp.append(a[:, i:i + 1])
        tmp.append(a[:, i:i + 1])
    b = torch.stack(tmp, dim=1).squeeze()
    return b

def expand4(a):
    tmp = []
    for i in range(a.size()[1]):
        tmp.append(a[:, i:i + 1])
        tmp.append(a[:, i:i + 1])
        tmp.append(a[:, i:i + 1])
        tmp.append(a[:, i:i + 1])
    b = torch.stack(tmp, dim=1).squeeze()
    return b

class Gate3(nn.Module):
    def __init__(
        self,
        input_size: int,

        dropout: float = 0.2,

    ):
        super().__init__()
        self.input_size = input_size
        self.liner=nn.Linear(input_size,input_size)
        self.dropout = dropout





    def forward(self, x1,x2,x3):
        list1=[]
        list2=[]
        list3=[]
        x1=expand4(x1)
        x2=expand2(x2)
        for i in range(16):
            r1=torch.mul(F.sigmoid(self.liner(x1[:,i:i+1])) , x1[:,i:i+1])
            r2=torch.mul(F.sigmoid(self.liner(x2[:, i:i + 1])), x2[:, i:i + 1])
            r3=torch.mul(F.sigmoid(self.liner(x3[:, i:i + 1])) , x3[:, i:i + 1])
            r=r1+r2+r3
            list3.append(r)

        rr=torch.stack(list3,dim=1)

        return rr



# a=torch.tensor([[[3,3],[2,2],[1,1]],[[4,4],[5,5],[6,6]]])
#
# print(1)