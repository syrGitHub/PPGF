import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)
from torch import nn
import torch
import torch.nn.functional as F

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True)

    def forward(self, input):
        # input应该为(batch_size,seq_len,input_szie)
        self.hidden = self.initHidden(input.size(0))
        out, self.hidden = self.lstm(input, self.hidden)
        return out, self.hidden

    def initHidden(self, batch_size):
        if self.lstm.bidirectional:
            return (torch.rand(self.num_layers * 2, batch_size, self.hidden_size).cuda(),
                    torch.rand(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        else:
            return (torch.rand(self.num_layers, batch_size, self.hidden_size).cuda(),
                    torch.rand(self.num_layers, batch_size, self.hidden_size).cuda())


class GruRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.GRU(input_size, hidden_size, num_layers)  # utilize the GRU model in torch.nn
  # 全连接层

    def forward(self, _x):
        x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # x = x.view(s * b, h)
        # x = self.linear1(x)
        # x = self.linear2(x)
        # x = x.view(s, b, -1)
        return x,_

