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


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = 0):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output


class AddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = None,
            skip_size: int = None,
            trainable_add: bool = False,
            dropout: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.2,
            context_size: int = None,
            residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        # x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context
        print('gate:', self.num_inputs, self.input_size, self.hidden_size)

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout, self.context)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout)

        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = F.softmax(sparse_weights, dim=1).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(
                self.single_variable_grns[i](embedding[:, (i * self.input_size): (i + 1) * self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=1)

        outputs = var_outputs * sparse_weights

        outputs = outputs.sum(axis=1).squeeze()

        return outputs, sparse_weights
