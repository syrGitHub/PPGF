from math import sqrt

import torch
import torch.nn as nn
import math
import random
seed = random.randint(1, 200)
seed = 1990
class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

class MultiHeadSelfAttention_graph(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention_graph, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


class PositionalEncoding10(nn.Module):

    def __init__(self, d_model=32, max_len=10):#50->60
        super(PositionalEncoding10, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(60,10,32)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm10(nn.Module):
    def __init__(self, feature_size=32, num_layers=1, dropout=0.3):
        super(TransAm10, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ipt_mask = None
        self.pos_encoder = PositionalEncoding10(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.transformer_encoder_ipt = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder = nn.Linear(64, 2)
        self.decoder_trans=nn.Linear(32,1)
        self.data_bn = nn.BatchNorm1d(10)
        self.data_bn.cuda()
        self.init_weights()
        self.src_key_padding_mask = None
        self.ipt_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding,ipt,ipt_padding):
        if self.src_key_padding_mask is None:
            mask_key = ~src_padding.bool()
            self.src_key_padding_mask = mask_key
        if self.ipt_key_padding_mask is None:
            ipt_key = ~src_padding.bool()
            self.ipt_key_padding_mask = ipt_key
        # src=torch.unsqueeze(src,dim=2).cuda()
        src = self.pos_encoder(src)
        ipt=self.pos_encoder(ipt)
        ipt_output= self.transformer_encoder_ipt( self.data_bn(ipt), self.ipt_mask, self.ipt_key_padding_mask)
        # ipt_output=self.decoder_trans(ipt_output.cuda())
        # ipt_output=F.sigmoid(ipt_output)

        output = self.transformer_encoder( self.data_bn(src), self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        output=torch.cat((output,ipt_output),dim=2)
        # output = self.decoder(output.cuda())
        return output,ipt_output


class PositionalEncoding15(nn.Module):

    def __init__(self, d_model=32, max_len=15):#50->60
        super(PositionalEncoding15, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(60,15,32)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm15(nn.Module):
    def __init__(self, feature_size=32, num_layers=1, dropout=0.3):
        super(TransAm15, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ipt_mask = None
        self.pos_encoder = PositionalEncoding15  (feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.transformer_encoder_ipt = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder = nn.Linear(64, 2)
        self.decoder_trans=nn.Linear(32,1)
        self.data_bn = nn.BatchNorm1d(15)
        self.data_bn.cuda()
        self.init_weights()
        self.src_key_padding_mask = None
        self.ipt_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding,ipt,ipt_padding):
        if self.src_key_padding_mask is None:
            mask_key = ~src_padding.bool()
            self.src_key_padding_mask = mask_key
        if self.ipt_key_padding_mask is None:
            ipt_key = ~src_padding.bool()
            self.ipt_key_padding_mask = ipt_key
        # src=torch.unsqueeze(src,dim=2).cuda()
        src = self.pos_encoder(src)
        ipt=self.pos_encoder(ipt)
        ipt_output= self.transformer_encoder_ipt( self.data_bn(ipt), self.ipt_mask, self.ipt_key_padding_mask)
        # ipt_output=self.decoder_trans(ipt_output.cuda())
        # ipt_output=F.sigmoid(ipt_output)

        output = self.transformer_encoder( self.data_bn(src), self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        output=torch.cat((output,ipt_output),dim=2)
        # output = self.decoder(output.cuda())
        return output,ipt_output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model=32, max_len=5):#50->60
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(60,5,32)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size=32, num_layers=1, dropout=0.3):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ipt_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.transformer_encoder_ipt = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder = nn.Linear(64, 2)
        self.decoder_trans=nn.Linear(32,1)
        self.data_bn = nn.BatchNorm1d(5)
        self.data_bn.cuda()
        self.init_weights()
        self.src_key_padding_mask = None
        self.ipt_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding,ipt,ipt_padding):
        if self.src_key_padding_mask is None:
            mask_key = ~src_padding.bool()
            self.src_key_padding_mask = mask_key
        if self.ipt_key_padding_mask is None:
            ipt_key = ~src_padding.bool()
            self.ipt_key_padding_mask = ipt_key
        # src=torch.unsqueeze(src,dim=2).cuda()
        src = self.pos_encoder(src)
        ipt=self.pos_encoder(ipt)
        ipt_output= self.transformer_encoder_ipt( self.data_bn(ipt), self.ipt_mask, self.ipt_key_padding_mask)
        # ipt_output=self.decoder_trans(ipt_output.cuda())
        # ipt_output=F.sigmoid(ipt_output)

        output = self.transformer_encoder( self.data_bn(src), self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        output=torch.cat((output,ipt_output),dim=2)
        # output = self.decoder(output.cuda())
        return output,ipt_output
