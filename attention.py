from typing import List, Union, Optional, Dict

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers.utils import PaddingStrategy

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,dropout:float,max_len:int=5000):

        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)/d_model*math.log(10000))

        pe[:,0::2] = torch.sin(pos/div_term)
        pe[:,1::2] = torch.cos(pos/div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

     def forward(self,x:torch.Tensor):

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,h:int,d_model:int,dropout:float=0.1):

        super(MultiHeadAttention,self).__init__()
        assert d_model%h==0
        self.d_h = d_model // h
        self.h = h
        ####四个深拷贝，相互独立的linear全连接层
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

     def attention(self, query: torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask: torch.Tensor=None, dropout: torch.nn.Module=None):

