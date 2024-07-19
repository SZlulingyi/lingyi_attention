from typing import List, Union, Optional, Dict

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers.utils import PaddingStrategy

class PositionalEncoding(nn.torch):
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
