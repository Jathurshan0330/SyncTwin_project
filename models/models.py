import copy
from typing import Optional, Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm, BatchNorm1d
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import math


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
class get_Embedding(nn.Module): 
    def __init__(self, in_channels: int = 3, emb_size: int = 64):
        super(get_Embedding, self).__init__()

        self.projection =  nn.Sequential(
            Rearrange('b s e -> b e s'),
            nn.Conv1d(in_channels, emb_size, kernel_size = 1, stride = 1),
            # Rearrange('b e s -> b s e')
            )
        # self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.arrange1 = Rearrange('b e s -> s b e')
        self.pos = PositionalEncoding(d_model=emb_size)
        self.arrange2 = Rearrange('s b e -> b s e  ')

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)  
        b = x.shape[0]
        # cls_tokens = repeat(self.cls_token, '() s e -> b e s', b=b)
        # prepend the cls token to the input
        # print(cls_tokens.shape,x.shape)
        # x = torch.cat([cls_tokens, x], dim=-1)
        # add position embedding
        x = self.arrange1(x)
        x = self.pos(x)
        x = self.arrange2(x)
        return x
    
class Atten_block(nn.Module): 
    def __init__(self, d_model=64, nhead=8, dropout=0.1,dim_feedforward=512,
                 layer_norm_eps=1e-5):
        super(Atten_block, self).__init__()
       
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)#, **factory_kwargs)  
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)#,
                                            # **factory_kwargs)
        self.dropout = Dropout(dropout)
        
        
        self.norm_ff = LayerNorm(d_model, eps=layer_norm_eps)#, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward)#, **factory_kwargs)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)#, **factory_kwargs)
        self.dropout2 = Dropout(dropout)
 

    def forward(self, x: Tensor) -> Tensor:
        src = x
        src2 = self.self_attn(src, src, src)[0]
        out = src + self.dropout(src2)
        out = self.norm(out)   ########
        
        src2 = self.linear2(self.dropout1(self.relu(self.linear1(out))))
        out = out + self.dropout2(src2)
        out = self.norm_ff(out)
        return out                           
    

class Trans_Encoder(nn.Module): 
    def __init__(self, in_channels = 3,d_model=64, nhead=8, dropout=0.1,dim_feedforward=512,
                 layer_norm_eps=1e-5):
        super(Trans_Encoder, self).__init__()
        
        self.embed = get_Embedding(in_channels = in_channels, emb_size = d_model)
        
        self.atten_1 =  Atten_block(d_model=d_model, nhead=nhead, dropout=dropout,
                                    dim_feedforward=dim_feedforward,layer_norm_eps=layer_norm_eps)
        self.atten_2 =  Atten_block(d_model=d_model, nhead=nhead, dropout=dropout,
                                    dim_feedforward=dim_feedforward,layer_norm_eps=layer_norm_eps)
        
        # self.final = Linear(d_model, 1)
    def forward(self, x):
        x = self.embed(x)
        x = self.atten_1(x)
        x = self.atten_2(x)
        # x = self.final(x)
        return x
    

class Trans_Decoder(nn.Module): 
    def __init__(self,out_channels= 3, d_model=64, nhead=8, dropout=0.1,dim_feedforward=512,
                 layer_norm_eps=1e-5):
        super(Trans_Decoder, self).__init__()
        
        # self.embed = get_Embedding(in_channels = 3, emb_size = d_model)
        # self.first = Linear(1, d_model)
        self.atten_1 =  Atten_block(d_model=d_model, nhead=nhead, dropout=dropout,
                                    dim_feedforward=dim_feedforward,layer_norm_eps=layer_norm_eps)
        self.atten_2 =  Atten_block(d_model=d_model, nhead=nhead, dropout=dropout,
                                    dim_feedforward=dim_feedforward,layer_norm_eps=layer_norm_eps)
        self.final = Linear(d_model, out_channels)
        # self.arrange1 = Rearrange('b e s -> s b e')
    def forward(self, x):
        # x = self.embed(x)
        # x = self.first(x)
        x = self.atten_1(x)
        x = self.atten_2(x)
        x = self.final(x)
        
        
        return x

class linear_cls(nn.Module):
    def __init__(self,y_times = 5):
        super(linear_cls, self).__init__()
        
        self.Q = nn.Linear(1600, y_times)
    def forward(self, x):
        # print(x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=- 1) 
        x = self.Q(x)
        return x
        