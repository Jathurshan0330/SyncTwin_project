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
    def __init__(self,in_dim,y_times = 5):
        super(linear_cls, self).__init__()
        
        self.Q = nn.Linear(in_dim, y_times)
    def forward(self, x):
        # print(x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=- 1) 
        x = self.Q(x)
        return x
    
    
class LSTM_Encoder(nn.Module): 
    def __init__(self, in_dim,hidden_dim):
        super(LSTM_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, bidirectional=True,batch_first=True)
        self.atten = nn.Parameter(torch.ones(1,1,hidden_dim*2))
    def forward(self, x):
        x,_ = self.lstm (x)
        atten = repeat(self.atten, '() s e -> b e s', b=x.shape[0])
        # print(x.shape,atten.shape)
        atten_weight = torch.softmax(torch.matmul(x, atten) / math.sqrt(self.hidden_dim), dim=1)
        # print(atten_weight.shape,x.shape)
        x = torch.sum(x * atten_weight, dim=1)
        return x
    

class LSTM_Decoder(nn.Module): 
    def __init__(self,hidden_dim,out_dim,time_points):
        super(LSTM_Decoder, self).__init__()
        
        self.time_points = time_points
        self.lstm = nn.LSTM(hidden_dim, hidden_dim,batch_first=True)
        self.final = Linear(hidden_dim, out_dim)
    def forward(self, c):
        out,h = self.lstm(c)
        
        out_main = self.final(out).unsqueeze(1)
        # print(out_main.shape)
        for i in range(self.time_points-1):
            out,h = self.lstm(c,h)
            out_main = torch.cat((out_main,self.final(out).unsqueeze(1)),dim=1)
            
        
        
        return out_main



class CNN_Encoder(nn.Module):
    
    def __init__(self, input_nc=1, ngf = 16):
        super(CNN_Encoder, self).__init__()
        self.enc1 = self.enc_block(in_ch = input_nc, out_ch = ngf, kernel_size=4, stride=2, padding = 1,bias = True, innermost = False  )
        self.enc2 = self.enc_block(in_ch = ngf, out_ch = ngf*2, kernel_size=4, stride=2, padding = 1,bias = True, innermost = False  )
        self.enc3 = self.enc_block(in_ch = ngf*2, out_ch = ngf*4, kernel_size=4, stride=2, padding = 1,bias = True, innermost = False  )
        self.enc4 = self.enc_block(in_ch = ngf*4, out_ch = ngf*8, kernel_size=4, stride=2, padding = 1,bias = True, innermost = False  )
        
        
    def enc_block(self, in_ch, out_ch, kernel_size=4, stride=2, padding = 1,bias = True, innermost = False):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU()
                )
      
            
    def forward(self, x):
        x = torch.moveaxis(x,1,-1)
        # print(x.shape)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        
        return x
    

class CNN_Decoder(nn.Module):
    
    def __init__(self, lat_chan=128,out_ch = 1, ngf = 16 ):
        super(CNN_Decoder, self).__init__()
        self.dec1 = self.conv_up_block(in_ch = lat_chan , out_ch = ngf*8, 
                                       kernel_size=4, stride=2, padding = 1,bias = True)
        self.dec2 = self.conv_up_block(in_ch = ngf*8 , out_ch = ngf*4, 
                                       kernel_size=4, stride=2, padding = 1,bias = True)
        self.dec3 = self.conv_up_block(in_ch = ngf*4 , out_ch = ngf*2, 
                                       kernel_size=4, stride=2, padding = 1,bias = True)
        self.dec4 = self.conv_up_block(in_ch = ngf*2 , out_ch = 25, 
                                       kernel_size=4, stride=2, padding = 1,bias = True)
        self.final = nn.Linear(16,out_ch)
        
        
    def conv_up_block(self,  in_ch, out_ch, kernel_size=4, stride=2, padding = 1,bias = True,outermost = False):
            return nn.Sequential(
                nn.ConvTranspose1d( in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding = padding,bias = bias),
                nn.LeakyReLU()
                )
        
    def forward(self,x): 
        x = self.dec1(x)
        
        x = self.dec2(x)
        
        
        x = self.dec3(x)
        
        x = self.dec4(x)
        x = self.final(x)
        # x = torch.moveaxis(x,-1,1)
        return x

class Linear_Encoder(nn.Module):
    
    def __init__(self, input_nc=1, ngf = 16,seq_length = 25):
        super(Linear_Encoder, self).__init__()
        self.enc1 = nn.Sequential(nn.Flatten(),
                                  nn.Linear(input_nc*seq_length,ngf),
                                  nn.ReLU(),
                                  nn.Linear(ngf,ngf*2),
                                  nn.ReLU(),
                                  nn.Linear(ngf*2,seq_length),
                                  nn.ReLU())
            
    
      
            
    def forward(self, x):
        
        # print(x.shape)
        x = self.enc1(x)
        return x.unsqueeze(-1)
    
    
class Linear_Decoder(nn.Module):
    
    def __init__(self, lat_chan=128,out_ch = 3, ngf = 16 ):
        super(Linear_Decoder, self).__init__()
        self.dec1 = nn.Sequential(
                                  nn.Linear(1,ngf),
                                  nn.ReLU(),
                                  nn.Linear(ngf,ngf*2),
                                  nn.ReLU(),
                                  nn.Linear(ngf*2,3),
                                  nn.ReLU())
        
        
    
    def forward(self,x): 
        x = self.dec1(x)
        
        
        return x