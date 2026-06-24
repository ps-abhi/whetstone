from torch import rsqrt, mean
import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight  = nn.Parameter(torch.ones(dim))
        self.eps = eps 
    
    def forward(self, x):
        rms_inverse = rsqrt(self.eps+mean(x**2, -1, keepdim=True))
        rms_norm = x*rms_inverse*self.weight
        return rms_norm


class RoPE(nn.Module):
    def __init__(self, head_dims, T):
        super().__init__()

        assert head_dims % 2 == 0
        i = torch.arange(0, head_dims/2)
        frequencies = 1 / 10000 ** (2*i / head_dims) # theta calc

        positions = torch.arange(T) # T is the sequence length

        angles = torch.outer(positions, frequencies)
        angles = torch.cat([angles, angles], dim =-1)
        
        cos_rotation_table = torch.cos(angles)
        sin_rotation_table = torch.sin(angles)

        self.register_buffer("cos", cos_rotation_table)
        self.register_buffer("sin", sin_rotation_table)


    # there are two ways we can proceed - interleaved and rotate-half. we're proceeding with rotate half (it's used by HF and LLama)

    def rotate_half(self, x):
        # helper splits into two, negates the second half and joins first half to it x1,x2 -> -x2,x1
        x1, x2 = x.chunk(2, dim=-1)
        x2 = torch.neg(x2)
        return torch.cat((x2, x1), dim=-1)
    

    def forward(self, x, start_pos):
        T = x.shape[-2] # seq length = second to last axis
        cos = self.cos[start_pos : start_pos + T]
        sin = self.sin[start_pos : start_pos + T]
        return x * cos + self.rotate_half(x) * sin  # out = x·cos + rotate_half(x)·sin