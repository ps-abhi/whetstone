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
    