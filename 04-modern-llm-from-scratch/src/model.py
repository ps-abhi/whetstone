from numpy import sqrt
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
    
class Attention(nn.Module):
    def __init__(self, head_dim, d_model, n_heads, n_kv_heads, max_seq_length):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads*head_dim, bias=False) # no additive bias
        self.k_proj = nn.Linear(d_model, n_kv_heads*head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads*head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads*head_dim, d_model, bias=False)  # 256 -> 256

        self.rope = RoPE(head_dim, max_seq_length)
    
    def forward(self, x, start_pos=0): #x: [B, T, d_model])
        B, T, _ = x.shape

        q = self.q_proj(x) # # [B, T, 256]
        q = q.view(B, T, self.n_heads, self.head_dim) ## [B, T, 8, 32] split that flat 256 into (heads, head_dim), then move heads next to batch
        q = q.transpose(1, 2)       

        k = self.k_proj(x)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = self.v_proj(x)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.transpose(1,2)

        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)

        def repeat_kv(x, n_rep):
            if n_rep == 1:
                return x 
            B, n_kv, T, Dh = x.shape
            x = x[:, :, None, :, :].expand(B, n_kv, n_rep, T, Dh)
            return x.reshape(B, n_kv * n_rep, T, Dh) # [B, n_heads, T, Dh]
            
        k = repeat_kv(k, self.n_rep) # # [B, 2, T, 32] -> [B, 8, T, 32]
        v = repeat_kv(v, self.n_rep)

        scores = (q @ k.transpose(-2, -1))/sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1) # [B, 8, T, T]
        out = attention @ v #  # [B, 8, T, 32]

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


attn= Attention(head_dim=32, d_model=256, n_heads=8, n_kv_heads=2, max_seq_length=64)
print(attn(torch.randn(2, 10, 256)).shape) 