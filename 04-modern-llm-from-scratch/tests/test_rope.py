import torch.nn as nn
import torch
from src.model import RoPE


def test_rope():
    L, dh, off = 12, 8, 5

    rope = RoPE(dh, T=64)
    q = torch.randn(L, dh)
    k = torch.randn(L, dh)

    #baseline 
    s1 = rope(q, 0) @ rope(k, 0).transpose(-2, -1) # [L, dh]@[dh, L] -> [L, L]

    #shifted
    s2 = rope(q, off) @ rope(k, off).transpose(-2, -1) #  # [L, L]

    assert torch.allclose(s1, s2, atol=1e-6)

