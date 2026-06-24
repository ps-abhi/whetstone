from numpy import repeat
import torch
import torch.nn.functional as F
from src.model import causal_attention, repeat_kv

def test_attention_sdpa():
    B, H, T, Dh = 2, 8, 16, 32
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    v = torch.randn(B, H, T, Dh)

    written = causal_attention(q,k,v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(written, ref, atol=1e-6)
    
def test_gqa_repeat_kv_sdpa_match():
    B, H, n_kv, T, Dh = 2, 8, 2, 16, 32
    n_rep = H // n_kv
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, n_kv, T, Dh)
    v = torch.randn(B, n_kv, T, Dh)

    written = causal_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    assert torch.allclose(written, ref, atol=1e-6)
