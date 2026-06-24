import torch.nn as nn
from src import model
import torch 

def test_RMSTest():
    dim = 16
    x = torch.randn(2,5,dim)
    rms_norm_torch = nn.RMSNorm(dim)
    result_torch = rms_norm_torch(x)

    rms_norm_implemented = model.RMSNorm(dim)
    result_implemented = rms_norm_implemented(x)
    
    assert torch.allclose(result_torch, result_implemented, atol=1e-6)

