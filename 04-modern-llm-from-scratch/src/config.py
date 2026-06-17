from attr import dataclass


@dataclass
class ModelConfig:
    d_model=256
    n_layers = 6
    n_heads = 8
    n_kv_heads = 2
    head_dims = 32
    mlp_ratio = 0.6