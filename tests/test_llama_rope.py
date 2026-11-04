import torch
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from nanovllm.models.llama import llama3_inv_freq


def test_llama31_rope_scaling_matches_transformers():
    rs = dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192)
    cfg = LlamaConfig(rope_theta=500000.0, max_position_embeddings=131072, hidden_size=4096, num_attention_heads=32, rope_scaling=rs)
    ref, _ = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cpu")
    assert torch.allclose(llama3_inv_freq(500000.0, 128, rs), ref, rtol=1e-6, atol=0)
