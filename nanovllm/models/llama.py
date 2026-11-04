"""Llama-3.x for nano-vllm: the Qwen3 module tree built from a Llama config without the q/k RMSNorms (Llama has none) and
with Llama-3 scaled RoPE (HF `llama3` rope_scaling rule, Llama-3.1/3.2/3.3).  Weight names (q/k/v_proj, gate/up/down_proj,
input/post_attention_layernorm, embed_tokens, lm_head) are identical, so the stock loader applies."""
import math
import torch
from torch import nn

from nanovllm.models.qwen3 import Qwen3ForCausalLM


def llama3_inv_freq(base: float, dim: int, rs: dict) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    factor, lo, hi, old = rs.get("factor", 8.0), rs.get("low_freq_factor", 1.0), rs.get("high_freq_factor", 4.0), rs.get("original_max_position_embeddings", 8192)
    low_wl, high_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    inv_l = torch.where(wl > low_wl, inv / factor, inv)
    smooth = (old / wl - lo) / (hi - lo)
    smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
    medium = ~(wl < high_wl) & ~(wl > low_wl)
    return torch.where(medium, smoothed, inv_l)


class LlamaForCausalLM(Qwen3ForCausalLM):

    def __init__(self, config) -> None:
        config.attention_bias = getattr(config, "attention_bias", False)
        super().__init__(config)
        rs = getattr(config, "rope_scaling", None) or {}
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.q_norm = nn.Identity()
            attn.k_norm = nn.Identity()
            if rs.get("rope_type", rs.get("type")) == "llama3":
                re = attn.rotary_emb
                inv = llama3_inv_freq(float(config.rope_theta), re.head_size, rs).to(re.cos_sin_cache.device)
                t = torch.arange(re.cos_sin_cache.shape[0], dtype=torch.float, device=inv.device)
                f = torch.einsum("i,j->ij", t, inv)
                re.cos_sin_cache = torch.cat((f.cos(), f.sin()), dim=-1).unsqueeze_(1).to(re.cos_sin_cache.dtype)
