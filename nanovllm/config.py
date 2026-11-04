import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # GAAC lossless compressed KV cache
    kv_compression: str = "none"          # "none" (stock paged BF16 cache) | "gaac"
    gaac_bases: str | None = None         # calibrated exponent bases (examples/calibrate.py); None = calibrate each sequence on its prompt
    gaac_ovf_bytes_per_tile: int = 128    # escape-stream reservation per 16-token tile
    num_kv_slots: int = -1                # set by the model runner: sequences the compressed cache holds

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.kv_compression in ("none", "gaac")
        if self.kv_compression == "gaac":
            hf = self.hf_config
            head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
            assert head_dim == 128, "GAAC kernel supports head_dim 128"
            assert 1 <= hf.num_attention_heads // hf.num_key_value_heads <= 8, "GAAC kernel supports GQA group size 1..8"
            assert self.tensor_parallel_size == 1, "GAAC mode is single-GPU"
            # prompts are prefilled in one piece (no chunked prefill, no prefix caching)
            assert self.max_num_batched_tokens >= self.max_model_len, "GAAC mode needs max_num_batched_tokens >= max_model_len"
