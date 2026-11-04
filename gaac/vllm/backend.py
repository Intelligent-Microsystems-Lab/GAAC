"""GAAC attention backend for vLLM V1 (0.17.1): the lossless compressed KV cache lives in vLLM's own paged block buffer and is read
directly by the fused decode kernel.  Enabled by GAAC_VLLM=1 in a vLLM patched with vllm/vllm-0.17.1-gaac.patch.

vLLM's per-layer KV buffer holds compressed 16-token kernel blocks (gaac/vllm/paged_kernel.py layout, 2*n_kv' tile slots per block),
so the block manager, scheduler, prefix caching and CUDA graphs stay vLLM's; the backend defines the block bytes (page_size_bytes,
called from the patched AttentionSpec), the store path (gaac/vllm/store.py) and the attention:
  decode (query_len == 1)  -> fused paged attention over the compressed blocks + raw side pool
  prefill without context  -> FlashAttention varlen over this step's K/V
  prefill with context     -> context blocks decompressed (Triton) into a flat bf16 scratch + FlashAttention varlen
                              (chunked prefill and prefix-cache hits)
Head dim 256 (Qwen3.5/3.6/3.8 hybrid models) is stored as n_kv' = 2*n_kv half-heads of 128, each with its own bases.  Hybrid models
size the attention manager block to the Mamba page; vLLM splits it into 16-token kernel blocks (get_supported_kernel_block_sizes).

Environment:
  GAAC_BASES=<file.pt>   per-layer exponent bases {hf layer index: {k3,k4,v3,v4: uint8 [n_kv', 128]}}
  GAAC_CALIB=<file.pt>   calibration run: collect exponent histograms over prefill K/V and write the bases file (use a long prompt,
                         enforce_eager, and no GAAC_BASES: every tile then stays raw, so size the raw pool to the prompt)
  GAAC_OVF_SLOT=128      escape bytes per tile slot (tiles that need more stay raw in the side pool, still lossless)
  GAAC_RAW_TILES=<n>     raw side-pool size per layer (default: filling blocks of max_num_seqs sequences + 1% of the blocks)
Limitation: multi-token decode (speculative decoding) is not supported.
"""
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import ClassVar, Optional
import numpy as np, torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionImpl, AttentionMetadataBuilder, AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec
from .store import LayerState, OVF_OFF
from .paged_kernel import paged_attention, choose_nsplit
from .paged_kernel256 import paged_attention256

try:
    from flash_attn import flash_attn_varlen_func                       # NGC vLLM containers ship flash-attn 2.7.4
except ImportError:                                                     # pip wheels of vLLM: same FlashAttention version as the stock backend
    import functools
    from vllm.vllm_flash_attn import flash_attn_varlen_func as _fa_varlen
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    flash_attn_varlen_func = functools.partial(_fa_varlen, fa_version=get_flash_attn_version())

OVF_SLOT = int(os.environ.get("GAAC_OVF_SLOT", "128"))
TS = OVF_OFF + OVF_SLOT
BLOCK = 16
BLOCKS_PER_KV_BLOCK = 1   # manager block / kernel block (set by the builder; used to expand freed manager blocks)
PENDING_FREE: list = []   # kernel-block ids freed by vLLM (queue_freed_blocks, called from the patched BlockPool), drained by the builder
LAYER_STATES: list = []
ATTN_LAYERS: set = set()  # hf layer indices with (full) attention; hybrid models attend only every 4th layer
CALIB: dict = {}
CALIB_FRESH = bool(os.environ.get("GAAC_CALIB")) and not os.path.exists(os.environ.get("GAAC_BASES", ""))
_bases = None


# ---------------------------------------------------------------- hooks called from the patched vLLM
def page_size_bytes(num_kv_heads: int, head_size: int, block_size: int) -> int:
    """Bytes of a KV block of block_size tokens (vLLM probes block_size=1 for hybrid models; real blocks are multiples of 16)."""
    return block_bytes(eff_heads(num_kv_heads, head_size)) * block_size // BLOCK


def queue_freed_blocks(blocks) -> None:
    """Blocks freed without prefix caching hold dead content: queue their kernel blocks so each layer returns their raw tiles."""
    k = BLOCKS_PER_KV_BLOCK
    PENDING_FREE.extend(b.block_id * k + i for b in blocks if b.block_id != 0 for i in range(k))


# ----------------------------------------------------------------
def block_bytes(n_kv: int) -> int:
    """Bytes of one 16-token kernel block holding n_kv (half-)heads of 128 dims: K and V tile slots per head."""
    return 2 * n_kv * TS


def eff_heads(n_kv: int, head_size: int) -> int:
    assert head_size in (128, 256), "GAAC: head_size 128 or 256"
    return n_kv * (head_size // 128)


def _flat_bases(n_eff: int) -> dict:
    """Placeholder bases for calibration runs without a bases file."""
    return dict(k3=torch.full((n_eff, 128), 124, dtype=torch.uint8), k4=torch.full((n_eff, 128), 120, dtype=torch.uint8),
                v3=torch.full((n_eff, 128), 124, dtype=torch.uint8), v4=torch.full((n_eff, 128), 120, dtype=torch.uint8))


def load_bases():
    global _bases
    if _bases is None:
        _bases = torch.load(os.environ["GAAC_BASES"], map_location="cpu", weights_only=False)
    return _bases


_scratch_kv = None


def _scratch(rows: int, n_kv: int, device):
    """Flat bf16 K/V scratch [rows, n_kv, 128] for prefill with context (shared by all layers, grown on demand)."""
    global _scratch_kv
    if _scratch_kv is None or _scratch_kv[0].shape[0] < rows:
        cap = max(rows, 16384)
        _scratch_kv = (torch.empty(cap, n_kv, 128, dtype=torch.bfloat16, device=device), torch.empty(cap, n_kv, 128, dtype=torch.bfloat16, device=device))
    return _scratch_kv[0][:rows], _scratch_kv[1][:rows]


class GAACBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [BLOCK]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"        # registered as AttentionBackendEnum.CUSTOM by the patch

    @staticmethod
    def get_impl_cls():
        return GAACImpl

    @staticmethod
    def get_metadata_cls():
        return GAACMetadata

    @staticmethod
    def get_builder_cls():
        return GAACMetadataBuilder

    @classmethod
    def get_supported_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 256]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_block_size(cls, block_size) -> bool:
        return block_size is None or block_size % BLOCK == 0

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str = "auto"):
        assert block_size == BLOCK, "GAAC: kernel block size 16"
        return (num_blocks, block_bytes(eff_heads(num_kv_heads, head_size)) // 2)      # bf16 elements per block (viewed as bytes in the impl)


@dataclass
class GAACMetadata:
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    ent: torch.Tensor           # int32 [E,4] (block, tok0, off0, n) for the store kernel
    freeq: torch.Tensor         # int32 [E*2*n_kv] scratch
    nsplit: int
    prefill_cu: Optional[torch.Tensor] = None
    prefill_max_q: int = 0
    prefill_context: bool = False
    ctx_pairs: Optional[torch.Tensor] = None      # int32 [P,3] (vLLM block, first scratch row, valid rows)
    ctx_cu_k: Optional[torch.Tensor] = None       # int32 [npf+1] cumulative (context + new) lengths = scratch row starts
    ctx_max_k: int = 0
    ctx_slots: Optional[torch.Tensor] = None      # int64 [npt] scratch row of each new prefill token
    ctx_rows: int = 0


class GAACMetadataBuilder(AttentionMetadataBuilder[GAACMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device):
        global BLOCKS_PER_KV_BLOCK
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.n_kv = eff_heads(kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
        assert kv_cache_spec.block_size == BLOCK, "GAAC: kernel block size 16 (the manager block may be any multiple of 16)"
        BLOCKS_PER_KV_BLOCK = vllm_config.cache_config.block_size // BLOCK
        # persistent buffers for pure-decode steps (read by kernels captured in full CUDA graphs; build() refills them each step)
        cc = vllm_config.compilation_config
        M = max(vllm_config.scheduler_config.max_num_seqs, int(getattr(cc, "max_cudagraph_capture_size", None) or getattr(cc, "max_capture_size", 0) or 0))
        self.ent_buf = torch.zeros(M, 4, dtype=torch.int32, device=device); self.ent_buf[:, 0] = -1
        self.freeq_buf = torch.empty(M * 2 * self.n_kv, dtype=torch.int32, device=device)
        # pinned staging ring for the per-step entry upload: a pageable H2D copy would block the host until the GPU drains and
        # defeat vLLM's async scheduling
        self.RING = 16; self.ent_pin = torch.zeros(self.RING, M, 4, dtype=torch.int32, pin_memory=True); self.ring = 0
        self.max_model_len = vllm_config.model_config.max_model_len; self.steps = 0

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> "GAACMetadata":
        md = self.build(0, common_attn_metadata)
        md.ent[:, 0] = -1                    # inactive store entries during capture
        md.seq_lens.fill_(1)
        return md

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False) -> GAACMetadata:
        common = common_attn_metadata
        nd, npf, ndt, npt = split_decodes_and_prefills(common, decode_threshold=1)
        R = common.num_reqs
        qsl = common.query_start_loc_cpu.numpy().astype(np.int64)
        start = common.num_computed_tokens_cpu.numpy()[:R].astype(np.int64)
        qlen = qsl[1:R + 1] - qsl[:R]
        # touched (request, logical block) pairs -> store entries
        lb0 = start // BLOCK; lb1 = (start + qlen - 1) // BLOCK; cnt = lb1 - lb0 + 1
        E = int(cnt.sum()); req = np.repeat(np.arange(R), cnt)
        lb = lb0[req] + (np.arange(E) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        pos0 = np.maximum(start[req], lb * BLOCK); pos1 = np.minimum(start[req] + qlen[req], lb * BLOCK + BLOCK)
        off0 = pos0 - lb * BLOCK; n = pos1 - pos0; tok0 = qsl[req] + (pos0 - start[req])
        ent_cpu = np.stack([tok0, tok0, off0, n], 1).astype(np.int32)
        if npf == 0 and E <= self.ent_buf.shape[0]:      # pure decode: persistent buffers (graph-replay safe)
            ent = self.ent_buf[:E]; stage = self.ent_pin[self.ring, :E]; self.ring = (self.ring + 1) % self.RING
            stage.copy_(torch.from_numpy(ent_cpu)); ent.copy_(stage, non_blocking=True)
            self.ent_buf[E:, 0].fill_(-1)                                        # graph padding rows stay inactive
            freeq = self.freeq_buf[:E * 2 * self.n_kv]
        else:
            ent = torch.from_numpy(ent_cpu).to(self.device); freeq = torch.empty(E * 2 * self.n_kv, dtype=torch.int32, device=self.device)
        ent[:, 0] = (common.slot_mapping[ent[:, 1].long()] // BLOCK).to(torch.int32)
        nsplit = choose_nsplit(max(nd, 1), self.n_kv, self.max_model_len)          # fixed per batch size (baked into graphs)
        # per-step housekeeping for all layers, outside any graph: release raw tiles of freed blocks, check pool exhaustion
        self.steps += 1
        if PENDING_FREE:
            fb = torch.tensor(PENDING_FREE, dtype=torch.int32, device=self.device); PENDING_FREE.clear()
            for st in LAYER_STATES: st.release_blocks(fb)
        if self.steps % 16 == 0:
            for st in LAYER_STATES:
                if int(st.err_host[0]) != 0: raise RuntimeError(f"GAAC: raw side pool exhausted ({int(st.err_host[0])} failed allocations); raise GAAC_RAW_TILES or GAAC_OVF_SLOT")
                st.err_host.copy_(st.err, non_blocking=True)
        md = GAACMetadata(num_actual_tokens=common.num_actual_tokens, num_decodes=nd, num_decode_tokens=ndt, num_prefills=npf,
                          max_query_len=common.max_query_len, max_seq_len=common.max_seq_len, query_start_loc=common.query_start_loc,
                          seq_lens=common.seq_lens, block_table=common.block_table_tensor, slot_mapping=common.slot_mapping,
                          ent=ent, freeq=freeq, nsplit=nsplit)
        if npf:
            md.prefill_cu = (common.query_start_loc[nd:R + 1] - ndt).to(torch.int32)
            md.prefill_max_q = int(qlen[nd:].max()); md.prefill_context = bool((start[nd:] > 0).any())
            if md.prefill_context:
                ps, pq = start[nd:], qlen[nd:]; tot = ps + pq; rs = np.concatenate([[0], np.cumsum(tot)])
                nctx = (ps + BLOCK - 1) // BLOCK
                preq = np.repeat(np.arange(npf), nctx); plb = np.arange(int(nctx.sum())) - np.repeat(np.cumsum(nctx) - nctx, nctx)
                nvalid = np.minimum(BLOCK, ps[preq] - plb * BLOCK)
                pairs = torch.from_numpy(np.stack([plb, rs[preq] + plb * BLOCK, nvalid], 1).astype(np.int32)).to(self.device)
                pairs[:, 0] = common.block_table_tensor[nd + torch.from_numpy(preq).to(self.device), pairs[:, 0].long()]
                treq = np.repeat(np.arange(npf), pq); pos = ps[treq] + (np.arange(int(pq.sum())) - np.repeat(np.cumsum(pq) - pq, pq))
                md.ctx_pairs = pairs; md.ctx_cu_k = torch.from_numpy(rs.astype(np.int32)).to(self.device)
                md.ctx_max_k = int(tot.max()); md.ctx_slots = torch.from_numpy((rs[treq] + pos).astype(np.int64)).to(self.device); md.ctx_rows = int(rs[-1])
        return md


class GAACImpl(AttentionImpl):
    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, alibi_slopes, sliding_window, kv_cache_dtype: str,
                 logits_soft_cap: Optional[float] = None, attn_type: str = AttentionType.DECODER, kv_sharing_target_layer_name=None, sinks=None, **kw):
        assert head_size in (128, 256) and alibi_slopes is None and sliding_window is None and attn_type == AttentionType.DECODER, "GAAC: unsupported attention config"
        assert kv_cache_dtype in ("auto", "bfloat16") and not logits_soft_cap and kv_sharing_target_layer_name is None and sinks is None
        self.num_heads, self.head_size, self.scale, self.num_kv_heads = num_heads, head_size, float(scale), num_kv_heads
        self.n_eff = eff_heads(num_kv_heads, head_size)          # 128-dim (half-)heads seen by the codec / store / bases
        self.state: Optional[LayerState] = None
        cfg = get_current_vllm_config()
        self.max_num_seqs = cfg.scheduler_config.max_num_seqs; self.max_model_len = cfg.model_config.max_model_len
        hf = cfg.model_config.hf_text_config if hasattr(cfg.model_config, "hf_text_config") else cfg.model_config.hf_config
        lt = getattr(hf, "layer_types", None)
        self.n_attn_layers = sum(1 for t in lt if t == "full_attention") if lt else hf.num_hidden_layers

    def _get_state(self, layer, kv_cache: torch.Tensor) -> LayerState:
        if self.state is None:
            li = int(re.search(r"layers\.(\d+)\.", layer.layer_name).group(1))      # hf layer index (model.layers.N / language_model.model.layers.N)
            num_blocks = kv_cache.shape[0]
            raw_tiles = int(os.environ.get("GAAC_RAW_TILES", "0")) or (2 * self.n_eff * 4 * self.max_num_seqs + int(0.01 * num_blocks * 2 * self.n_eff) + 64)
            if CALIB_FRESH:      # calibration without bases: flat windows, every tile of the prompt may stay raw
                bases = _flat_bases(self.n_eff); raw_tiles = min(num_blocks, (self.max_model_len // 16 + 16) * self.max_num_seqs) * 2 * self.n_eff
            else:
                bases = load_bases()[li]
            self.state = LayerState(num_blocks, self.n_eff, raw_tiles, bases, OVF_SLOT, kv_cache.device)
            self.state.err_host = torch.zeros(1, dtype=torch.int32, pin_memory=True); LAYER_STATES.append(self.state)
            self.li = li; ATTN_LAYERS.add(li)
        return self.state

    def _calibrate(self, k, v):
        """Accumulate per-(head, channel) exponent histograms; once every attention layer has data, (re)write the bases file."""
        n = self.n_eff
        h = CALIB.setdefault(self.li, torch.zeros(2, n * 128 * 256, dtype=torch.int64, device=k.device))
        for i, x in enumerate((k, v)):
            e = (x.contiguous().view(torch.int16).to(torch.int32) >> 7) & 0xFF
            idx = (torch.arange(n * 128, device=k.device).view(1, -1) * 256 + e.view(e.shape[0], -1)).view(-1)
            h[i] += torch.bincount(idx, minlength=h.shape[1])
        if len(ATTN_LAYERS) == self.n_attn_layers and self.li == max(ATTN_LAYERS) and len(CALIB) == self.n_attn_layers:
            from ..codec import bases_from_hist
            out = {}
            for li, hh in sorted(CALIB.items()):
                d = {}
                for i, kind in enumerate("kv"):
                    b3, b4 = bases_from_hist(hh[i].view(-1, 256), n)
                    d[kind + "3"], d[kind + "4"] = b3.cpu(), b4.cpu()
                out[li] = d
            torch.save(out, os.environ["GAAC_CALIB"])
            print(f"[gaac] calibration written: {os.environ['GAAC_CALIB']} ({len(out)} attention layers, {n} heads of 128)", flush=True)

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: Optional[GAACMetadata],
                output: Optional[torch.Tensor] = None, output_scale=None, output_block_scale=None) -> torch.Tensor:
        assert output is not None
        if attn_metadata is None:            # profiling run
            return output
        md = attn_metadata; st = self._get_state(layer, kv_cache)
        buf = kv_cache.view(torch.uint8).view(-1)
        self._store(key, value, buf, md, st)
        self._attend(query, key, value, buf, md, st, output)
        return output

    def _store(self, key, value, buf, md, st):
        na = md.num_actual_tokens
        if os.environ.get("GAAC_CALIB") and md.num_prefills:
            self._calibrate(key[md.num_decode_tokens:na], value[md.num_decode_tokens:na])
        st.store(key[:na].reshape(na, self.n_eff, 128).view(torch.int16), value[:na].reshape(na, self.n_eff, 128).view(torch.int16), buf, md.ent, md.freeq)

    def _attend(self, query, key, value, buf, md, st, output):
        na = md.num_actual_tokens; nd = md.num_decodes
        if nd:
            assert output.is_contiguous()
            if self.head_size == 128:
                paged_attention(query[:nd].contiguous(), output[:nd].view(nd, -1), buf, st.raw, st.mt, md.block_table, md.seq_lens, st.bases, st.TS,
                                self.num_heads, self.num_kv_heads, md.nsplit)
            else:
                paged_attention256(query[:nd].contiguous(), output[:nd].view(nd, -1), buf, st.raw, st.mt, md.block_table, md.seq_lens, st.bases, st.TS,
                                   self.num_heads, self.num_kv_heads, md.nsplit, scale=self.scale)
        if md.num_prefills:
            if not md.prefill_context:
                o = flash_attn_varlen_func(query[nd:na], key[nd:na], value[nd:na], cu_seqlens_q=md.prefill_cu, cu_seqlens_k=md.prefill_cu,
                                           max_seqlen_q=md.prefill_max_q, max_seqlen_k=md.prefill_max_q, softmax_scale=self.scale, causal=True)
            else:   # context blocks -> bf16 scratch (shared across layers), new rows scattered in, FlashAttention varlen
                sk, sv = _scratch(md.ctx_rows, self.n_eff, query.device)
                st.decode_blocks(buf, md.ctx_pairs, sk, sv)
                sk = sk.view(-1, self.num_kv_heads, self.head_size); sv = sv.view(-1, self.num_kv_heads, self.head_size)
                sk[md.ctx_slots] = key[nd:na]; sv[md.ctx_slots] = value[nd:na]
                o = flash_attn_varlen_func(query[nd:na], sk, sv, cu_seqlens_q=md.prefill_cu, cu_seqlens_k=md.ctx_cu_k, max_seqlen_q=md.prefill_max_q,
                                           max_seqlen_k=md.ctx_max_k, softmax_scale=self.scale, causal=True)
            output[nd:na].copy_(o)
        return output
