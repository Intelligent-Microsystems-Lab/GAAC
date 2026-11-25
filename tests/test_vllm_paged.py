"""vLLM backend kernels: store K/V into the paged block buffer, then decode attention (head_dim 128 and 256) vs a reference.

The "high" placement puts the sequence's blocks past byte offset 2^31 of the block buffer, as happens when one layer's
buffer is larger than 2 GiB (large GPUs, long max_model_len); needs ~2.3 GB of free GPU memory."""
import pytest, torch
from common import kv_like, ref_attention

pytest.importorskip("triton")
from gaac.codec import bases_from_hist, exponent_hist
from gaac.vllm.store import LayerState, OVF_OFF
from gaac.vllm.paged_kernel import paged_attention
from gaac.vllm.paged_kernel256 import paged_attention256

OVF_SLOT = 128
TS = OVF_OFF + OVF_SLOT
N_EFF = 8                                   # 128-dim (half-)heads per block: 8 KV heads of 128, or 4 KV heads of 256
HIGH = (2 ** 31) // (2 * N_EFF * TS) + 64   # first block id whose byte offset is past 2^31, plus margin


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("place", ["low", "high"])
def test_paged_store_attention_matches_reference(head_dim, place):
    S = 2000; nb = (S + 15) // 16; n_kv = N_EFF * 128 // head_dim; G = 4; n_q = n_kv * G
    num_blocks = HIGH + nb + 8
    if torch.cuda.mem_get_info()[0] < num_blocks * 2 * N_EFF * TS + (1 << 30):
        pytest.skip("not enough free GPU memory for a >2 GiB block buffer")
    dev = "cuda"
    k = kv_like(N_EFF, S, seed=11).transpose(0, 1).contiguous()     # [S, N_EFF, 128] token-major, as the model produces it
    v = kv_like(N_EFF, S, seed=12).transpose(0, 1).contiguous()
    b3k, b4k = bases_from_hist(exponent_hist(k, N_EFF), N_EFF); b3v, b4v = bases_from_hist(exponent_hist(v, N_EFF), N_EFF)
    st = LayerState(num_blocks, N_EFF, nb * 2 * N_EFF + 64, dict(k3=b3k, k4=b4k, v3=b3v, v4=b4v), OVF_SLOT, dev)
    buf = torch.zeros(num_blocks * 2 * N_EFF * TS, dtype=torch.uint8, device=dev)
    first = HIGH if place == "high" else 1
    blocks = torch.arange(first, first + nb, dtype=torch.int32, device=dev)
    j = torch.arange(nb, device=dev)
    ent = torch.stack([blocks, (16 * j).int(), torch.zeros_like(blocks), torch.clamp(S - 16 * j, max=16).int()], 1).to(torch.int32).contiguous()
    freeq = torch.empty(nb * 2 * N_EFF, dtype=torch.int32, device=dev)
    st.store(k.view(torch.int16), v.view(torch.int16), buf, ent, freeq)
    torch.cuda.synchronize()
    assert int(st.err.item()) == 0

    q = (torch.randn(1, n_q, head_dim, device=dev) * 0.1).to(torch.bfloat16)
    out = torch.empty(1, n_q * head_dim, dtype=torch.bfloat16, device=dev)
    bt = blocks.view(1, nb).contiguous(); seq_lens = torch.tensor([S], dtype=torch.int32, device=dev)
    scale = head_dim ** -0.5
    if head_dim == 128:
        paged_attention(q, out, buf, st.raw, st.mt, bt, seq_lens, st.bases, st.TS, n_q, n_kv, 4)
    else:
        paged_attention256(q, out, buf, st.raw, st.mt, bt, seq_lens, st.bases, st.TS, n_q, n_kv, 4, scale=scale)
    kf = k.view(S, n_kv, head_dim); vf = v.view(S, n_kv, head_dim)              # half-heads (kvh, half) -> full heads
    ref = ref_attention(q[0], kf, vf, scale)
    err = (out.view(n_q, head_dim).float() - ref).abs().max().item(); mag = ref.abs().max().item()
    assert torch.isfinite(out).all() and err <= 2e-2 * mag + 1e-3, (place, head_dim, err, mag)
