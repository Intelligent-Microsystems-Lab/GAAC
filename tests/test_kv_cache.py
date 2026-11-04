import pytest, torch
from gaac import CompressedKVCache, Codec
from gaac.codec import exponent_hist, bases_from_hist
from common import kv_like, ref_attention

N_KV, G, SCALE = 4, 4, 128 ** -0.5


def _tok(x):          # [n_kv, L, D] -> token-major [L, n_kv, D]
    return x.transpose(0, 1).contiguous()


def _bases(seed):
    k, v = _tok(kv_like(N_KV, 4096, seed=seed)), _tok(kv_like(N_KV, 4096, seed=seed + 1))
    k3, k4 = bases_from_hist(exponent_hist(k, N_KV), N_KV); v3, v4 = bases_from_hist(exponent_hist(v, N_KV), N_KV)
    return dict(k3=k3, k4=k4, v3=v3, v4=v4)


def _check(cache, hist, slots, q):
    out = cache.attention(q, slots, SCALE)
    for b, s in enumerate(slots.tolist()):
        if s < 0:
            assert out[b].abs().max().item() == 0
            continue
        k, v = hist[s]
        ref = ref_attention(q[b], k, v, SCALE)
        assert (out[b].float() - ref).abs().max().item() <= 2e-2 * ref.abs().max().item() + 1e-3
        dk, dv = cache.decompress(s)
        assert torch.equal(dk.view(torch.int16), k.view(torch.int16)) and torch.equal(dv.view(torch.int16), v.view(torch.int16))


@pytest.mark.parametrize("static_bases", [True, False])
def test_slots_prefill_append(static_bases):
    cache = CompressedKVCache(n_slots=4, n_kv=N_KV, capacity=1024, bases=_bases(10) if static_bases else None)
    lens = {0: 37, 2: 500, 3: 16}
    hist = {}
    for s, L in lens.items():
        k, v = _tok(kv_like(N_KV, L, seed=20 + s)), _tok(kv_like(N_KV, L, seed=40 + s))
        cache.prefill(s, k, v); hist[s] = (k, v)
    # decode steps over a changing batch (slot order != batch order, plus a padded row)
    for step in range(40):
        slots_l = [3, 0, 2, -1] if step % 3 else [2, 3, -1, 0]
        slots = torch.tensor(slots_l, dtype=torch.int32, device="cuda")
        k = (torch.randn(4, N_KV, 128, device="cuda") * 0.5).to(torch.bfloat16)
        v = (torch.randn(4, N_KV, 128, device="cuda") * 0.5).to(torch.bfloat16)
        cache.append(slots, k, v)
        for b, s in enumerate(slots_l):
            if s >= 0: hist[s] = (torch.cat([hist[s][0], k[b:b + 1]]), torch.cat([hist[s][1], v[b:b + 1]]))
        if step % 16 == 15: cache.check_capacity()
        if step % 13 == 0 or step == 39:
            q = (torch.randn(4, N_KV * G, 128, device="cuda") * 0.1).to(torch.bfloat16)
            _check(cache, hist, slots, q)
    # slot reuse: a new sequence replaces slot 0
    k, v = _tok(kv_like(N_KV, 64, seed=99)), _tok(kv_like(N_KV, 64, seed=98))
    cache.prefill(0, k, v); hist[0] = (k, v)
    _check(cache, hist, torch.tensor([0], dtype=torch.int32, device="cuda"), (torch.randn(1, N_KV * G, 128, device="cuda") * 0.1).to(torch.bfloat16))


def test_cuda_graph_decode_step():
    cache = CompressedKVCache(n_slots=2, n_kv=N_KV, capacity=512, bases=_bases(11))
    k0, v0 = _tok(kv_like(N_KV, 200, seed=5)), _tok(kv_like(N_KV, 200, seed=6))
    cache.prefill(1, k0, v0)
    slots = torch.tensor([1, -1], dtype=torch.int32, device="cuda")
    q = torch.zeros(2, N_KV * G, 128, dtype=torch.bfloat16, device="cuda")
    kn = torch.zeros(2, N_KV, 128, dtype=torch.bfloat16, device="cuda"); vn = torch.zeros_like(kn)
    run = lambda: cache.attention(q, slots, SCALE, max_len=512)
    def step():
        cache.append(slots, kn, vn); return run()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):          # warm up (compile) outside capture, on a scratch slot
        slots.fill_(-1); step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    slots.copy_(torch.tensor([1, -1], dtype=torch.int32))
    hist = {1: (k0, v0)}
    for _ in range(20):
        kn.copy_((torch.randn(2, N_KV, 128, device="cuda") * 0.5).to(torch.bfloat16)); vn.copy_((torch.randn(2, N_KV, 128, device="cuda") * 0.5).to(torch.bfloat16))
        q.copy_((torch.randn(2, N_KV * G, 128, device="cuda") * 0.1).to(torch.bfloat16))
        g.replay(); torch.cuda.synchronize()
        hist[1] = (torch.cat([hist[1][0], kn[:1]]), torch.cat([hist[1][1], vn[:1]]))
        ref = ref_attention(q[0], *hist[1], SCALE)
        assert (out[0].float() - ref).abs().max().item() <= 2e-2 * ref.abs().max().item() + 1e-3
    dk, dv = cache.decompress(1)
    assert torch.equal(dk.view(torch.int16), hist[1][0].view(torch.int16))
