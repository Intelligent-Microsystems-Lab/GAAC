import pytest, torch
from gaac import Codec, TileEncoder, attention
from common import kv_like, ref_attention


@pytest.mark.parametrize("B,n_kv,G", [(1, 8, 4), (2, 4, 8), (3, 2, 1), (1, 8, 2)])
@pytest.mark.parametrize("S", [16, 333, 4096])
def test_attention_matches_reference(B, n_kv, G, S):
    k, v = kv_like(B * n_kv, S, seed=3), kv_like(B * n_kv, S, seed=4)
    ck = Codec(n_kv).calibrate(k); cv = Codec(n_kv).calibrate(v)
    kp, vp = TileEncoder("k").encode(ck, k), TileEncoder("v").encode(cv, v)
    q = (torch.randn(B, n_kv * G, 128, device="cuda") * 0.1).to(torch.bfloat16)
    scale = 128 ** -0.5
    out = attention(q, kp, vp, scale)
    for b in range(B):
        kb = k[b * n_kv:(b + 1) * n_kv].transpose(0, 1); vb = v[b * n_kv:(b + 1) * n_kv].transpose(0, 1)
        ref = ref_attention(q[b], kb, vb, scale)
        err = (out[b].float() - ref).abs().max().item(); mag = ref.abs().max().item()
        assert err <= 2e-2 * mag + 1e-3, (b, err, mag)
