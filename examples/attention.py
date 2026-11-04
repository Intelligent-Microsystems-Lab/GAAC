"""Fused decode attention over the compressed KV vs FlashAttention-2 over the BF16 KV.

  python examples/attention.py --batch 1 --n-kv 8 --group 4 --context 65536
"""
import argparse, sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaac import Codec, TileEncoder, attention

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1); ap.add_argument("--n-kv", type=int, default=8)
ap.add_argument("--group", type=int, default=4); ap.add_argument("--context", type=int, default=65536)
a = ap.parse_args()
B, n_kv, G, S = a.batch, a.n_kv, a.group, a.context
H = B * n_kv

g = torch.Generator(device="cuda").manual_seed(0)
def kv():
    scale = torch.exp2(torch.rand(1, 1, 128, generator=g, device="cuda") * 8 - 4)
    return (torch.randn(H, S, 128, generator=g, device="cuda") * scale).to(torch.bfloat16)
k, v = kv(), kv()
kp = TileEncoder("k").encode(Codec(n_kv).calibrate(k[:, :4096]), k)
vp = TileEncoder("v").encode(Codec(n_kv).calibrate(v[:, :4096]), v)
q = (torch.randn(B, n_kv * G, 128, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
scale = 128 ** -0.5

out = attention(q, kp, vp, scale)
kk = k.view(B, n_kv, S, 128).transpose(1, 2).contiguous(); vv = v.view(B, n_kv, S, 128).transpose(1, 2).contiguous()
try:
    from flash_attn import flash_attn_with_kvcache
    ref = lambda: flash_attn_with_kvcache(q.unsqueeze(1), kk, vv, softmax_scale=scale, causal=True).squeeze(1)
except ImportError:
    ref = lambda: torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(2), kk.transpose(1, 2), vv.transpose(1, 2), scale=scale, enable_gqa=True).squeeze(2)
r = ref()
print(f"max |GAAC - reference| = {(out.float() - r.float()).abs().max().item():.2e} (max |out| {r.float().abs().max().item():.2e})")

def t(fn, n=200, warm=20):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; ts = []
    for _ in range(warm): fn()
    for _ in range(n):
        ev[0].record(); fn(); ev[1].record(); torch.cuda.synchronize(); ts.append(ev[0].elapsed_time(ev[1]))
    return sorted(ts)[n // 2]
tg, tr = t(lambda: attention(q, kp, vp, scale)), t(ref)
print(f"B={B} n_kv={n_kv} G={G} context={S}: GAAC {tg:.3f} ms ({(kp.nbytes + vp.nbytes) / 2**20:.0f} MiB), "
      f"reference {tr:.3f} ms ({2 * k.numel() * 2 / 2**20:.0f} MiB), {tr / tg:.2f}x  [per-call CUDA events, median of 200]")
