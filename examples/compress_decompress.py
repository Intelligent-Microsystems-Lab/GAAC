"""Compression / decompression workflow on one layer's K or V.

  python examples/compress_decompress.py                       # synthetic KV (kernel sanity check)
  python examples/compress_decompress.py --input kv.pt --kind k --n-kv 8

--input: a BF16 tensor [H, S, 128] (head-major) saved with torch.save; the first --calib-tokens tokens calibrate the
exponent bases and the rest is compressed (calibrating on the tensor itself is only a sanity check, not the protocol).
"""
import argparse, sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaac import Codec, TileEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--input"); ap.add_argument("--kind", default="k", choices=["k", "v"])
ap.add_argument("--n-kv", type=int, default=8); ap.add_argument("--tokens", type=int, default=65536)
ap.add_argument("--calib-tokens", type=int, default=4096)
a = ap.parse_args()

if a.input:
    x = torch.load(a.input).cuda().to(torch.bfloat16)
    calib, x = x[:, :a.calib_tokens], x[:, a.calib_tokens:]
else:
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = torch.exp2(torch.rand(a.n_kv, 1, 128, generator=g, device="cuda") * 8 - 4)     # per-channel magnitudes
    data = (torch.randn(a.n_kv, a.calib_tokens + a.tokens, 128, generator=g, device="cuda") * scale).to(torch.bfloat16)
    calib, x = data[:, :a.calib_tokens], data[:, a.calib_tokens:]

# 1. calibrate the per-(head, channel) exponent windows (offline, once per model in deployment)
codec = Codec(a.n_kv).calibrate(calib)
# 2. compress (Triton, one program per head x 16-token tile)
enc = TileEncoder(a.kind)
page = enc.encode(codec, x)
# 3. decompress (reference decoder) and check bitwise equality
y = Codec.decode(page)
assert torch.equal(x.view(torch.int16), y.view(torch.int16)), "round trip is not bit-exact"

f = page.flags[:, :page.S].to(torch.int64) & 0xFFFFFFFF
t1 = sum(int((((f >> (2 * g)) & 3) == 1).sum()) for g in range(16)); t2 = sum(int((((f >> (2 * g)) & 3) == 3).sum()) for g in range(16))
ng = page.H * page.S * 16
print(f"{tuple(x.shape)} {a.kind}: {x.numel() * 2 / 2**20:.1f} MiB -> {page.nbytes / 2**20:.1f} MiB, ratio {page.ratio:.3f}x, "
      f"groups: {100 * (1 - (t1 + t2) / ng):.2f}% 3-bit, {100 * t1 / ng:.2f}% 4-bit, {100 * t2 / ng:.3f}% raw; bit-exact round trip")

# encode time per call (CUDA events; no graph replay)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
for _ in range(5): enc.encode(codec, x)
ts = []
for _ in range(20):
    ev[0].record(); enc.encode(codec, x); ev[1].record(); torch.cuda.synchronize(); ts.append(ev[0].elapsed_time(ev[1]))
ts.sort(); print(f"encode: {ts[len(ts) // 2]:.3f} ms median ({x.numel() * 2 / ts[len(ts) // 2] / 1e6:.1f} GB/s, includes page allocation)")
