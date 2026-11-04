import pytest, torch
from gaac import Codec, TileEncoder
from common import kv_like


@pytest.mark.parametrize("kind", ["k", "v"])
@pytest.mark.parametrize("S", [16, 100, 1024])
def test_triton_matches_reference_and_roundtrips(kind, S):
    n_kv = 4
    calib, x = kv_like(2 * n_kv, 2048, seed=1), kv_like(2 * n_kv, S, seed=2)
    codec = Codec(n_kv).calibrate(calib)
    ref = codec.encode(x, kind)
    fast = TileEncoder(kind).encode(codec, x)
    for f in ("sm", "codes", "flags", "ovf_off", "ovf_head_start", "tile_off"):
        assert torch.equal(getattr(ref, f), getattr(fast, f)), f
    n = int(ref.ovf_head_start[-1])
    assert torch.equal(ref.ovf[:n], fast.ovf[:n])
    for page in (ref, fast):
        assert torch.equal(Codec.decode(page).view(torch.int16), x.view(torch.int16))


def test_adversarial_never_expands_beyond_raw():
    """Uniformly random bit patterns (every group raw): still lossless, at most raw size + metadata."""
    x = torch.randint(-32768, 32767, (4, 256, 128), dtype=torch.int16, device="cuda")
    x = torch.where(((x >> 7) & 0xFF) == 0xFF, x & ~0x4000, x).view(torch.bfloat16)   # avoid NaN/Inf patterns
    codec = Codec(4).calibrate(x)
    page = TileEncoder("k").encode(codec, x)
    assert torch.equal(Codec.decode(page).view(torch.int16), x.view(torch.int16))
    assert page.nbytes <= x.numel() * 2 + 4 * 256 * 5
