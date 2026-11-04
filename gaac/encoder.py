"""Triton tile encoder: one program per (head, 16-token tile), bit-identical to Codec.encode.

Single pass over the BF16 input: tiers, tag words and 3-bit codes are computed in registers, sign/mantissa and code
bytes are written directly in the kernel's fragment order, and the tile's escape bytes go to a bounded per-tile
scratch slot (16 x 16 groups x 5 B = 1280 B worst case, ~32 B on average).  A second kernel packs the slots into
each head's escape stream (tile_off = exclusive prefix of the per-tile byte counts).
"""
from __future__ import annotations
import torch, triton, triton.language as tl
from .codec import Codec, Page, D, T, CB, perm, storage_index, row_offsets

SLOT = tl.constexpr(1280)
SLOT_P2 = tl.constexpr(2048)   # power-of-2 range for the compaction copy


@triton.jit
def _encode_tile(x_ptr, b3_ptr, b4_ptr, perm_ptr, inv_sm_ptr, inv_cd_ptr,
                 sm_out, cd_out, flags_out, slot_out, tile_bytes_out,
                 S, S_in, S_pad, n_kv, ntiles, bidx_ptr, USE_BIDX: tl.constexpr):
    h = tl.program_id(0); tile = tl.program_id(1)
    if USE_BIDX:
        kvh = tl.load(bidx_ptr + h)
    else:
        kvh = h % n_kv
    tok = tl.arange(0, 16)[:, None]
    dd = tl.arange(0, 128)[None, :]
    perm = tl.load(perm_ptr + dd)                   # stored position -> channel
    row = tile * 16 + tok
    valid = row < S
    x = tl.load(x_ptr + (h * S_in + row) * 128 + perm, mask=valid & (dd >= 0), other=0).to(tl.int32) & 0xFFFF
    e = (x >> 7) & 0xFF
    smb = ((x >> 8) & 0x80) | (x & 0x7F)
    b3 = tl.load(b3_ptr + kvh * 128 + perm).to(tl.int32)
    b4 = tl.load(b4_ptr + kvh * 128 + perm).to(tl.int32)
    d3 = e - b3; d4 = e - b4
    out3 = ((d3 < 0) | (d3 > 7)).to(tl.int32); out4 = ((d4 < 0) | (d4 > 15)).to(tl.int32)
    g3 = tl.sum(tl.reshape(out3, [16, 16, 8]), axis=2) > 0
    g4 = tl.sum(tl.reshape(out4, [16, 16, 8]), axis=2) > 0
    tier = tl.where(g3, tl.where(g4, 2, 1), 0)
    tier = tl.where(valid, tier, 0)                 # padded tokens: tier 0, no escapes
    tier_e = tl.reshape(tl.broadcast_to(tl.reshape(tier, [16, 16, 1]), [16, 16, 8]), [16, 128])
    code = tl.where(tier_e == 0, d3 & 7, tl.where(tier_e == 1, d4 & 7, e & 7))
    code = tl.where(valid, code, 0)
    inv_sm = tl.load(inv_sm_ptr + tok * 128 + dd)
    tl.store(sm_out + (h * S_pad + tile * 16) * 128 + inv_sm, smb.to(tl.uint8))
    sh = (tl.arange(0, 8) * 3)[None, None, :]
    w = tl.sum(tl.reshape(code, [16, 16, 8]) << sh, axis=2)      # 8 codes -> 24-bit word -> bytes 3g+p
    gg = tl.arange(0, 16)[None, :]
    for p in tl.static_range(3):
        by = (w >> (8 * p)) & 0xFF
        inv = tl.load(inv_cd_ptr + tok * 48 + 3 * gg + p)
        tl.store(cd_out + (h * S_pad + tile * 16) * 48 + inv, by.to(tl.uint8))
    fl = tl.sum(((tier >= 1).to(tl.int32) << (2 * gg)) | ((tier == 2).to(tl.int32) << (2 * gg + 1)), axis=1)
    tl.store(flags_out + h * S_pad + tile * 16 + tl.arange(0, 16), fl)
    size = tl.where(tier == 1, 1, tl.where(tier == 2, 5, 0))
    size_f = tl.reshape(size, [256])
    start = tl.cumsum(size_f, axis=0) - size_f
    total = tl.sum(size_f, axis=0)
    slot = slot_out + (h * ntiles + tile) * SLOT
    bit4 = tl.sum(tl.reshape((d4 >> 3) & 1, [16, 16, 8]) << tl.arange(0, 8)[None, None, :], axis=2)
    hi5 = tl.sum(tl.reshape((e >> 3).to(tl.int64), [16, 16, 8]) << (5 * tl.arange(0, 8))[None, None, :].to(tl.int64), axis=2)
    tier_f = tl.reshape(tier, [256]); bit4_f = tl.reshape(bit4, [256]); hi5_f = tl.reshape(hi5, [256])
    tl.store(slot + start, bit4_f.to(tl.uint8), mask=tier_f == 1)
    for k in tl.static_range(5):
        tl.store(slot + start + k, ((hi5_f >> (8 * k)) & 0xFF).to(tl.uint8), mask=tier_f == 2)
    tl.store(tile_bytes_out + h * ntiles + tile, total)


@triton.jit
def _compact(slot_ptr, tile_bytes_ptr, tile_off_ptr, head_start_ptr, ovf_out, ntiles):
    h = tl.program_id(0); tile = tl.program_id(1)
    n = tl.load(tile_bytes_ptr + h * ntiles + tile)
    dst = tl.load(head_start_ptr + h) + tl.load(tile_off_ptr + h * ntiles + tile)
    i = tl.arange(0, SLOT_P2)
    v = tl.load(slot_ptr + (h * ntiles + tile) * SLOT + i, mask=i < n, other=0)
    tl.store(ovf_out + dst + i, v, mask=i < n)


class TileEncoder:
    def __init__(self, kind: str, device="cuda"):
        self.kind = kind
        self.perm = perm(kind).to(device).to(torch.int32)
        si, ci = storage_index(kind, device)
        inv_s = torch.empty_like(si); inv_s[si] = torch.arange(si.numel(), device=device)
        inv_c = torch.empty_like(ci); inv_c[ci] = torch.arange(ci.numel(), device=device)
        self.inv_sm = inv_s.to(torch.int32).contiguous(); self.inv_cd = inv_c.to(torch.int32).contiguous()
        self._slots = None

    def _scratch(self, n: int, dev):
        if self._slots is None or self._slots.numel() < n:
            self._slots = torch.empty(n, dtype=torch.uint8, device=dev)
        return self._slots

    def encode_tiles(self, x, base3, base4, sm, cd, flags, S, S_pad, n_kv, bidx=None) -> torch.Tensor:
        """Encode x [H, S, D] (tiles 0..ceil(S/16)-1) into sm/cd/flags with token stride S_pad; escape bytes stay in the
        scratch slots.  Returns tile_bytes int32 [H, ntiles]; finish with compact()."""
        H = x.shape[0]; ntiles = (S + T - 1) // T; dev = x.device
        slots = self._scratch(H * ntiles * SLOT, dev)
        tile_bytes = torch.empty(H, ntiles, dtype=torch.int32, device=dev)
        _encode_tile[(H, ntiles)](x.contiguous().view(torch.int16), base3, base4, self.perm, self.inv_sm, self.inv_cd, sm, cd, flags, slots, tile_bytes,
                                  S, x.shape[1], S_pad, n_kv, ntiles, bidx if bidx is not None else tile_bytes, USE_BIDX=bidx is not None, num_warps=4)
        return tile_bytes

    def compact(self, tile_bytes, head_start, ovf) -> torch.Tensor:
        """Pack the scratch slots into ovf at head_start[h] + tile_off[h, t]; returns tile_off int32 [H, ntiles]."""
        H, ntiles = tile_bytes.shape
        tile_off = (torch.cumsum(tile_bytes, 1) - tile_bytes).to(torch.int32).contiguous()
        _compact[(H, ntiles)](self._slots, tile_bytes, tile_off, head_start, ovf, ntiles, num_warps=4)
        return tile_off

    def encode(self, codec: Codec, x: torch.Tensor) -> Page:
        """x: [H, S, D] bf16 -> Page, bit-identical to codec.encode(x, kind)."""
        H, S, _ = x.shape; dev = x.device
        S_pad = (S + T - 1) // T * T
        sm = torch.empty(H, S_pad * D, dtype=torch.uint8, device=dev)
        cd = torch.empty(H, S_pad * CB, dtype=torch.uint8, device=dev)
        flags = torch.zeros(H, S_pad, dtype=torch.int32, device=dev)
        tile_bytes = self.encode_tiles(x, codec.base3, codec.base4, sm, cd, flags, S, S_pad, codec.n_kv)
        head_bytes = tile_bytes.sum(1)
        head_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(head_bytes, 0)]).to(torch.int32).contiguous()
        ovf = torch.zeros(int(head_start[-1]) + 8, dtype=torch.uint8, device=dev)
        tile_off = self.compact(tile_bytes, head_start, ovf)
        return Page(sm, cd, flags, row_offsets(flags, S), ovf, head_start, tile_off, codec.base3, codec.base4, H, S, codec.n_kv, self.kind)
