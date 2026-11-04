"""Compressed KV cache of one attention layer for serving (used by the bundled nano-vllm).

Each sequence owns a slot = n_kv cache heads with a fixed token capacity.  A head holds compressed 16-token tiles
plus a raw BF16 tail of 0..15 tokens; when a decode step fills the tail, the tile is compressed in place (Triton) and
its escape bytes are appended to the head's escape stream.  All decode-step work (append + attention) runs on device
with no host synchronization and is CUDA-graph capturable: batch rows reach their slot through a device slot map,
and rows with slot -1 (graph padding) are skipped.

Exponent bases are either calibrated per model (static [n_kv, D] tables, gaac.calibration) or per sequence from its
own prompt at prefill (tables of [n_slots*n_kv, D]).
"""
from __future__ import annotations
import torch, triton, triton.language as tl
from .codec import Codec, Page, D, T, CB, perm, storage_index, row_offsets
from .encoder import TileEncoder
from .kernel import _ext, choose_nsplit

TAIL = 16
TILE_MAX_ESC = 1280    # worst-case escape bytes of one 16-token tile (all 256 groups raw)


@triton.jit
def _append_one(x_ptr, xrow, tail_ptr, b3_ptr, b4_ptr, brow, perm_ptr, inv_sm_ptr, inv_cd_ptr,
                sm_out, cd_out, flags_out, ovf_out, tile_off_out, ovf_used_ptr, h, pos, full, tile, S_cap, ovf_cap):
    """One cache head: write the new token into tail slot `pos`; if the tail is now full, compress the 16-token tile."""
    dd = tl.arange(0, 128)[None, :]
    xn = tl.load(x_ptr + xrow * 128 + dd)
    tl.store(tail_ptr + (h * 16 + pos) * 128 + dd, xn)
    tl.debug_barrier()
    if full != 0:
        tok = tl.arange(0, 16)[:, None]
        perm = tl.load(perm_ptr + dd)
        x = tl.load(tail_ptr + (h * 16 + tok) * 128 + perm).to(tl.int32) & 0xFFFF
        e = (x >> 7) & 0xFF; smb = ((x >> 8) & 0x80) | (x & 0x7F)
        b3 = tl.load(b3_ptr + brow * 128 + perm).to(tl.int32); b4 = tl.load(b4_ptr + brow * 128 + perm).to(tl.int32)
        d3 = e - b3; d4 = e - b4
        out3 = ((d3 < 0) | (d3 > 7)).to(tl.int32); out4 = ((d4 < 0) | (d4 > 15)).to(tl.int32)
        g3 = tl.sum(tl.reshape(out3, [16, 16, 8]), axis=2) > 0; g4 = tl.sum(tl.reshape(out4, [16, 16, 8]), axis=2) > 0
        tier = tl.where(g3, tl.where(g4, 2, 1), 0)
        tier_e = tl.reshape(tl.broadcast_to(tl.reshape(tier, [16, 16, 1]), [16, 16, 8]), [16, 128])
        code = tl.where(tier_e == 0, d3 & 7, tl.where(tier_e == 1, d4 & 7, e & 7))
        inv_sm = tl.load(inv_sm_ptr + tok * 128 + dd)
        tl.store(sm_out + (h * S_cap + tile * 16) * 128 + inv_sm, smb.to(tl.uint8))
        sh = (tl.arange(0, 8) * 3)[None, None, :]
        w = tl.sum(tl.reshape(code, [16, 16, 8]) << sh, axis=2)
        gg = tl.arange(0, 16)[None, :]
        for p in tl.static_range(3):
            by = (w >> (8 * p)) & 0xFF
            inv = tl.load(inv_cd_ptr + tok * 48 + 3 * gg + p)
            tl.store(cd_out + (h * S_cap + tile * 16) * 48 + inv, by.to(tl.uint8))
        fl = tl.sum(((tier >= 1).to(tl.int32) << (2 * gg)) | ((tier == 2).to(tl.int32) << (2 * gg + 1)), axis=1)
        tl.store(flags_out + h * S_cap + tile * 16 + tl.arange(0, 16), fl)
        size = tl.where(tier == 1, 1, tl.where(tier == 2, 5, 0)); size_f = tl.reshape(size, [256])
        start = tl.cumsum(size_f, axis=0) - size_f; total = tl.sum(size_f, axis=0)
        used = tl.load(ovf_used_ptr + h)
        base = ovf_out + h * ovf_cap + used
        bit4 = tl.sum(tl.reshape((d4 >> 3) & 1, [16, 16, 8]) << tl.arange(0, 8)[None, None, :], axis=2)
        hi5 = tl.sum(tl.reshape((e >> 3).to(tl.int64), [16, 16, 8]) << (5 * tl.arange(0, 8))[None, None, :].to(tl.int64), axis=2)
        tier_f = tl.reshape(tier, [256]); bit4_f = tl.reshape(bit4, [256]); hi5_f = tl.reshape(hi5, [256])
        tl.store(base + start, bit4_f.to(tl.uint8), mask=tier_f == 1)
        for k in tl.static_range(5):
            tl.store(base + start + k, ((hi5_f >> (8 * k)) & 0xFF).to(tl.uint8), mask=tier_f == 2)
        tl.store(tile_off_out + h * (S_cap // 16) + tile, used)
        tl.store(ovf_used_ptr + h, used + total)


@triton.jit
def _append_kv(kx_ptr, vx_ptr, slots_ptr, tail_len_ptr, S_ptr,
               ktail, kb3, kb4, kperm, kinvsm, kinvcd, ksm, kcd, kfl, kovf, ktoff, kused,
               vtail, vb3, vb4, vperm, vinvsm, vinvcd, vsm, vcd, vfl, vovf, vtoff, vused,
               S_cap, ovf_cap, n_kv, NB):
    """grid (B*n_kv, 2): row i = b*n_kv + kvh of the step's new K (axis 1 = 0) or V (= 1) goes to cache head SLOT[b]*n_kv + kvh.
    The shared counters (tail_len, S) are read here and advanced afterwards by _advance."""
    i = tl.program_id(0); kind = tl.program_id(1)
    slot = tl.load(slots_ptr + i // n_kv)
    if slot >= 0:
        h = slot * n_kv + i % n_kv
        pos = tl.load(tail_len_ptr + h)
        full = ((pos + 1) == 16).to(tl.int32)
        tile = tl.load(S_ptr + h) // 16
        if kind == 0:
            _append_one(kx_ptr, i, ktail, kb3, kb4, h % NB, kperm, kinvsm, kinvcd, ksm, kcd, kfl, kovf, ktoff, kused, h, pos, full, tile, S_cap, ovf_cap)
        else:
            _append_one(vx_ptr, i, vtail, vb3, vb4, h % NB, vperm, vinvsm, vinvcd, vsm, vcd, vfl, vovf, vtoff, vused, h, pos, full, tile, S_cap, ovf_cap)


@triton.jit
def _advance(slots_ptr, tail_len_ptr, S_ptr, n_kv):
    i = tl.program_id(0)
    slot = tl.load(slots_ptr + i // n_kv)
    if slot >= 0:
        h = slot * n_kv + i % n_kv
        t = tl.load(tail_len_ptr + h) + 1
        full = (t == 16).to(tl.int32)
        tl.store(tail_len_ptr + h, t - 16 * full)
        tl.store(S_ptr + h, tl.load(S_ptr + h) + 16 * full)


class _Store:
    """Compressed buffers of one kind ('k' or 'v') for H cache heads."""
    def __init__(self, kind, H, cap, ovf_cap, base3, base4, dev):
        self.kind, self.H, self.ovf_cap = kind, H, ovf_cap
        self.sm = torch.zeros(H, cap * D, dtype=torch.uint8, device=dev)
        self.codes = torch.zeros(H, cap * CB, dtype=torch.uint8, device=dev)
        self.flags = torch.zeros(H, cap, dtype=torch.int32, device=dev)
        self.tile_off = torch.zeros(H, cap // T, dtype=torch.int32, device=dev)
        self.ovf = torch.zeros(H * ovf_cap, dtype=torch.uint8, device=dev)
        self.head_start = (torch.arange(H + 1, device=dev, dtype=torch.int32) * ovf_cap).contiguous()
        self.ovf_used = torch.zeros(H, dtype=torch.int32, device=dev)
        self.tail = torch.zeros(H, TAIL, D, dtype=torch.bfloat16, device=dev)
        self.base3, self.base4 = base3, base4
        self.enc = TileEncoder(kind, dev)
        self.perm, self.inv_sm, self.inv_cd = self.enc.perm, self.enc.inv_sm, self.enc.inv_cd

    def grow(self, ovf_cap):
        new = torch.zeros(self.H * ovf_cap, dtype=torch.uint8, device=self.ovf.device)
        new.view(self.H, ovf_cap)[:, :self.ovf_cap] = self.ovf.view(self.H, self.ovf_cap)
        self.ovf, self.ovf_cap = new, ovf_cap
        self.head_start = (torch.arange(self.H + 1, device=new.device, dtype=torch.int32) * ovf_cap).contiguous()


class CompressedKVCache:
    def __init__(self, n_slots: int, n_kv: int, capacity: int, bases: dict | None = None,
                 ovf_bytes_per_tile: int = 128, device="cuda"):
        """capacity: max tokens per sequence; bases: {'k3','k4','v3','v4'} uint8 [n_kv, D] (None = per-sequence calibration);
        ovf_bytes_per_tile: escape-stream reservation per 16-token tile (mean use ~32 B with calibrated bases)."""
        self.n_slots, self.n_kv = n_slots, n_kv
        self.cap = (capacity + T - 1) // T * T
        self.H = H = n_slots * n_kv
        self.per_sequence_bases = bases is None
        NB = H if bases is None else n_kv
        def table(key):
            if bases is None: return torch.zeros(NB, D, dtype=torch.uint8, device=device)
            return bases[key].to(device, torch.uint8).reshape(n_kv, D).contiguous()
        ovf_cap = (self.cap // T) * ovf_bytes_per_tile + 8
        self.k = _Store("k", H, self.cap, ovf_cap, table("k3"), table("k4"), device)
        self.v = _Store("v", H, self.cap, ovf_cap, table("v3"), table("v4"), device)
        self.S = torch.zeros(H, dtype=torch.int32, device=device)          # compressed tokens per head
        self.tail_len = torch.zeros(H, dtype=torch.int32, device=device)
        self.growable = True       # set False once CUDA graphs captured the buffer addresses

    @staticmethod
    def bytes_per_slot(n_kv: int, capacity: int, ovf_bytes_per_tile: int = 128, per_sequence_bases: bool = True) -> int:
        cap = (capacity + T - 1) // T * T
        per_head = cap * (D + CB + 4) + (cap // T) * 4 + (cap // T) * ovf_bytes_per_tile + 8 + 8 + TAIL * D * 2 + (2 * D if per_sequence_bases else 0)
        return 2 * n_kv * per_head + 8 * n_kv

    # ------------------------------------------------------------------ prefill
    def prefill(self, slot: int, k: torch.Tensor, v: torch.Tensor):
        """Store a sequence's prompt K/V ([L, n_kv, D] bf16, token-major as produced by the model) in `slot`."""
        L = k.shape[0]
        assert L <= self.cap, f"sequence of {L} tokens exceeds the cache capacity {self.cap}"
        h0, h1 = slot * self.n_kv, (slot + 1) * self.n_kv
        Sf = L // T * T
        for st, x in ((self.k, k), (self.v, v)):
            xh = x.transpose(0, 1).contiguous()                                # [n_kv, L, D]
            if self.per_sequence_bases:
                c = Codec(self.n_kv, x.device).calibrate(xh)
                st.base3[h0:h1] = c.base3; st.base4[h0:h1] = c.base4
            st.ovf_used[h0:h1] = 0
            if Sf:
                bidx = (torch.arange(h0, h1, device=x.device, dtype=torch.int32) % st.base3.shape[0]).contiguous()
                tb = st.enc.encode_tiles(xh[:, :Sf], st.base3, st.base4, st.sm[h0:h1], st.codes[h0:h1], st.flags[h0:h1], Sf, self.cap, self.n_kv, bidx=bidx)
                need = int(tb.sum(1).max()) + TILE_MAX_ESC + 8        # headroom for the next decode tile before check_capacity runs
                if need > st.ovf_cap: self._grow(need)
                st.tile_off[h0:h1, :Sf // T] = st.enc.compact(tb, st.head_start[h0:h1 + 1], st.ovf)
                st.ovf_used[h0:h1] = tb.sum(1).to(torch.int32)
            if L > Sf: st.tail[h0:h1, :L - Sf] = xh[:, Sf:]
        self.S[h0:h1] = Sf; self.tail_len[h0:h1] = L - Sf

    # ------------------------------------------------------------------ decode step
    def append(self, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """One new token per batch row: slots int32 [B] (-1 = skip), k/v [B, n_kv, D] bf16.  Two launches, graph-safe."""
        k, v = k.contiguous(), v.contiguous()
        K, V = self.k, self.v
        n = slots.numel() * self.n_kv
        _append_kv[(n, 2)](k.view(torch.int16), v.view(torch.int16), slots, self.tail_len, self.S,
                           K.tail.view(torch.int16), K.base3, K.base4, K.perm, K.inv_sm, K.inv_cd, K.sm, K.codes, K.flags, K.ovf, K.tile_off, K.ovf_used,
                           V.tail.view(torch.int16), V.base3, V.base4, V.perm, V.inv_sm, V.inv_cd, V.sm, V.codes, V.flags, V.ovf, V.tile_off, V.ovf_used,
                           self.cap, K.ovf_cap, self.n_kv, K.base3.shape[0], num_warps=4)
        _advance[(n,)](slots, self.tail_len, self.S, self.n_kv, num_warps=1)

    def attention(self, q: torch.Tensor, slots: torch.Tensor, scale: float, max_len: int | None = None, nsplit: int | None = None):
        """q: [B, n_q, D] bf16 (row b attends over slot slots[b]) -> [B, n_q, D] bf16."""
        B, n_q, _ = q.shape
        if nsplit is None: nsplit = choose_nsplit(B, self.n_kv, max_len or self.cap)
        K, V = self.k, self.v
        out = _ext().gaac_attn_launch(K.sm, K.codes, K.flags, K.tile_off, K.ovf, K.head_start, K.base3, K.base4,
                                      V.sm, V.codes, V.flags, V.tile_off, V.ovf, V.head_start, V.base3, V.base4,
                                      q.reshape(B * n_q, D).contiguous(), slots, n_q, self.n_kv, self.S, self.cap, scale, nsplit,
                                      K.tail, V.tail, self.tail_len, TAIL)
        return out.view(B, n_q, D)

    # ------------------------------------------------------------------ housekeeping
    def max_escape_bytes(self) -> torch.Tensor:
        return torch.maximum(self.k.ovf_used.max(), self.v.ovf_used.max())

    def check_capacity(self, used: int | None = None):
        """Call at least once every 16 decode steps: each head finalizes at most one tile in 16 steps, so keeping one
        worst-case tile of headroom makes the escape streams overflow-proof between checks."""
        used = int(self.max_escape_bytes()) if used is None else used
        if used + TILE_MAX_ESC + 8 > self.k.ovf_cap: self._grow(used + TILE_MAX_ESC + 8)

    def _grow(self, need: int):
        if not self.growable:
            raise RuntimeError(f"escape stream needs {need} B per head but holds {self.k.ovf_cap} B and CUDA graphs are captured; "
                               "raise gaac_ovf_bytes_per_tile or calibrate the exponent bases on representative prompts")
        cap = max(2 * self.k.ovf_cap, need + 4096 + 7) // 8 * 8
        self.k.grow(cap); self.v.grow(cap)

    def decompress(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Bit-exact BF16 K/V of `slot` ([L, n_kv, D] each) from the compressed tiles + raw tail (reference decoder)."""
        h0, h1 = slot * self.n_kv, (slot + 1) * self.n_kv
        S, tl_ = int(self.S[h0]), int(self.tail_len[h0])
        out = []
        for st in (self.k, self.v):
            parts = []
            if S:
                per_seq = st.base3.shape[0] != self.n_kv
                b3, b4 = (st.base3[h0:h1], st.base4[h0:h1]) if per_seq else (st.base3, st.base4)
                pg = Page(st.sm[h0:h1, :S * D].contiguous(), st.codes[h0:h1, :S * CB].contiguous(), st.flags[h0:h1, :S].contiguous(), row_offsets(st.flags[h0:h1], S),
                          st.ovf, st.head_start[h0:h1 + 1], st.tile_off[h0:h1], b3, b4, self.n_kv, S, self.n_kv, st.kind)
                parts.append(Codec.decode(pg))
            parts.append(st.tail[h0:h1, :tl_])
            out.append(torch.cat(parts, 1).transpose(0, 1).contiguous())
        return out[0], out[1]

    def stats(self) -> dict:
        """Bytes in use (compressed tiles + escapes + metadata + raw tails) vs the BF16 bytes of the same tokens."""
        S = int(self.S.sum()); tl_ = int(self.tail_len.sum())
        esc = int(self.k.ovf_used.sum() + self.v.ovf_used.sum())
        used = 2 * (S * (D + CB + 4) + (S // T) * 4 + tl_ * D * 2) + esc
        bf16 = 2 * (S + tl_) * D * 2
        return dict(tokens=(S + tl_) // max(1, self.n_kv), bytes_used=used, bf16_bytes=bf16, ratio=bf16 / max(1, used))
