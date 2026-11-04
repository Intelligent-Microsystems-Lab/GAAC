"""Store path of the GAAC vLLM backend: writes new K/V tokens into vLLM's compressed block buffer (gaac/vllm/paged_kernel.py layout).

One Triton program per (touched block entry e, tile slot t = 2*kvh + kind).  Entry e (built on the host from query_start_loc /
num_computed_tokens) says: block id, first token row of this step in the K/V tensors, block offset of that row, number of new rows.
  * block not full after this step  -> new rows go to the raw side pool RAW[r] (r = the block's raw tile, popped from a per-layer
                                       free stack when the block starts), MT[p] = ~r                          (filling block)
  * block completes this step       -> gather 16 rows (old rows from RAW[r], new rows from K/V), GAAC-compress into the block's
                                       tile slot when the overflow fits the slot (MT[p] = p, raw tile queued for release) or
                                       keep the tile raw (MT[p] = ~r, bounded metadata, lossless)            (finalize)
Pops (atomic) happen only in `_store_tiles`, pushes only in `_free_tiles`, so the free stack never sees a pop racing a push.
A raw tile of a block's previous life (MT[p] < 0 at block offset 0) is simply reused for the new life.
"""
from __future__ import annotations
import torch, triton, triton.language as tl
from ..codec import perm as _perm, storage_index

D = 128
SM_OFF, CD_OFF, FL_OFF, OVF_OFF = 0, 2048, 2816, 2880   # byte layout of a tile slot; TS = OVF_OFF + OVF_SLOT
SM_OFF_C, CD_OFF_C, FL_OFF_C, OVF_OFF_C = tl.constexpr(SM_OFF), tl.constexpr(CD_OFF), tl.constexpr(FL_OFF), tl.constexpr(OVF_OFF)


@triton.jit
def _store_tiles(k_ptr, v_ptr, k_stride, v_stride, ent_ptr, n_ent,
                 buf_ptr, bufi_ptr, raw_ptr, mt_ptr, stack_ptr, top_ptr, err_ptr, freeq_ptr,
                 b3k_ptr, b4k_ptr, b3v_ptr, b4v_ptr, permk_ptr, permv_ptr, invsmk_ptr, invcdk_ptr, invsmv_ptr, invcdv_ptr,
                 n_kv, TS, OVF_SLOT):
    e = tl.program_id(0); t = tl.program_id(1)
    kind = t & 1; kvh = t >> 1
    blk = tl.load(ent_ptr + e * 4 + 0); tok0 = tl.load(ent_ptr + e * 4 + 1); off0 = tl.load(ent_ptr + e * 4 + 2); n = tl.load(ent_ptr + e * 4 + 3)
    act = blk >= 0                                   # blk < 0: inactive entry (graph capture dummy)
    p = tl.where(act, blk, 0) * (2 * n_kv) + t
    old = tl.load(mt_ptr + p, mask=act, other=0)
    have = (old < 0) & act
    full = ((off0 + n) == 16) & act
    tok = tl.arange(0, 16)[:, None]; dd = tl.arange(0, 128)[None, :]
    is_new = (tok >= off0) & (tok < off0 + n)
    is_old = (tok < off0) & have
    r_old = tl.where(have, -old - 1, 0)
    # per-kind tables (scalar branch: both K and V variants are loaded through the same code path)
    if kind == 0:
        perm = tl.load(permk_ptr + dd); b3 = tl.load(b3k_ptr + kvh * 128 + perm).to(tl.int32); b4 = tl.load(b4k_ptr + kvh * 128 + perm).to(tl.int32)
        inv_sm = tl.load(invsmk_ptr + tok * 128 + dd)
        xn = tl.load(k_ptr + (tok0 + tok - off0).to(tl.int64) * k_stride + kvh * 128 + perm, mask=is_new, other=0).to(tl.int32) & 0xFFFF
        xr = tl.load(k_ptr + (tok0 + tok - off0).to(tl.int64) * k_stride + kvh * 128 + dd, mask=is_new, other=0)
    else:
        perm = tl.load(permv_ptr + dd); b3 = tl.load(b3v_ptr + kvh * 128 + perm).to(tl.int32); b4 = tl.load(b4v_ptr + kvh * 128 + perm).to(tl.int32)
        inv_sm = tl.load(invsmv_ptr + tok * 128 + dd)
        xn = tl.load(v_ptr + (tok0 + tok - off0).to(tl.int64) * v_stride + kvh * 128 + perm, mask=is_new, other=0).to(tl.int32) & 0xFFFF
        xr = tl.load(v_ptr + (tok0 + tok - off0).to(tl.int64) * v_stride + kvh * 128 + dd, mask=is_new, other=0)
    # ---- gather the tile: old rows from the raw tile (if any), new rows from this step's K/V (int16 views: raw bf16 bits)
    xo = tl.load(raw_ptr + (r_old.to(tl.int64) * 16 + tok) * 128 + perm, mask=is_old, other=0).to(tl.int32) & 0xFFFF
    x = tl.where(is_new, xn, xo)
    if full:                                          # scalar branch: compression math + compressed stores only on finalizing entries
        # ---- tile compression (same math as gaac/kv_cache._append_one)
        e8 = (x >> 7) & 0xFF; smb = ((x >> 8) & 0x80) | (x & 0x7F)
        d3 = e8 - b3; d4 = e8 - b4
        out3 = ((d3 < 0) | (d3 > 7)).to(tl.int32); out4 = ((d4 < 0) | (d4 > 15)).to(tl.int32)
        g3 = tl.sum(tl.reshape(out3, [16, 16, 8]), axis=2) > 0; g4 = tl.sum(tl.reshape(out4, [16, 16, 8]), axis=2) > 0
        tier = tl.where(g3, tl.where(g4, 2, 1), 0)
        tier_e = tl.reshape(tl.broadcast_to(tl.reshape(tier, [16, 16, 1]), [16, 16, 8]), [16, 128])
        code = tl.where(tier_e == 0, d3 & 7, tl.where(tier_e == 1, d4 & 7, e8 & 7))
        size = tl.where(tier == 1, 1, tl.where(tier == 2, 5, 0)); size_f = tl.reshape(size, [256])
        start = tl.cumsum(size_f, axis=0) - size_f; total = tl.sum(size_f, axis=0)
        fits = total <= OVF_SLOT
        # ---- raw tile needed only when the tile does not fit
        need = fits == 0
        pop = need & (have == 0) & act
        idx = tl.atomic_add(top_ptr, tl.where(pop, -1, 0)) - 1
        r_new = tl.load(stack_ptr + tl.maximum(idx, 0), mask=pop & (idx >= 0), other=0)
        tl.atomic_add(err_ptr, tl.where(pop & (idx < 0), 1, 0))
        r = tl.where(have, r_old, r_new)
        # ---- compressed store (fits): tile slot p of the block buffer
        base = buf_ptr + p.to(tl.int64) * TS
        tl.store(base + SM_OFF_C + inv_sm, smb.to(tl.uint8), mask=(inv_sm >= 0) & fits)
        sh = (tl.arange(0, 8) * 3)[None, None, :]
        w = tl.sum(tl.reshape(code, [16, 16, 8]) << sh, axis=2); gg = tl.arange(0, 16)[None, :]
        for q in tl.static_range(3):
            by = (w >> (8 * q)) & 0xFF
            if kind == 0:
                inv = tl.load(invcdk_ptr + tok * 48 + 3 * gg + q)
            else:
                inv = tl.load(invcdv_ptr + tok * 48 + 3 * gg + q)
            tl.store(base + CD_OFF_C + inv, by.to(tl.uint8), mask=(inv >= 0) & fits)
        fl = tl.sum(((tier >= 1).to(tl.int32) << (2 * gg)) | ((tier == 2).to(tl.int32) << (2 * gg + 1)), axis=1)
        tl.store(bufi_ptr + (p.to(tl.int64) * TS + FL_OFF_C) // 4 + tl.arange(0, 16), fl, mask=(tl.arange(0, 16) >= 0) & fits)
        ob = base + OVF_OFF_C
        bit4 = tl.sum(tl.reshape((d4 >> 3) & 1, [16, 16, 8]) << tl.arange(0, 8)[None, None, :], axis=2)
        hi5 = tl.sum(tl.reshape((e8 >> 3).to(tl.int64), [16, 16, 8]) << (5 * tl.arange(0, 8))[None, None, :].to(tl.int64), axis=2)
        tier_f = tl.reshape(tier, [256]); bit4_f = tl.reshape(bit4, [256]); hi5_f = tl.reshape(hi5, [256])
        tl.store(ob + start, bit4_f.to(tl.uint8), mask=(tier_f == 1) & fits)
        for q in tl.static_range(5):
            tl.store(ob + start + q, ((hi5_f >> (8 * q)) & 0xFF).to(tl.uint8), mask=(tier_f == 2) & fits)
    else:                                             # filling block: the new rows go to the raw tile (popped when the block starts)
        fits = full
        need = act
        pop = need & (have == 0)
        idx = tl.atomic_add(top_ptr, tl.where(pop, -1, 0)) - 1
        r_new = tl.load(stack_ptr + tl.maximum(idx, 0), mask=pop & (idx >= 0), other=0)
        tl.atomic_add(err_ptr, tl.where(pop & (idx < 0), 1, 0))
        r = tl.where(have, r_old, r_new)
    # ---- raw store (not fits): new rows (natural dim order) into raw tile r
    tl.store(raw_ptr + (r.to(tl.int64) * 16 + tok) * 128 + dd, xr, mask=is_new & need)
    # ---- mode table + release queue
    tl.store(mt_ptr + p, tl.where(fits, p, -r - 1), mask=act)
    tl.store(freeq_ptr + e * (2 * n_kv) + t, tl.where(fits & have, r_old, -1))


@triton.jit
def _free_tiles(freeq_ptr, n_items, stack_ptr, top_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0); i = pid * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(freeq_ptr + i, mask=i < n_items, other=-1)
    push = r >= 0
    idx = tl.atomic_add(top_ptr + tl.zeros([BLOCK], dtype=tl.int32), tl.full([BLOCK], 1, tl.int32), mask=push)
    tl.store(stack_ptr + idx, r, mask=push)


@triton.jit
def _decode_tiles(buf_ptr, bufi_ptr, raw_ptr, mt_ptr, pairs_ptr, sk_ptr, sv_ptr, row_stride,
                  b3k_ptr, b4k_ptr, b3v_ptr, b4v_ptr, permk_ptr, permv_ptr, invsmk_ptr, invcdk_ptr, invsmv_ptr, invcdv_ptr, n_kv, TS):
    """Expand one (vLLM block, tile slot) into bf16 rows of a flat scratch K/V [rows, n_kv, D] (varlen layout, int16 views).
    pairs[e] = (vLLM block id, first scratch row, number of valid rows <= 16)."""
    e = tl.program_id(0); t = tl.program_id(1); kind = t & 1; kvh = t >> 1
    blk = tl.load(pairs_ptr + e * 3); row0 = tl.load(pairs_ptr + e * 3 + 1); nvalid = tl.load(pairs_ptr + e * 3 + 2)
    p = blk * (2 * n_kv) + t; m = tl.load(mt_ptr + p)
    tok = tl.arange(0, 16)[:, None]; dd = tl.arange(0, 128)[None, :]
    if kind == 0:
        perm = tl.load(permk_ptr + dd); b3 = tl.load(b3k_ptr + kvh * 128 + perm).to(tl.int32); b4 = tl.load(b4k_ptr + kvh * 128 + perm).to(tl.int32)
        inv_sm = tl.load(invsmk_ptr + tok * 128 + dd); gg = tl.arange(0, 16)[None, :]
        c0 = tl.load(invcdk_ptr + tok * 48 + 3 * gg); c1 = tl.load(invcdk_ptr + tok * 48 + 3 * gg + 1); c2 = tl.load(invcdk_ptr + tok * 48 + 3 * gg + 2)
    else:
        perm = tl.load(permv_ptr + dd); b3 = tl.load(b3v_ptr + kvh * 128 + perm).to(tl.int32); b4 = tl.load(b4v_ptr + kvh * 128 + perm).to(tl.int32)
        inv_sm = tl.load(invsmv_ptr + tok * 128 + dd); gg = tl.arange(0, 16)[None, :]
        c0 = tl.load(invcdv_ptr + tok * 48 + 3 * gg); c1 = tl.load(invcdv_ptr + tok * 48 + 3 * gg + 1); c2 = tl.load(invcdv_ptr + tok * 48 + 3 * gg + 2)
    is_raw = m < 0
    r = tl.where(is_raw, -m - 1, 0)
    xraw = tl.load(raw_ptr + (r.to(tl.int64) * 16 + tok) * 128 + dd, mask=is_raw & (dd >= 0), other=0)
    base = buf_ptr + p.to(tl.int64) * TS
    cm = (is_raw == 0)
    smb = tl.load(base + SM_OFF_C + inv_sm, mask=cm & (inv_sm >= 0), other=0).to(tl.int32)
    w = tl.load(base + CD_OFF_C + c0, mask=cm & (c0 >= 0), other=0).to(tl.int32) | (tl.load(base + CD_OFF_C + c1, mask=cm & (c1 >= 0), other=0).to(tl.int32) << 8) | (tl.load(base + CD_OFF_C + c2, mask=cm & (c2 >= 0), other=0).to(tl.int32) << 16)   # [16,16] group words
    fl = tl.load(bufi_ptr + (p.to(tl.int64) * TS + FL_OFF_C) // 4 + tl.arange(0, 16)[:, None], mask=cm & (tok >= 0), other=0)            # [16,1]
    tier = (fl >> (2 * gg)) & 3                                                                                                        # [16,16] 0/1/3
    size = tl.where(tier == 1, 1, tl.where(tier == 3, 5, 0)); size_f = tl.reshape(size, [256]); start_f = tl.cumsum(size_f, axis=0) - size_f
    ob = base + OVF_OFF_C
    ext1 = tl.load(ob + start_f, mask=cm & (tl.reshape(tier, [256]) == 1), other=0).to(tl.int32)
    hi5 = tl.zeros([256], dtype=tl.int64)
    for q in tl.static_range(5):
        hi5 = hi5 | (tl.load(ob + start_f + q, mask=cm & (tl.reshape(tier, [256]) == 3), other=0).to(tl.int64) << (8 * q))
    ext1 = tl.reshape(ext1, [16, 16]); hi5 = tl.reshape(hi5, [16, 16])
    # per element: group g = dd // 8, i = dd % 8
    w_e = tl.reshape(tl.broadcast_to(tl.reshape(w, [16, 16, 1]), [16, 16, 8]), [16, 128]); i8 = dd & 7
    code = (w_e >> (3 * i8)) & 7
    tier_e = tl.reshape(tl.broadcast_to(tl.reshape(tier, [16, 16, 1]), [16, 16, 8]), [16, 128])
    ext1_e = tl.reshape(tl.broadcast_to(tl.reshape(ext1, [16, 16, 1]), [16, 16, 8]), [16, 128]); hi5_e = tl.reshape(tl.broadcast_to(tl.reshape(hi5, [16, 16, 1]), [16, 16, 8]), [16, 128])
    e_exp = tl.where(tier_e == 0, b3 + code, tl.where(tier_e == 1, b4 + code + (((ext1_e >> i8) & 1) << 3), ((hi5_e >> (5 * i8)) & 31).to(tl.int32) << 3 | code))
    val = ((smb & 0x80) << 8) | (e_exp << 7) | (smb & 0x7F)
    ok = tok < nvalid
    if kind == 0:
        orow = sk_ptr + (row0.to(tl.int64) + tok) * row_stride + kvh * 128
    else:
        orow = sv_ptr + (row0.to(tl.int64) + tok) * row_stride + kvh * 128
    tl.store(orow + perm, val.to(tl.int16), mask=cm & ok)
    tl.store(orow + dd, xraw, mask=is_raw & ok)


@triton.jit
def _release_blocks(blk_ptr, n_blk, mt_ptr, stack_ptr, top_ptr, NT: tl.constexpr):
    """Blocks freed by vLLM (prefix caching off): give their raw tiles back to the free stack and reset the mode table."""
    e = tl.program_id(0); t = tl.arange(0, NT)
    blk = tl.load(blk_ptr + e); p = blk * NT + t
    m = tl.load(mt_ptr + p); push = m < 0
    idx = tl.atomic_add(top_ptr + tl.zeros([NT], dtype=tl.int32), tl.full([NT], 1, tl.int32), mask=push)
    tl.store(stack_ptr + idx, -m - 1, mask=push)
    tl.store(mt_ptr + p, p, mask=push)


class LayerState:
    """Per-layer device state outside vLLM's block buffer: raw side pool, mode table, free stack, bases and layout tables."""
    _tables = {}

    def __init__(self, num_blocks: int, n_kv: int, raw_tiles: int, bases: dict, ovf_slot: int, device):
        self.n_kv, self.ovf_slot, self.TS = n_kv, ovf_slot, OVF_OFF + ovf_slot
        assert self.TS % 16 == 0, "OVF_SLOT must be a multiple of 16"
        self.mt = torch.arange(num_blocks * 2 * n_kv, dtype=torch.int32, device=device)          # every tile starts 'compressed at itself'
        self.raw = torch.zeros(raw_tiles, 16, D, dtype=torch.bfloat16, device=device)
        self.stack = torch.arange(raw_tiles, dtype=torch.int32, device=device); self.top = torch.tensor([raw_tiles], dtype=torch.int32, device=device)
        self.err = torch.zeros(1, dtype=torch.int32, device=device)
        self.b3k, self.b4k, self.b3v, self.b4v = (bases[k].to(device).to(torch.uint8).contiguous() for k in ("k3", "k4", "v3", "v4"))
        key = str(device)
        if key not in LayerState._tables:
            tabs = []
            for kind in ("k", "v"):
                perm = _perm(kind).to(device).to(torch.int32).contiguous()
                si, ci = storage_index(kind, device)
                inv_s = torch.empty_like(si); inv_s[si] = torch.arange(si.numel(), device=device); inv_c = torch.empty_like(ci); inv_c[ci] = torch.arange(ci.numel(), device=device)
                tabs += [perm, inv_s.to(torch.int32).contiguous(), inv_c.to(torch.int32).contiguous()]
            LayerState._tables[key] = tabs
        self.permk, self.invsmk, self.invcdk, self.permv, self.invsmv, self.invcdv = LayerState._tables[key]

    @property
    def bases(self):
        return (self.b3k, self.b4k, self.b3v, self.b4v)

    def store(self, key: torch.Tensor, value: torch.Tensor, buf: torch.Tensor, ent: torch.Tensor, freeq: torch.Tensor):
        """key/value: [num_tokens, n_kv, D] int16 views of the bf16 tensors; buf: uint8 flat block buffer; ent: int32 [E,4] = (blk, tok0, off0, n); freeq: int32 [E*2*n_kv] scratch."""
        E = ent.shape[0]
        if E == 0: return
        assert key.stride(-1) == 1 and key.stride(1) == D and value.stride(-1) == 1 and value.stride(1) == D
        _store_tiles[(E, 2 * self.n_kv)](key, value, key.stride(0), value.stride(0), ent, E, buf, buf.view(torch.int32), self.raw.view(torch.int16), self.mt, self.stack, self.top, self.err, freeq,
                                          self.b3k, self.b4k, self.b3v, self.b4v, self.permk, self.permv, self.invsmk, self.invcdk, self.invsmv, self.invcdv,
                                          self.n_kv, self.TS, self.ovf_slot, num_warps=4)
        n = E * 2 * self.n_kv
        _free_tiles[((n + 255) // 256,)](freeq, n, self.stack, self.top, BLOCK=256, num_warps=4)

    def decode_blocks(self, buf: torch.Tensor, pairs: torch.Tensor, sk: torch.Tensor, sv: torch.Tensor):
        """pairs: int32 [P,3] (vLLM block, first scratch row, valid rows); sk/sv: flat scratch K/V [rows, n_kv, D] bf16 (written)."""
        P = pairs.shape[0]
        if P == 0: return
        assert sk.stride(0) == self.n_kv * D and sv.stride(0) == self.n_kv * D
        _decode_tiles[(P, 2 * self.n_kv)](buf, buf.view(torch.int32), self.raw.view(torch.int16), self.mt, pairs, sk.view(torch.int16), sv.view(torch.int16), sk.stride(0),
                                           self.b3k, self.b4k, self.b3v, self.b4v, self.permk, self.permv, self.invsmk, self.invcdk, self.invsmv, self.invcdv, self.n_kv, self.TS, num_warps=4)

    def release_blocks(self, blocks: torch.Tensor):
        """blocks: int32 [F] vLLM block ids whose content is dead (freed without prefix caching)."""
        F = blocks.shape[0]
        if F == 0: return
        _release_blocks[(F,)](blocks, F, self.mt, self.stack, self.top, NT=2 * self.n_kv, num_warps=1)
