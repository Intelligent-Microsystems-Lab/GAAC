"""Lossless BF16 KV format (GAAC tiles): calibration, reference encoder and bit-exact reference decoder (PyTorch).

A BF16 value is split into its sign+mantissa byte (stored as is) and its 8-bit exponent, which is coded against a
per-(kv head, channel) exponent window ("base"):

  * tier 0: 3-bit code c = e - base3[kvh, d] in [0, 8)
  * elements form 8-element GROUPS; a group with any element outside the 3-bit window is promoted as a whole:
      tier 1: 4-bit window c = e - base4[kvh, d] in [0, 16); low 3 bits stay in the code stream, the 4th bits of the
              8 elements form ONE escape byte;
      tier 2: raw exponent; low 3 bits stay in the code stream, the high 5 bits of the 8 elements form 5 escape bytes.
  * per token row (128 elements = 16 groups): a u32 tag word, 2 bits per group (bit0 = promoted, bit1 = raw).
  * per 16-token tile: the byte offset of its escape bytes in the head's escape stream (tile_off).

A 16 x 128 tile therefore costs 2048 B (sign/mantissa) + 768 B (codes) + 64 B (tags) + escape bytes + 4 B (tile
offset), about 11.4 bits per value against 16.  Storage order inside a tile is the m16n8k16 mma fragment order of
the decode kernel (K: two 8-token octets, V: one 16-token tile), so reconstructed values land directly in the
kernel's operand registers.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

D = 128     # head dim
T = 16      # tokens per tile
GS = 8      # elements per group
CB = D * 3 // 8   # code bytes per token row


def perm_k(D: int = D) -> torch.Tensor:
    """K storage order: lane j (0..3) owns stored positions [32j, 32j+32); position 32j+4s+{0,1,2,3} = dim 16s+2j+{0,1,8,9}."""
    p = torch.empty(D, dtype=torch.long, device="cpu")
    for j in range(4):
        for s in range(8):
            for q, off in enumerate((0, 1, 8, 9)):
                p[32 * j + 4 * s + q] = 16 * s + 2 * j + off
    return p


def perm_v(D: int = D) -> torch.Tensor:
    """V storage order: lane dim-class r (0..7) owns stored positions [16r, 16r+16); position 16r+n = dim 8n+r."""
    p = torch.empty(D, dtype=torch.long, device="cpu")
    for r in range(8):
        for n in range(16):
            p[16 * r + n] = 8 * n + r
    return p


def perm(kind: str) -> torch.Tensor:
    return perm_k() if kind == "k" else perm_v()


def storage_index(kind: str, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """(sm_idx [T*D], code_idx [T*CB]): storage byte -> token-major byte (tok*D + e / tok*CB + b) within one tile."""
    sm_idx, cd_idx = [], []
    if kind == "k":     # two octets of 8 tokens; lane l -> tok l>>2, j=l&3: elems [32j,32j+32), code bytes [12j,12j+12)
        for o in range(T // 8):
            for l in range(32):
                tok, j = 8 * o + (l >> 2), l & 3
                sm_idx += [tok * D + 32 * j + q for q in range(32)]
                cd_idx += [tok * CB + 12 * j + q for q in range(12)]
    else:               # 16-token tile; lane l -> j=l&3, r=l>>2: tokens (2j,2j+1,2j+8,2j+9), elems [16r,16r+16), code bytes [6r,6r+6)
        for l in range(32):
            j, r = l & 3, l >> 2
            for tok in (2 * j, 2 * j + 1, 2 * j + 8, 2 * j + 9):
                sm_idx += [tok * D + 16 * r + q for q in range(16)]
                cd_idx += [tok * CB + 6 * r + q for q in range(6)]
    return torch.tensor(sm_idx, device=dev), torch.tensor(cd_idx, device=dev)


def bases_from_hist(hist: torch.Tensor, n_kv: int) -> tuple[torch.Tensor, torch.Tensor]:
    """hist: [n_kv*D, 256] exponent counts -> (base3, base4) uint8 [n_kv, D]: start of the densest 8- / 16-wide window."""
    hist = hist.double()
    cs = torch.cat([torch.zeros(hist.shape[0], 1, dtype=hist.dtype, device=hist.device), torch.cumsum(hist, 1)], 1)
    out = []
    for w in (8, 16):
        out.append(torch.argmax(cs[:, w:] - cs[:, :-w], 1).to(torch.uint8).view(n_kv, D).contiguous())
    return out[0], out[1]


def exponent_hist(x: torch.Tensor, n_kv: int) -> torch.Tensor:
    """x: [..., n_kv, D] bf16 (token-major, as produced by the model) -> int64 [n_kv*D, 256] exponent histogram."""
    e = (x.contiguous().view(torch.int16).to(torch.int32) >> 7) & 0xFF
    idx = torch.arange(n_kv * D, device=x.device).view(1, -1) * 256 + e.reshape(-1, n_kv * D)
    return torch.bincount(idx.reshape(-1), minlength=n_kv * D * 256).view(n_kv * D, 256)


@dataclass
class Page:
    """Compressed K or V of H heads x S tokens (all heads the same length)."""
    sm: torch.Tensor              # uint8 [H, S_pad*D]   sign+mantissa bytes, storage order
    codes: torch.Tensor           # uint8 [H, S_pad*CB]  3-bit codes, storage order
    flags: torch.Tensor           # int32 [H, S_pad]     per-row tag words (0 for padded rows)
    ovf_off: torch.Tensor         # int32 [H, S]         row offset within the head's escape stream (reference decoder only)
    ovf: torch.Tensor             # uint8 [total + 8]    escape bytes of all heads
    ovf_head_start: torch.Tensor  # int32 [H+1]
    tile_off: torch.Tensor        # int32 [H, S_pad/16]  tile offset within the head's escape stream (kernel metadata)
    base3: torch.Tensor           # uint8 [n_kv, D]
    base4: torch.Tensor           # uint8 [n_kv, D]
    H: int
    S: int
    n_kv: int
    kind: str                     # 'k' | 'v'

    @property
    def nbytes(self) -> int:
        """Stored bytes: sign/mantissa + codes + escapes + tags + tile offsets (bases excluded: 4 KiB per layer)."""
        esc = int(self.ovf_head_start[-1] - self.ovf_head_start[0])
        return self.H * self.S * (D + CB + 4) + esc + self.H * ((self.S + T - 1) // T) * 4

    @property
    def ratio(self) -> float:
        return self.H * self.S * D * 2 / self.nbytes


class Codec:
    """Per-(kv head, channel) exponent bases + reference (PyTorch) encoder/decoder."""
    def __init__(self, n_kv: int, device: str = "cuda"):
        self.n_kv, self.device = n_kv, device
        self.base3 = self.base4 = None

    def calibrate(self, sample: torch.Tensor) -> "Codec":
        """sample: [H, T, D] bf16 with head h using bases of h % n_kv."""
        H = sample.shape[0]
        x = sample.to(self.device).reshape(H // self.n_kv, self.n_kv, -1, D).permute(0, 2, 1, 3)   # [., T, n_kv, D]
        self.base3, self.base4 = bases_from_hist(exponent_hist(x, self.n_kv), self.n_kv)
        return self

    def set_bases(self, base3: torch.Tensor, base4: torch.Tensor) -> "Codec":
        self.base3 = base3.to(self.device, torch.uint8).reshape(self.n_kv, D).contiguous()
        self.base4 = base4.to(self.device, torch.uint8).reshape(self.n_kv, D).contiguous()
        return self

    def encode(self, x: torch.Tensor, kind: str) -> Page:
        """x: [H, S, D] bf16 -> Page (reference implementation; gaac.encoder.TileEncoder is the fast, bit-identical one)."""
        H, S, _ = x.shape
        dev = self.device
        p = perm(kind).to(dev)
        x16 = x.to(dev).contiguous().view(torch.int16).to(torch.int32)[..., p] & 0xFFFF
        e = (x16 >> 7) & 0xFF
        sm = (((x16 >> 8) & 0x80) | (x16 & 0x7F)).to(torch.uint8)
        hidx = torch.arange(H, device=dev) % self.n_kv
        b3 = self.base3[hidx][:, p].to(torch.int32).unsqueeze(1)
        b4 = self.base4[hidx][:, p].to(torch.int32).unsqueeze(1)
        d3, d4 = e - b3, e - b4
        ng = D // GS
        g3 = ((d3 >= 0) & (d3 < 8)).view(H, S, ng, GS).all(-1)
        g4 = ((d4 >= 0) & (d4 < 16)).view(H, S, ng, GS).all(-1)
        tier = torch.where(g3, 0, torch.where(g4, 1, 2)).to(torch.int32)                       # [H,S,ng]
        tier_e = tier.unsqueeze(-1).expand(H, S, ng, GS).reshape(H, S, D)
        code = torch.where(tier_e == 0, d3 & 7, torch.where(tier_e == 1, d4 & 7, e & 7))
        bits = ((code.unsqueeze(-1) >> torch.arange(3, device=dev)) & 1).view(H, S, D * 3)     # bit k of code i at 3i+k
        codes_tok = (bits.view(H, S, CB, 8) << torch.arange(8, device=dev)).sum(-1).to(torch.uint8)
        g = torch.arange(ng, device=dev)
        flags = (((tier >= 1).to(torch.int64) << (2 * g)) | ((tier == 2).to(torch.int64) << (2 * g + 1))).sum(-1)
        # escape bytes: per row, groups ascending; tier 1 -> 1 byte (4th bit of d4), tier 2 -> 5 bytes (high 5 bits of e, 40-bit LE)
        gsize = torch.where(tier == 1, 1, torch.where(tier == 2, 5, 0)).to(torch.int64)
        row_bytes = gsize.sum(-1)
        head_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(row_bytes.sum(-1), 0)])
        ovf_off = torch.cumsum(row_bytes, 1) - row_bytes
        total = int(head_start[-1])
        ovf = torch.zeros(total + 8, dtype=torch.uint8, device=dev)    # +8: the kernel may read a whole 5-byte group at the end
        if total:
            gstart = head_start[:-1].view(H, 1, 1) + ovf_off.unsqueeze(-1) + (torch.cumsum(gsize, -1) - gsize)
            bit4 = (((d4 >> 3) & 1).view(H, S, ng, GS) << torch.arange(GS, device=dev)).sum(-1)
            m1 = tier == 1
            ovf[gstart[m1]] = bit4[m1].to(torch.uint8)
            m2 = tier == 2
            if bool(m2.any()):
                hi = (e.view(H, S, ng, GS)[m2] >> 3).to(torch.int64)
                w40 = (hi << (5 * torch.arange(GS, device=dev))).sum(-1)
                pos = gstart[m2].unsqueeze(-1) + torch.arange(5, device=dev)
                ovf[pos.reshape(-1)] = ((w40.unsqueeze(-1) >> (8 * torch.arange(5, device=dev))) & 0xFF).reshape(-1).to(torch.uint8)
        S_pad = (S + T - 1) // T * T
        if S_pad != S:
            sm = torch.cat([sm, torch.zeros(H, S_pad - S, D, dtype=torch.uint8, device=dev)], 1)
            codes_tok = torch.cat([codes_tok, torch.zeros(H, S_pad - S, CB, dtype=torch.uint8, device=dev)], 1)
        si, ci = storage_index(kind, dev)
        sm_st = sm.view(H, S_pad // T, T * D)[:, :, si].reshape(H, S_pad * D)
        cd_st = codes_tok.view(H, S_pad // T, T * CB)[:, :, ci].reshape(H, S_pad * CB)
        off32 = ovf_off.to(torch.int32).contiguous()
        flags32 = torch.zeros(H, S_pad, dtype=torch.int32, device=dev)
        flags32[:, :S] = flags.to(torch.int32)
        return Page(sm_st.contiguous(), cd_st.contiguous(), flags32, off32, ovf, head_start.to(torch.int32).contiguous(),
                    off32[:, ::T].contiguous(), self.base3, self.base4, H, S, self.n_kv, kind)

    @staticmethod
    def decode(page: Page) -> torch.Tensor:
        """Bit-exact reconstruction: Page -> [H, S, D] bf16."""
        H, S = page.H, page.S
        dev = page.sm.device
        S_pad = page.sm.shape[1] // D
        si, ci = storage_index(page.kind, dev)
        inv_s = torch.empty_like(si); inv_s[si] = torch.arange(si.numel(), device=dev)
        inv_c = torch.empty_like(ci); inv_c[ci] = torch.arange(ci.numel(), device=dev)
        sm = page.sm.view(H, S_pad // T, T * D)[:, :, inv_s].reshape(H, S_pad, D)[:, :S].to(torch.int32)
        cd = page.codes.view(H, S_pad // T, T * CB)[:, :, inv_c].reshape(H, S_pad, CB)[:, :S].to(torch.int32)
        bits = ((cd.unsqueeze(-1) >> torch.arange(8, device=dev)) & 1).view(H, S, D, 3)
        code = bits[..., 0] | (bits[..., 1] << 1) | (bits[..., 2] << 2)
        p = perm(page.kind).to(dev)
        hidx = torch.arange(H, device=dev) % page.n_kv
        b3 = page.base3[hidx][:, p].to(torch.int32).unsqueeze(1)
        b4 = page.base4[hidx][:, p].to(torch.int32).unsqueeze(1)
        ng = D // GS
        f = page.flags[:, :S].to(torch.int64) & 0xFFFFFFFF
        tier = torch.stack([(f >> (2 * g)) & 3 for g in range(ng)], -1)                      # 0, 1, 3
        gsize = torch.where(tier == 1, 1, torch.where(tier == 3, 5, 0)).to(torch.int64)
        gstart = page.ovf_head_start[:-1].to(torch.int64).view(H, 1, 1) + page.ovf_off.to(torch.int64).unsqueeze(-1) + (torch.cumsum(gsize, -1) - gsize)
        ovf = page.ovf.to(torch.int32)
        gstart = gstart.clamp(max=ovf.numel() - 1)
        bit4 = ((ovf[gstart].unsqueeze(-1) >> torch.arange(GS, device=dev)) & 1).view(H, S, D)
        w40 = (ovf[(gstart.unsqueeze(-1) + torch.arange(5, device=dev)).clamp(max=ovf.numel() - 1)].to(torch.int64) << (8 * torch.arange(5, device=dev))).sum(-1)
        hi5 = ((w40.unsqueeze(-1) >> (5 * torch.arange(GS, device=dev))) & 31).view(H, S, D).to(torch.int32)
        tier_e = tier.unsqueeze(-1).expand(H, S, ng, GS).reshape(H, S, D)
        exp = torch.where(tier_e == 0, b3 + code, torch.where(tier_e == 1, b4 + (code | (bit4 << 3)), code | (hi5 << 3)))
        raw16 = ((sm & 0x80) << 8) | (exp << 7) | (sm & 0x7F)
        out = torch.empty(H, S, D, dtype=torch.int32, device=dev)
        out[..., p] = raw16.to(torch.int32)
        return out.to(torch.int16).view(torch.bfloat16)


def row_offsets(flags: torch.Tensor, S: int) -> torch.Tensor:
    """Row offsets within each head's escape stream from the tag words (tiles are stored back to back)."""
    f = flags[:, :S].to(torch.int64) & 0xFFFFFFFF
    rb = sum(((f >> (2 * g)) & 1) + 4 * ((f >> (2 * g + 1)) & 1) for g in range(D // GS))
    return (torch.cumsum(rb, 1) - rb).to(torch.int32).contiguous()
