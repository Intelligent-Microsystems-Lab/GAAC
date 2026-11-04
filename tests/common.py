import torch


def kv_like(H, S, seed=0, device="cuda", outliers=1e-3):
    """Synthetic BF16 KV: Gaussian channels with per-channel scales over ~8 binades plus rare large outliers
    (exercises all three tiers)."""
    g = torch.Generator(device=device).manual_seed(seed)
    scale = torch.exp2(torch.rand(1, 1, 128, generator=g, device=device) * 8 - 4)
    x = torch.randn(H, S, 128, generator=g, device=device) * scale
    m = torch.rand(H, S, 128, generator=g, device=device) < outliers
    x = torch.where(m, x * 1e4, x)
    return x.to(torch.bfloat16)


def ref_attention(q, k, v, scale):
    """q [n_q, D], k/v [L, n_kv, D] -> [n_q, D] in fp32 (GQA: q head j uses kv head j // G)."""
    n_q, n_kv = q.shape[0], k.shape[1]
    G = n_q // n_kv
    kk = k.float().repeat_interleave(G, 1).transpose(0, 1)       # [n_q, L, D]
    vv = v.float().repeat_interleave(G, 1).transpose(0, 1)
    p = torch.softmax((kk @ q.float().unsqueeze(-1)).squeeze(-1) * scale, -1)
    return (p.unsqueeze(1) @ vv).squeeze(1)
