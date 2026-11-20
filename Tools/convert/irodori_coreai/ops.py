"""Small building blocks that keep the Irodori graphs exact in fp16 on the Apple Neural Engine (ANE).

Findings behind each helper (measured on M5, macOS 27, coreai-torch 0.4.3):
- `sigmoid`/`silu`: the ANE sigmoid is noticeably inexact (DiT velocity cos 0.995 after one step); the tanh form is exact.
- `rms`/`layer_norm`: activations reach ~1700 in the DiT and ~1500 in ModernBERT, so squaring overflows fp16. Divide each row
  by its max |x| (at least 1) first, and floor the mean square at the smallest fp16 normal so all-zero padding rows stay finite.
- `SubPixel`: ConvTranspose1d runs on the GPU; a stride-times-wider Conv1d plus an interleave runs on the ANE.
- `Snake`: precompute 1/alpha in fp32 (the smallest alpha is 2.2e-5; `alpha + 1e-9` underflows in fp16).
"""

import torch
import torch.nn.functional as F

NEG = -40000.0
TINY = 2.0 ** -14


def sigmoid(x):
    return 0.5 * torch.tanh(0.5 * x) + 0.5


def silu(x):
    return x * sigmoid(x)


def rms(x, weight, eps):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1.0)
    xs = x / s
    return x * torch.rsqrt(((xs * xs).mean(-1, keepdim=True) + eps / (s * s)).clamp_min(TINY)) / s * weight


def layer_norm(x, weight, eps):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1.0)
    xs = x / s
    xs = xs - xs.mean(-1, keepdim=True)
    return xs * torch.rsqrt(((xs * xs).mean(-1, keepdim=True) + eps / (s * s)).clamp_min(TINY)) * weight


def attend(q, k, v, heads, bias):
    """Per-head attention in the ANE-friendly (B, C, 1, S) layout.

    q (1, Lq, H·D) already scaled by 1/sqrt(D); k, v (1, Lk, H·D); bias (1, Lk, 1, 1) additive key mask. Returns (1, Lq, H·D).
    """
    d = q.shape[-1] // heads
    qb = q.transpose(1, 2).unsqueeze(2)
    kb, vb = k.unsqueeze(2), v.unsqueeze(2)
    out = []
    for i in range(heads):
        s = slice(i * d, (i + 1) * d)
        w = torch.softmax(torch.einsum("bchq,bkhc->bkhq", qb[:, s], kb[..., s]) + bias, dim=1)
        out.append(torch.einsum("bkhq,bkhc->bchq", w, vb[..., s]))
    return torch.cat(out, 1).squeeze(2).transpose(1, 2)


def key_bias(mask):
    """(1, L) 0/1 mask → (1, L, 1, 1) additive bias."""
    return ((mask - 1) * -NEG).reshape(1, -1, 1, 1)


def rope_pairs(x, cos, sin):
    """Interleaved-pair rotary embedding (complex view of adjacent pairs). x (1, L, H, D); cos/sin (L, D/2)."""
    pair = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pair[..., 0], pair[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([a * c - b * s, a * s + b * c], -1).reshape(x.shape)


class SubPixel(torch.nn.Module):
    """ConvTranspose1d whose output length is input × stride, rewritten as a Conv1d with stride × out channels and an interleave."""

    def __init__(self, ct):
        super().__init__()
        w, b = ct.weight.detach(), ct.bias.detach()  # (Cin, Cout, K)
        cin, cout, k = w.shape
        s, p = ct.stride[0], ct.padding[0]
        rng = [(-(c // s), (k - 1 - c) // s) for c in (f + p for f in range(s))]
        jmin, jmax = min(r[0] for r in rng), max(r[1] for r in rng)
        assert jmin <= 0 <= jmax
        wc = torch.zeros(s * cout, cin, jmax - jmin + 1)
        for f in range(s):
            c = f + p
            for j in range(jmin, jmax + 1):
                if 0 <= j * s + c < k:
                    wc[f * cout:(f + 1) * cout, :, jmax - j] = w[:, :, j * s + c].T
        self.conv = torch.nn.Conv1d(cin, s * cout, jmax - jmin + 1)
        self.conv.weight.data, self.conv.bias.data = wc, b.repeat(s)
        self.pad, self.s, self.cout = (jmax, -jmin), s, cout

    def forward(self, x):
        b, _, n = x.shape
        y = self.conv(F.pad(x, list(self.pad)))
        return y.reshape(b, self.s, self.cout, n).permute(0, 2, 3, 1).reshape(b, self.cout, n * self.s)


class Snake(torch.nn.Module):
    """dacvae Snake1d: x + sin²(αx) / (α + 1e-9), with the reciprocal precomputed in fp32."""

    def __init__(self, alpha):
        super().__init__()
        a = alpha.detach().float()
        self.register_buffer("alpha", a.clone())
        self.register_buffer("inv", (a + 1e-9).reciprocal().clamp(max=60000.0))

    def forward(self, x):
        return x + self.inv * torch.sin(self.alpha * x) ** 2


def replace_snakes(module):
    for parent in list(module.modules()):
        for name, child in parent.named_children():
            if type(child).__name__ == "Snake1d":
                setattr(parent, name, Snake(child.alpha))


def remove_weight_norm(module):
    for m in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(m)
        except ValueError:
            pass
