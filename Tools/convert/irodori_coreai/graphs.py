"""ANE-friendly rewrites of the Irodori-TTS v4.1 Small MF modules. Each wraps the original weights; outputs match the PyTorch
model in fp32 (see verify.py)."""

import torch

from .ops import SubPixel, attend, key_bias, layer_norm, remove_weight_norm, replace_snakes, rms, rope_pairs, sigmoid, silu

TEXT_LEN = 64     # text and caption tokens (BOS included)
SPEAKER_LEN = 64  # 1 mean token + 63 patches of 4 latent frames (~10 s of reference audio)
STEPS = 4         # MeanFlow sampling steps (fixed schedule 1 → 0)


# ---- text / caption ----------------------------------------------------------------------------------------------------

def _rope_half_table(theta, n, dim):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    f = torch.arange(n, dtype=torch.float32)[:, None] * inv[None]
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()


class Backbone(torch.nn.Module):
    """ModernBERT-ja-310m. Input: token embeddings (1, T, 768) (the table lookup is done by the caller) and a 0/1 mask (1, T).
    With T ≤ 64 the 128-wide local attention equals global attention, so layers differ only in the RoPE theta."""

    def __init__(self, bert, length=TEXT_LEN):
        super().__init__()
        self.b = bert
        c = bert.config
        self.H, self.D = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        assert c.local_attention >= 2 * length
        for kind in ("full_attention", "sliding_attention"):
            cos, sin = _rope_half_table(c.rope_parameters[kind]["rope_theta"], length, self.D)
            self.register_buffer(f"cos_{kind[0]}", cos)
            self.register_buffer(f"sin_{kind[0]}", sin)

    def _rope(self, x, kind):
        cos, sin = getattr(self, f"cos_{kind[0]}")[None, :, None], getattr(self, f"sin_{kind[0]}")[None, :, None]
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat([-b, a], -1) * sin

    def forward(self, emb, mask):
        b, n = self.b, emb.shape[1]
        bias = key_bias(mask)
        x = layer_norm(emb, b.embeddings.norm.weight, b.embeddings.norm.eps)
        for i, layer in enumerate(b.layers):
            h = x if i == 0 else layer_norm(x, layer.attn_norm.weight, layer.attn_norm.eps)
            q, k, v = layer.attn.Wqkv(h).reshape(1, n, 3, self.H, self.D).unbind(2)
            q = self._rope(q, layer.attention_type) * (self.D ** -0.5)
            k = self._rope(k, layer.attention_type)
            x = x + layer.attn.Wo(attend(q.reshape(1, n, -1), k.reshape(1, n, -1), v.reshape(1, n, -1), self.H, bias))
            h = layer_norm(x, layer.mlp_norm.weight, layer.mlp_norm.eps)
            a, g = layer.mlp.Wi(h).chunk(2, -1)
            x = x + layer.mlp.Wo(torch.nn.functional.gelu(a) * g)
        return layer_norm(x, b.final_norm.weight, b.final_norm.eps) * mask.unsqueeze(-1)


class Projector(torch.nn.Module):
    """Backbone output → text_state or caption_state (projection + residual MLP + RMSNorm)."""

    def __init__(self, proj, norm):
        super().__init__()
        self.p, self.norm = proj, norm

    def forward(self, state, mask):
        p = self.p
        y = p.projector(state) + p.residual_down(silu(p.residual_up(rms(state, p.residual_norm.weight, p.residual_norm.eps))))
        return rms(y * mask.unsqueeze(-1), self.norm.weight, self.norm.eps)


# ---- DiT ---------------------------------------------------------------------------------------------------------------

def cond_table(model):
    """The four MeanFlow (t, Δt) condition vectors (4, 3840) and the schedule (5,)."""
    from irodori_tts.model import get_timestep_embedding

    sched = torch.linspace(1.0, 0.0, STEPS + 1)
    t, d = sched[:-1], sched[:-1] - sched[1:]
    dim = model.cfg.timestep_embed_dim
    with torch.no_grad():
        c = model.cond_module(get_timestep_embedding(t, dim)) + model.delta_cond_module(get_timestep_embedding(d, dim))
    return c, sched


class Context(torch.nn.Module):
    """Text, speaker and caption states → per-layer context keys/values (12, C, 1280), keys already k-normed. Once per line."""

    def __init__(self, model):
        super().__init__()
        self.blocks = model.blocks

    def forward(self, text, speaker, caption):
        ks, vs = [], []
        for b in self.blocks:
            a = b.attention
            k = torch.cat([a.wk_text(text), a.wk_speaker(speaker), a.wk_caption(caption)], 1)
            v = torch.cat([a.wv_text(text), a.wv_speaker(speaker), a.wv_caption(caption)], 1)
            k = rms(k.reshape(1, k.shape[1], a.heads, a.head_dim), a.k_norm.weight, a.k_norm.eps)
            ks.append(k.reshape(1, k.shape[1], -1))
            vs.append(v)
        return torch.cat(ks, 0), torch.cat(vs, 0)


class Step(torch.nn.Module):
    """One DiT evaluation (12 blocks) for a latent of fixed length L. Returns the velocity (1, L, 32)."""

    def __init__(self, model, length):
        super().__init__()
        from irodori_tts.model import precompute_freqs_cis

        # Hold only the needed submodules; the whole model would drag ModernBERT and complex RoPE caches into the export.
        self.in_proj, self.blocks, self.out_proj = model.in_proj, model.blocks, model.out_proj
        self.out_w, self.out_eps = model.out_norm.weight, model.out_norm.eps
        self.L = length
        f = precompute_freqs_cis(model.blocks[0].attention.head_dim, length)
        self.register_buffer("cos", f.real.float().clone())
        self.register_buffer("sin", f.imag.float().clone())

    def _rope_half_heads(self, x):
        h = x.shape[2] // 2
        return torch.cat([rope_pairs(x[:, :, :h], self.cos, self.sin), x[:, :, h:]], 2)

    def _attention(self, a, x, ck, cv, bias):
        L, H, D = self.L, a.heads, a.head_dim
        q = self._rope_half_heads(rms(a.wq(x).reshape(1, L, H, D), a.q_norm.weight, a.q_norm.eps)) * (D ** -0.5)
        k = self._rope_half_heads(rms(a.wk(x).reshape(1, L, H, D), a.k_norm.weight, a.k_norm.eps))
        y = attend(q.reshape(1, L, -1), torch.cat([k.reshape(1, L, -1), ck], 1), torch.cat([a.wv(x), cv], 1), H, bias)
        return a.wo(y * sigmoid(a.gate(x)))

    @staticmethod
    def _adaln(ad, x, cond):
        shift, scale, gate = cond.chunk(3, -1)
        shift = ad.shift_up(ad.shift_down(silu(shift))) + shift
        scale = ad.scale_up(ad.scale_down(silu(scale))) + scale
        gate = ad.gate_up(ad.gate_down(silu(gate))) + gate
        return rms(x, 1.0, ad.eps) * (1 + scale) + shift, torch.tanh(gate)

    def forward(self, x, cond, ck, cv, mask, ctx_mask):
        """x (1, L, 32); cond (1, 1, 3840); ck, cv (12, C, 1280); mask (1, L); ctx_mask (1, C)."""
        bias = key_bias(torch.cat([mask, ctx_mask], 1))
        h = self.in_proj(x)
        for i, b in enumerate(self.blocks):
            y, g = self._adaln(b.attention_adaln, h, cond)
            h = h + g * self._attention(b.attention, y, ck[i:i + 1], cv[i:i + 1], bias)
            y, g = self._adaln(b.mlp_adaln, h, cond)
            h = h + g * b.mlp.w2(silu(b.mlp.w1(y)) * b.mlp.w3(y))
        return self.out_proj(rms(h, self.out_w, self.out_eps))


# ---- duration and speaker ----------------------------------------------------------------------------------------------

class Duration(torch.nn.Module):
    """Duration predictor (token_sum_dual_adarn_zero_no_aux). text (1, T, 512), speaker vector (1, 768), caption vector (1, 512)
    → per-token logits (1, T). The caller sums softplus(logits) over valid tokens to get latent frames."""

    def __init__(self, model):
        super().__init__()
        d = model.duration_predictor
        assert d.duration_architecture == "token_sum_dual_adarn_zero_no_aux"
        self.inp, self.blocks, self.out_norm, self.out = d.token_input_proj, d.token_blocks, d.token_out_norm, d.token_out_proj

    def forward(self, text, spk, cap):
        h = self.inp(text)
        for b in self.blocks:
            y = rms(h, b.norm.weight, b.norm.eps)
            shift, scale, gate = (b.modulation(silu(spk)) + b.caption_modulation(silu(cap))).unsqueeze(1).chunk(3, -1)
            y = y * (1 + scale) + shift
            h = h + torch.tanh(gate) * b.mlp.w2(silu(b.mlp.w1(y)) * b.mlp.w3(y))
        return self.out(rms(h, self.out_norm.weight, self.out_norm.eps)).squeeze(-1)


class Speaker(torch.nn.Module):
    """Reference latent patches (1, N, 128) and mask (1, N) → speaker state (1, N+1, 768) with the masked-mean token first."""

    def __init__(self, model, n=SPEAKER_LEN - 1):
        super().__init__()
        from irodori_tts.model import precompute_freqs_cis

        e = model.speaker_encoder
        self.inp, self.blocks, self.norm, self.N = e.in_proj, e.blocks, model.speaker_norm, n
        f = precompute_freqs_cis(e.head_dim, n)
        self.register_buffer("cos", f.real.float().clone())
        self.register_buffer("sin", f.imag.float().clone())

    def forward(self, latent, mask):
        m = mask.unsqueeze(-1)
        bias = key_bias(mask)
        x = self.inp(latent) / 6.0 * m
        for b in self.blocks:
            a, n = b.attention, self.N
            h = rms(x, b.attention_norm.weight, b.attention_norm.eps)
            q = rope_pairs(rms(a.wq(h).reshape(1, n, a.heads, a.head_dim), a.q_norm.weight, a.q_norm.eps), self.cos, self.sin)
            k = rope_pairs(rms(a.wk(h).reshape(1, n, a.heads, a.head_dim), a.k_norm.weight, a.k_norm.eps), self.cos, self.sin)
            y = attend(q.reshape(1, n, -1) * (a.head_dim ** -0.5), k.reshape(1, n, -1), a.wv(h), a.heads, bias)
            x = x + a.wo(y * sigmoid(a.gate(h)))
            y = rms(x, b.mlp_norm.weight, b.mlp_norm.eps)
            x = (x + b.mlp.w2(silu(b.mlp.w1(y)) * b.mlp.w3(y))) * m
        x = rms(x * m, self.norm.weight, self.norm.eps)
        mean = (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
        return torch.cat([mean, x], 1)


# ---- codec -------------------------------------------------------------------------------------------------------------

class Decoder(torch.nn.Module):
    """Semantic-DACVAE decoder: latent (1, 32, T) at 25 frames/s → 48 kHz waveform (1, 1, T·1920). The watermark branch is
    skipped exactly as Irodori does. Run in chunks of ≤ 32 frames (61440 samples); longer waveforms leave the ANE."""

    def __init__(self, dacvae):
        super().__init__()
        remove_weight_norm(dacvae)
        self.out_proj, self.dec = dacvae.quantizer.out_proj, dacvae.decoder
        for block in self.dec.model[1:]:
            for i, layer in enumerate(block.block):
                if isinstance(layer, torch.nn.ConvTranspose1d) and layer.padding[0] == layer.stride[0] // 2 and layer.stride[0] % 2 == 0:
                    block.block[i] = SubPixel(layer)
        self.final = self.dec.wm_model.encoder_block.pre[:-1]
        replace_snakes(self)

    def forward(self, z):
        x = self.out_proj(z)
        for layer in self.dec.model:
            x = layer(x)
        return self.final(x)


class Encoder(torch.nn.Module):
    """Semantic-DACVAE encoder (deterministic mean): waveform (1, 1, n·1920) → latent (1, 32, n). Used once per voice, on the GPU
    in fp32: in fp16 the last frames of a clip drift and the speaker encoder amplifies it."""

    def __init__(self, dacvae):
        super().__init__()
        remove_weight_norm(dacvae)
        self.enc, self.in_proj = dacvae.encoder, dacvae.quantizer.in_proj
        replace_snakes(self)

    def forward(self, wav):
        return self.in_proj(self.enc(wav))[:, :32]


def word_table(model):
    return model.pretrained_text_backbone.backbone.embeddings.tok_embeddings.weight.detach()


def null_vectors(model):
    d = model.duration_predictor
    return d.null_speaker.detach(), d.null_caption.detach()

