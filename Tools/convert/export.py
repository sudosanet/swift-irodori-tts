"""Convert Irodori-TTS v4.1 Small MF to Core AI models for the IrodoriTTS Swift package.

    python export.py --irodori <Irodori-TTS clone> --out <dir>

Writes into <dir>:
  text.aimodel     backbone, text_proj, caption_proj           int8 weights, ANE
  dit.aimodel      context, step_L160, step_L256                int8 weights, ANE
  aux.aimodel      duration, speaker                            int8 weights (speaker runs on the GPU at voice registration)
  codec.aimodel    decode                                       fp16, ANE (32 latent frames per call)
  encoder.aimodel  encode (dynamic length)                      fp32, GPU (voice registration only)
  word_embeddings.f16, constants.json, tokenizer.json, README.md, LICENSE-*.txt
All inputs and outputs are fp16 except the encoder (fp32).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small-MF"
LENGTHS = (160, 256)  # latent frames per DiT step function (6.4 s and 10.24 s)
CODEC_FRAMES, CODEC_CONTEXT = 16, 8  # decode 16 frames per call with 8 frames of context on each side (32 in total)


MODEL_CARD = """# Irodori-TTS v4.1 Small MF for Core AI

Core AI conversion of [Irodori-TTS v4.1 Small MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) for the
[IrodoriTTS](https://github.com/sudosanet/swift-irodori-tts) Swift package. Unofficial; not affiliated with the authors.

| File | Contents | Weights |
|---|---|---|
| text.aimodel | ModernBERT-ja-310m backbone, text and caption projections | int8 Linear |
| dit.aimodel | DiT context and MeanFlow steps (160 and 256 latent frames) | int8 Linear |
| aux.aimodel | duration predictor, speaker encoder | int8 Linear |
| codec.aimodel | Semantic-DACVAE decoder (32 latent frames per call) | fp16 |
| encoder.aimodel | Semantic-DACVAE encoder (2 to 1500 frames) | fp32 |
| word_embeddings.f16 | backbone token embeddings | fp16 |
| tokenizer.json | ModernBERT-ja tokenizer | |
| constants.json | shapes, MeanFlow schedule and conditioning, null embeddings | |

Japanese text-to-speech at 48 kHz. Against the PyTorch model: same predicted lengths, waveform cosine 0.86 to 0.99,
the same Whisper character error rate.

## Licenses

- Irodori-TTS (weights): MIT, LICENSE-Irodori-TTS.txt
- ModernBERT-ja-310m (backbone, tokenizer): MIT, LICENSE-ModernBERT-ja.txt
- Semantic-DACVAE-Japanese-32dim (codec): MIT, LICENSE-Semantic-DACVAE.txt
- DACVAE (base of the codec): Apache-2.0, LICENSE-DACVAE.txt

The Irodori-TTS model card asks users not to impersonate people without their consent and not to create misleading
deepfakes, and notes that voices generated from text alone may coincidentally resemble real people.
"""

SEMANTIC_DACVAE_LICENSE = """MIT License (as stated on https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)

Copyright (c) Aratako

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def write_notices(irodori, out):
    import urllib.request

    from huggingface_hub import hf_hub_download

    shutil.copy(Path(irodori) / "LICENSE", out / "LICENSE-Irodori-TTS.txt")
    shutil.copy(hf_hub_download("sbintuitions/modernbert-ja-310m", "LICENSE"), out / "LICENSE-ModernBERT-ja.txt")
    (out / "LICENSE-Semantic-DACVAE.txt").write_text(SEMANTIC_DACVAE_LICENSE)
    with urllib.request.urlopen("https://raw.githubusercontent.com/facebookresearch/dacvae/main/LICENSE") as r:
        (out / "LICENSE-DACVAE.txt").write_bytes(r.read())
    (out / "README.md").write_text(MODEL_CARD)


def load_runtime(irodori, checkpoint):
    sys.path.insert(0, str(irodori))
    from huggingface_hub import hf_hub_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

    return InferenceRuntime.from_key(RuntimeKey(checkpoint=hf_hub_download(checkpoint, "model.safetensors"),
                                                model_device="cpu", codec_device="cpu"))


def int8_linear(module, example):
    """Weight-only int8 (per output channel) for nn.Linear only. Conv1d stays fp16: int8 convolutions failed ANE compilation."""
    from coreai_opt.quantization import Quantizer, QuantizerConfig

    cfg = QuantizerConfig.presets.w8()
    base, cfg.global_config = cfg.global_config, None
    cfg.module_type_configs["torch.nn.modules.linear.Linear"] = base
    q = Quantizer(module, cfg)
    q.prepare(example)
    return q.finalize()


def add(conv, module, example, inputs, outputs, entry, quantize=False, dynamic_shapes=None):
    from coreai_torch import get_decomp_table

    m = int8_linear(module, example) if quantize else module
    ep = torch.export.export(m, example, dynamic_shapes=dynamic_shapes).run_decompositions(get_decomp_table())
    conv.add_exported_program(ep, input_names=inputs, output_names=outputs, entrypoint_name=entry)


def r16(*shape):
    return torch.randn(*shape).half() * 0.1


def o16(*shape):
    return torch.ones(*shape).half()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irodori", required=True, type=Path, help="path to a clone of https://github.com/Aratako/Irodori-TTS")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import copy

    from coreai_torch import TorchConverter

    from irodori_coreai import graphs as g

    rt = load_runtime(a.irodori, a.checkpoint)
    model = copy.deepcopy(rt.model).eval()  # .half() would turn the original's complex RoPE caches real
    T, S, C = g.TEXT_LEN, g.SPEAKER_LEN, g.TEXT_LEN + g.SPEAKER_LEN + g.TEXT_LEN

    conv = TorchConverter()
    add(conv, g.Backbone(model.pretrained_text_backbone.backbone).half().eval(), (r16(1, T, 768), o16(1, T)), ["emb", "mask"],
        ["hidden"], "backbone", True)
    for name, proj, norm in (("text_proj", model.text_encoder, model.text_norm), ("caption_proj", model.caption_encoder, model.caption_norm)):
        add(conv, g.Projector(proj, norm).half().eval(), (r16(1, T, 768), o16(1, T)), ["hidden", "mask"], ["state"], name, True)
    conv.to_coreai().save_asset(a.out / "text.aimodel")
    print("text.aimodel")

    conv = TorchConverter()
    add(conv, g.Context(model).half().eval(), (r16(1, T, 512), r16(1, S, 768), r16(1, T, 512)), ["text", "speaker", "caption"],
        ["ck", "cv"], "context", True)
    for L in LENGTHS:
        add(conv, g.Step(model, L).half().eval(), (r16(1, L, 32), r16(1, 1, 3840), r16(12, C, 1280), r16(12, C, 1280), o16(1, L), o16(1, C)),
            ["x", "cond", "ck", "cv", "mask", "ctx_mask"], ["v"], f"step_L{L}", True)
    conv.to_coreai().save_asset(a.out / "dit.aimodel")
    print("dit.aimodel")

    conv = TorchConverter()
    add(conv, g.Duration(model).half().eval(), (r16(1, T, 512), r16(1, 768), r16(1, 512)), ["text", "speaker", "caption"], ["logits"],
        "duration", True)
    add(conv, g.Speaker(model).half().eval(), (r16(1, S - 1, 128), o16(1, S - 1)), ["latent", "mask"], ["state"], "speaker", True)
    conv.to_coreai().save_asset(a.out / "aux.aimodel")
    print("aux.aimodel")

    conv = TorchConverter()
    add(conv, g.Decoder(rt.codec.model).half().eval(), (r16(1, 32, CODEC_FRAMES + 2 * CODEC_CONTEXT),), ["z"], ["wav"], "decode")
    conv.to_coreai().save_asset(a.out / "codec.aimodel")
    print("codec.aimodel")

    conv = TorchConverter()
    frames = torch.export.Dim("frames", min=2, max=1500)
    add(conv, g.Encoder(rt.codec.model).float().eval(), (torch.randn(1, 1, 1920 * 100) * 0.1,), ["wav"], ["latent"], "encode",
        dynamic_shapes={"wav": {2: 1920 * frames}})
    conv.to_coreai().save_asset(a.out / "encoder.aimodel")
    print("encoder.aimodel")

    g.word_table(model).numpy().astype(np.float16).tofile(a.out / "word_embeddings.f16")
    conds, sched = g.cond_table(model)
    null_spk, null_cap = g.null_vectors(model)
    json.dump({
        "format": 1, "checkpoint": a.checkpoint, "sampleRate": 48000, "hop": 1920, "latentDim": 32, "textLength": T,
        "speakerLength": S, "speakerPatch": 4, "lengths": list(LENGTHS), "codecFrames": CODEC_FRAMES, "codecContext": CODEC_CONTEXT,
        "vocabSize": int(g.word_table(model).shape[0]), "embeddingDim": int(g.word_table(model).shape[1]),
        "loudnessTarget": -16.0, "minFrames": 13, "maxFrames": 750,
        "schedule": sched.tolist(), "conditions": conds.tolist(), "nullSpeaker": null_spk.tolist(), "nullCaption": null_cap.tolist(),
    }, open(a.out / "constants.json", "w"))
    from huggingface_hub import hf_hub_download

    repo = model.cfg.text_tokenizer_repo
    shutil.copy(hf_hub_download(repo, "tokenizer.json", revision=getattr(model.cfg, "text_encoder_revision", None)), a.out / "tokenizer.json")
    write_notices(a.irodori, a.out)
    print("constants.json, word_embeddings.f16, tokenizer.json, README.md, licenses")


if __name__ == "__main__":
    main()
