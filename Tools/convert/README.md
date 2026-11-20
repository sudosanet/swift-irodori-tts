# Converter

Converts Irodori-TTS v4.1 Small MF to the Core AI models used by the Swift package. Needs macOS 27 with Xcode 27
(`xcrun coreai-build`) and Python 3.12.

```sh
git clone https://github.com/Aratako/Irodori-TTS
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r Irodori-TTS/requirements.txt -r requirements.txt
python export.py --irodori Irodori-TTS --out ../../models     # downloads the checkpoint from Hugging Face
python fixtures.py --irodori Irodori-TTS --out ../../Tests/IrodoriTTSTests/Fixtures
python loudness_fixtures.py --out ../../Tests/IrodoriTTSTests/Fixtures
```

Converted with Irodori-TTS f01c08c, torch 2.13.0, coreai-torch 0.4.3, coreai-opt 0.4.0.

| File | Functions | Weights | Runs on |
|---|---|---|---|
| text.aimodel | backbone, text_proj, caption_proj | int8 Linear | Neural Engine |
| dit.aimodel | context, step_L160, step_L256 | int8 Linear | Neural Engine |
| aux.aimodel | duration, speaker | int8 Linear | Neural Engine (speaker: GPU) |
| codec.aimodel | decode (32 latent frames) | fp16 | Neural Engine |
| encoder.aimodel | encode (2–1500 frames) | fp32 | GPU |

What keeps the graphs on the Neural Engine in fp16 (`irodori_coreai/ops.py`):

- Attention per head as einsum in a channels-first layout, masks of -40000, queries scaled before the product
- `sigmoid` written with `tanh` (the Neural Engine sigmoid is too coarse for the DiT gates)
- RMSNorm and LayerNorm divide each row by its largest magnitude before squaring, and clamp the mean square at 2^-14
- ConvTranspose replaced by convolution + pixel shuffle; Snake uses a precomputed 1/α
- Codec calls limited to 32 frames (61440 samples); longer calls fall back to the GPU
- int8 convolutions fail Neural Engine compilation, so only Linear layers are quantized and the codec stays fp16
- The codec encoder needs fp32 for the speaker state to stay accurate, so voice registration runs on the GPU

Quality against PyTorch (5 sentences, caption and reference): waveform cosine 0.86–0.99, the same Whisper error rate
(0.9 %), identical predicted lengths. `ParityTests` checks the Swift port against the Python Core AI pipeline.
