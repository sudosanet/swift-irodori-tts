"""Expected integrated loudness (audiotools / pyloudnorm BS.1770-4, as Irodori uses for reference audio) of deterministic
test signals, for Tests/IrodoriTTSTests/AudioTests.swift.

    python loudness_fixtures.py --out ../../Tests/IrodoriTTSTests/Fixtures
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from audiotools import AudioSignal


def signals(sr=48000):
    t = np.arange(int(3.0 * sr)) / sr
    yield "sine1k_0.1", 0.1 * np.sin(2 * math.pi * 1000 * t)
    yield "sines_mix", 0.3 * np.sin(2 * math.pi * 220 * t) + 0.05 * np.sin(2 * math.pi * 5000 * t) + 0.02 * np.sin(2 * math.pi * 60 * t)
    burst = np.zeros_like(t)
    burst[sr // 2: sr] = 0.5 * np.sin(2 * math.pi * 440 * t[sr // 2: sr])
    yield "burst", burst
    yield "short_0.3s", 0.2 * np.sin(2 * math.pi * 700 * t[: int(0.3 * sr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for name, x in signals():
        s = AudioSignal(torch.tensor(x, dtype=torch.float32)[None, None], 48000)
        loud = float(s.loudness())
        s.normalize(-16.0)
        s.ensure_max_of_audio()
        y = s.audio_data[0, 0].numpy()
        rows.append({"name": name, "loudness": loud, "peak_after": float(np.abs(y).max()), "rms_after": float(np.sqrt((y ** 2).mean()))})
        print(rows[-1])
    json.dump(rows, open(a.out / "loudness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
