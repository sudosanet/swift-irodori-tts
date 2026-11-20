# swift-irodori-tts

Japanese text-to-speech for Apple platforms: an unofficial Swift port of
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1 Small MF running on Core AI.
Speech runs entirely on the Neural Engine. [日本語](#日本語)

- Describe a voice in words (`caption`), or imitate a short recording (`reference`), or both
- Streams 48 kHz audio in 0.64 s chunks
- About 0.09 × real time on an M5 Mac (7.4 s of speech in 0.66 s); about 1.1 GB of models
- macOS 27 / iOS 27, Swift 6 (iOS builds; not yet tested on a device)

## Models

Download `IrodoriTTS-v4.1-Small-MF-coreai.zip` from [Releases](../../releases) and unzip it anywhere
(an app can download it on first launch or bundle it).

```sh
shasum -a 256 -c IrodoriTTS-v4.1-Small-MF-coreai.zip.sha256
unzip IrodoriTTS-v4.1-Small-MF-coreai.zip -d models
```

The first load specializes the models for the device (about 45 s on an M5); Core AI caches the result, later loads take
well under a second.

## Usage

```swift
.package(url: "https://github.com/sudosanet/swift-irodori-tts.git", from: "0.1.0")
```

```swift
import IrodoriTTS

let tts = try await IrodoriTTS(modelDirectory: modelsURL)

// From a description of the voice (written in Japanese works best)
let voice = try await tts.voice(caption: "明るくやさしい、少し高めの若い女性の声。")
// From a recording (about 10 s is used; any format AVFoundation reads), optionally with a caption
let cloned = try await tts.voice(reference: recordingURL)

// All at once
let samples = try await tts.synthesize("こんにちは！今日もいっしょにがんばろうね。", voice: voice)

// Or as it is decoded
for try await chunk in tts.speak("ビルドが通ったよ。", voice: voice, seed: 42) {
    // mono Float samples at IrodoriTTS.sampleRate (48 kHz)
}
```

- A `Voice` is reusable; create it once per caption or recording.
- Long text is cut at sentence ends into pieces of up to about 8 s. Without a reference the voice is sampled for each
  piece, so it can drift between pieces. For a steady voice, use a reference: generate a sample you like from a caption,
  then pass it back with `voice(caption:referenceSamples:)`.
- `durationScale` above 1 speaks slower.
- Registering a reference runs the codec encoder once on the GPU; everything else runs on the Neural Engine.
- Japanese only (the model is trained on Japanese).

Command line:

```sh
swift run -c release irodori-say --models models --caption "落ち着いた低めの女性の声" -o out.wav "こんにちは。"
```

## How it is built

`Tools/convert` converts the PyTorch checkpoint with coreai-torch: int8 weights (Linear layers) for the text encoder,
DiT and auxiliary heads, fp16 for the codec decoder, fp32 for the codec encoder; layers are rewritten to stay on the
Neural Engine in fp16 (see [Tools/convert/README.md](Tools/convert/README.md)). The Swift side ports the tokenizer, text
normalization, loudness normalization (BS.1770), duration and MeanFlow sampling; tests check them against the Python
implementation.

## License

Apache License 2.0 for this repository. The model release carries the MIT licenses of Irodori-TTS, ModernBERT-ja and
Semantic-DACVAE; see [NOTICE](NOTICE), which also reproduces the ethical notes from the Irodori-TTS model card.
The original Irodori-TTS runtime can add a SilentCipher watermark; this port does not.

---

## 日本語

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1 Small MF を Core AI で動かす、非公式の Swift 版です。
喋る処理はすべてニューラルエンジンで動きます。

- 声を言葉で指定する（`caption`）、短い録音をまねる（`reference`）、またはその両方
- 48kHz の音声を 0.64 秒ずつ順に返す
- M5 の Mac で実時間の約 0.09 倍（7.4 秒の音声を 0.66 秒）。モデルは約 1.1GB
- macOS 27・iOS 27、Swift 6（iOS はビルドのみ確認、実機は未確認）

モデルは [Releases](../../releases) の zip を展開して、そのディレクトリを `IrodoriTTS(modelDirectory:)` に渡します。
初回の読み込みは端末向けの最適化に約 45 秒かかりますが、結果は Core AI がキャッシュするので、2 回目からは 1 秒もかかりません。
使い方は上の Usage と同じです。長い文は文末で約 8 秒ずつに分けて作ります。参照の録音なしでは分けた部分ごとに声が少し変わることがあるので、声をそろえたいときは、キャプションで作った気に入った声を `voice(caption:referenceSamples:)` で参照として渡してください。
日本語専用です（モデルが日本語で学習されています）。

ライセンスはこのリポジトリが Apache-2.0、モデルは Irodori-TTS などの MIT です（[NOTICE](NOTICE)）。
NOTICE には Irodori-TTS のモデルカードにある倫理面の注意も載せています。
