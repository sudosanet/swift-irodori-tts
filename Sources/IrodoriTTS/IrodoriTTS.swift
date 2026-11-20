import CoreAI
import Foundation

/// Irodori-TTS v4.1 Small MF on Core AI. Speech runs on the Neural Engine; registering a reference voice runs once on the GPU.
public actor IrodoriTTS {
    public static let sampleRate = 48000

    let constants: Constants
    let words: WordTable
    let textTokenizer: Tokenizer
    let captionTokenizer: Tokenizer
    let backbone, textProjection, captionProjection, duration, context, decode: InferenceFunction
    let steps: [Int: InferenceFunction]
    let directory: URL
    private var registration: (encode: InferenceFunction, speaker: InferenceFunction)?

    /// - Parameter modelDirectory: the unzipped model release (text.aimodel, dit.aimodel, …, constants.json).
    public init(modelDirectory: URL) async throws {
        directory = modelDirectory
        let constants = try JSONDecoder().decode(Constants.self, from: Data(contentsOf: modelDirectory.appending(path: "constants.json")))
        guard constants.format == 1 else { throw IrodoriError.invalidModel("unsupported model format \(constants.format)") }
        self.constants = constants
        words = try WordTable(contentsOf: modelDirectory.appending(path: "word_embeddings.f16"), vocab: constants.vocabSize,
                              dim: constants.embeddingDim)
        let tokenizer = modelDirectory.appending(path: "tokenizer.json")
        textTokenizer = try Tokenizer(contentsOf: tokenizer, splitRareTokens: true)
        captionTokenizer = try Tokenizer(contentsOf: tokenizer, splitRareTokens: false)
        let ane = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        async let text = Self.load(modelDirectory, "text", ane)
        async let dit = Self.load(modelDirectory, "dit", ane)
        async let aux = Self.load(modelDirectory, "aux", ane)
        async let codec = Self.load(modelDirectory, "codec", ane)
        let (t, d, a, c) = try await (text, dit, aux, codec)
        backbone = try Self.function(t, "backbone")
        textProjection = try Self.function(t, "text_proj")
        captionProjection = try Self.function(t, "caption_proj")
        context = try Self.function(d, "context")
        steps = try Dictionary(uniqueKeysWithValues: constants.lengths.map { ($0, try Self.function(d, "step_L\($0)")) })
        duration = try Self.function(a, "duration")
        decode = try Self.function(c, "decode")
    }

    static func load(_ directory: URL, _ name: String, _ options: SpecializationOptions) async throws -> AIModel {
        try await AIModel(contentsOf: directory.appending(path: "\(name).aimodel"), options: options)
    }

    static func function(_ model: AIModel, _ name: String) throws -> InferenceFunction {
        guard let f = try model.loadFunction(named: name) else { throw IrodoriError.invalidModel("missing function \(name)") }
        return f
    }

    // MARK: - Voices

    /// A voice from a style caption (any language the backbone reads, e.g. "落ち着いた低めの女性の声") and/or reference audio
    /// (about the first 10 s shape the voice; at most 60 s are read). Without either, the model picks a voice at random per utterance.
    public func voice(caption: String = "", reference: URL? = nil) async throws -> Voice {
        try await voice(caption: caption, referenceSamples: reference.map(Audio.load))
    }

    /// - Parameter referenceSamples: mono 48 kHz samples.
    public func voice(caption: String = "", referenceSamples: [Float]?) async throws -> Voice {
        let (cap, capMask) = try await encode(caption, captionTokenizer, captionProjection)
        let capCount = capMask.reduce(0, +)
        let captionVector = capCount > 0
            ? (0..<512).map { j in (0..<constants.textLength).reduce(Float(0)) { $0 + cap[$1 * 512 + j] * capMask[$1] } / capCount }
            : constants.nullCaption
        let S = constants.speakerLength
        guard let samples = referenceSamples else {
            return Voice(caption: cap, captionMask: capMask, speaker: [Float](repeating: 0, count: S * 768),
                         speakerMask: [Float](repeating: 0, count: S), captionVector: captionVector, speakerVector: constants.nullSpeaker)
        }
        let (speaker, speakerMask) = try await register(samples)
        return Voice(caption: cap, captionMask: capMask, speaker: speaker, speakerMask: speakerMask, captionVector: captionVector,
                     speakerVector: Array(speaker[0..<768]))
    }

    private func register(_ samples: [Float]) async throws -> (state: [Float], mask: [Float]) {
        let hop = constants.hop, patch = constants.speakerPatch, rows = constants.speakerLength - 1
        guard samples.count > patch * hop else { throw IrodoriError.audio("reference audio is too short") }
        // The encoder takes up to 60 s (1500 frames); only the first rows × patch frames reach the speaker encoder.
        let used = Array(samples.prefix(1499 * hop))
        let fns = try await registrationFunctions()
        let wav = Audio.padToHop(Audio.normalize(used, target: constants.loudnessTarget), hop: hop)
        let latent = try await fns.encode(["wav": NDArray(scalars: wav, shape: [1, 1, wav.count])], "latent")[0]
        let frames = latent.shape[2], d = constants.latentDim
        let lat = latent.floats  // (1, d, frames)
        let count = min(frames / patch, rows)
        var input = [Float](repeating: 0, count: rows * d * patch)
        for r in 0..<count {
            for p in 0..<patch {
                for c in 0..<d { input[r * d * patch + p * d + c] = lat[c * frames + r * patch + p] }
            }
        }
        let mask = (0..<rows).map { Float($0 < count ? 1 : 0) }
        let state = try await fns.speaker(["latent": .f16(input, [1, rows, d * patch]), "mask": .f16(mask, [1, rows])], "state")[0]
        return (state.floats, [1] + mask)
    }

    private func registrationFunctions() async throws -> (encode: InferenceFunction, speaker: InferenceFunction) {
        if let registration { return registration }
        let gpu = SpecializationOptions(preferredComputeUnitKind: .gpu)
        async let encoder = Self.load(directory, "encoder", gpu)
        async let aux = Self.load(directory, "aux", gpu)
        let fns = (try Self.function(try await encoder, "encode"), try Self.function(try await aux, "speaker"))
        registration = fns
        return fns
    }

    // MARK: - Speech

    /// Whole utterance as mono 48 kHz samples. Text is cut at sentence ends into pieces of up to about 8 s, each
    /// synthesized separately (without a reference voice, the voice may vary between pieces).
    /// - Parameters:
    ///   - durationScale: multiplies the predicted length (> 1 speaks slower).
    ///   - seed: fixes the initial noise; `nil` draws a new one each call.
    public func synthesize(_ text: String, voice: Voice, durationScale: Float = 1, seed: UInt64? = nil) async throws -> [Float] {
        var out = [Float]()
        for try await chunk in speak(text, voice: voice, durationScale: durationScale, seed: seed) { out += chunk }
        return out
    }

    /// Streams mono 48 kHz samples as they are decoded (0.64 s per chunk).
    public nonisolated func speak(_ text: String, voice: Voice, durationScale: Float = 1, seed: UInt64? = nil)
        -> AsyncThrowingStream<[Float], any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var noise = Noise(seed: seed ?? .random(in: .min ... .max))
                    for chunk in self.chunks(text) {
                        try await self.utterance(chunk, voice: voice, durationScale: durationScale,
                                                 noise: { noise.normal($0) }) { continuation.yield($0) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// One call of the model; text that does not fit (64 tokens, the longest DiT bucket) is halved and retried.
    /// `noise(n)` returns n standard normal samples (latent frames × latentDim, row-major).
    func utterance(_ text: String, voice: Voice, durationScale: Float, noise: (Int) -> [Float],
                   emit: ([Float]) -> Void) async throws {
        let T = constants.textLength, S = constants.speakerLength, D = constants.latentDim, longest = constants.lengths.max()!
        let normalized = TextNormalizer.normalize(text)
        if textTokenizer.encode(normalized).count > T, let (a, b) = Self.halves(text) {
            try await utterance(a, voice: voice, durationScale: durationScale, noise: noise, emit: emit)
            try await utterance(b, voice: voice, durationScale: durationScale, noise: noise, emit: emit)
            return
        }
        let (state, mask) = try await encode(normalized, textTokenizer, textProjection)
        let logits = try await duration(["text": .f16(state, [1, T, 512]), "speaker": .f16(voice.speakerVector, [1, 768]),
                                         "caption": .f16(voice.captionVector, [1, 512])], "logits")[0].floats
        let predicted = zip(logits, mask).reduce(0.0) { $0 + log1p(exp(Double($1.0))) * Double($1.1) }
        let wanted = max(constants.minFrames, min(constants.maxFrames, Int((predicted * Double(durationScale)).rounded(.toNearestOrEven))))
        if wanted > longest, let (a, b) = Self.halves(text) {
            try await utterance(a, voice: voice, durationScale: durationScale, noise: noise, emit: emit)
            try await utterance(b, voice: voice, durationScale: durationScale, noise: noise, emit: emit)
            return
        }
        let n = min(wanted, longest)
        let L = constants.lengths.sorted().first { $0 >= n }!
        let kv = try await context(["text": .f16(state, [1, T, 512]), "speaker": .f16(voice.speaker, [1, S, 768]),
                                    "caption": .f16(voice.caption, [1, T, 512])], "ck", "cv")
        let ctxMask = NDArray.f16(mask + voice.speakerMask + voice.captionMask, [1, 2 * T + S])
        let frameMask = NDArray.f16((0..<L).map { Float($0 < n ? 1 : 0) }, [1, L])
        var x = noise(n * D) + [Float](repeating: 0, count: (L - n) * D)
        for k in 0..<(constants.schedule.count - 1) {
            try Task.checkCancellation()
            let v = try await steps[L]!(["x": .f16(x, [1, L, D]), "cond": .f16(constants.conditions[k], [1, 1, constants.conditions[k].count]),
                                         "ck": kv[0], "cv": kv[1], "mask": frameMask, "ctx_mask": ctxMask], "v")[0].floats
            let dt = constants.schedule[k + 1] - constants.schedule[k]
            for i in x.indices { x[i] += v[i] * dt }
        }
        try await decodeChunks(x, frames: n, emit: emit)
    }

    /// Decodes `codecFrames` latent frames per call with `codecContext` frames of context on each side.
    private func decodeChunks(_ x: [Float], frames n: Int, emit: ([Float]) -> Void) async throws {
        let D = constants.latentDim, F = constants.codecFrames, C = constants.codecContext, W = F + 2 * C, hop = constants.hop
        for start in stride(from: 0, to: n, by: F) {
            try Task.checkCancellation()
            var z = [Float](repeating: 0, count: D * W)  // (1, D, W), zero outside [0, n)
            for w in 0..<W {
                let t = start - C + w
                guard t >= 0, t < n else { continue }
                for c in 0..<D { z[c * W + w] = x[t * D + c] }
            }
            let wav = try await decode(["z": .f16(z, [1, D, W])], "wav")[0].floats
            let keep = min(F, n - start) * hop
            emit(Array(wav[(C * hop)..<(C * hop + keep)]))
        }
    }

    /// Backbone + projection: (textLength × 512 state, mask). An empty string gives an all-zero mask.
    private func encode(_ text: String, _ tokenizer: Tokenizer, _ projection: InferenceFunction) async throws -> ([Float], [Float]) {
        let T = constants.textLength
        var (ids, mask) = tokenizer.encode(text, length: T)
        if text.isEmpty { mask = [Float](repeating: 0, count: T) }
        let count = Int(mask.reduce(0, +))
        let emb = NDArray(scalars: words.rows(Array(ids.prefix(count)), length: T), shape: [1, T, words.dim])
        let m = NDArray.f16(mask, [1, T])
        let hidden = try await backbone(["emb": emb, "mask": m], "hidden")[0]
        return (try await projection(["hidden": hidden, "mask": m], "state")[0].floats, mask)
    }

    /// Sentences packed greedily into pieces that fit one call.
    nonisolated func chunks(_ text: String) -> [String] {
        // ponytail: about 8 s of Japanese per piece; utterance() halves anything that still runs past the longest bucket
        let maxCharacters = 40
        var out = [String]()
        for sentence in Self.split(text, after: "。．！？!?") {
            if let last = out.last, last.count + sentence.count <= maxCharacters,
               textTokenizer.encode(TextNormalizer.normalize(last + sentence)).count <= constants.textLength {
                out[out.count - 1] = last + sentence
            } else {
                out.append(sentence)
            }
        }
        return out
    }

    /// Two halves of similar length, cut after a comma when there is one.
    static func halves(_ text: String) -> (String, String)? {
        let parts = split(text, after: "、，,")
        if parts.count > 1 {
            var best = 1, total = parts.reduce(0) { $0 + $1.count }, left = 0
            var bestGap = Int.max
            for i in 1..<parts.count {
                left += parts[i - 1].count
                if abs(total - 2 * left) < bestGap { (best, bestGap) = (i, abs(total - 2 * left)) }
            }
            return (parts[..<best].joined(), parts[best...].joined())
        }
        guard text.count > 1 else { return nil }
        let mid = text.index(text.startIndex, offsetBy: text.count / 2)
        return (String(text[..<mid]), String(text[mid...]))
    }

    /// Splits after any of `enders` and at line breaks, keeping the punctuation; closing brackets and further
    /// punctuation stay with the preceding piece.
    static func split(_ text: String, after enders: String) -> [String] {
        var out = [String](), current = "", ended = false
        for ch in text {
            if ch.isNewline {
                out.append(current)
                (current, ended) = ("", false)
                continue
            }
            if ended, !enders.contains(ch), !"」』）)】…".contains(ch) {
                out.append(current)
                (current, ended) = ("", false)
            }
            current.append(ch)
            if enders.contains(ch) { ended = true }
        }
        out.append(current)
        return out.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}
