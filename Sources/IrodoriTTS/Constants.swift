import Foundation

/// constants.json written by Tools/convert/export.py.
struct Constants: Decodable {
    let format: Int
    let sampleRate: Int
    let hop: Int
    let latentDim: Int
    let textLength: Int
    let speakerLength: Int
    let speakerPatch: Int
    let lengths: [Int]
    let codecFrames: Int
    let codecContext: Int
    let vocabSize: Int
    let embeddingDim: Int
    let loudnessTarget: Double
    let minFrames: Int
    let maxFrames: Int
    /// MeanFlow time points, one more than the number of steps.
    let schedule: [Float]
    /// Precomputed timestep conditioning per step.
    let conditions: [[Float]]
    let nullSpeaker: [Float]
    let nullCaption: [Float]
}

/// Token embedding table of the text backbone (vocabSize × embeddingDim, fp16), memory-mapped.
struct WordTable: @unchecked Sendable {
    let data: Data
    let dim: Int

    init(contentsOf url: URL, vocab: Int, dim: Int) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count == vocab * dim * 2 else { throw IrodoriError.invalidModel("word_embeddings.f16 has an unexpected size") }
        self.dim = dim
    }

    /// Rows for `ids` followed by zero rows up to `length`.
    func rows(_ ids: [Int], length: Int) -> [Float16] {
        var out = [Float16](repeating: 0, count: length * dim)
        data.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self)
            for (i, id) in ids.prefix(length).enumerated() {
                out.withUnsafeMutableBufferPointer { o in
                    for j in 0..<dim { o[i * dim + j] = table[id * dim + j] }
                }
            }
        }
        return out
    }
}
