/// Conditioning for a voice: a style caption, an optional reference speaker, or both. Reusable across utterances.
public struct Voice: Sendable {
    /// Caption encoder output (textLength × 512) and its mask.
    let caption: [Float]
    let captionMask: [Float]
    /// Speaker encoder output (speakerLength × 768) and its mask; zeros without a reference.
    let speaker: [Float]
    let speakerMask: [Float]
    /// Pooled vectors for the duration predictor.
    let captionVector: [Float]
    let speakerVector: [Float]
}
