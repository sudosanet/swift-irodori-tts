import AVFoundation
import Accelerate

/// Reference-audio preprocessing, matching Irodori's codec path: mono, 48 kHz, loudness normalized to -16 LUFS
/// (audiotools / pyloudnorm, ITU-R BS.1770-4) with peak safety scaling, reflect-padded to a multiple of the hop.
enum Audio {
    static let sampleRate = 48000.0

    /// Mono samples at 48 kHz from any file AVFoundation can read.
    static func load(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw IrodoriError.audio("cannot allocate a buffer for \(url.lastPathComponent)")
        }
        try file.read(into: input)
        let channels = Int(format.channelCount), n = Int(input.frameLength)
        guard let data = input.floatChannelData else { throw IrodoriError.audio("unsupported sample format") }
        var mono = [Float](repeating: 0, count: n)
        for c in 0..<channels {
            vDSP.add(mono, UnsafeBufferPointer(start: data[c], count: n), result: &mono)
        }
        if channels > 1 { vDSP.multiply(1 / Float(channels), mono, result: &mono) }
        return format.sampleRate == sampleRate ? mono : try resample(mono, from: format.sampleRate)
    }

    static func resample(_ x: [Float], from rate: Double) throws -> [Float] {
        guard let src = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
              let dst = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let converter = AVAudioConverter(from: src, to: dst),
              let input = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(x.count)),
              let output = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: AVAudioFrameCount(Double(x.count) * sampleRate / rate) + 1024)
        else { throw IrodoriError.audio("cannot resample from \(rate) Hz") }
        converter.sampleRateConverterQuality = .max
        input.frameLength = AVAudioFrameCount(x.count)
        x.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: x.count) }
        var fed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if fed {
                status.pointee = .endOfStream
                return nil
            }
            fed = true
            status.pointee = .haveData
            return input
        }
        if let error { throw IrodoriError.audio(error.localizedDescription) }
        return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }

    /// Integrated loudness in LUFS (BS.1770-4: K-weighting, 400 ms blocks with 75 % overlap, -70 LUFS absolute and -10 LU
    /// relative gates), floored at -70 like audiotools. Signals shorter than 0.5 s are zero-padded to 0.5 s.
    static func loudness(_ input: [Float], rate: Double = sampleRate) -> Double {
        var x = input.map(Double.init)
        if x.count < Int(0.5 * rate) { x += [Double](repeating: 0, count: Int(0.5 * rate) - x.count) }
        for (b, a) in [biquad(.highShelf, gain: 4, q: 1 / 2.0.squareRoot(), fc: 1500, rate: rate),
                       biquad(.highPass, gain: 0, q: 0.5, fc: 38, rate: rate)] {
            x = filter(x, b: b, a: a)
        }
        let size = Int(0.4 * rate), stride = Int(0.4 * rate * 0.25)
        let frames = Int((Double(max(x.count, size) - size) / Double(stride)).rounded(.up)) + 1
        let padded = x + [Double](repeating: 0, count: max(0, (frames - 1) * stride + size - x.count))
        let z = (0..<frames).map { f in vDSP.sumOfSquares(padded[(f * stride)..<(f * stride + size)]) / Double(size) }
        let l = z.map { -0.691 + 10 * log10($0) }
        func mean(_ keep: (Int) -> Bool) -> Double {
            let kept = z.indices.filter(keep)
            return kept.isEmpty ? .nan : kept.map { z[$0] }.reduce(0, +) / Double(kept.count)
        }
        let relative = -0.691 + 10 * log10(mean { l[$0] > -70 }) - 10
        var gated = mean { l[$0] > -70 && l[$0] > relative }
        if gated.isNaN { gated = 0 }
        return max(-70, -0.691 + 10 * log10(gated))
    }

    /// Normalize to `target` LUFS, then scale down if the peak exceeds 1.
    static func normalize(_ x: [Float], target: Double) -> [Float] {
        let gain = Float(pow(10, (target - loudness(x)) / 20))
        var y = vDSP.multiply(gain, x)
        let peak = vDSP.maximumMagnitude(y)
        if peak > 1 { y = vDSP.multiply(1 / peak, y) }
        return y
    }

    /// Reflect-pad the end to a multiple of `hop` (torch.nn.functional.pad mode "reflect").
    static func padToHop(_ x: [Float], hop: Int) -> [Float] {
        let r = x.count % hop
        guard r != 0 else { return x }
        let need = hop - r
        precondition(need < x.count, "reference audio is too short")
        return x + (0..<need).map { x[x.count - 2 - $0] }
    }

    enum Kind { case highShelf, highPass }

    /// pyloudnorm IIRfilter coefficients, normalized by a0.
    static func biquad(_ kind: Kind, gain: Double, q: Double, fc: Double, rate: Double) -> (b: [Double], a: [Double]) {
        let A = pow(10, gain / 40), w0 = 2 * Double.pi * fc / rate, alpha = sin(w0) / (2 * q), c = cos(w0)
        let b: [Double], a: [Double]
        switch kind {
        case .highShelf:
            b = [A * ((A + 1) + (A - 1) * c + 2 * A.squareRoot() * alpha), -2 * A * ((A - 1) + (A + 1) * c),
                 A * ((A + 1) + (A - 1) * c - 2 * A.squareRoot() * alpha)]
            a = [(A + 1) - (A - 1) * c + 2 * A.squareRoot() * alpha, 2 * ((A - 1) - (A + 1) * c),
                 (A + 1) - (A - 1) * c - 2 * A.squareRoot() * alpha]
        case .highPass:
            b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
            a = [1 + alpha, -2 * c, 1 - alpha]
        }
        return (b.map { $0 / a[0] }, a.map { $0 / a[0] })
    }

    static func filter(_ x: [Double], b: [Double], a: [Double]) -> [Double] {
        var y = [Double](repeating: 0, count: x.count)
        var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0
        for i in x.indices {
            let v = b[0] * x[i] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x2 = x1; x1 = x[i]; y2 = y1; y1 = v
            y[i] = v
        }
        return y
    }
}
