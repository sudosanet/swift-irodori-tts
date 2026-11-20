import Foundation
import Testing
@testable import IrodoriTTS

/// Compares against the Python Core AI pipeline and PyTorch with the same initial noise. Needs IRODORI_MODELS (the model
/// release) and IRODORI_PARITY, a directory with manifest.json and per row `<name>.noise.f32` (frames × 32),
/// `<name>.coreai.f32` and `<name>.torch.f32` (48 kHz waveforms); skipped otherwise.
struct ParityTests {
    struct Manifest: Decodable {
        struct Row: Decodable {
            let name: String
            let text: String
            let reference: Bool
            let frames: Int
        }
        let caption: String
        let reference: String
        let rows: [Row]
    }

    static let env = ProcessInfo.processInfo.environment

    static func floats(_ url: URL) throws -> [Float] {
        try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        var ab = 0.0, aa = 0.0, bb = 0.0
        for i in 0..<n {
            ab += Double(a[i] * b[i]); aa += Double(a[i] * a[i]); bb += Double(b[i] * b[i])
        }
        return ab / (aa * bb).squareRoot()
    }

    @Test(.enabled(if: env["IRODORI_MODELS"] != nil && env["IRODORI_PARITY"] != nil))
    func matchesPython() async throws {
        let dir = URL(filePath: Self.env["IRODORI_PARITY"]!)
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: dir.appending(path: "manifest.json")))
        let tts = try await IrodoriTTS(modelDirectory: URL(filePath: Self.env["IRODORI_MODELS"]!))
        let captionOnly = try await tts.voice(caption: manifest.caption)
        let withReference = try await tts.voice(caption: manifest.caption, reference: URL(filePath: manifest.reference))
        for row in manifest.rows {
            let noise = try Self.floats(dir.appending(path: "\(row.name).noise.f32"))
            var wav = [Float]()
            try await tts.utterance(row.text, voice: row.reference ? withReference : captionOnly, durationScale: 1,
                                    noise: { #expect($0 == noise.count); return noise }) { wav += $0 }
            let coreai = try Self.floats(dir.appending(path: "\(row.name).coreai.f32"))
            let torch = try Self.floats(dir.appending(path: "\(row.name).torch.f32"))
            let c1 = Self.cosine(wav, coreai), c2 = Self.cosine(wav, torch)
            print(row.name, wav.count / 1920, row.frames, String(format: "coreai %.4f torch %.4f", c1, c2))
            #expect(wav.count == row.frames * 1920)
            #expect(c1 > 0.97, "\(row.name) vs Python Core AI")
            #expect(c2 > 0.8, "\(row.name) vs PyTorch")
        }
    }
}
