import Foundation
import Testing
@testable import IrodoriTTS

/// Fixtures come from Tools/convert/loudness_fixtures.py (audiotools, as Irodori uses for reference audio).
struct AudioTests {
    struct Row: Decodable {
        let name: String
        let loudness: Double
        let peak_after: Double
        let rms_after: Double
    }

    static func signal(_ name: String) -> [Float] {
        let sr = 48000.0
        let t = (0..<Int(3 * sr)).map { Double($0) / sr }
        func sine(_ f: Double) -> [Double] { t.map { sin(2 * .pi * f * $0) } }
        switch name {
        case "sine1k_0.1": return sine(1000).map { Float(0.1 * $0) }
        case "sines_mix":
            return zip(zip(sine(220), sine(5000)), sine(60)).map { Float(0.3 * $0.0 + 0.05 * $0.1 + 0.02 * $1) }
        case "burst":
            let s = sine(440)
            return s.indices.map { (24000..<48000).contains($0) ? Float(0.5 * s[$0]) : 0 }
        default: return sine(700).prefix(Int(0.3 * sr)).map { Float(0.2 * $0) }
        }
    }

    @Test func loudnessMatchesAudiotools() throws {
        let url = Bundle.module.url(forResource: "loudness", withExtension: "json", subdirectory: "Fixtures")!
        for row in try JSONDecoder().decode([Row].self, from: Data(contentsOf: url)) {
            let x = Self.signal(row.name)
            #expect(abs(Audio.loudness(x) - row.loudness) < 0.01, "\(row.name): \(Audio.loudness(x)) vs \(row.loudness)")
            let y = Audio.normalize(x, target: -16)
            let rms = (y.map { Double($0 * $0) }.reduce(0, +) / Double(y.count)).squareRoot()
            #expect(abs(rms - row.rms_after) / row.rms_after < 0.002, "\(row.name) rms")
        }
    }

    @Test func reflectPad() {
        #expect(Audio.padToHop([1, 2, 3, 4, 5], hop: 4) == [1, 2, 3, 4, 5, 4, 3, 2])
    }
}
