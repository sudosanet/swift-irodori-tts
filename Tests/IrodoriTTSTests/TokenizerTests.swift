import Foundation
import Testing
@testable import IrodoriTTS

/// Fixtures come from Tools/convert/fixtures.py (Irodori's Python tokenizer and normalizer).
struct TokenizerTests {
    struct Row: Decodable {
        let text: String
        let normalized: String
        let ids: [Int]
        let caption_ids: [Int]
    }

    static let rows: [Row] = {
        let url = Bundle.module.url(forResource: "tokenizer", withExtension: "json", subdirectory: "Fixtures")!
        return try! JSONDecoder().decode([Row].self, from: Data(contentsOf: url))
    }()

    static var tokenizerURL: URL? {
        let dir = ProcessInfo.processInfo.environment["IRODORI_MODELS"].map { URL(fileURLWithPath: $0) }
        return dir?.appending(path: "tokenizer.json")
    }

    @Test func normalizer() {
        for row in Self.rows {
            #expect(TextNormalizer.normalize(row.text) == row.normalized, "\(row.text)")
        }
    }

    @Test(.enabled(if: tokenizerURL != nil, "set IRODORI_MODELS to the model directory"))
    func textAndCaptionTokens() throws {
        let text = try Tokenizer(contentsOf: Self.tokenizerURL!, splitRareTokens: true)
        let caption = try Tokenizer(contentsOf: Self.tokenizerURL!, splitRareTokens: false)
        for row in Self.rows {
            #expect(text.encode(row.normalized) == row.ids, "\(row.normalized)")
            #expect(caption.encode(row.text) == row.caption_ids, "\(row.text)")
        }
    }
}
