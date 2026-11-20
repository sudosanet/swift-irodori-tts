import Foundation

/// Irodori's `normalize_text` (irodori_tts/text_normalization.py, MIT), applied to the text to speak (not to captions).
public enum TextNormalizer {
    public static func normalize(_ input: String) -> String {
        var text = input
        for (old, new) in [("\t", ""), ("[n]", ""), ("\\[n\\]", ""), ("\u{3000}", ""), ("？", "?"), ("！", "!"), ("♥", "♡"),
                           ("●", "○"), ("◯", "○"), ("〇", "○")] {
            text = text.replacingOccurrences(of: old, with: new)
        }
        for (pattern, replacement) in [("[;▼♀♂《》≪≫①②③④⑤⑥]", ""),
                                       ("[\\u02d7\\u2010-\\u2015\\u2043\\u2212\\u23af\\u23e4\\u2500\\u2501\\u2e3a\\u2e3b]", ""),
                                       ("[\\uff5e\\u301C]", "ー"), ("…{3,}", "……")] {
            text = text.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
        }
        text = stripOuterBrackets(text).precomposedStringWithCompatibilityMapping
        text = text.replacingOccurrences(of: "...", with: "…").replacingOccurrences(of: "..", with: "…")
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func stripOuterBrackets(_ input: String) -> String {
        let pairs: [Character: Character] = ["「": "」", "『": "』", "（": "）", "【": "】", "(": ")"]
        var text = Array(input)
        while text.count >= 2, let close = pairs[text[0]], text[text.count - 1] == close {
            let open = text[0]
            var depth = 0, enclosing = true
            for (i, c) in text.enumerated() {
                if c == open { depth += 1 } else if c == close { depth -= 1 }
                if depth == 0, i < text.count - 1 {
                    enclosing = false
                    break
                }
            }
            guard enclosing, depth == 0 else { break }
            text = Array(text[1..<(text.count - 1)])
        }
        return String(text)
    }
}
