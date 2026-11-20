import Foundation

/// SentencePiece Unigram tokenizer for `sbintuitions/modernbert-ja-310m`, matching Hugging Face `tokenizers`
/// (Metaspace without prefix, byte fallback, unknown penalty 10) and Irodori's rare-token splitting for the text input.
public struct Tokenizer: Sendable {
    public let bosID: Int
    public let padID: Int
    let unkID: Int
    let scores: [Double]
    /// Keys are scalar values: Swift `String` equality is canonical equivalence, `tokenizers` matches exact bytes.
    let pieces: [[UInt32]: Int]
    let byteIDs: [Int]
    let maxPieceLength: Int
    let unkScore: Double
    /// Token id → ids of the pieces it is replaced with (Irodori's forced splits).
    let forced: [Int: [Int]]

    /// - Parameter splitRareTokens: Irodori disables rare long tokens for the text input (not for captions).
    public init(contentsOf url: URL, splitRareTokens: Bool) throws {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any], model["type"] as? String == "Unigram",
              let vocab = model["vocab"] as? [[Any]] else { throw IrodoriError.invalidModel("tokenizer.json is not a Unigram tokenizer") }
        var scores = [Double](), pieces = [[UInt32]: Int](), maxLength = 0
        scores.reserveCapacity(vocab.count)
        for (i, item) in vocab.enumerated() {
            guard let piece = item.first as? String, let score = (item.last as? NSNumber)?.doubleValue else {
                throw IrodoriError.invalidModel("bad vocabulary entry \(i)")
            }
            scores.append(splitRareTokens && RareTokens.disabled(piece, score) ? -1e9 : score)
            let key = piece.unicodeScalars.map(\.value)
            if pieces[key] == nil { pieces[key] = i }
            maxLength = max(maxLength, piece.unicodeScalars.count)
        }
        self.scores = scores
        self.pieces = pieces
        self.maxPieceLength = maxLength
        self.unkID = model["unk_id"] as? Int ?? 0
        self.unkScore = (vocab.compactMap { ($0.last as? NSNumber)?.doubleValue }.min() ?? 0) - 10
        let id = { (s: String) in pieces[s.unicodeScalars.map(\.value)] }
        self.byteIDs = (0..<256).map { id(String(format: "<0x%02X>", $0)) ?? -1 }
        self.bosID = id("<s>") ?? 1
        self.padID = id("<pad>") ?? 3
        var forced = [Int: [Int]]()
        if splitRareTokens {
            let plain = Tokenizer(scores: scores, pieces: pieces, byteIDs: byteIDs, maxPieceLength: maxLength, unkID: unkID,
                                  unkScore: unkScore, bosID: bosID, padID: padID)
            for (word, parts) in RareTokens.forceSplits {
                guard let wordID = id(word) else { continue }
                var ids = [Int](), length = 0
                for part in parts {
                    if let partID = id(part) {
                        ids.append(partID)
                    } else {
                        ids += plain.encodeBody(part)
                    }
                    length += part.unicodeScalars.count
                }
                if length == word.unicodeScalars.count { forced[wordID] = ids }
            }
        }
        self.forced = forced
    }

    private init(scores: [Double], pieces: [[UInt32]: Int], byteIDs: [Int], maxPieceLength: Int, unkID: Int, unkScore: Double,
                 bosID: Int, padID: Int) {
        self.scores = scores
        self.pieces = pieces
        self.byteIDs = byteIDs
        self.maxPieceLength = maxPieceLength
        self.unkID = unkID
        self.unkScore = unkScore
        self.bosID = bosID
        self.padID = padID
        self.forced = [:]
    }

    /// BOS + tokens, without truncation.
    public func encode(_ text: String) -> [Int] {
        [bosID] + encodeBody(text).flatMap { forced[$0] ?? [$0] }
    }

    /// Ids padded or truncated to `length` (BOS first) and the 0/1 mask.
    public func encode(_ text: String, length: Int) -> (ids: [Int], mask: [Float]) {
        let ids = Array(encode(text).prefix(length))
        return (ids + Array(repeating: padID, count: length - ids.count),
                Array(repeating: 1, count: ids.count) + Array(repeating: 0, count: length - ids.count))
    }

    /// Viterbi over Unicode scalars, as `tokenizers`' Unigram `encode_optimized`: strictly better scores win, so on ties the
    /// shorter piece found first is kept; positions no single-scalar piece covers get the unknown token, consecutive
    /// unknowns are fused and then byte-fallbacked.
    func encodeBody(_ text: String) -> [Int] {
        let scalars = Array(text.replacingOccurrences(of: " ", with: "\u{2581}").unicodeScalars)
        let n = scalars.count
        guard n > 0 else { return [] }
        var best = [Double](repeating: -.infinity, count: n + 1)
        var back = [(start: Int, id: Int)](repeating: (-1, -1), count: n + 1)
        best[0] = 0
        for start in 0..<n where best[start] > -.infinity {
            var single = false
            var key = [UInt32]()
            for end in (start + 1)...min(n, start + maxPieceLength) {
                key.append(scalars[end - 1].value)
                guard let id = pieces[key] else { continue }
                let candidate = best[start] + scores[id]
                if back[end].id < 0 || candidate > best[end] {
                    best[end] = candidate
                    back[end] = (start, id)
                }
                if end == start + 1 { single = true }
            }
            if !single {
                let candidate = best[start] + unkScore
                if back[start + 1].id < 0 || candidate > best[start + 1] {
                    best[start + 1] = candidate
                    back[start + 1] = (start, unkID)
                }
            }
        }
        var spans = [(start: Int, end: Int, id: Int)]()
        var end = n
        while end > 0 {
            let (start, id) = back[end]
            if id == unkID, let last = spans.last, last.id == unkID {
                spans[spans.count - 1] = (start, last.end, unkID)
            } else {
                spans.append((start, end, id))
            }
            end = start
        }
        var ids = [Int]()
        for span in spans.reversed() {
            guard span.id == unkID else {
                ids.append(span.id)
                continue
            }
            var piece = String.UnicodeScalarView()
            piece.append(contentsOf: scalars[span.start..<span.end])
            let bytes = Array(String(piece).utf8).map { byteIDs[Int($0)] }
            ids += bytes.contains(-1) ? [unkID] : bytes
        }
        return ids
    }
}

/// Irodori's rare-token rule (irodori_tts/token_split.py, MIT): tokens of ≥ 4 characters with a score ≤ -13 are disabled,
/// except hiragana-only tokens and `keep`; `forceSplits` are split as listed.
enum RareTokens {
    static func disabled(_ piece: String, _ score: Double) -> Bool {
        let text = piece.replacingOccurrences(of: "\u{2581}", with: "")
        return score <= -13 && text.unicodeScalars.count >= 4 && !keep.contains(text) && forceSplits[text] == nil
            && !text.unicodeScalars.allSatisfy { (0x3041...0x3096).contains($0.value) || [0x309D, 0x309E, 0x30FC].contains($0.value) }
    }

    static let keep: Set<String> = [
        "心筋梗塞", "日常茶飯事", "瀬戸内海", "老若男女", "興味津々", "高田馬場", "アクリル板", "年モデル", "オフショア", "モスクワ",
        "テラス席", "缶バッジ", "ジュリー", "ッチャー", "ティーン", "トゥルー", "マリッジ", "リフィル", "リベット", "デパート",
        "カリウム", "タピオカ", "ボーイズ", "フレックス", "フリース", "ファブリック", "ユナイテッド", "ブルース", "ミスター", "タイムズ",
        "ソケット", "コスチューム", "プレビュー", "リーズナブル", "プロダクション", "ファイト", "アバター", "産婦人科", "行方不明", "直射日光",
    ]

    static let forceSplits: [String: [String]] = [
        "セブンイレブン": ["セブン", "イレブン"], "楽天市場": ["楽天", "市場"], "外壁塗装": ["外壁", "塗装"],
        "イベルメクチン": ["イベルメ", "クチン"], "ミノキシジル": ["ミノ", "キシ", "ジル"], "送料無料": ["送料", "無料"],
        "ヴィンテージ": ["ヴィン", "テージ"], "点セット": ["点", "セット"], "フォルダ": ["フォル", "ダ"], "消費者金融": ["消費者", "金融"],
        "代表取締役": ["代表", "取締役"], "本セット": ["本", "セット"], "厚生労働省": ["厚生", "労働", "省"],
        "オンラインカジノ": ["オンライン", "カジノ"], "当サイト": ["当", "サイト"], "プロペシア": ["プロ", "ペ", "シア"],
        "リノベーション": ["リノベ", "ー", "ション"], "メール便": ["メール", "便"], "不動産投資": ["不動産", "投資"],
        "ティック": ["ティ", "ック"], "フジテレビ": ["フジ", "テレビ"], "東日本大震災": ["東日本", "大震災"],
        "新品未使用": ["新品", "未使用"], "ネクスガード": ["ネ", "クス", "ガード"], "全身脱毛": ["全身", "脱毛"], "債務整理": ["債務", "整理"],
        "ス感染症": ["ス", "感染症"], "代表取締役社長": ["代表", "取締役", "社長"], "枚セット": ["枚", "セット"],
        "フィナステリド": ["フィ", "ナス", "テリ", "ド"], "クチコミ": ["クチ", "コミ"], "ブロガー": ["ブロ", "ガー"],
        "ルイヴィトン": ["ルイ", "ヴィ", "トン"], "転職エージェント": ["転職", "エージェント"], "数量限定": ["数量", "限定"],
        "ホワイトニング": ["ホワイト", "ニング"], "無料ダウンロード": ["無料", "ダウンロード"], "スーパーコピー": ["スーパー", "コピー"],
        "一般社団法人": ["一般", "社", "団", "法人"], "日本語版": ["日本語", "版"], "ペット用品": ["ペット", "用品"], "投資信託": ["投資", "信託"],
        "脱毛サロン": ["脱毛", "サロン"], "東京都港区": ["東京都", "港区"], "注文住宅": ["注文", "住宅"], "マイナビ": ["マイ", "ナビ"],
        "行政書士": ["行政", "書", "士"], "無料相談": ["無料", "相談"], "市区町村": ["市", "区", "町", "村"], "ザガーロ": ["ザ", "ガー", "ロ"],
        "リハビリテーショ": ["リハビリ", "テーショ"], "株主優待": ["株主", "優待"], "オンラインストア": ["オンライン", "ストア"],
        "プレゼンテーショ": ["プレゼン", "テーショ"], "個人事業主": ["個人", "事業主"], "社会福祉法人": ["社会", "福祉", "法人"],
        "カルティエ": ["カル", "ティエ"], "イオンモール": ["イオン", "モール"], "ピーター": ["ピー", "ター"], "銀行振込": ["銀行", "振込"],
        "スタッフ一同": ["スタッフ", "一同"], "即日融資": ["即日", "融資"], "プラセンタ": ["プラ", "センタ"], "大学大学院": ["大学", "大学院"],
        "モンクレール": ["モン", "ク", "レール"], "賃貸物件": ["賃貸", "物件"], "協同組合": ["協同", "組合"], "年以上前": ["年", "以上", "前"],
        "土日祝日": ["土日", "祝日"], "ウェイト": ["ウェ", "イト"],
    ]
}
