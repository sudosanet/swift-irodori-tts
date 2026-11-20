import Testing
@testable import IrodoriTTS

struct ChunkTests {
    @Test func split() {
        #expect(IrodoriTTS.split("こんにちは！今日は「晴れ」。\nまたね……！？", after: "。！？") == ["こんにちは！", "今日は「晴れ」。", "またね……！？"])
        #expect(IrodoriTTS.split("「おはよう。」と言った。", after: "。") == ["「おはよう。」", "と言った。"])
    }

    @Test func halves() {
        #expect(IrodoriTTS.halves("あいう、えお、かきくけこ、さ")! == ("あいう、えお、", "かきくけこ、さ"))
        #expect(IrodoriTTS.halves("あいうえ")! == ("あい", "うえ"))
        #expect(IrodoriTTS.halves("あ") == nil)
    }
}
