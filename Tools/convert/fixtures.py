"""Generate test fixtures for the Swift package from the reference Python implementation.

    python fixtures.py --irodori <Irodori-TTS clone> --out ../../Tests/IrodoriTTSTests/Fixtures

tokenizer.json: [{"text", "normalized", "ids"}] — Irodori's normalize_text + its tokenizer (rare-token splitting on, BOS
prepended, no truncation).
"""

import argparse
import json
import sys
from pathlib import Path

SENTENCES = [
    "こんにちは！今日もいっしょにがんばろうね。",
    "ビルドが通ったよ。テストも全部成功してる！",
    "ちょっと休憩しない？もう二時間も画面を見てるよ。",
    "えっ、エラーが出ちゃった。ログを一緒に見てみよう。",
    "今日の東京は晴れ、最高気温は二十三度だって。",
    "「おはようございます」",
    "（小声で）ねえ、聞こえてる……？",
    "ねえねえ😆、新しいバージョンが出たんだって！🥰 いっしょに見てみようよ〜😊",
    "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。",
    "Swift 6.4 と Xcode 27 で macOS 27 のアプリを作ろう。",
    "Hello, world! This is a test.",
    "価格は1,980円（税込）です。",
    "2026年11月20日、午前9時30分。",
    "浦和レッズの試合を見に行った。",
    "ゼルダの伝説、セブンイレブンで予約した？",
    "代表取締役社長が東日本大震災について話した。",
    "心筋梗塞と老若男女、興味津々の高田馬場。",
    "ｶﾀｶﾅとＡＢＣ１２３を全角半角で。",
    "～ウェーブ〜と―ダッシュ―と…………三点。",
    "♥●◯〇の記号。",
    "タブ\tと[n]改行と　全角空白。",
    "a",
    "",
    "ヴィンテージのルイヴィトン、ホワイトニング。",
    "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ",
    "𠮷野家で🍣を食べた。",
    "日本語版の無料ダウンロード、オンラインストアで。",
    "ゆっくり　はっきり　しゃべってね",
    "っっっっっ！！！",
    "AIがAIを作る時代。GPT、Claude、Gemini。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irodori", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.irodori))
    from huggingface_hub import hf_hub_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
    from irodori_tts.text_normalization import normalize_text

    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=hf_hub_download("Aratako/Irodori-TTS-v4.1-Small-MF", "model.safetensors"),
                                              model_device="cpu", codec_device="cpu"))
    rows = []
    for s in SENTENCES:
        n = normalize_text(s).strip()
        rows.append({"text": s, "normalized": n, "ids": rt.tokenizer.encode(n).tolist(), "caption_ids": rt.caption_tokenizer.encode(s).tolist()})
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out / "tokenizer.json", "w"), ensure_ascii=False, indent=1)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
