import AVFAudio
import Foundation
import IrodoriTTS

let usage = """
usage: irodori-say --models <dir> [--caption <text>] [--reference <audio>] [--seed <n>] [--duration-scale <x>] [-o speech.wav] <text>
"""

var options = [String: String](), words = [String]()
var rest = CommandLine.arguments.dropFirst()
while let arg = rest.popFirst() {
    if arg.hasPrefix("-"), let value = rest.popFirst() { options[arg] = value } else { words.append(arg) }
}
let models = options["--models"] ?? ProcessInfo.processInfo.environment["IRODORI_MODELS"]
let caption = options["--caption"] ?? ""
let reference = options["--reference"]
let seed = options["--seed"].flatMap(UInt64.init)
let scale = options["--duration-scale"].flatMap(Float.init) ?? 1
let output = options["-o"] ?? "speech.wav"
guard let models, !words.isEmpty else {
    FileHandle.standardError.write(Data(usage.utf8))
    exit(2)
}

let clock = ContinuousClock()
var start = clock.now
let tts = try await IrodoriTTS(modelDirectory: URL(filePath: models))
print("load: \(clock.now - start)")
start = clock.now
let voice = try await tts.voice(caption: caption, reference: reference.map { URL(filePath: $0) })
print("voice: \(clock.now - start)")
start = clock.now
let samples = try await tts.synthesize(words.joined(separator: " "), voice: voice, durationScale: scale, seed: seed)
let seconds = Double(samples.count) / Double(IrodoriTTS.sampleRate)
print("speech: \(String(format: "%.2f", seconds)) s in \(clock.now - start)")

let format = AVAudioFormat(standardFormatWithSampleRate: Double(IrodoriTTS.sampleRate), channels: 1)!
let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
buffer.frameLength = buffer.frameCapacity
samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
do {  // the header is finalized when the file is released
    let file = try AVAudioFile(forWriting: URL(filePath: output), settings: format.settings)
    try file.write(from: buffer)
}
