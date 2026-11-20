// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "swift-irodori-tts",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "IrodoriTTS", targets: ["IrodoriTTS"]),
        .executable(name: "irodori-say", targets: ["irodori-say"]),
    ],
    targets: [
        .target(name: "IrodoriTTS"),
        .executableTarget(name: "irodori-say", dependencies: ["IrodoriTTS"]),
        .testTarget(name: "IrodoriTTSTests", dependencies: ["IrodoriTTS"], resources: [.copy("Fixtures")]),
    ]
)
