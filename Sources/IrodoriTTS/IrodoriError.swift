import Foundation

public enum IrodoriError: Error, LocalizedError {
    case invalidModel(String)
    case inference(String)
    case audio(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidModel(s): "Invalid model files: \(s)"
        case let .inference(s): "Inference failed: \(s)"
        case let .audio(s): "Audio error: \(s)"
        }
    }
}
