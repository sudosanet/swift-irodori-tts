import Foundation


/// Seeded standard normal samples (SplitMix64 + Box–Muller); not bit-compatible with PyTorch.
struct Noise: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func normal(_ count: Int) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(count + 1)
        while out.count < count {
            let u = Double.random(in: Double.ulpOfOne..<1, using: &self), v = Double.random(in: 0..<1, using: &self)
            let r = (-2 * log(u)).squareRoot()
            out += [Float(r * cos(2 * .pi * v)), Float(r * sin(2 * .pi * v))]
        }
        return Array(out.prefix(count))
    }
}
