import CoreAI

extension NDArray {
    static func f16(_ values: [Float], _ shape: [Int]) -> NDArray {
        NDArray(scalars: values.map { Float16($0) }, shape: shape)
    }

    /// Scalars in row-major order, whatever the layout and fp16/fp32 type.
    var floats: [Float] {
        func read<T: BinaryFloatingPoint & BitwiseCopyable>(_: T.Type) -> [Float] {
            view(as: T.self).withUnsafePointer { p, shapeSpan, stridesSpan in
                let shape = (0..<shapeSpan.count).map { shapeSpan[$0] }
                let strides = (0..<stridesSpan.count).map { stridesSpan[$0] }
                let count = shape.reduce(1, *)
                var out = [Float](repeating: 0, count: count)
                var index = [Int](repeating: 0, count: shape.count)
                for i in 0..<count {
                    var offset = 0
                    for d in shape.indices { offset += index[d] * strides[d] }
                    out[i] = Float(p[offset])
                    var d = shape.count - 1
                    while d >= 0 {
                        index[d] += 1
                        if index[d] < shape[d] { break }
                        index[d] = 0
                        d -= 1
                    }
                }
                return out
            }
        }
        return scalarType == .float16 ? read(Float16.self) : read(Float.self)
    }
}

extension InferenceFunction {
    /// Runs and takes the named outputs.
    func callAsFunction(_ inputs: [String: NDArray], _ names: String...) async throws -> [NDArray] {
        var out = try await run(inputs: inputs)
        return try names.map { name in
            guard let value = out.remove(name)?.ndArray else { throw IrodoriError.inference("missing output \(name)") }
            return value
        }
    }
}
