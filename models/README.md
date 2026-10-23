# Bundled inference model

`asl_mediapipe_mlp_model.h5` is the original trained model moved from the
repository root, without changing its bytes.

- Input: 63 float32 coordinates (21 landmarks in x/y/z order).
- Dense layers: 128 ReLU, 64 ReLU, 28 softmax.
- Output order: A–Z, `del`, `space`.
- SHA-256: `0c3ac5d88059ac588a969b40ab0362c1b6d94d3207049b4a0ce007d02cbeaac0`.

The production loader reads only inference weights; no optimizer state or
TensorFlow runtime is loaded. See [architecture](../docs/architecture.md)
and [training instructions](../training/README.md) before replacing it.
