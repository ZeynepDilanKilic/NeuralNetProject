# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- CMake build with presets (`debug`, `release`, `asan`, `coverage`, `msvc`) and a strict,
  compiler-specific warning set enabled as errors.
- Catch2 test suite: numerical gradient check over every parameter for all activations
  and several topologies, activation/derivative tests, XOR convergence, initialization
  and input-validation tests.
- GitHub Actions CI: GCC/Clang/MSVC/AppleClang matrix, ASan+UBSan, clang-tidy,
  cppcheck, clang-format and gcovr coverage with a 90 % line-coverage gate.
- `.clang-format`, `.clang-tidy`, `.editorconfig`, `.gitattributes`.
- `NeuralNetwork::computeLoss`, seeded construction, layer/neuron accessors.
- Architecture Decision Records under `docs/adr/`.

### Changed
- Sources moved to `include/neuralnet/` and `src/`; everything lives in
  `namespace neuralnet`; `ActivationFunctionType` became `enum class ActivationType`.
- Neuron members are private; activation helpers are `static`.
- `tanh` uses `std::tanh` (the hand-rolled formula overflowed to NaN for |x| > ~709).
- Comments and documentation are in English and UTF-8.

### Fixed
- Weights are now randomly initialized (Glorot/He). Constant `0.1` initialization made
  every neuron in a layer identical, so hidden layers could not learn (see ADR 0002).

### Removed
- Dead code: `Neuron::calculateDeltas`, `Neuron::softmax`, the `(output, expected)`
  overload of `updateParameters`, `NeuralNetwork::derivativeOf*`, `feedForward`
  (duplicate of `predict`).
