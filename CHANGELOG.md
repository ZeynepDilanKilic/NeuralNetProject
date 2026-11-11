# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `LICENSE` file (MIT); the README already referenced it.
- The `xor` and `training_check` examples are registered as CTest tests and `xor` exits
  non-zero when XOR is not learned.
- Tests for NaN/infinity handling, the strong exception guarantee of `train()`, the
  pre-update loss returned by `train()`, moved-from networks and `computeLoss` validation,
  plus a test that documents the dead-ReLU behaviour.
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
- `NeuralNetwork::train` returns the sample's loss before the update (previously `void`).
- `train` rejects non-finite inputs and targets with `std::invalid_argument`.
- Using a moved-from `NeuralNetwork` throws `std::logic_error`; its shape accessors report 0.
- The gradient check moves ReLU units into their active region and fails if fewer than half
  of the gradients are non-zero; its data RNG no longer replays the weight-init stream.
- The single-neuron training test sets its weight explicitly instead of relying on the seed.
- `CMAKE_BUILD_TYPE` is only defaulted when NeuralNet is the top-level project, and the
  warnings target is linked as `$<BUILD_INTERFACE:...>`.
- README: Ninja requirement, accurate scope of CI, lint and gradient-check claims.
- Sources moved to `include/neuralnet/` and `src/`; everything lives in
  `namespace neuralnet`; `ActivationFunctionType` became `enum class ActivationType`.
- Neuron members are private; activation helpers are `static`.
- `tanh` uses `std::tanh` (the hand-rolled formula overflowed to NaN for |x| > ~709).
- Comments and documentation are in English and UTF-8.

### Fixed
- ReLU mapped NaN to 0, hiding NaN inputs and NaN-poisoned weights.
- `getInputSize()`, `getOutputSize()` and `Layer::getNumberOfInputs()` dereferenced an empty
  vector on a moved-from object (undefined behaviour in `noexcept` functions).
- The CI "Run examples" step could never fail because `find -exec` discarded exit codes.
- Weights are now randomly initialized (Glorot/He). Constant `0.1` initialization made
  every neuron in a layer identical, so hidden layers could not learn (see ADR 0002).

### Removed
- Dead code: `Neuron::calculateDeltas`, `Neuron::softmax`, the `(output, expected)`
  overload of `updateParameters`, `NeuralNetwork::derivativeOf*`, `feedForward`
  (duplicate of `predict`).
