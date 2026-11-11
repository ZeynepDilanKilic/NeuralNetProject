# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking
- `NeuralNetwork(layerSizes, activation, seed)` is replaced by
  `NeuralNetwork(inputSize, std::vector<LayerSpec> layers, seed)`. The input width is no
  longer a trainable layer (`{2, 4, 1}` used to build three layers, the first a hidden 2×2
  layer), and every layer has its own activation. Old call sites fail to compile instead of
  changing meaning. See ADR 0004.
- Initial weights for a given seed changed (and are now identical on every platform), so
  networks built from a seed differ from earlier builds. See ADR 0005.

### Added
- `LayerSpec` and `ActivationType::Linear` (identity) for regression outputs. The
  constructor validates every layer before allocating and names the offending one.
- Portable weight initialization and a golden-value test that pins seed 42's weights.
- Install rules and a CMake package: `find_package(NeuralNet)` with the
  `neuralnet::neuralnet` target, a version file, and the license under
  `share/licenses/neuralnet`. `NEURALNET_INSTALL` (on when top-level) controls them.
- CI: a `minimum-versions` job (GCC 11, Clang 14, CMake 3.21), a `package` job that builds
  `tests/package` against the installed library, run cancellation for superseded pushes,
  and Dependabot for GitHub Actions.
- Tests: gradient check over mixed hidden/output activations, a learning rate other than 1
  and a new 1→6→3 topology; statistical Glorot/He initialization test; exact one-step
  update test; ReLU + Linear regression test with targets outside [0, 1].
- ADRs 0004 (explicit input width and per-layer activations) and 0005 (portable
  initialization).
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
- CI static analysis runs on `ubuntu-24.04` with pinned LLVM 18 (`clang-tidy-18`,
  `clang-format-18`) and that image's cppcheck; gcovr is pinned to 8.8.
- The XOR example and test use a 2 → 4 Tanh → 1 Sigmoid network, which converges for every
  seed tested instead of failing for about 8.5 % of seeds.
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
