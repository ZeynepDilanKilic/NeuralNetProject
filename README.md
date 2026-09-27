# NeuralNet

[![CI](https://github.com/ZeynepDilanKilic/NeuralNetProject/actions/workflows/ci.yml/badge.svg)](https://github.com/ZeynepDilanKilic/NeuralNetProject/actions/workflows/ci.yml)

A dependency-free C++20 library implementing feed-forward neural networks with
backpropagation, written with the engineering discipline expected of production code:
every gradient is verified numerically, every build runs with warnings as errors, and
every commit is checked by sanitizers, static analysis and a coverage gate.

## Features

- Fully connected feed-forward networks of arbitrary depth
- Sigmoid, tanh, ReLU and LeakyReLU activations
- Backpropagation with per-sample stochastic gradient descent on the squared-error loss
- Glorot/He weight initialization from a caller-supplied seed (reproducible runs)
- No third-party dependencies at runtime; Catch2 is used for tests only

## Quality gates

| Gate | Tooling | Where |
|------|---------|-------|
| Compiles warning-free | `-Wall -Wextra -Wpedantic -Wconversion …` / `/W4 /permissive-`, as errors | every build |
| Gradients are correct | Central-finite-difference check of every weight and bias, 4 activations × 3 topologies × 3 seeds | `tests/gradient_check_test.cpp` |
| Memory- and UB-safe | AddressSanitizer + UndefinedBehaviorSanitizer | CI job `sanitizers` |
| Lint | clang-tidy (`bugprone`, `cppcoreguidelines`, `modernize`, `performance`, `readability`), cppcheck | CI job `static-analysis` |
| Consistent style | clang-format, checked with `--Werror` | CI job `static-analysis` |
| Coverage | gcovr, ≥ 90 % line coverage of `src/` and `include/` | CI job `coverage` |
| Portability | GCC, Clang, MSVC and AppleClang | CI job `build-and-test` |

## Building

Requirements: CMake ≥ 3.21 and a C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+).
Catch2 is fetched automatically if it is not installed.

```bash
cmake --preset release     # or: debug, asan, coverage, msvc
cmake --build --preset release
ctest --preset release
```

Without presets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Useful options: `NEURALNET_WARNINGS_AS_ERRORS`, `NEURALNET_ENABLE_SANITIZERS`,
`NEURALNET_ENABLE_COVERAGE`, `NEURALNET_BUILD_TESTS`, `NEURALNET_BUILD_EXAMPLES`.

## Usage

```cpp
#include "neuralnet/NeuralNetwork.h"

using neuralnet::ActivationType;
using neuralnet::NeuralNetwork;

// Layer sizes {2, 4, 1}: a trainable 2-input layer, one hidden layer of 4, one output.
NeuralNetwork network({2, 4, 1}, ActivationType::Tanh, /*seed=*/42);

for (int epoch = 0; epoch < 3000; ++epoch) {
    network.train({0.0, 1.0}, {1.0}, /*learningRate=*/0.5);
    // ... remaining samples
}

double loss = network.computeLoss({0.0, 1.0}, {1.0});
std::vector<double> y = network.predict({0.0, 1.0});
```

See `examples/xor.cpp` for the complete XOR demo and `examples/training_check.cpp` for the
smallest possible end-to-end check.

## Repository layout

```
include/neuralnet/   public headers (Neuron, Layer, NeuralNetwork)
src/                 library sources
tests/               Catch2 unit tests
examples/            runnable examples
cmake/               shared CMake modules (compiler warnings)
docs/adr/            architecture decision records
.github/workflows/   CI definition
```

## Design notes

- **Derivatives are evaluated from the activation output, not the pre-activation.**
  The neuron caches only `a = f(z)`; see [ADR 0003](docs/adr/0003-derivatives-from-activation-output.md).
- **Random initialization is not optional.** Constant initialization silently collapses
  every layer to one effective neuron; see [ADR 0002](docs/adr/0002-random-weight-initialization.md).
- **Errors are exceptions.** Shape mismatches and invalid hyper-parameters throw
  `std::invalid_argument`; out-of-range accessors throw `std::out_of_range`.

## Roadmap

1. **Architecture** — matrix-backed layers (cache-friendly, SIMD-ready), pluggable
   activation / loss / optimizer strategies (SGD with momentum, Adam), mini-batch training,
   explicit input width in the constructor, `std::span` interfaces.
2. **Embedded** — heap-free inference path, int8/fixed-point quantized inference, ARM
   cross-compile toolchain file, Google Benchmark numbers in this README.
3. **Robustness** — versioned model serialization with integrity check, libFuzzer harness
   for the loader and input paths.
4. **Application** — a non-toy demo (neural syndrome decoder for a small quantum
   error-correcting code, or network-traffic anomaly detection).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
