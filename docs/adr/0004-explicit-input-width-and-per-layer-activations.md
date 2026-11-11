# ADR 0004: Explicit input width and per-layer activations

**Status:** Accepted
**Date:** 2026-11-11

## Context

The original constructor was `NeuralNetwork(const std::vector<std::size_t>& layerSizes,
ActivationType activation, seed)`. It had two problems that a review measured:

- **`layerSizes[0]` was both the input width and a trainable layer.** `{2, 4, 1}` built
  three layers (2→2, 2→4, 4→1): an extra square layer applied the activation to the raw
  input. `{784, 128, 10}` carried a hidden 784×784 layer, about 7× the intended compute
  and memory. A plain perceptron or logistic regression (inputs → one unit) could not be
  expressed at all.
- **One activation for every layer, the output included.** A ReLU output cannot produce
  negative values and often starts dead (a single ReLU neuron was dead for 511 of 1000
  seeds); a sigmoid or tanh output cannot reach regression targets outside its range. With
  ReLU, the XOR network failed for every seed tried.

## Decision

- The constructor is `NeuralNetwork(std::size_t inputSize, const std::vector<LayerSpec>&
  layers, std::uint32_t seed)`, with `struct LayerSpec { std::size_t units; ActivationType
  activation; }`. The input is not a layer and has no weights; `layers` lists the trainable
  layers from the first hidden layer to the output layer. `LayerSpec` has a constructor, so
  the activation cannot be left out (an aggregate would default it to Sigmoid).
- Each layer has its own activation. `ActivationType::Linear` (identity, derivative 1) is
  added as the regression output activation. Its derivative is trivially expressible in
  terms of the output, so ADR 0003's convention still holds.
- The old constructor is removed rather than deprecated. Its arguments have a different
  meaning, so keeping it would preserve the trap it caused; the library has not been
  released, and old call sites fail to compile instead of silently changing meaning
  (`{2, 4, 1}` cannot convert to `std::size_t`).

Usage: `NeuralNetwork net(2, {{4, ActivationType::Tanh}, {1, ActivationType::Sigmoid}});`

## Alternatives considered

- **Keep `layerSizes`, but treat `layerSizes[0]` as input-only.** Smallest diff, but every
  existing call would silently lose a layer, and it still cannot express per-layer
  activations.
- **A builder (`Network::Builder().input(2).dense(4, Tanh).dense(1, Sigmoid)`).** Reads
  well, but adds a second way to describe a network before there is a need for one; it can
  be layered on top of `LayerSpec` later.
- **`std::span<const LayerSpec>`.** A span cannot bind a braced list, so the convenient
  `{{4, Tanh}, {1, Sigmoid}}` call would not compile.

## Consequences

- Networks are exactly the size the caller writes down, and mixed networks such as
  ReLU hidden layers with a Linear output (regression) or Tanh hidden layers with a
  Sigmoid output (binary classification) are possible. The XOR network
  2 → 4 Tanh → 1 Sigmoid now converges for all 200 seeds tested (previously 17 of 200
  failed with the extra layer and a Tanh output).
- Every call site had to change; the CHANGELOG lists this as a breaking change.
- Loss is still the squared error. A sigmoid/softmax output paired with cross-entropy is
  the natural next step and needs a loss abstraction, which is out of scope here.
