# ADR 0002: Seeded Glorot/He weight initialization

**Status:** Accepted
**Date:** 2026-09-25

## Context

The original implementation initialized every weight to `0.1` and every bias to `0.1`.
With identical parameters, every neuron in a layer computes the same output, receives the
same gradient and therefore stays identical forever: a layer of N neurons has the capacity
of a single neuron. The network could not learn XOR.

## Decision

- Weights are drawn from a uniform distribution whose half-width depends on the activation:
  Glorot/Xavier (`sqrt(6 / (fan_in + fan_out))`) for sigmoid and tanh, He
  (`sqrt(6 / fan_in)`) for the ReLU family. Biases start at zero.
- The random engine is a `std::mt19937` seeded from a `NeuralNetwork` constructor
  argument (default `42`), so a given seed always produces the same network on the same
  standard library. Tests and examples rely on this for reproducibility.

## Consequences

- Symmetry between neurons is broken; the XOR test passes and gradient checks can exercise
  every parameter independently.
- `std::uniform_real_distribution` is implementation-defined, so a seed reproduces the same
  network within one standard library (libstdc++, libc++, MSVC STL) but not across them.
  A hand-rolled distribution would fix this and is noted as a follow-up.
- Initialization is currently coupled to `Layer`. When layers move to a matrix
  representation, initialization should become a pluggable strategy.
