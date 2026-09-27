# ADR 0003: Activation derivatives take the activation output

**Status:** Accepted
**Date:** 2026-09-25

## Context

Backpropagation needs `f'(z)` for every neuron, where `z` is the pre-activation. The
neuron caches its output `a = f(z)` after each forward pass; caching `z` as well would
double the per-neuron state.

## Decision

`Neuron::activationDerivative(type, output)` is defined in terms of `a`, not `z`:

| Activation | `f'` in terms of `a` |
|------------|----------------------|
| Sigmoid    | `a * (1 - a)`        |
| Tanh       | `1 - a^2`            |
| ReLU       | `a > 0 ? 1 : 0`      |
| LeakyReLU  | `a > 0 ? 1 : slope`  |

For the ReLU family this is valid because `sign(a) == sign(z)`. The convention is
documented on the function and enforced by `tests/activation_test.cpp`, which compares
each analytic derivative against a finite difference of `f` at the corresponding `z`.

## Consequences

- One cached value per neuron and no redundant evaluation of `f` during backpropagation.
- Any future activation must either be expressible this way or the neuron must start
  caching `z`. That trade-off should be revisited when the activation set grows (softmax,
  GELU, etc.).
