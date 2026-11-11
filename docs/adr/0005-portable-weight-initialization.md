# ADR 0005: Portable weight initialization

**Status:** Accepted (amends the "Consequences" of [ADR 0002](0002-random-weight-initialization.md))
**Date:** 2026-11-11

## Context

ADR 0002 draws weights with `std::uniform_real_distribution<double>` from a seeded
`std::mt19937`. The engine's output sequence is fully specified by the C++ standard, but the
distribution's algorithm is not: libstdc++, libc++ and the MSVC STL turn the same seed into
different weights. A seed therefore identified a network only on one standard library, which
made seed-dependent tests and examples pass or fail per platform and made published results
irreproducible elsewhere.

## Decision

Weights are drawn with a hand-written conversion from the engine's output:

```cpp
u = ((rng() >> 5) * 2^26 + (rng() >> 6)) * 2^-53   // 53 random bits, u in [0, 1)
w = limit * (2u - 1)                                // w in [-limit, limit)
```

This is the `genrand_res53` construction from the reference Mersenne Twister code. Every
step is exact except the final multiplication, which is a single correctly rounded IEEE-754
operation: integers below 2^53 convert exactly, scaling by a power of two is exact, and
`2u - 1` is exact for `u` in `[0, 1)` (so even a fused multiply-add cannot change it).
`limit` comes from `std::sqrt`, which IEEE-754 also requires to be correctly rounded. The
engine output is masked to 32 bits by the standard even where `result_type` is 64 bits wide,
so the bit split is the same everywhere.

## Alternatives considered

- **`std::generate_canonical`.** Its specification has had portability and range defects
  that implementations resolved differently, so it does not reliably solve the problem.
- **A different engine (PCG, xoshiro).** Faster and smaller, but adds code for no benefit at
  this scale; `std::mt19937` is already portable.
- **Accept per-library results.** Rejected: golden-value tests and reproducible
  experiments are worth a ten-line function.

## Consequences

- A seed produces bit-identical initial weights with every compiler and standard library on
  platforms that evaluate `double` in IEEE-754 binary64 (`FLT_EVAL_METHOD == 0`): x86-64,
  ARM64, and MSVC on x86/x64. 32-bit x86 builds that use the x87 FPU (GCC/Clang without
  `-mfpmath=sse`) round intermediates to 80 bits first and can differ in the last bit. A
  golden-value test (`tests/training_test.cpp`) pins the first network for seed 42 with
  exact hexadecimal literals; changing the initialization is a deliberate, visible decision.
- Changing the conversion again would change every seeded network; treat it as breaking.
- Training results are still only reproducible per build: floating-point contraction
  settings and `libm` implementations of `exp`/`tanh` can differ between compilers. Only
  initialization is guaranteed portable.
