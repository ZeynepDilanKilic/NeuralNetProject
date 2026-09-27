# ADR 0001: Record architecture decisions

**Status:** Accepted
**Date:** 2026-09-25

## Context

The library is small today but is intended to grow (matrix-based layers, optimizers,
quantized inference, an embedded target). Design choices made now will constrain that
growth, and the reasoning behind them is lost quickly if it only lives in commit messages.

## Decision

Keep lightweight Architecture Decision Records in `docs/adr/`, one file per decision,
following the Nygard format: Context, Decision, Consequences. New records are numbered
sequentially and listed in `docs/adr/README.md`.

## Consequences

- Every non-obvious design choice has a written rationale that reviewers and future
  contributors can read and challenge.
- Reversing a decision means writing a new record that supersedes the old one, which
  keeps the history of *why* intact.
