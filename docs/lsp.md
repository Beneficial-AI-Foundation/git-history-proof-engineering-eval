# LSP-Based Fast Feedback Mode

## Motivation

The current evaluation harness uses `lake build` for verification, which takes minutes per attempt. This is fine for one-shot evaluation but brutal for RL, which wants millions of iterations.

## Proposal

Add an LSP-based fast-feedback mode using `lean_goal` and `lean_diagnostic_messages` to get per-tactic feedback in ~ms instead of full rebuilds.

## Benefits for RL

- **Speed**: ~100ms per tactic vs minutes per build
- **Shaped rewards**: Can give partial credit based on:
  - Number of remaining goals (fewer = better)
  - Type errors vs proof errors (proof errors = closer)
  - Goal state similarity to closed
- **Multi-turn episodes**: Agent can iterate based on error messages rather than one-shot
- **Curriculum learning**: Start with omega/simp-solvable goals, progress to harder

## Implementation Sketch

1. Extract the broken proof's goal state at the sorry/error location
2. Let agent propose tactic sequences
3. Use `lean_goal` to check goal state after each tactic
4. Reward = f(initial_goals - remaining_goals, error_type, ...)
5. Episode ends when goals = 0 or max steps reached

## Prior Art

DeepMind's AlphaProof worked because they could run many Lean instances in parallel with fast per-tactic feedback.

## Status

Idea captured. Not yet implemented.
