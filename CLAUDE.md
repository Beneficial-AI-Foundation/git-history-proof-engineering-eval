# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project mines git history from Lean 4 repositories to extract proof repair challenges. It identifies commits where definition changes broke existing proofs, packages them as evaluation challenges with the broken state, and outputs JSONL files that can be used to benchmark AI systems on proof repair.

The primary target repository is `curve25519-dalek-lean-verify` (included as a git submodule).

## Commands

```bash
# Install dependencies
uv sync

# Run the mining pipeline (all commits, default config)
uv run mine generate

# Dry run to see candidates without validation
uv run mine generate --dry-run --limit 50

# Custom output path
uv run mine generate -o ./my_challenges.jsonl

# Limit commits processed
uv run mine generate --limit 100
```

## Architecture

The pipeline has two phases:

**Phase A (Identification)**: Iterates through git history finding commits that modify both definition files (Aeneas-generated code, `def`, `structure`, `inductive`) and proof files (files with `theorem`, `lemma`, or tactic blocks).

**Phase B (Validation)**: For each candidate commit:
1. Checkout the "fixed" state (commit C where proof passes)
2. Verify `lake build` succeeds
3. Restore the proof file to its parent state (creating "broken" state)
4. Run `lake build` - if it fails with a verification error, this is a valid challenge
5. Capture the codebase snapshot and author's fix diff

### Key Modules

- `__init__.py` - Main pipeline orchestration (`run_mining`, `identify_candidates`, `validate_candidate`)
- `file_classifier.py` - Heuristics to classify Lean files as definition/proof based on content patterns
- `git_ops.py` - GitPython operations for traversing history and manipulating working tree
- `verification.py` - Lake build execution and error parsing
- `snapshot.py` - Captures codebase state for challenge packaging
- `structures.py` - Pydantic-style dataclasses for `CommitCandidate`, `Challenge`, `MiningResult`
- `config.py` - TOML configuration loading (defaults in `config.toml`)

### Configuration

`src/git_history_proof_engineering_eval/config.toml` contains:
- `[mining]` - repo path, start ref, max commits
- `[filtering]` - minimum theorem/tactic counts, excluded paths
- `[verification]` - Lake build timeout
- `[output]` - JSONL output path, log directory

### Output Format

JSONL with one challenge per line:
```json
{
  "task_id": "abc123_FileName",
  "metadata": { "original_commit": "...", "author_fix_diff": "...", "error_message": "..." },
  "setup": { "instructions": "...", "codebase_state": { "path/file.lean": "content" } },
  "verification": { "command": "lake build Module.Name", "timeout_seconds": 300 }
}
```

## Development Notes

- Requires Python 3.14+
- Uses `uv` for dependency management (uv_build backend)
- The target repo (curve25519-dalek-lean-verify) must be initialized: `git submodule update --init`
- Lake must be available in PATH for verification to work
