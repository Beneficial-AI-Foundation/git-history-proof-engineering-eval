"""Data structures for the git history mining pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SorryLocation:
    """Location of a sorry that was filled in."""

    line: int  # 1-indexed line number
    column: int  # 0-indexed column
    context: str  # The line containing the sorry
    enclosing_decl: str | None  # Name of enclosing theorem/lemma/def


@dataclass
class CommitCandidate:
    """A commit where sorries were filled with actual proofs."""

    commit_hash: str
    commit_message: str
    author: str
    date: datetime
    filled_sorries: dict[
        Path, list[SorryLocation]
    ]  # file -> sorries filled in that file


@dataclass
class VerificationResult:
    """Result from a Lake build attempt."""

    success: bool
    stdout: str
    stderr: str
    error_message: str | None
    timeout: bool


@dataclass
class Challenge:
    """A valid proof repair challenge extracted from git history."""

    task_id: str  # Format: "{commit_hash[:8]}_{decl_name}"
    commit_hash: str  # Child commit where sorry was filled
    parent_commit: str  # Parent commit where sorry exists (codebase snapshot source)
    proof_file: Path
    sorry_location: SorryLocation  # Specific sorry to fill
    author_fix_diff: str  # Diff showing how sorry was filled
    error_message: str  # "sorry in {decl_name}"
    codebase_snapshot: list[dict[str, str]] = field(
        default_factory=list
    )  # List of {"path": ..., "content": ...} for HF datasets compatibility
    verification_command: str = ""  # "lake build Module.Name"
    manifest_hash: str = ""  # SHA256 of lake-manifest.json (from parent commit)


@dataclass
class MiningResult:
    """Overall statistics from the mining process."""

    total_commits: int
    candidates: int
    valid_challenges: int
    challenges: list[Challenge]
    skipped_reasons: dict[str, int]  # reason -> count
