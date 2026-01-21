"""Metrics and reporting for mining results."""

import logging

from git_history_proof_engineering_eval.structures import CommitCandidate, MiningResult

logger = logging.getLogger(__name__)


def print_candidates(candidates: list[CommitCandidate]) -> None:
    """Print summary of candidate commits (for dry-run mode).

    Args:
        candidates: List of candidate commits found.
    """
    print("\n" + "=" * 70)
    print("CANDIDATE COMMITS (sorry → proof transitions)")
    print("=" * 70)

    for i, candidate in enumerate(candidates, 1):
        total_sorries = sum(len(s) for s in candidate.filled_sorries.values())
        print(f"\n[{i}] {candidate.commit_hash[:8]} - {candidate.commit_message[:60]}")
        print(f"    Author: {candidate.author}")
        print(f"    Date: {candidate.date}")
        print(f"    Filled sorries: {total_sorries}")

        # Show files and declarations with filled sorries
        for file_path, sorries in list(candidate.filled_sorries.items())[:3]:
            decls = [s.enclosing_decl or "anon" for s in sorries]
            print(f"      - {file_path}: {', '.join(decls[:3])}")
            if len(decls) > 3:
                print(f"        ... and {len(decls) - 3} more declarations")

        if len(candidate.filled_sorries) > 3:
            print(f"      ... and {len(candidate.filled_sorries) - 3} more files")

    print("\n" + "=" * 70)
    print(f"Total candidates: {len(candidates)}")
    total_sorries_all = sum(
        sum(len(s) for s in c.filled_sorries.values()) for c in candidates
    )
    print(f"Total filled sorries: {total_sorries_all}")
    print("=" * 70)


def print_mining_summary(result: MiningResult) -> None:
    """Print summary of mining results.

    Args:
        result: Mining result with statistics.
    """
    print("\n" + "=" * 70)
    print("MINING SUMMARY")
    print("=" * 70)

    print(f"Commits analyzed:     {result.total_commits}")
    print(f"Candidates found:     {result.candidates}")
    print(f"Valid challenges:     {result.valid_challenges}")

    # Discovery rate: what % of commits had sorry→proof transitions
    if result.total_commits > 0:
        discovery_rate = (result.candidates / result.total_commits) * 100
        print(f"Discovery rate:       {discovery_rate:.1f}% of commits were candidates")

    # Yield: average challenges per candidate commit
    if result.candidates > 0:
        yield_rate = result.valid_challenges / result.candidates
        print(f"Yield:                {yield_rate:.1f} challenges per candidate")

    if result.skipped_reasons:
        print("\nSkipped reasons:")
        for reason, count in sorted(
            result.skipped_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {reason:25s} {count:4d}")

    print("=" * 70)

    # Log detailed statistics
    logger.info(f"Mining completed: {result.valid_challenges} challenges extracted")
    logger.info(f"Total commits: {result.total_commits}")
    logger.info(f"Candidates: {result.candidates}")
    logger.info(f"Skipped reasons: {result.skipped_reasons}")


def format_progress(current: int, total: int, message: str) -> str:
    """Format a progress message.

    Args:
        current: Current item number.
        total: Total items.
        message: Progress message.

    Returns:
        Formatted string like "[5/100] Processing commit abc123..."
    """
    return f"[{current}/{total}] {message}"
