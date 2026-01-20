"""Git history proof repair mining pipeline.

This module extracts proof repair challenges from the curve25519-dalek-lean-verify
git history by identifying commits where sorry placeholders were filled with actual proofs.
"""

import logging
from pathlib import Path

from git import Repo
from git.exc import GitCommandError

from git_history_proof_engineering_eval.config import Config, setup_logging
from git_history_proof_engineering_eval.file_classifier import find_filled_sorries, should_exclude_path
from git_history_proof_engineering_eval.git_ops import (
    get_commit_iterator,
    get_file_content_at_commit,
    get_file_diff,
    get_modified_files,
    safe_checkout,
)
from git_history_proof_engineering_eval.jsonl_writer import append_challenge
from git_history_proof_engineering_eval.metrics import (
    format_progress,
    print_candidates,
    print_mining_summary,
)
from git_history_proof_engineering_eval.snapshot import capture_codebase_snapshot
from git_history_proof_engineering_eval.structures import Challenge, CommitCandidate, MiningResult

logger = logging.getLogger(__name__)


def identify_candidates(repo: Repo, config: Config) -> list[CommitCandidate]:
    """Phase A: Identify commits where sorries were filled with actual proofs.

    Args:
        repo: GitPython repository object.
        config: Configuration object.

    Returns:
        List of candidate commits with filled_sorries info.
    """
    candidates = []
    processed = 0

    logger.info("Phase A: Identifying commits where sorries were filled")

    for commit in get_commit_iterator(Path(repo.working_dir), config.mining.start_ref):
        processed += 1

        # Check max_commits limit
        if config.mining.max_commits > 0 and processed > config.mining.max_commits:
            logger.info(f"Reached max_commits limit ({config.mining.max_commits})")
            break

        if processed % 10 == 0:
            logger.info(
                f"Processed {processed} commits, found {len(candidates)} candidates"
            )

        # Skip commits without parents (initial commit)
        if not commit.parents:
            continue

        parent = commit.parents[0]

        # Get modified files
        current_files, _ = get_modified_files(commit)

        # Filter out excluded paths and non-.lean files
        current_files = [
            f
            for f in current_files
            if not should_exclude_path(f, config.filtering.exclude_paths)
            and str(f).endswith(".lean")
        ]

        if not current_files:
            continue

        # Check each modified file for filled sorries
        filled_sorries: dict[Path, list] = {}

        for file_path in current_files:
            # Get file content at parent and child
            parent_content = get_file_content_at_commit(repo, parent, file_path)
            child_content = get_file_content_at_commit(repo, commit, file_path)

            if parent_content is None or child_content is None:
                continue

            # Find sorries that were filled in this commit
            sorries = find_filled_sorries(parent_content, child_content)

            if sorries:
                filled_sorries[file_path] = sorries

        # If any sorries were filled, this is a candidate
        if filled_sorries:
            total_filled = sum(len(s) for s in filled_sorries.values())
            candidate = CommitCandidate(
                commit_hash=commit.hexsha,
                commit_message=commit.message.strip(),
                author=str(commit.author),
                date=commit.committed_datetime,
                filled_sorries=filled_sorries,
            )
            candidates.append(candidate)
            logger.info(
                f"Found candidate: {commit.hexsha[:8]} - {total_filled} sorry(s) filled - "
                f"{commit.message.strip()[:40]}"
            )

    logger.info(
        f"Phase A complete: {len(candidates)} candidates from {processed} commits"
    )
    return candidates


def validate_candidate(
    repo: Repo, candidate: CommitCandidate, config: Config
) -> list[Challenge]:
    """Phase B: Create challenges from filled sorries.

    Creates one challenge per filled sorry. The challenge captures:
    - Codebase at parent commit (containing sorry)
    - Diff showing how the sorry was filled
    - Specific sorry location info

    Args:
        repo: GitPython repository object.
        candidate: Candidate commit to validate.
        config: Configuration object.

    Returns:
        List of challenges (one per filled sorry).
    """
    challenges = []
    repo_path = Path(repo.working_dir)
    commit = repo.commit(candidate.commit_hash)
    parent = commit.parents[0]

    # Create challenges for each filled sorry
    for file_path, sorry_list in candidate.filled_sorries.items():
        # Get the diff for this file
        file_diff = get_file_diff(repo, commit, file_path)

        # Create module name for verification command
        module_name = str(file_path).replace("/", ".").replace(".lean", "")
        verification_command = f"lake build {module_name}"

        for sorry in sorry_list:
            # Generate task_id from commit and declaration name
            decl_name = sorry.enclosing_decl or "anon"
            task_id = f"{candidate.commit_hash[:8]}_{decl_name}"

            challenge = Challenge(
                task_id=task_id,
                commit_hash=candidate.commit_hash,
                proof_file=file_path,
                sorry_location=sorry,
                author_fix_diff=file_diff,
                error_message=f"sorry in {decl_name}",
                verification_command=verification_command,
            )
            challenges.append(challenge)

            logger.info(
                f"Challenge created: {task_id} - {sorry.enclosing_decl or 'anonymous'} "
                f"at line {sorry.line}"
            )

    # Checkout parent and capture snapshot (codebase with sorries)
    if challenges:
        if not safe_checkout(repo, parent.hexsha):
            logger.warning(f"Failed to checkout parent {parent.hexsha}")
            return []

        codebase_snapshot = capture_codebase_snapshot(repo_path)

        # Attach snapshot to all challenges from this candidate
        for challenge in challenges:
            challenge.codebase_snapshot = codebase_snapshot

    return challenges


def run_mining(
    repo_path: Path | None = None,
    output_path: Path | None = None,
    limit: int = -1,
    dry_run: bool = False,
) -> MiningResult:
    """Run the proof repair challenge mining pipeline.

    Args:
        repo_path: Path to repository (overrides config).
        output_path: Path to output JSONL (overrides config).
        limit: Max commits to process (overrides config).
        dry_run: If True, only identify candidates without validation.

    Returns:
        MiningResult with statistics.
    """
    # Load configuration
    config = Config.from_file()

    # Override config with CLI arguments
    if repo_path:
        config.mining.repo_path = repo_path
    if output_path:
        config.output.jsonl_path = output_path
    if limit > 0:
        config.mining.max_commits = limit

    # Setup logging
    setup_logging(config)
    logger.info("Starting git history mining")
    logger.info(f"Repository: {config.mining.repo_path}")
    logger.info(f"Max commits: {config.mining.max_commits}")
    logger.info(f"Dry run: {dry_run}")

    # Open repository
    try:
        repo = Repo(config.mining.repo_path)
    except GitCommandError as e:
        logger.error(f"Failed to open repository: {e}")
        raise

    # Count total commits for statistics
    total_commits = sum(1 for _ in repo.iter_commits(config.mining.start_ref))
    if config.mining.max_commits > 0:
        total_commits = min(total_commits, config.mining.max_commits)

    logger.info(f"Total commits to analyze: {total_commits}")

    # Phase A: Identify candidates
    candidates = identify_candidates(repo, config)

    if dry_run:
        print_candidates(candidates)
        return MiningResult(
            total_commits=total_commits,
            candidates=len(candidates),
            valid_challenges=0,
            challenges=[],
            skipped_reasons={},
        )

    # Phase B: Validate and package
    logger.info("Phase B: Validating candidates and packaging challenges")

    # Initialize output file (clear any existing content)
    output_path = config.output.jsonl_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")  # Clear/create empty file
    logger.info(f"Output file initialized: {output_path}")

    all_challenges = []
    skipped_reasons: dict[str, int] = {}

    for i, candidate in enumerate(candidates, 1):
        progress_msg = format_progress(
            i,
            len(candidates),
            f"Validating {candidate.commit_hash[:8]} - {candidate.commit_message[:40]}...",
        )
        logger.info(progress_msg)
        print(progress_msg)

        try:
            challenges = validate_candidate(repo, candidate, config)

            if challenges:
                # Write challenges incrementally as they're found
                for challenge in challenges:
                    append_challenge(challenge, output_path)
                all_challenges.extend(challenges)
                logger.info(f"  Found {len(challenges)} challenge(s)")
                print(f"  Found {len(challenges)} challenge(s)")
            else:
                reason = "no_challenges"
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                logger.debug(f"  Skipped: {reason}")

        except Exception as e:
            logger.error(f"  Exception while validating: {e}")
            reason = "exception"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    # Final summary
    if all_challenges:
        print(f"\nWrote {len(all_challenges)} challenges to {output_path}")

    # Build result
    result = MiningResult(
        total_commits=total_commits,
        candidates=len(candidates),
        valid_challenges=len(all_challenges),
        challenges=all_challenges,
        skipped_reasons=skipped_reasons,
    )

    # Print summary
    print_mining_summary(result)

    return result


# CLI interface will be added separately
__all__ = [
    "run_mining",
    "identify_candidates",
    "validate_candidate",
]
