"""Build .lake caches for each manifest version in the target repo."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from git import Repo
from huggingface_hub import hf_hub_download

app = typer.Typer(help="Build .lake caches for each manifest version")

DEFAULT_DATASET = "beneficial-ai-foundation/dalek-lean-githistory-eval"


def log(msg: str) -> None:
    """Print and flush immediately."""
    typer.echo(msg)
    sys.stdout.flush()


def compute_manifest_hash(repo_path: Path) -> str:
    """Compute SHA256 hash of lake-manifest.json."""
    manifest_path = repo_path / "lake-manifest.json"
    if not manifest_path.exists():
        return ""
    content = manifest_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


def get_manifest_data_from_dataset(
    dataset_name: str, repo: Repo | None = None
) -> dict[str, str]:
    """Load manifest hash -> commit mapping from the dataset.

    Downloads the raw JSONL to avoid HF schema issues.
    Uses parent_commit (where manifest_hash was computed) if available,
    otherwise falls back to original_commit.

    Args:
        dataset_name: HuggingFace dataset name.
        repo: GitPython Repo object (needed if manifest_hash not in dataset).

    Returns:
        Dict mapping manifest_hash -> representative commit (parent_commit preferred).
    """
    log(f"Loading dataset: {dataset_name}")

    # Download raw JSONL to avoid schema mismatch issues
    jsonl_path = hf_hub_download(
        repo_id=dataset_name,
        filename="test.jsonl",
        repo_type="dataset",
    )
    log(f"Downloaded JSONL from HuggingFace")

    hash_to_commit: dict[str, str] = {}
    commits_needing_hash: set[str] = set()

    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            metadata = row.get("metadata", {})
            # Prefer parent_commit (where manifest_hash was actually computed)
            # Fall back to original_commit for backwards compatibility
            commit = metadata.get("parent_commit") or metadata.get("original_commit", "")
            manifest_hash = metadata.get("manifest_hash", "")

            if not commit:
                continue

            if manifest_hash:
                # Dataset has manifest_hash - use it directly
                if manifest_hash not in hash_to_commit:
                    hash_to_commit[manifest_hash] = commit
            else:
                # Need to compute hash from commit
                commits_needing_hash.add(commit)

    if hash_to_commit:
        log(f"Found {len(hash_to_commit)} unique manifest hashes from dataset metadata")

    # Compute missing hashes if needed
    if commits_needing_hash and repo:
        log(f"Computing manifest hashes for {len(commits_needing_hash)} commits...")
        repo_path = Path(repo.working_dir)
        for commit in commits_needing_hash:
            try:
                repo.git.checkout(commit, force=True)
                manifest_hash = compute_manifest_hash(repo_path)
                if manifest_hash and manifest_hash not in hash_to_commit:
                    hash_to_commit[manifest_hash] = commit
            except Exception as e:
                log(f"  Warning: couldn't checkout {commit[:8]}: {e}")
        log(f"Total unique manifest versions: {len(hash_to_commit)}")
    elif commits_needing_hash and not repo:
        log(f"WARNING: {len(commits_needing_hash)} commits need hash computation but no repo provided")

    return hash_to_commit


def build_cache_for_commit(
    repo: Repo,
    commit_hash: str,
    output_dir: Path,
    timeout: int,
) -> tuple[str, bool, str]:
    """Build .lake cache for a specific commit.

    Returns:
        Tuple of (manifest_hash, success, message).
    """
    repo_path = Path(repo.working_dir)

    # Checkout the commit
    log(f"Checking out {commit_hash[:8]}...")
    repo.git.checkout(commit_hash, force=True)

    # Compute manifest hash
    manifest_hash = compute_manifest_hash(repo_path)
    if not manifest_hash:
        return "", False, "No lake-manifest.json found"

    # Check if cache already exists
    cache_dir = output_dir / manifest_hash
    log(f"  Manifest hash: {manifest_hash}")
    if cache_dir.exists():
        log(f"  Cache already exists, skipping")
        return manifest_hash, True, "already exists"

    # Clean any existing .lake
    lake_dir = repo_path / ".lake"
    if lake_dir.exists():
        shutil.rmtree(lake_dir)

    # Run lake update first
    log("  Running: lake update...")
    try:
        update_result = subprocess.run(
            ["lake", "update"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min for update
        )
        log(f"  lake update returned: {update_result.returncode}")
        if update_result.returncode != 0:
            log(f"  update stderr: {update_result.stderr[:500]}")
            # Continue anyway - update might fail but build could work
    except subprocess.TimeoutExpired:
        log("  lake update TIMEOUT, continuing...")

    # Run lake build
    timeout_msg = f"timeout={timeout}s" if timeout > 0 else "no timeout"
    log(f"  Running: lake build ({timeout_msg})...")
    try:
        result = subprocess.run(
            ["lake", "build"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout if timeout > 0 else None,
        )
        log(f"  lake build returned: {result.returncode}")
        if result.returncode != 0:
            log(f"  stderr: {result.stderr[:500]}")
            return manifest_hash, False, f"lake build failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        log("  TIMEOUT!")
        return manifest_hash, False, "lake build timed out"

    # Verify .lake was created
    if not lake_dir.exists():
        return manifest_hash, False, ".lake not created after build"

    log(f"  .lake exists at {lake_dir}")

    # Copy .lake to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_lake = cache_dir / ".lake"
    log(f"  Copying to {cache_lake}...")
    try:
        shutil.copytree(lake_dir, cache_lake, symlinks=True)
    except Exception as e:
        return manifest_hash, False, f"copytree failed: {e}"

    # Verify copy worked
    if cache_lake.exists():
        log(f"  SUCCESS: Cached to {cache_lake}")
    else:
        return manifest_hash, False, f"copytree succeeded but {cache_lake} doesn't exist"
    return manifest_hash, True, "built successfully"


@app.command()
def build(
    repo_path: Path = typer.Option(
        Path("data/curve25519-dalek-lean-verify"),
        "--repo",
        "-r",
        help="Path to the target repository",
    ),
    output_dir: Path = typer.Option(
        Path("lake_caches"),
        "--output",
        "-o",
        help="Output directory for caches",
    ),
    timeout: int = typer.Option(
        0,
        "--timeout",
        "-t",
        help="Timeout for each lake build in seconds (0 = no timeout)",
    ),
    dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        "-d",
        help="HuggingFace dataset to get manifest hashes from",
    ),
    commits: list[str] = typer.Option(
        None,
        "--commit",
        "-c",
        help="Specific commit(s) to build (overrides dataset discovery)",
    ),
) -> None:
    """Build .lake caches for manifest-changing commits."""
    # Resolve to absolute paths before any git operations
    output_dir = output_dir.resolve()
    repo_path = repo_path.resolve()

    repo = Repo(repo_path)
    original_head = repo.head.commit.hexsha

    log(f"Output directory: {output_dir}")
    log(f"Repo path: {repo_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if commits:
        # Use explicit commits - build each one
        log(f"Building caches for {len(commits)} explicit commits")
        commit_list = [(None, c) for c in commits]  # (expected_hash, commit)
    else:
        # Get manifest hash -> commit mapping from dataset
        hash_to_commit = get_manifest_data_from_dataset(dataset, repo)
        log(f"Building caches for {len(hash_to_commit)} unique manifest versions")
        commit_list = list(hash_to_commit.items())  # (manifest_hash, commit)

    results = []

    for expected_hash, commit in commit_list:
        manifest_hash, success, msg = build_cache_for_commit(
            repo, commit, output_dir, timeout
        )

        # Sanity check: if we expected a specific hash, verify it matches
        if expected_hash and manifest_hash and expected_hash != manifest_hash:
            log(f"  WARNING: Expected hash {expected_hash} but got {manifest_hash}")

        results.append((commit, manifest_hash, success, msg))

    # Restore original HEAD
    log(f"\nRestoring to {original_head[:8]}...")
    repo.git.checkout(original_head, force=True)

    # Print summary
    log("\n" + "=" * 60)
    log("BUILD SUMMARY")
    log("=" * 60)
    for commit, manifest_hash, success, msg in results:
        status = "OK" if success else "FAIL"
        log(f"{commit[:8]}  {manifest_hash or 'N/A':12}  [{status}] {msg}")

    built = sum(1 for _, _, s, m in results if s and m == "built successfully")
    cached = sum(1 for _, _, s, m in results if s and m == "already exists")
    failed = sum(1 for _, _, s, _ in results if not s)
    log(f"\nBuilt: {built}, Already cached: {cached}, Failed: {failed}")


@app.command()
def list_hashes(
    repo_path: Path = typer.Option(
        Path("data/curve25519-dalek-lean-verify"),
        "--repo",
        "-r",
        help="Path to the target repository",
    ),
    dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        "-d",
        help="HuggingFace dataset to get manifest hashes from",
    ),
) -> None:
    """List manifest hashes from dataset with their representative commits."""
    repo_path = repo_path.resolve()
    repo = Repo(repo_path)
    original_head = repo.head.commit.hexsha

    # Get hash -> commit mapping from dataset
    hash_to_commit = get_manifest_data_from_dataset(dataset, repo)

    log("\nManifest Hash   Commit      Verified?")
    log("-" * 45)

    for manifest_hash, commit in sorted(hash_to_commit.items()):
        # Verify the commit actually has this hash
        try:
            repo.git.checkout(commit, force=True)
            actual_hash = compute_manifest_hash(repo_path)
            if actual_hash == manifest_hash:
                status = "OK"
            else:
                status = f"MISMATCH (got {actual_hash})"
        except Exception as e:
            status = f"ERROR: {e}"

        log(f"{manifest_hash}    {commit[:8]}    {status}")

    repo.git.checkout(original_head, force=True)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
