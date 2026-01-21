"""Build .lake caches for each manifest version in the target repo."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import typer
from git import Repo

app = typer.Typer(help="Build .lake caches for each manifest version")

# Commits that changed lake-manifest.json (in chronological order)
MANIFEST_COMMITS = [
    "f1c2e12",  # Initial commit
    "9d82afb",  # Oct 23
    "9c411c6",  # Oct 30
    "349a411",  # Nov 11
    "5234ce6",  # Nov 26
    "aef8b10",  # Dec 8
    "917e323",  # Dec 11
    "faa69fc",  # Jan 2
    "23d60b1",  # Jan 6 (current)
]


def compute_manifest_hash(repo_path: Path) -> str:
    """Compute SHA256 hash of lake-manifest.json."""
    manifest_path = repo_path / "lake-manifest.json"
    if not manifest_path.exists():
        return ""
    content = manifest_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


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
    typer.echo(f"Checking out {commit_hash}...")
    repo.git.checkout(commit_hash, force=True)

    # Compute manifest hash
    manifest_hash = compute_manifest_hash(repo_path)
    if not manifest_hash:
        return "", False, "No lake-manifest.json found"

    # Check if cache already exists
    cache_dir = output_dir / manifest_hash
    if cache_dir.exists():
        typer.echo(f"  Cache already exists for {manifest_hash}")
        return manifest_hash, True, "already exists"

    # Clean any existing .lake
    lake_dir = repo_path / ".lake"
    if lake_dir.exists():
        shutil.rmtree(lake_dir)

    # Run lake build
    typer.echo(f"  Building (manifest hash: {manifest_hash})...")
    try:
        result = subprocess.run(
            ["lake", "build"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return manifest_hash, False, f"lake build failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return manifest_hash, False, "lake build timed out"

    # Copy .lake to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_lake = cache_dir / ".lake"
    shutil.copytree(lake_dir, cache_lake, symlinks=True)

    typer.echo(f"  Cached to {cache_dir}")
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
        1800,
        "--timeout",
        "-t",
        help="Timeout for each lake build in seconds",
    ),
    commits: list[str] = typer.Option(
        None,
        "--commit",
        "-c",
        help="Specific commit(s) to build (default: all manifest commits)",
    ),
) -> None:
    """Build .lake caches for manifest-changing commits."""
    repo = Repo(repo_path)
    original_head = repo.head.commit.hexsha

    commits_to_build = commits if commits else MANIFEST_COMMITS
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for commit in commits_to_build:
        manifest_hash, success, msg = build_cache_for_commit(
            repo, commit, output_dir, timeout
        )
        results.append((commit, manifest_hash, success, msg))

    # Restore original HEAD
    typer.echo(f"\nRestoring to {original_head[:8]}...")
    repo.git.checkout(original_head, force=True)

    # Print summary
    typer.echo("\n" + "=" * 60)
    typer.echo("BUILD SUMMARY")
    typer.echo("=" * 60)
    for commit, manifest_hash, success, msg in results:
        status = "OK" if success else "FAIL"
        typer.echo(f"{commit[:8]}  {manifest_hash or 'N/A':12}  [{status}] {msg}")


@app.command()
def list_hashes(
    repo_path: Path = typer.Option(
        Path("data/curve25519-dalek-lean-verify"),
        "--repo",
        "-r",
        help="Path to the target repository",
    ),
) -> None:
    """List manifest hashes for each manifest-changing commit."""
    repo = Repo(repo_path)
    original_head = repo.head.commit.hexsha
    repo_path = Path(repo.working_dir)

    typer.echo("Commit    Manifest Hash")
    typer.echo("-" * 30)

    for commit in MANIFEST_COMMITS:
        repo.git.checkout(commit, force=True)
        manifest_hash = compute_manifest_hash(repo_path)
        typer.echo(f"{commit[:8]}  {manifest_hash or 'N/A'}")

    repo.git.checkout(original_head, force=True)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
