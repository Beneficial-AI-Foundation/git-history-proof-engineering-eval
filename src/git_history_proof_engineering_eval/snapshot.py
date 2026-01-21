"""Codebase snapshot capture for challenge packaging."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


EXCLUDED_DIRS = {".lake", ".git", "build", ".elan"}

# Build configuration files that should always be included
BUILD_CONFIG_FILES = [
    "lakefile.lean",
    "lakefile.toml",
    "lake-manifest.json",
    "lean-toolchain",
]


def capture_codebase_snapshot(
    repo_path: Path, relevant_files: set[Path] | None = None
) -> list[dict[str, str]]:
    """Capture content of all relevant Lean files and build config.

    Returns a list of {path, content} dicts for HuggingFace datasets compatibility.
    The list format ensures a homogeneous schema across all challenges.

    Args:
        repo_path: Path to repository root.
        relevant_files: Set of specific files to include. If None, captures all .lean files
                       excluding .lake/, .git/, build/, etc.

    Returns:
        List of {"path": relative_path, "content": file_content} dicts.
    """
    snapshot: dict[str, str] = {}

    # Always include build configuration files if they exist
    for config_file in BUILD_CONFIG_FILES:
        config_path = repo_path / config_file
        if config_path.is_file():
            try:
                content = config_path.read_text(encoding="utf-8")
                snapshot[config_file] = content
                logger.debug(f"Included build config: {config_file}")
            except Exception as e:
                logger.warning(f"Failed to read {config_path}: {e}")

    if relevant_files is None:
        # Capture all .lean files, excluding package cache and build dirs
        relevant_files = set()
        for file_path in repo_path.rglob("*.lean"):
            # Check if any parent is an excluded directory
            rel_path = file_path.relative_to(repo_path)
            if not any(part in EXCLUDED_DIRS for part in rel_path.parts):
                relevant_files.add(file_path)

    for file_path in relevant_files:
        if file_path.is_file():
            try:
                # Get relative path from repo root
                rel_path = file_path.relative_to(repo_path)
                content = file_path.read_text(encoding="utf-8")
                snapshot[str(rel_path)] = content
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

    logger.info(f"Captured snapshot of {len(snapshot)} files")

    # Convert to list format for HuggingFace datasets compatibility
    return [{"path": path, "content": content} for path, content in sorted(snapshot.items())]


def snapshot_to_dict(snapshot: list[dict[str, str]]) -> dict[str, str]:
    """Convert list-format snapshot back to dict for easy file access.

    Args:
        snapshot: List of {"path": ..., "content": ...} dicts.

    Returns:
        Dict mapping path -> content.
    """
    return {item["path"]: item["content"] for item in snapshot}


def identify_dependencies(repo_path: Path, proof_file: Path) -> set[Path]:
    """Identify transitive dependencies of a proof file by parsing imports.

    This is a simplified implementation that parses import statements.
    A complete implementation would need to:
    1. Parse all imports in the file
    2. Recursively find imports in those files
    3. Handle Mathlib and other external dependencies

    For MVP, we just return all .lean files. This can be optimized later.

    Args:
        repo_path: Path to repository root.
        proof_file: Path to the proof file.

    Returns:
        Set of file paths that are dependencies.
    """
    # MVP: Return all .lean files
    # TODO: Implement proper import-based dependency tracking
    return set(repo_path.rglob("*.lean"))


def extract_imports(file_path: Path) -> list[str]:
    """Extract import statements from a Lean file.

    Args:
        file_path: Path to Lean file.

    Returns:
        List of imported module names.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return []

    # Match "import ModuleName" statements
    imports = re.findall(r"^\s*import\s+([A-Za-z0-9._]+)", content, re.MULTILINE)
    return imports


def module_name_to_path(module_name: str, repo_path: Path) -> Path | None:
    """Convert a Lean module name to a file path.

    Args:
        module_name: Module name (e.g., "Curve25519Dalek.Defs.Edwards.Curve")
        repo_path: Repository root path.

    Returns:
        Path to the .lean file, or None if not found.
    """
    # Convert dots to slashes and add .lean extension
    file_path = repo_path / f"{module_name.replace('.', '/')}.lean"

    if file_path.exists():
        return file_path

    return None
