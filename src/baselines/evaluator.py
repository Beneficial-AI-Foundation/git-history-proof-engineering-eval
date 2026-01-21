"""Evaluator for dalek-lean-githistory-eval using pydantic-ai."""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from pydantic import BaseModel
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class SorryLocation(BaseModel):
    """Location of a sorry to fill."""

    line: int
    column: int
    context: str
    enclosing_decl: str | None


class ChallengeMetadata(BaseModel):
    """Metadata about the challenge."""

    original_commit: str
    author_fix_diff: str
    error_message: str
    proof_file: str
    sorry_location: SorryLocation
    manifest_hash: str = ""  # SHA256 prefix of lake-manifest.json for cache selection


class CodebaseFile(BaseModel):
    """A single file in the codebase snapshot."""

    path: str
    content: str


class ChallengeSetup(BaseModel):
    """Setup information for the challenge."""

    instructions: str
    # List format for HuggingFace datasets compatibility
    codebase_state: list[CodebaseFile]


class ChallengeVerification(BaseModel):
    """Verification config for the challenge."""

    command: str
    expected_output: str
    timeout_seconds: int


class Challenge(BaseModel):
    """A proof repair challenge."""

    task_id: str
    metadata: ChallengeMetadata
    setup: ChallengeSetup
    verification: ChallengeVerification


class ProofFix(BaseModel):
    """Generated proof fix from the model."""

    fixed_content: str
    explanation: str | None = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    model: str = "anthropic:claude-sonnet-4-5"
    max_challenges: int = -1  # -1 = all
    timeout_seconds: int = 900
    dataset_name: str = "beneficial-ai-foundation/dalek-lean-githistory-eval"
    dataset_split: str = "test"
    verbose: bool = False
    lake_caches_dir: Path = Path("lake_caches")  # Directory containing per-manifest caches
    artifacts_dir: Path = Path("eval_artifacts")  # Directory to persist attempts


@dataclass
class ChallengeResult:
    """Result of evaluating a single challenge."""

    task_id: str
    success: bool
    generated_fix: str | None
    build_stdout: str
    build_stderr: str
    error: str | None = None


@dataclass
class EvaluationResult:
    """Overall evaluation results."""

    total: int
    passed: int
    failed: int
    errors: int
    results: list[ChallengeResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


SYSTEM_PROMPT = """\
You are a Lean 4 proof repair assistant. Your task is to fill in `sorry` placeholders
with valid proof terms.

You will be given:
1. The file containing the sorry to fill
2. The surrounding codebase for context
3. The specific location of the sorry

Your response must contain the complete fixed file content. Replace the `sorry` with
a valid Lean 4 proof. Use tactics like `simp`, `ring`, `omega`, `rfl`, `exact`, `apply`,
`intro`, `constructor`, etc. as appropriate.

Important:
- Return ONLY the complete file content with the sorry replaced
- Do not add explanatory comments unless they help the proof
- Ensure the proof type-checks
- Use the codebase context to understand the types and available lemmas
"""


def create_proof_agent(model: str) -> Agent[None, ProofFix]:
    """Create the pydantic-ai agent for proof repair."""
    return Agent(
        model,
        output_type=ProofFix,
        system_prompt=SYSTEM_PROMPT,
    )


def setup_workspace(
    challenge: Challenge, workspace: Path, lake_caches_dir: Path | None = None
) -> Path:
    """Set up workspace with codebase state.

    Args:
        challenge: Challenge to set up.
        workspace: Workspace directory.
        lake_caches_dir: Directory containing per-manifest .lake caches.

    Returns:
        Path to the proof file within workspace.
    """
    for file_entry in challenge.setup.codebase_state:
        file_path = workspace / file_entry.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_entry.content, encoding="utf-8")

    # Copy pre-built .lake cache if available for this manifest
    manifest_hash = challenge.metadata.manifest_hash
    if lake_caches_dir and manifest_hash:
        lake_cache = lake_caches_dir / manifest_hash / ".lake"
        if lake_cache.exists():
            dest_lake = workspace / ".lake"
            shutil.copytree(lake_cache, dest_lake, symlinks=True)
            logger.debug(f"Copied .lake cache from {lake_cache}")
        else:
            logger.debug(f"No cache found for manifest {manifest_hash}")

    return workspace / challenge.metadata.proof_file


def save_artifacts(
    artifacts_dir: Path,
    task_id: str,
    generated_fix: str | None,
    stdout: str,
    stderr: str,
    success: bool,
    error: str | None = None,
) -> None:
    """Save evaluation artifacts for a challenge attempt.

    Args:
        artifacts_dir: Base directory for artifacts.
        task_id: Challenge task ID.
        generated_fix: The model's generated code.
        stdout: Build stdout.
        stderr: Build stderr.
        success: Whether the build succeeded.
        error: Any error message.
    """
    task_dir = artifacts_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save generated fix
    if generated_fix:
        (task_dir / "generated_fix.lean").write_text(generated_fix, encoding="utf-8")

    # Save build output
    (task_dir / "build_stdout.txt").write_text(stdout, encoding="utf-8")
    (task_dir / "build_stderr.txt").write_text(stderr, encoding="utf-8")

    # Save metadata
    metadata = {
        "task_id": task_id,
        "success": success,
        "error": error,
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.debug(f"Saved artifacts to {task_dir}")


def run_verification(workspace: Path, command: str, timeout: int) -> tuple[bool, str, str]:
    """Run lake build verification.

    Args:
        workspace: Workspace directory.
        command: Build command to run.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    try:
        result = subprocess.run(
            command.split(),
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Build timed out"
    except Exception as e:
        return False, "", str(e)


def build_prompt(challenge: Challenge) -> str:
    """Build the prompt for the agent.

    Args:
        challenge: Challenge to build prompt for.

    Returns:
        Prompt string.
    """
    proof_file = challenge.metadata.proof_file
    sorry_loc = challenge.metadata.sorry_location

    # Convert list to dict for easy lookup
    codebase = {f.path: f.content for f in challenge.setup.codebase_state}

    # Get the file content
    file_content = codebase.get(proof_file, "")

    # Build context from other files
    context_files = []
    for path, content in codebase.items():
        if path != proof_file:
            # Include a summary of other files
            lines = content.split("\n")
            if len(lines) > 50:
                preview = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more lines)"
            else:
                preview = content
            context_files.append(f"--- {path} ---\n{preview}")

    context_str = "\n\n".join(context_files[:10])  # Limit to 10 files

    prompt = f"""\
## Task
{challenge.setup.instructions}

## File to Fix: {proof_file}
```lean
{file_content}
```

## Sorry Location
- Line: {sorry_loc.line}
- Column: {sorry_loc.column}
- Context: `{sorry_loc.context}`
- Enclosing declaration: {sorry_loc.enclosing_decl or "anonymous"}

## Error Message
{challenge.metadata.error_message}

## Codebase Context (other files)
{context_str}

## Instructions
Replace the sorry at line {sorry_loc.line} with a valid proof term.
Return the complete fixed file content.
"""
    return prompt


async def evaluate_challenge(
    agent: Agent[None, ProofFix],
    challenge: Challenge,
    config: EvaluationConfig,
) -> ChallengeResult:
    """Evaluate a single challenge.

    Args:
        agent: The pydantic-ai agent.
        challenge: Challenge to evaluate.
        config: Evaluation configuration.

    Returns:
        Result of the evaluation.
    """
    logger.info(f"Evaluating {challenge.task_id}")

    try:
        # Build prompt and get fix from agent
        prompt = build_prompt(challenge)
        result = await agent.run(prompt)
        fix = result.output

        # Set up workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            proof_file = setup_workspace(challenge, workspace, config.lake_caches_dir)

            # Apply fix
            proof_file.write_text(fix.fixed_content, encoding="utf-8")

            # Run verification
            success, stdout, stderr = run_verification(
                workspace,
                challenge.verification.command,
                config.timeout_seconds,
            )

            # Save artifacts
            save_artifacts(
                config.artifacts_dir,
                challenge.task_id,
                fix.fixed_content,
                stdout,
                stderr,
                success,
            )

            return ChallengeResult(
                task_id=challenge.task_id,
                success=success,
                generated_fix=fix.fixed_content,
                build_stdout=stdout,
                build_stderr=stderr,
            )

    except Exception as e:
        logger.error(f"Error evaluating {challenge.task_id}: {e}")
        # Save artifacts even on error
        save_artifacts(
            config.artifacts_dir,
            challenge.task_id,
            None,
            "",
            "",
            False,
            str(e),
        )
        return ChallengeResult(
            task_id=challenge.task_id,
            success=False,
            generated_fix=None,
            build_stdout="",
            build_stderr="",
            error=str(e),
        )


def load_challenges(config: EvaluationConfig) -> list[Challenge]:
    """Load challenges from HuggingFace dataset.

    Args:
        config: Evaluation configuration.

    Returns:
        List of challenges.
    """
    logger.info(f"Loading dataset: {config.dataset_name}")

    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    logger.info(f"Loaded {len(dataset)} challenges from HuggingFace")

    challenges = []
    for row in dataset:
        # Parse codebase_state from list format
        codebase_state = [
            CodebaseFile(path=f["path"], content=f["content"])
            for f in row["setup"]["codebase_state"]
        ]

        # Parse the row into Challenge object
        challenge = Challenge(
            task_id=row["task_id"],
            metadata=ChallengeMetadata(
                original_commit=row["metadata"]["original_commit"],
                author_fix_diff=row["metadata"]["author_fix_diff"],
                error_message=row["metadata"]["error_message"],
                proof_file=row["metadata"]["proof_file"],
                sorry_location=SorryLocation(
                    line=row["metadata"]["sorry_location"]["line"],
                    column=row["metadata"]["sorry_location"]["column"],
                    context=row["metadata"]["sorry_location"]["context"],
                    enclosing_decl=row["metadata"]["sorry_location"]["enclosing_decl"],
                ),
                manifest_hash=row["metadata"].get("manifest_hash", ""),
            ),
            setup=ChallengeSetup(
                instructions=row["setup"]["instructions"],
                codebase_state=codebase_state,
            ),
            verification=ChallengeVerification(
                command=row["verification"]["command"],
                expected_output=row["verification"]["expected_output"],
                timeout_seconds=row["verification"]["timeout_seconds"],
            ),
        )
        challenges.append(challenge)

        if config.max_challenges > 0 and len(challenges) >= config.max_challenges:
            break

    logger.info(f"Returning {len(challenges)} challenges")
    return challenges


async def evaluate(config: EvaluationConfig | None = None) -> EvaluationResult:
    """Run evaluation on the dataset.

    Args:
        config: Evaluation configuration. Uses defaults if None.

    Returns:
        Evaluation results.
    """
    if config is None:
        config = EvaluationConfig()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load challenges
    challenges = load_challenges(config)

    # Create agent
    agent = create_proof_agent(config.model)

    # Evaluate each challenge
    results = []
    passed = 0
    failed = 0
    errors = 0

    for i, challenge in enumerate(challenges, 1):
        logger.info(f"[{i}/{len(challenges)}] {challenge.task_id}")

        result = await evaluate_challenge(agent, challenge, config)
        results.append(result)

        if result.error:
            errors += 1
            logger.warning(f"  ERROR: {result.error}")
        elif result.success:
            passed += 1
            logger.info(f"  PASSED")
        else:
            failed += 1
            logger.info(f"  FAILED: {result.build_stderr[:100]}")

    return EvaluationResult(
        total=len(challenges),
        passed=passed,
        failed=failed,
        errors=errors,
        results=results,
    )


def write_results(results: EvaluationResult, output_path: Path) -> None:
    """Write evaluation results to JSON file.

    Args:
        results: Evaluation results.
        output_path: Path to output file.
    """
    output = {
        "summary": {
            "total": results.total,
            "passed": results.passed,
            "failed": results.failed,
            "errors": results.errors,
            "pass_rate": results.pass_rate,
        },
        "results": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "error": r.error,
                "build_stderr": r.build_stderr[:500] if r.build_stderr else None,
            }
            for r in results.results
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results written to {output_path}")
