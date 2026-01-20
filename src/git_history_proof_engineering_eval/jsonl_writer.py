"""JSONL output generation for proof repair challenges."""

import json
import logging
from pathlib import Path

from git_history_proof_engineering_eval.structures import Challenge

logger = logging.getLogger(__name__)


def format_challenge_jsonl(challenge: Challenge) -> dict:
    """Convert Challenge to JSONL schema.

    Output schema for sorry-filling challenges:
    {
      "task_id": "commit_hash_decl_name",
      "metadata": {
        "original_commit": "hash",
        "author_fix_diff": "git_diff_string",
        "error_message": "sorry in {decl_name}",
        "proof_file": "...",
        "sorry_location": {
          "line": 42,
          "column": 8,
          "context": "sorry",
          "enclosing_decl": "theorem_name"
        }
      },
      "setup": {
        "instructions": "Fill in the sorry placeholder...",
        "codebase_state": {...}
      },
      "verification": {
        "command": "lake build [Module.Name]",
        "expected_output": "Success",
        "timeout_seconds": 300
      }
    }

    Args:
        challenge: Challenge object to convert.

    Returns:
        Dictionary in JSONL format.
    """
    # Generate instruction text
    decl_name = challenge.sorry_location.enclosing_decl or "anonymous declaration"
    instructions = (
        f"Fill in the sorry placeholder in {decl_name} "
        f"(file: {challenge.proof_file}, line {challenge.sorry_location.line}). "
        "Replace the sorry with a valid proof term."
    )

    return {
        "task_id": challenge.task_id,
        "metadata": {
            "original_commit": challenge.commit_hash,
            "author_fix_diff": challenge.author_fix_diff,
            "error_message": challenge.error_message,
            "proof_file": str(challenge.proof_file),
            "sorry_location": {
                "line": challenge.sorry_location.line,
                "column": challenge.sorry_location.column,
                "context": challenge.sorry_location.context,
                "enclosing_decl": challenge.sorry_location.enclosing_decl,
            },
        },
        "setup": {
            "instructions": instructions,
            "codebase_state": challenge.codebase_snapshot,
        },
        "verification": {
            "command": challenge.verification_command,
            "expected_output": "Success",
            "timeout_seconds": 300,
        },
    }


def append_challenge(challenge: Challenge, output_path: Path) -> None:
    """Append a single challenge to JSONL file.

    Args:
        challenge: Challenge to append.
        output_path: Path to output JSONL file.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_obj = format_challenge_jsonl(challenge)
    json_line = json.dumps(jsonl_obj, ensure_ascii=False)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json_line + "\n")

    logger.debug(f"Appended challenge {challenge.task_id} to {output_path}")


def write_challenges(challenges: list[Challenge], output_path: Path) -> None:
    """Write challenges to JSONL file.

    Each challenge is written as one line of JSON.

    Args:
        challenges: List of challenges to write.
        output_path: Path to output JSONL file.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(challenges)} challenges to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for challenge in challenges:
            jsonl_obj = format_challenge_jsonl(challenge)
            json_line = json.dumps(jsonl_obj, ensure_ascii=False)
            f.write(json_line + "\n")

    logger.info(f"Successfully wrote {len(challenges)} challenges")


def read_challenges(input_path: Path) -> list[dict]:
    """Read challenges from JSONL file.

    Args:
        input_path: Path to JSONL file.

    Returns:
        List of challenge dictionaries.
    """
    challenges = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                challenges.append(json.loads(line))

    return challenges
