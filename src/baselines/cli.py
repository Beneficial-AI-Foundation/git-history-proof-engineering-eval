"""CLI for running proof repair evaluation."""

import asyncio
import os
import shutil
from pathlib import Path

import typer
from dotenv import load_dotenv

from baselines.evaluator import EvaluationConfig, evaluate, write_results

# Load .env file if present
load_dotenv()

app = typer.Typer(help="Evaluate proof repair models on dalek-lean-githistory-eval")


def check_prerequisites() -> list[str]:
    """Check for required prerequisites.

    Returns:
        List of missing prerequisite descriptions.
    """
    missing = []

    # Check for lake
    if not shutil.which("lake"):
        missing.append("'lake' not found in PATH (required for Lean builds)")

    # Check for API key based on model provider
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        missing.append(
            "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "(depending on your --model choice)"
        )

    return missing


@app.command()
def run(
    model: str = typer.Option(
        "anthropic:claude-sonnet-4-5",
        "--model",
        "-m",
        help="pydantic-ai model identifier",
    ),
    max_challenges: int = typer.Option(
        -1,
        "--max",
        "-n",
        help="Max challenges to evaluate (-1 = all)",
    ),
    timeout: int = typer.Option(
        900,
        "--timeout",
        "-t",
        help="Lake build timeout in seconds",
    ),
    output: Path = typer.Option(
        Path("./eval_results.json"),
        "--output",
        "-o",
        help="Output file for results",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    dataset: str = typer.Option(
        "beneficial-ai-foundation/dalek-lean-githistory-eval",
        "--dataset",
        "-d",
        help="HuggingFace dataset name",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        "-s",
        help="Dataset split to evaluate",
    ),
    lake_caches_dir: Path = typer.Option(
        Path("lake_caches"),
        "--lake-caches",
        "-c",
        help="Directory containing per-manifest .lake caches",
    ),
    artifacts_dir: Path = typer.Option(
        Path("eval_artifacts"),
        "--artifacts",
        "-a",
        help="Directory to save evaluation artifacts",
    ),
) -> None:
    """Run proof repair evaluation."""
    # Check prerequisites
    missing = check_prerequisites()
    if missing:
        typer.echo("Missing prerequisites:", err=True)
        for item in missing:
            typer.echo(f"  - {item}", err=True)
        typer.echo("\nPlease fix these issues before running evaluation.", err=True)
        raise typer.Exit(1)

    config = EvaluationConfig(
        model=model,
        max_challenges=max_challenges,
        timeout_seconds=timeout,
        dataset_name=dataset,
        dataset_split=split,
        verbose=verbose,
        lake_caches_dir=lake_caches_dir,
        artifacts_dir=artifacts_dir,
    )

    typer.echo(f"Running evaluation with model: {model}")
    typer.echo(f"Dataset: {dataset} ({split})")
    typer.echo(f"Max challenges: {max_challenges if max_challenges > 0 else 'all'}")
    typer.echo("")

    results = asyncio.run(evaluate(config))

    typer.echo("")
    typer.echo("=" * 50)
    typer.echo("EVALUATION RESULTS")
    typer.echo("=" * 50)
    typer.echo(f"Total:     {results.total}")
    typer.echo(f"Passed:    {results.passed}")
    typer.echo(f"Failed:    {results.failed}")
    typer.echo(f"Errors:    {results.errors}")
    typer.echo(f"Pass Rate: {results.pass_rate:.1%}")
    typer.echo("=" * 50)

    write_results(results, output)
    typer.echo(f"\nResults saved to {output}")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
