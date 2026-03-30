from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Sequence

from .config import (
    ConfigError,
    DEFAULT_SUBJECT_TEMPLATE,
    PaperScoutConfig,
    describe_config,
    load_config,
)
from .delivery import DeliveryError, build_delivery_channels
from .digest import DigestRenderer
from .fetcher import ArxivFetcher
from .models import DigestContext, DigestEntry, ScoredPaper
from .scorer import AnthropicScorer, ScoringError
from .state import StateError, determine_since, load_last_run, save_last_run
from .summarizer import AnthropicSummarizer, SummarizationError

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "test-config":
        return _cmd_test_config(args)
    if args.command == "test-fetch":
        return _cmd_test_fetch(args)

    parser.print_help()
    return 1


def run_pipeline(config: PaperScoutConfig, dry_run: bool = False) -> int:
    started_at = datetime.now(timezone.utc)

    try:
        last_run = load_last_run(config.state_file)
    except StateError as exc:
        LOGGER.warning("Failed to load state file (%s). Falling back to lookback_hours.", exc)
        last_run = None

    since = determine_since(last_run, config.arxiv.lookback_hours, now=started_at)
    LOGGER.info("Fetching papers since %s", since.isoformat())

    fetcher = ArxivFetcher(config.arxiv)
    try:
        papers = fetcher.fetch_new_papers(since)
    except Exception as exc:
        LOGGER.error("Fetch stage failed: %s", exc)
        return 3

    paper_by_id = {paper.arxiv_id: paper for paper in papers}

    scored: list[ScoredPaper] = []
    if papers:
        if not config.anthropic_api_key:
            LOGGER.error(
                "No Anthropic API key configured. Set anthropic_api_key in YAML or ANTHROPIC_API_KEY."
            )
            return 4

        scorer = AnthropicScorer(config.profile, config.scoring, config.anthropic_api_key)
        try:
            scored = scorer.score_papers(papers)
        except ScoringError:
            LOGGER.exception("Scoring stage failed.")
            scored = []
    else:
        LOGGER.info("No new papers found for this run window.")

    score_by_id = {item.arxiv_id: item for item in scored if item.arxiv_id in paper_by_id}
    selected_scored = sorted(
        (
            item
            for item in score_by_id.values()
            if item.relevance_score >= config.scoring.threshold
        ),
        key=lambda item: item.relevance_score,
        reverse=True,
    )[: config.scoring.max_papers]

    summary_by_id: dict[str, str] = {}
    if selected_scored and config.anthropic_api_key:
        selected_papers = [paper_by_id[item.arxiv_id] for item in selected_scored]
        summarizer = AnthropicSummarizer(config.profile, config.scoring, config.anthropic_api_key)
        try:
            summaries = summarizer.summarize_papers(selected_papers, score_by_id)
            summary_by_id = {item.arxiv_id: item.summary for item in summaries}
        except SummarizationError:
            LOGGER.exception("Summary stage failed; falling back to scoring rationales.")
            summary_by_id = {}

    entries = [
        DigestEntry(
            paper=paper_by_id[item.arxiv_id],
            relevance_score=item.relevance_score,
            rationale=item.rationale,
            novelty_signal=item.novelty_signal,
            summary=summary_by_id.get(item.arxiv_id, item.rationale),
        )
        for item in selected_scored
    ]

    digest_context = DigestContext(
        generated_at=started_at,
        total_reviewed=len(papers),
        threshold=config.scoring.threshold,
        entries=entries,
    )

    renderer = DigestRenderer()
    rendered = renderer.render(
        digest_context,
        subject_template=_select_subject_template(config),
    )

    if dry_run:
        print(rendered.markdown)
        LOGGER.info("Dry run complete. Delivery and state update skipped.")
        return 0

    try:
        channels = build_delivery_channels(config.delivery_channels)
    except DeliveryError as exc:
        LOGGER.error("Invalid delivery configuration: %s", exc)
        return 5

    successful_deliveries = 0
    if not channels:
        LOGGER.warning("No enabled delivery channels configured.")
    else:
        for channel in channels:
            try:
                channel.deliver(
                    subject=rendered.subject,
                    markdown_body=rendered.markdown,
                    html_body=rendered.html,
                )
                successful_deliveries += 1
            except DeliveryError:
                LOGGER.exception(
                    "Delivery failed for channel '%s'.",
                    getattr(channel, "channel_type", "unknown"),
                )

    if channels and successful_deliveries == 0:
        LOGGER.error("All delivery channels failed. State file not updated.")
        return 6

    try:
        save_last_run(config.state_file, started_at)
    except StateError as exc:
        LOGGER.error("Digest delivered but failed to update state file: %s", exc)
        return 7

    LOGGER.info(
        "Run complete: reviewed=%d, selected=%d, delivered=%d channel(s).",
        len(papers),
        len(entries),
        successful_deliveries,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2

    return run_pipeline(config, dry_run=args.dry_run)


def _cmd_test_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2

    print(describe_config(config))
    return 0


def _cmd_test_fetch(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2

    now = datetime.now(timezone.utc)
    since = determine_since(last_run=None, lookback_hours=config.arxiv.lookback_hours, now=now)
    fetcher = ArxivFetcher(config.arxiv)

    try:
        papers = fetcher.fetch_new_papers(since)
    except Exception as exc:
        LOGGER.error("Fetch test failed: %s", exc)
        return 3

    print(f"Fetched {len(papers)} unique papers since {since.isoformat()}")
    for idx, paper in enumerate(papers[:10], start=1):
        categories = ", ".join(paper.categories)
        print(f"{idx:2d}. {paper.arxiv_id} [{categories}] {paper.title}")

    return 0


def _select_subject_template(config: PaperScoutConfig) -> str:
    for channel in config.delivery_channels:
        if channel.type.lower() == "email" and channel.subject_template:
            return channel.subject_template
    for channel in config.delivery_channels:
        if channel.subject_template:
            return channel.subject_template
    return DEFAULT_SUBJECT_TEMPLATE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scout",
        description="Scout: personalized arXiv digest generator",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: scout.yml or SCOUT_CONFIG).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v=INFO, -vv=DEBUG).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run full fetch → score → summarize → deliver")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but skip delivery and state updates; print markdown digest to stdout.",
    )

    subparsers.add_parser("test-config", help="Validate config and print summary")
    subparsers.add_parser("test-fetch", help="Fetch papers only (no LLM calls)")

    return parser


def _configure_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )