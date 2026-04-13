from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from .config import (
    ConfigError,
    DEFAULT_SUBJECT_TEMPLATE,
    PaperScoutConfig,
    WatchlistConfig,
    describe_config,
    load_config,
)
from .deep_reader import DeepReadAgent, DeepReadError
from .delivery import DeliveryError, build_delivery_channels
from .digest import DigestRenderer
from .fetcher import ArxivFetcher
from .knowledge_base import KnowledgeBase, KnowledgeBaseError
from .models import (
    DigestContext,
    DigestEntry,
    DeepReadBreakdown,
    DeepReadEntry,
    DeepReadResult,
    KBPaperRecord,
    Paper,
    ScoredPaper,
    WatchlistMatch,
)
from .scorer import AnthropicScorer, ScoringError
from .semantic_scholar import SemanticScholarClient
from .costs import CostTracker
from .state import (
    StateError,
    determine_since,
    load_cumulative_cost,
    load_last_run,
    save_last_run,
)
from .kb_writer import KBNoteWriter
from .watchlist import WatchlistMatcher

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class HotAlert:
    paper: Paper
    scored: ScoredPaper
    reason: str
    deep_read: DeepReadEntry | None = None


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


def run_pipeline(
    config: PaperScoutConfig,
    dry_run: bool = False,
    weekly: bool = False,
) -> int:
    started_at = datetime.now(timezone.utc)
    cost_tracker = CostTracker()
    prior_total_cost_usd = 0.0

    try:
        last_run = load_last_run(config.state_file)
        prior_total_cost_usd = load_cumulative_cost(config.state_file)
    except StateError as exc:
        LOGGER.warning(
            "Failed to load state file (%s). Falling back to lookback_hours and zero cumulative cost.",
            exc,
        )
        last_run = None
        prior_total_cost_usd = 0.0

    since = determine_since(last_run, config.arxiv.lookback_hours, now=started_at)
    LOGGER.info("Fetching papers since %s", since.isoformat())

    fetcher = ArxivFetcher(config.arxiv)
    try:
        papers = fetcher.fetch_new_papers(since)
    except Exception as exc:
        LOGGER.error("Fetch stage failed: %s", exc)
        return 3

    paper_by_id = {paper.arxiv_id: paper for paper in papers}

    if config.web_sources.enabled:
        from .web_fetcher import WebFetcher

        web_fetcher = WebFetcher(config.web_sources, logger=LOGGER)
        try:
            web_papers = web_fetcher.fetch_new_posts(since)
            LOGGER.info("Web sources: fetched %d posts.", len(web_papers))
            seen_titles: set[str] = {_normalize_merge_title(p.title) for p in papers}
            for web_paper in web_papers:
                if web_paper.arxiv_id in paper_by_id:
                    continue
                norm_title = _normalize_merge_title(web_paper.title)
                if norm_title and norm_title in seen_titles:
                    LOGGER.debug(
                        "Skipping web paper %s (title duplicate of existing paper: %s).",
                        web_paper.arxiv_id,
                        web_paper.title,
                    )
                    continue
                papers.append(web_paper)
                paper_by_id[web_paper.arxiv_id] = web_paper
                if norm_title:
                    seen_titles.add(norm_title)
            papers.sort(key=lambda item: item.published, reverse=True)
        except Exception as exc:
            LOGGER.error("Web source fetch failed: %s", exc)

    scored: list[ScoredPaper] = []
    if papers:
        if not config.anthropic_api_key:
            LOGGER.error(
                "No Anthropic API key configured. Set anthropic_api_key in YAML or ANTHROPIC_API_KEY."
            )
            return 4

        scorer = AnthropicScorer(
            config.profile,
            config.scoring,
            config.anthropic_api_key,
            cost_tracker=cost_tracker,
        )
        try:
            scored = scorer.score_papers(papers)
        except ScoringError:
            LOGGER.exception("Scoring stage failed.")
            scored = []
    else:
        LOGGER.info("No new papers found for this run window.")

    score_by_id = {item.arxiv_id: item for item in scored if item.arxiv_id in paper_by_id}

    # Load KB from the single papers directory (markdown files only, no JSON).
    knowledge_base = KnowledgeBase(
        config.knowledge_base.papers_path,
        max_topic_references=config.knowledge_base.max_topic_references,
    )
    try:
        knowledge_base.load()
    except KnowledgeBaseError as exc:
        LOGGER.warning("Could not load knowledge base (%s). Proceeding with empty KB.", exc)

    s2_client = SemanticScholarClient(
        request_timeout_seconds=config.arxiv.request_timeout_seconds,
        pause_seconds=max(0.0, config.arxiv.query_pause_seconds / 2.0),
    )
    watchlist_matcher = WatchlistMatcher(config.watchlist)
    watchlist_matches = _match_watchlist_papers(papers, watchlist_matcher, s2_client)

    selected_scored = _select_scored_for_digest(
        papers=papers,
        score_by_id=score_by_id,
        watchlist_matches=watchlist_matches,
        watchlist_config=config.watchlist,
        threshold=config.scoring.threshold,
        max_papers=config.scoring.max_papers,
        knowledge_base=knowledge_base,
    )

    selected_papers_for_s2 = [
        paper_by_id[s.arxiv_id] for s in selected_scored
        if s.arxiv_id in paper_by_id and s.arxiv_id not in watchlist_matches
    ]
    if selected_papers_for_s2:
        _enrich_watchlist_with_affiliations(
            selected_papers_for_s2, watchlist_matches, watchlist_matcher, s2_client,
        )

    deep_entries: list[DeepReadEntry] = []
    deep_results_by_id: dict[str, DeepReadResult] = {}
    noteworthy_scored: list[ScoredPaper] = selected_scored

    if weekly and selected_scored:
        deep_target_count = max(0, config.analysis.deep_read_count)
        deep_targets = selected_scored[:deep_target_count]
        noteworthy_scored = selected_scored[deep_target_count:]

        if deep_targets:
            if not config.anthropic_api_key:
                LOGGER.error("Deep read stage requires Anthropic API key.")
                return 4

            LOGGER.info(
                "Starting deep reads for %d papers (this may take several minutes).",
                len(deep_targets),
            )

            deep_reader = DeepReadAgent(
                api_key=config.anthropic_api_key,
                profile=config.profile,
                analysis=config.analysis,
                knowledge_base=knowledge_base,
                s2_client=s2_client,
                cost_tracker=cost_tracker,
            )

            for deep_index, scored_item in enumerate(deep_targets, start=1):
                paper = paper_by_id.get(scored_item.arxiv_id)
                if paper is None:
                    continue

                if cost_tracker.total_cost_usd >= config.analysis.max_run_cost_usd:
                    LOGGER.warning(
                        "Cost budget exceeded ($%.2f >= $%.2f). Skipping remaining %d deep read(s).",
                        cost_tracker.total_cost_usd,
                        config.analysis.max_run_cost_usd,
                        len(deep_targets) - deep_index + 1,
                    )
                    for remaining in deep_targets[deep_index - 1 :]:
                        if remaining.arxiv_id not in deep_results_by_id:
                            noteworthy_scored.append(remaining)
                    break

                LOGGER.info(
                    "Deep read %d/%d: %s (score=%.1f)",
                    deep_index,
                    len(deep_targets),
                    paper.title[:80],
                    scored_item.relevance_score,
                )

                watch_match = watchlist_matches.get(paper.arxiv_id)
                try:
                    result = deep_reader.analyze_paper(paper, scored_item)
                    entry = result.entry
                    LOGGER.info(
                        "Deep read %d/%d complete: %s",
                        deep_index,
                        len(deep_targets),
                        paper.arxiv_id,
                    )
                except DeepReadError:
                    LOGGER.exception("Deep read failed for %s", paper.arxiv_id)
                    entry = _fallback_deep_entry(
                        paper=paper,
                        scored=scored_item,
                        watchlist_match=watch_match.matched_name if watch_match else None,
                    )
                    result = _deep_entry_to_result(entry)

                deep_results_by_id[paper.arxiv_id] = result

                if watch_match and not entry.watchlist_match:
                    entry.watchlist_match = watch_match.matched_name

                deep_entries.append(entry)

                # Update in-memory KB so subsequent deep reads can see this paper.
                kb_record = _deep_result_to_kb_record(
                    result=result,
                    fallback_entry=entry,
                    timestamp=started_at,
                )
                knowledge_base.add_paper(kb_record)

    noteworthy_entries: list[DigestEntry] = []
    for scored_item in noteworthy_scored:
        paper = paper_by_id.get(scored_item.arxiv_id)
        if paper is None:
            continue
        watch_match = watchlist_matches.get(scored_item.arxiv_id)
        noteworthy_entries.append(
            DigestEntry(
                paper=paper,
                relevance_score=scored_item.relevance_score,
                rationale=scored_item.rationale,
                novelty_signal=scored_item.novelty_signal,
                summary=_compact_text(scored_item.rationale, max_chars=240),
                watchlist_match=watch_match.matched_name if watch_match else None,
            )
        )

    digest_context = DigestContext(
        generated_at=started_at,
        total_reviewed=len(papers),
        threshold=config.scoring.threshold,
        run_cost_usd=0.0,
        total_cost_usd=prior_total_cost_usd,
        entries=noteworthy_entries,
        deep_reads=deep_entries,
        noteworthy_entries=noteworthy_entries,
    )

    renderer = DigestRenderer()

    hot_alerts = _collect_hot_alerts(
        papers=papers,
        score_by_id=score_by_id,
        watchlist_matches=watchlist_matches,
        watchlist_config=config.watchlist,
        config=config,
        knowledge_base=knowledge_base,
    )

    # Deep-read hot alert papers immediately so the alert email includes the full breakdown.
    if hot_alerts and config.alerts.enabled and config.anthropic_api_key:
        alert_reader = DeepReadAgent(
            api_key=config.anthropic_api_key,
            profile=config.profile,
            analysis=config.analysis,
            knowledge_base=knowledge_base,
            s2_client=s2_client,
            cost_tracker=cost_tracker,
        )
        for alert in hot_alerts:
            already_read = any(
                entry.paper.arxiv_id == alert.paper.arxiv_id for entry in deep_entries
            )
            if already_read:
                matching_entry = next(
                    entry for entry in deep_entries
                    if entry.paper.arxiv_id == alert.paper.arxiv_id
                )
                alert.deep_read = matching_entry
                if alert.paper.arxiv_id not in deep_results_by_id:
                    deep_results_by_id[alert.paper.arxiv_id] = _deep_entry_to_result(matching_entry)
                continue

            watch_match = watchlist_matches.get(alert.paper.arxiv_id)

            if cost_tracker.total_cost_usd >= config.analysis.max_run_cost_usd:
                LOGGER.warning(
                    "Cost budget exceeded ($%.2f >= $%.2f). Skipping hot alert deep read for %s.",
                    cost_tracker.total_cost_usd,
                    config.analysis.max_run_cost_usd,
                    alert.paper.arxiv_id,
                )
                continue

            LOGGER.info(
                "Hot alert deep read: %s (score=%.1f, reason=%s)",
                alert.paper.title[:80],
                alert.scored.relevance_score,
                alert.reason,
            )
            try:
                result = alert_reader.analyze_paper(alert.paper, alert.scored)
                alert.deep_read = result.entry
                deep_results_by_id[alert.paper.arxiv_id] = result
                kb_record = _deep_result_to_kb_record(
                    result=result,
                    fallback_entry=result.entry,
                    timestamp=started_at,
                )
                knowledge_base.add_paper(kb_record)
                LOGGER.info("Hot alert deep read complete: %s", alert.paper.arxiv_id)
            except DeepReadError:
                LOGGER.exception("Hot alert deep read failed for %s", alert.paper.arxiv_id)

                fallback_entry = _fallback_deep_entry(
                    paper=alert.paper,
                    scored=alert.scored,
                    watchlist_match=watch_match.matched_name if watch_match else None,
                )
                fallback_result = _deep_entry_to_result(fallback_entry)

                alert.deep_read = fallback_entry
                deep_results_by_id[alert.paper.arxiv_id] = fallback_result

                kb_record = _deep_result_to_kb_record(
                    result=fallback_result,
                    fallback_entry=fallback_entry,
                    timestamp=started_at,
                )
                knowledge_base.add_paper(kb_record)

    # Compute costs before rendering so the digest footer has accurate numbers.
    run_cost_usd = cost_tracker.total_cost_usd
    total_cost_usd = prior_total_cost_usd + run_cost_usd
    digest_context.run_cost_usd = run_cost_usd
    digest_context.total_cost_usd = total_cost_usd

    rendered = renderer.render(
        digest_context,
        subject_template=_select_subject_template(config),
    )

    LOGGER.info(
        "LLM usage summary: %s | this digest cost: $%.4f | scout total to date: $%.4f",
        cost_tracker.summary(),
        run_cost_usd,
        total_cost_usd,
    )

    if dry_run:
        print(rendered.markdown)
        if hot_alerts and config.alerts.enabled:
            LOGGER.info("Dry run hot alerts:")
            for alert in hot_alerts:
                LOGGER.info(
                    "ALERT: %s | score=%.1f | reason=%s",
                    alert.paper.title,
                    alert.scored.relevance_score,
                    alert.reason,
                )
        LOGGER.info("Dry run complete. Delivery, KB note writes, and state update skipped.")
        return 0

    # --- Beyond this point: real run only (not dry-run) ---

    # Write KB paper notes to the papers directory (same dir the KB loaded from).
    kb_stub_notes = 0
    kb_full_notes = 0
    kb_note_writer: KBNoteWriter | None = None
    if config.knowledge_base.papers_path:
        try:
            kb_note_writer = KBNoteWriter(config.knowledge_base.papers_path)
        except OSError as exc:
            LOGGER.warning(
                "Could not initialize KB note writer at %s (%s). Proceeding without KB note writes.",
                config.knowledge_base.papers_path,
                exc,
            )

    if kb_note_writer is not None:
        try:
            kb_stub_notes, kb_full_notes = _generate_kb_notes(
                kb_note_writer=kb_note_writer,
                generated_at=started_at,
                paper_by_id=paper_by_id,
                scored=scored,
                digest_selected_ids={s.arxiv_id for s in selected_scored},
                deep_results_by_id=deep_results_by_id,
                watchlist_matches=watchlist_matches,
                knowledge_base=knowledge_base,
            )
        except Exception:
            LOGGER.exception("Failed to generate KB paper notes.")
        else:
            LOGGER.info(
                "Generated %d KB note(s): %d full, %d stub.",
                kb_stub_notes + kb_full_notes,
                kb_full_notes,
                kb_stub_notes,
            )

    try:
        channels = build_delivery_channels(config.delivery_channels)
    except DeliveryError as exc:
        LOGGER.error("Invalid delivery configuration: %s", exc)
        return 5

    if config.alerts.enabled and hot_alerts:
        _deliver_hot_alerts(hot_alerts, channels)

    successful_deliveries = 0
    channels_attempted = 0
    if not channels:
        LOGGER.warning("No enabled delivery channels configured.")
    else:
        selected_count = len(deep_entries) + len(noteworthy_entries)
        for channel in channels:
            channel_type = getattr(channel, "channel_type", "unknown")
            # Skip non-file delivery channels when the digest is empty.
            if selected_count == 0 and channel_type != "markdown":
                LOGGER.info(
                    "Empty digest — skipping delivery for channel '%s'.",
                    channel_type,
                )
                continue
            channels_attempted += 1
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

    if channels_attempted > 0 and successful_deliveries == 0:
        LOGGER.error("All delivery channels failed. State file not updated.")
        return 6

    try:
        save_last_run(
            config.state_file,
            started_at,
            cumulative_cost_usd=total_cost_usd,
        )
    except StateError as exc:
        LOGGER.error("Digest delivered but failed to update state file: %s", exc)
        return 7

    LOGGER.info(
        "Run complete: reviewed=%d, selected=%d, deep_reads=%d, kb_notes=%d (full=%d, stub=%d), delivered=%d channel(s), digest_cost=$%.4f, total_cost=$%.4f.",
        len(papers),
        len(deep_entries) + len(noteworthy_entries),
        len(deep_results_by_id),
        kb_stub_notes + kb_full_notes,
        kb_full_notes,
        kb_stub_notes,
        successful_deliveries,
        run_cost_usd,
        total_cost_usd,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2

    return run_pipeline(config, dry_run=args.dry_run, weekly=args.weekly)


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


def _match_watchlist_papers(
    papers: Sequence[Paper],
    matcher: WatchlistMatcher,
    s2_client: SemanticScholarClient,
) -> dict[str, WatchlistMatch]:
    matches: dict[str, WatchlistMatch] = {}

    for paper in papers:
        match = matcher.match_paper(paper)
        if match:
            matches[paper.arxiv_id] = match

    if matches:
        LOGGER.info(
            "Watchlist: %d author-name matches found from %d papers.",
            len(matches),
            len(papers),
        )

    return matches


def _enrich_watchlist_with_affiliations(
    papers: Sequence[Paper],
    existing_matches: dict[str, WatchlistMatch],
    matcher: WatchlistMatcher,
    s2_client: SemanticScholarClient,
) -> None:
    """Query S2 for author affiliations on a small set of papers (not all)."""
    if not matcher.config.organizations:
        return

    candidates = [p for p in papers if p.arxiv_id not in existing_matches]
    if not candidates:
        return

    LOGGER.info(
        "Watchlist: checking S2 affiliations for %d candidate papers.",
        len(candidates),
    )

    for index, paper in enumerate(candidates):
        try:
            affiliations = s2_client.get_author_affiliations(paper.arxiv_id)
        except Exception as exc:
            LOGGER.debug(
                "Failed to fetch affiliations for watchlist matching (%s): %s",
                paper.arxiv_id,
                exc,
            )
            continue

        match = matcher.match_paper(paper, affiliations=affiliations)
        if match:
            existing_matches[paper.arxiv_id] = match
            LOGGER.info(
                "Watchlist: organization match found for %s (%s).",
                paper.arxiv_id,
                match.matched_name,
            )


def _select_scored_for_digest(
    *,
    papers: Sequence[Paper],
    score_by_id: dict[str, ScoredPaper],
    watchlist_matches: dict[str, WatchlistMatch],
    watchlist_config: WatchlistConfig,
    threshold: float,
    max_papers: int,
    knowledge_base: KnowledgeBase,
) -> list[ScoredPaper]:
    watchlist_items: list[ScoredPaper] = []
    scored_items: list[ScoredPaper] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()

    for paper in papers:
        arxiv_id = paper.arxiv_id
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        norm_title = _normalize_merge_title(paper.title)
        if norm_title and norm_title in seen_titles:
            LOGGER.debug("Skipping %s (title duplicate: %s).", arxiv_id, paper.title)
            continue
        if norm_title:
            seen_titles.add(norm_title)

        if knowledge_base.known_paper(arxiv_id=arxiv_id, url=paper.url, title=paper.title):
            LOGGER.debug("Skipping %s (already present in knowledge base).", arxiv_id)
            continue

        scored_item = score_by_id.get(arxiv_id)
        watch_match = watchlist_matches.get(arxiv_id)

        include_by_watchlist = (
            watch_match is not None and _watchlist_always_include(watch_match, watchlist_config)
        )
        include_by_score = scored_item is not None and scored_item.relevance_score >= threshold

        if not include_by_watchlist and not include_by_score:
            continue

        if scored_item is None:
            scored_item = ScoredPaper(
                arxiv_id=arxiv_id,
                relevance_score=0.0,
                rationale=(
                    f"Included due to watchlist match: {watch_match.matched_name}."
                    if watch_match
                    else "Included due to watchlist match."
                ),
                novelty_signal="incremental",
            )

        if include_by_watchlist:
            watchlist_items.append(scored_item)
        else:
            scored_items.append(scored_item)

    watchlist_items.sort(key=lambda item: item.relevance_score, reverse=True)
    scored_items.sort(key=lambda item: item.relevance_score, reverse=True)

    ordered: list[ScoredPaper] = []
    watch_ids: set[str] = set()

    for item in watchlist_items:
        if item.arxiv_id in watch_ids:
            continue
        watch_ids.add(item.arxiv_id)
        ordered.append(item)

    for item in scored_items:
        if item.arxiv_id in watch_ids:
            continue
        ordered.append(item)

    if len(watchlist_items) >= max_papers:
        return ordered
    return ordered[:max_papers]


def _watchlist_always_include(match: WatchlistMatch, config: WatchlistConfig) -> bool:
    if match.match_type == "author":
        return True
    if match.match_type != "organization":
        return False

    normalized = match.matched_name.strip().lower()
    for org in config.organizations:
        if org.name.strip().lower() == normalized:
            return org.always_include
    return True


def _fallback_deep_entry(
    *,
    paper: Paper,
    scored: ScoredPaper,
    watchlist_match: str | None,
) -> DeepReadEntry:
    triage = (
        "Read the full paper — deep reader fallback could not extract full structure."
        if scored.relevance_score >= 8.0
        else "TL;DR captures it — deep reader fallback could not extract full structure."
    )

    tldr = [
        _compact_text(scored.rationale, max_chars=180) or "Relevance rationale unavailable.",
        "Automated deep-read extraction failed for this paper in this run.",
        "Use the paper link for manual follow-up.",
    ]

    breakdown = DeepReadBreakdown(
        triage=triage,
        tldr=tldr,
        motivation=_compact_text(scored.rationale, max_chars=240),
        hypothesis="Unable to extract automatically in this run.",
        methodology="Unable to extract automatically in this run.",
        results="Unable to extract automatically in this run.",
        interpretation="Unable to extract automatically in this run.",
        context="No additional contextual investigation completed.",
        limitations="Structured deep-read generation failed in this run.",
        relevance=_compact_text(scored.rationale, max_chars=220),
    )

    return DeepReadEntry(
        paper=paper,
        relevance_score=scored.relevance_score,
        rationale=scored.rationale,
        novelty_signal=scored.novelty_signal,
        breakdown=breakdown,
        watchlist_match=watchlist_match,
    )


def _deep_result_to_kb_record(
    *,
    result: DeepReadResult | None,
    fallback_entry: DeepReadEntry,
    timestamp: datetime,
) -> KBPaperRecord:
    if result is not None:
        entry = result.entry
        topics = _dedupe(_sanitize_list(result.topics)) or _fallback_topics(entry.paper)
        key_findings = _dedupe(_sanitize_list(result.key_findings)) or entry.breakdown.tldr[:3]
        builds_on = _dedupe(_sanitize_list(result.builds_on))
    else:
        entry = fallback_entry
        topics = _fallback_topics(entry.paper)
        key_findings = entry.breakdown.tldr[:3]
        builds_on = []

    return KBPaperRecord(
        arxiv_id=entry.paper.arxiv_id,
        title=entry.paper.title,
        authors=list(entry.paper.authors),
        date_read=timestamp.date().isoformat(),
        score=entry.relevance_score,
        topics=topics,
        key_findings=key_findings,
        builds_on=builds_on,
        tldr=" ".join(entry.breakdown.tldr[:3]).strip(),
    )


def _scored_to_kb_record(
    *,
    paper: Paper,
    scored: ScoredPaper,
    timestamp: datetime,
) -> KBPaperRecord:
    summary = _compact_text(scored.rationale, max_chars=240) or paper.title
    finding = _compact_text(scored.rationale, max_chars=180)
    key_findings = [finding] if finding else []

    return KBPaperRecord(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=list(paper.authors),
        date_read=timestamp.date().isoformat(),
        score=scored.relevance_score,
        topics=_stub_note_tags(paper, scored) or _fallback_topics(paper),
        key_findings=key_findings,
        builds_on=[],
        tldr=summary,
    )


def _fallback_topics(paper: Paper) -> list[str]:
    topics = [topic for topic in paper.categories if topic.strip()]
    if topics:
        return topics[:3]
    return ["uncategorized"]


def _collect_hot_alerts(
    *,
    papers: Sequence[Paper],
    score_by_id: dict[str, ScoredPaper],
    watchlist_matches: dict[str, WatchlistMatch],
    watchlist_config: WatchlistConfig,
    config: PaperScoutConfig,
    knowledge_base: KnowledgeBase,
) -> list[HotAlert]:
    if not config.alerts.enabled:
        return []

    alerts_by_id: dict[str, HotAlert] = {}

    for paper in papers:
        if knowledge_base.known_paper(arxiv_id=paper.arxiv_id, url=paper.url, title=paper.title):
            continue

        scored = score_by_id.get(paper.arxiv_id)
        watch_match = watchlist_matches.get(paper.arxiv_id)
        score = scored.relevance_score if scored else 0.0

        reason: str | None = None
        if watch_match and watch_match.match_type == "author":
            reason = f"Watchlist author match: {watch_match.matched_name}"
        elif score >= config.alerts.score_threshold:
            reason = f"High relevance score: {score:.1f}/10"
        elif (
            watch_match
            and watch_match.match_type == "organization"
            and _watchlist_always_include(watch_match, watchlist_config)
            and score >= config.alerts.watchlist_score_threshold
        ):
            reason = (
                f"Watchlist organization match: {watch_match.matched_name} "
                f"with score {score:.1f}/10"
            )

        if not reason:
            continue

        scored_payload = scored or ScoredPaper(
            arxiv_id=paper.arxiv_id,
            relevance_score=score,
            rationale=reason,
            novelty_signal="incremental",
        )
        alerts_by_id[paper.arxiv_id] = HotAlert(
            paper=paper,
            scored=scored_payload,
            reason=reason,
        )

    alerts = list(alerts_by_id.values())
    alerts.sort(key=lambda alert: alert.scored.relevance_score, reverse=True)
    return alerts


def _deliver_hot_alerts(hot_alerts: Sequence[HotAlert], channels: Sequence[object]) -> None:
    non_file_channels = [
        channel
        for channel in channels
        if getattr(channel, "channel_type", "").lower() != "markdown"
    ]

    if not non_file_channels:
        for alert in hot_alerts:
            LOGGER.info(
                "HOT ALERT: %s | %.1f/10 | %s",
                alert.paper.title,
                alert.scored.relevance_score,
                alert.reason,
            )
        return

    for alert in hot_alerts:
        subject = f"🔔 Scout Alert: {alert.paper.title}"
        markdown_body = _render_alert_markdown(alert)
        html_body = _render_alert_html(alert)

        for channel in non_file_channels:
            try:
                channel.deliver(
                    subject=subject,
                    markdown_body=markdown_body,
                    html_body=html_body,
                )
            except DeliveryError:
                LOGGER.exception(
                    "Hot alert delivery failed for channel '%s'.",
                    getattr(channel, "channel_type", "unknown"),
                )


def _render_alert_markdown(alert: HotAlert) -> str:
    lines = [
        f"🔔 **Scout Alert**\n",
        f"**Reason:** {alert.reason}\n",
        f"**[{alert.paper.title}]({alert.paper.url})**\n",
        f"Score: {alert.scored.relevance_score:.1f}/10 · Novelty: {alert.scored.novelty_signal}",
        f"Authors: {', '.join(alert.paper.authors)}",
        f"[📄 PDF]({alert.paper.pdf_url}) · [🔗 Abstract]({alert.paper.url})\n",
    ]

    if alert.deep_read is not None:
        bd = alert.deep_read.breakdown
        lines.append(f"---\n")
        lines.append(f"**Triage:** {bd.triage}\n")
        lines.append("**TL;DR**")
        for bullet in bd.tldr:
            lines.append(f"- {bullet}")
        lines.append("")
        lines.append(f"**Motivation:** {bd.motivation}\n")
        lines.append(f"**Hypothesis:** {bd.hypothesis}\n")
        lines.append(f"**Methodology:** {bd.methodology}\n")
        lines.append(f"**Results:** {bd.results}\n")
        lines.append(f"**Interpretation:** {bd.interpretation}\n")
        lines.append(f"**Context:** {bd.context}\n")
        lines.append(f"**Limitations:** {bd.limitations}\n")
        lines.append(f"**Why it matters to you:** {bd.relevance}")
    else:
        lines.append("Deep read could not be completed for this paper.")

    return "\n".join(lines)


def _render_alert_html(alert: HotAlert) -> str:
    parts = [
        "<html><body>",
        "<p>🔔 <strong>Scout Alert</strong></p>",
        f"<p><strong>Reason:</strong> {alert.reason}</p>",
        f'<p><a href="{alert.paper.url}"><strong>{alert.paper.title}</strong></a></p>',
        f"<p>Score: {alert.scored.relevance_score:.1f}/10 · Novelty: {alert.scored.novelty_signal}</p>",
        f"<p>Authors: {', '.join(alert.paper.authors)}</p>",
        f'<p><a href="{alert.paper.pdf_url}">📄 PDF</a> · <a href="{alert.paper.url}">🔗 Abstract</a></p>',
    ]

    if alert.deep_read is not None:
        bd = alert.deep_read.breakdown
        parts.append("<hr>")
        parts.append(f"<p><strong>Triage:</strong> {bd.triage}</p>")
        parts.append("<p><strong>TL;DR</strong></p><ul>")
        for bullet in bd.tldr:
            parts.append(f"<li>{bullet}</li>")
        parts.append("</ul>")
        parts.append(f"<p><strong>Motivation:</strong> {bd.motivation}</p>")
        parts.append(f"<p><strong>Hypothesis:</strong> {bd.hypothesis}</p>")
        parts.append(f"<p><strong>Methodology:</strong> {bd.methodology}</p>")
        parts.append(f"<p><strong>Results:</strong> {bd.results}</p>")
        parts.append(f"<p><strong>Interpretation:</strong> {bd.interpretation}</p>")
        parts.append(f"<p><strong>Context:</strong> {bd.context}</p>")
        parts.append(f"<p><strong>Limitations:</strong> {bd.limitations}</p>")
        parts.append(f"<p><strong>Why it matters to you:</strong> {bd.relevance}</p>")
    else:
        parts.append("<p>Deep read could not be completed for this paper.</p>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _generate_kb_notes(
    *,
    kb_note_writer: KBNoteWriter,
    generated_at: datetime,
    paper_by_id: dict[str, Paper],
    scored: Sequence[ScoredPaper],
    digest_selected_ids: set[str],
    deep_results_by_id: dict[str, DeepReadResult],
    watchlist_matches: dict[str, WatchlistMatch],
    knowledge_base: KnowledgeBase,
) -> tuple[int, int]:
    full_count = 0
    for arxiv_id in sorted(deep_results_by_id):
        result = deep_results_by_id[arxiv_id]
        paper = paper_by_id.get(arxiv_id) or result.entry.paper
        watch_match = watchlist_matches.get(arxiv_id)
        watchlist_label = (
            result.entry.watchlist_match
            or (watch_match.matched_name if watch_match else None)
        )
        kb_note_writer.write_deep_read_note(
            paper=paper,
            result=result,
            watchlist_match=watchlist_label,
        )
        knowledge_base.add_paper(
            _deep_result_to_kb_record(
                result=result,
                fallback_entry=result.entry,
                timestamp=generated_at,
            )
        )
        full_count += 1

    deep_read_ids = set(deep_results_by_id)
    seen_stub_ids: set[str] = set()
    stub_count = 0

    for scored_item in sorted(scored, key=lambda item: item.relevance_score, reverse=True):
        if scored_item.arxiv_id not in digest_selected_ids:
            continue
        if scored_item.arxiv_id in deep_read_ids or scored_item.arxiv_id in seen_stub_ids:
            continue

        paper = paper_by_id.get(scored_item.arxiv_id)
        if paper is None:
            continue
        if knowledge_base.known_paper(
            arxiv_id=scored_item.arxiv_id, url=paper.url, title=paper.title,
        ):
            continue

        kb_note_writer.write_stub_note(
            paper=paper,
            scored=scored_item,
            tags=_stub_note_tags(paper, scored_item),
        )
        knowledge_base.add_paper(
            _scored_to_kb_record(
                paper=paper,
                scored=scored_item,
                timestamp=generated_at,
            )
        )
        seen_stub_ids.add(scored_item.arxiv_id)
        stub_count += 1

    return stub_count, full_count


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

    run_parser = subparsers.add_parser(
        "run",
        help="Run full fetch → score → digest pipeline (weekly mode enables deep reads).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but skip delivery, KB note writes, and state updates; print markdown digest to stdout.",
    )
    run_parser.add_argument(
        "--weekly",
        action="store_true",
        help="Enable weekly mode: run deep reads for top-ranked papers before rendering digest.",
    )

    subparsers.add_parser("test-config", help="Validate config and print summary")
    subparsers.add_parser("test-fetch", help="Fetch papers only (no LLM calls)")

    return parser


def _normalize_merge_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace for cross-source title dedup."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _compact_text(text: str, *, max_chars: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "…"


def _sanitize_list(values: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _deep_entry_to_result(entry: DeepReadEntry) -> DeepReadResult:
    return DeepReadResult(
        entry=entry,
        topics=_fallback_topics(entry.paper),
        key_findings=list(entry.breakdown.tldr[:3]),
        builds_on=[],
    )


def _stub_note_tags(paper: Paper, scored: ScoredPaper) -> list[str]:
    raw_tags = list(paper.categories)
    raw_tags.append(f"novelty-{scored.novelty_signal}")
    return _dedupe(_sanitize_list(raw_tags))


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