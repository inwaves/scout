# Scout

A personalized, autonomous agent that monitors arXiv and major AI lab publications daily, surfacing the most relevant new papers and posts based on your research interests.

Scout fetches new papers from arXiv and web sources (Anthropic, OpenAI, Google DeepMind), scores each one against your research profile using Claude, generates summaries for the top matches, and delivers a ranked digest via email, Slack, Discord, or markdown file.

## How It Works

1. **Fetch** — Pulls new papers from your watched arXiv categories (last 24–28 hours) and optionally scrapes web sources (lab blogs, sitemaps) for recent posts, system cards, and risk reports.
2. **Score** — Sends each paper's title and abstract to Claude with your research profile. Gets back a relevance score (1–10), rationale, and novelty signal.
3. **Watchlist** — Checks author names and affiliations against your configured watchlist. Papers from priority labs (Anthropic, OpenAI, DeepMind) or specific authors are always included. Web source posts are automatically matched against watchlist organizations.
4. **Deep Read** *(weekly mode)* — For the top papers, an agentic reader analyzes the full PDF (or web post content) via Claude. It can query Scout's knowledge base and Semantic Scholar for context, then produces a structured breakdown: TL;DR, motivation, hypothesis, methodology, results, interpretation, context, limitations, and personal relevance.
5. **Knowledge Base** — Each deep-read paper is stored with its topics, key findings, and references. On future runs, the agent uses this accumulated knowledge to situate new papers.
6. **Deliver** — Renders a two-tier digest (Deep Reads + Also Noteworthy) in Markdown and HTML, then delivers via your configured channels.
7. **Hot Alerts** — Papers scoring ≥9 or matching your watchlist trigger immediate notifications.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy the example config and customize your research profile:

```bash
cp config.example.yml scout.yml
```

Edit `scout.yml`:
- Set your research interests in `profile.description` (natural language)
- Define your scoring rubric in `profile.scoring_rubric`
- Choose arXiv categories to watch
- Enable web sources to monitor lab blogs and publications
- Configure delivery channels

### 3. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Test the connection

```bash
# Validate config
python -m paper_scout test-config

# Fetch papers without scoring (no API cost)
python -m paper_scout test-fetch
```

### 5. Run

```bash
# Full run with delivery
python -m paper_scout -v run

# Dry run — print digest to stdout, skip delivery
python -m paper_scout -v run --dry-run
```

## Configuration

Scout is configured via a single YAML file. See `config.example.yml` for the full reference.

### Research Profile

The profile is written in natural language — this is the key advantage over keyword systems:

```yaml
profile:
  name: "ML/AI Agents Research"
  description: |
    I am a researcher focused on autonomous AI agents, particularly:

    HIGH INTEREST:
    - Agent architectures (planning, reasoning, memory, tool use)
    - Multi-agent systems and collaboration
    - Agent safety, alignment, and controllability

    LOW INTEREST (only if groundbreaking):
    - Pure computer vision without agent/reasoning components
    - Speech and audio models
```

### Web Sources

Scout can monitor major AI lab websites for blog posts, research announcements, system cards, and risk reports that never appear on arXiv. This is opt-in and configured via the `web_sources` section:

```yaml
web_sources:
  enabled: true
  sources:
    - type: "anthropic_alignment"   # alignment.anthropic.com blog
      enabled: true
    - type: "anthropic_news"        # anthropic.com/news + /research (via sitemap)
      enabled: true
    - type: "openai"                # openai.com/index + /research (via sitemap)
      enabled: true
    - type: "deepmind"              # deepmind.google/discover/blog + /research (via sitemap)
      enabled: true
  request_timeout_seconds: 30.0
  query_pause_seconds: 1.0
  max_items_per_source: 50
  fetch_page_metadata: true         # Fetch each page's title/description/PDF links
```

**Built-in sources:**

| Source Key | What It Covers | Method |
|-----------|---------------|--------|
| `anthropic_alignment` | Anthropic Alignment Science blog (alignment.anthropic.com) | HTML index parsing |
| `anthropic_news` | Anthropic news and research pages, including system cards and risk reports linked from www-cdn.anthropic.com | Sitemap XML |
| `openai` | OpenAI research index and blog posts | Sitemap XML |
| `deepmind` | Google DeepMind blog and research publications | Sitemap XML |

When `fetch_page_metadata` is enabled, Scout fetches each discovered page to extract its title, description meta tag, and any linked PDFs (e.g., system cards on `www-cdn.anthropic.com`). If a linked PDF is found, it becomes available for deep reading.

Web source posts flow through the same scoring, watchlist, deep-read, and delivery pipeline as arXiv papers. In the digest, they are labeled with their source (e.g., "Source: Anthropic Alignment").

### Delivery Channels

| Channel | Config Key | Notes |
|---------|-----------|-------|
| Markdown file | `type: markdown` | Written to `output_dir`. Default. |
| Email | `type: email` | Via SMTP. Sends HTML + text. |
| Slack | `type: slack` | Via incoming webhook URL. |
| Discord | `type: discord` | Via webhook URL. Auto-chunks long messages. |

### Environment Variables

Config values can reference environment variables with `${VAR_NAME}` syntax:

```yaml
anthropic_api_key: "${ANTHROPIC_API_KEY}"
delivery:
  channels:
    - type: email
      smtp_password: "${SMTP_PASSWORD}"
```

## Deployment

### GitHub Actions (Recommended)

The included workflow at `.github/workflows/daily-digest.yml` runs Scout on a daily schedule.

1. Push this repo to GitHub.
2. Add secrets in **Settings → Secrets and variables → Actions**:
   - `ANTHROPIC_API_KEY`
   - `SMTP_USERNAME` / `SMTP_PASSWORD` (if using email delivery)
3. The workflow runs daily at 7 AM UTC and commits digests to the repo.
4. Trigger manually via **Actions → Scout Daily Digest → Run workflow**.

### Local Cron Job

```bash
crontab -e
# Add:
0 7 * * * cd /path/to/scout && python -m paper_scout -v run >> /tmp/scout.log 2>&1
```

## Cost

With Claude Sonnet 4.6 and the Anthropic batch API (50% discount):

| Component | Daily Cost |
|-----------|-----------|
| Scoring ~200 papers | ~\$0.19 |
| Summarizing ~10 papers | ~\$0.02 |
| **Total** | **~\$0.21/day (~\$6.30/month)** |

Web source fetching has no LLM cost — only the scoring of discovered posts adds to the bill.

## CLI Reference

```
python -m paper_scout [OPTIONS] COMMAND

Commands:
  run           Run the full pipeline (fetch → score → summarize → deliver)
  test-config   Validate configuration and print summary
  test-fetch    Fetch papers only, no LLM calls

Options:
  --config PATH   Path to YAML config file (default: scout.yml)
  -v, --verbose   Increase logging (-v = INFO, -vv = DEBUG)

Run Options:
  --dry-run       Print digest to stdout; skip delivery and state update
  --weekly        Enable weekly mode: run agentic deep reads for top papers
```

## Project Structure

```
scout/
├── paper_scout/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # python -m entry point
│   ├── cli.py               # CLI and pipeline orchestration
│   ├── config.py            # YAML config loading and validation
│   ├── models.py            # Data models (Paper, ScoredPaper, etc.)
│   ├── fetcher.py           # arXiv API client
│   ├── web_fetcher.py       # Web source fetcher (lab blogs, sitemaps)
│   ├── scorer.py            # LLM scoring via Anthropic
│   ├── summarizer.py        # Summary generation
│   ├── deep_reader.py       # Agentic deep-read analysis
│   ├── digest.py            # Markdown + HTML rendering
│   ├── state.py             # Last-run state persistence
│   ├── watchlist.py         # Author/org watchlist matching
│   ├── knowledge_base.py    # KB loading and topic indexing
│   ├── kb_writer.py         # KB note generation
│   ├── semantic_scholar.py  # Semantic Scholar API client
│   ├── costs.py             # LLM cost tracking
│   ├── delivery/            # Pluggable delivery channels
│   │   ├── base.py
│   │   ├── file.py
│   │   ├── email_delivery.py
│   │   ├── slack.py
│   │   └── discord.py
│   └── templates/           # Jinja2 digest templates
│       ├── digest.md.j2
│       └── digest.html.j2
├── config.example.yml       # Example configuration
├── requirements.txt
├── pyproject.toml
├── .github/workflows/
│   └── daily-digest.yml     # GitHub Actions workflow
└── tests/
    ├── test_config.py
    ├── test_fetcher.py
    ├── test_web_fetcher.py
    ├── test_digest.py
    ├── test_delivery.py
    └── test_state.py
```

## License

MIT