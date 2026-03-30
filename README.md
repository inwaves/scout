# Scout

A personalized, autonomous agent that monitors arXiv daily and surfaces the most relevant new papers based on your research interests.

Scout fetches new papers from arXiv, scores each one against your research profile using Claude, generates summaries for the top matches, and delivers a ranked digest via email, Slack, Discord, or markdown file.

## How It Works

1. **Fetch** — Pulls new papers from your watched arXiv categories (last 24–28 hours).
2. **Score** — Sends each paper's title and abstract to Claude with your research profile. Gets back a relevance score (1–10), rationale, and novelty signal.
3. **Summarize** — For papers above your threshold, generates a ~150-word summary explaining what the paper does, why it matters to you, and the key technical contribution.
4. **Deliver** — Renders a digest in Markdown and HTML, then delivers via your configured channels.

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
│   ├── scorer.py            # LLM scoring via Anthropic
│   ├── summarizer.py        # Summary generation
│   ├── digest.py            # Markdown + HTML rendering
│   ├── state.py             # Last-run state persistence
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
```

## License

MIT