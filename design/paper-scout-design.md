# Paper Scout — Design Document

**A personalized, autonomous agent that monitors arXiv daily and surfaces the most relevant new papers based on your research interests.**

---

## 1. Goals

- **Passive discovery**: Runs on a schedule (daily or custom), requires no manual interaction after setup.
- **Personalized relevance**: You define your research interests in natural language. The agent scores every new paper against your profile using an LLM, not keyword matching.
- **Quality over quantity**: Surfaces 5–15 papers per day (configurable), ranked by relevance, with a one-paragraph summary explaining *why* each paper matters to you.
- **Low cost**: Designed to cost < $1/day even with a capable model.
- **Simple deployment**: Runs as a GitHub Actions workflow (free), a cron job, or a local script. No infrastructure to manage.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     SCHEDULER                           │
│          (GitHub Actions / cron / manual)                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  1. FETCH STAGE                         │
│                                                         │
│  arXiv API  ──►  Fetch new papers from watched          │
│                  categories (last 24h / since last run)  │
│                                                         │
│  Output: List of papers (title, abstract, authors,      │
│          categories, arxiv_id, url)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              2. PRE-FILTER STAGE (optional)             │
│                                                         │
│  Embedding similarity against research profile          │
│  using sentence-transformers (local, free).             │
│  Keep top N candidates (e.g., top 100 from 300+).       │
│                                                         │
│  Purpose: Reduce LLM calls for cost control.            │
│  Can be skipped if paper volume is manageable.           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               3. LLM SCORING STAGE                      │
│                                                         │
│  For each candidate paper, send to Claude Sonnet 4.6:   │
│                                                         │
│  Input:                                                 │
│    - System prompt: research profile + scoring rubric   │
│    - User prompt: paper title + abstract                │
│                                                         │
│  Output (structured JSON):                              │
│    - relevance_score: 1–10                              │
│    - rationale: one sentence explaining the score       │
│    - novelty_signal: "incremental" | "notable" | "breakthrough"│
│                                                         │
│  Batched: multiple papers per LLM call (up to ~10)     │
│  to reduce overhead and latency.                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              4. RANKING & SELECTION                      │
│                                                         │
│  Sort by relevance_score descending.                    │
│  Apply configurable threshold (default: score >= 7).     │
│  Take top K papers (default: 10).                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              5. SUMMARY GENERATION                      │
│                                                         │
│  For each selected paper, generate a ~150-word summary: │
│    - What the paper does                                │
│    - Why it matters to your interests                   │
│    - Key technical contribution                         │
│                                                         │
│  Uses the same Sonnet 4.6 model.                        │
│  Can be batched with scoring in a single call.          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               6. DELIVERY STAGE                         │
│                                                         │
│  Render digest and deliver via configured channel:      │
│    - Email (SMTP / SendGrid / AWS SES)                  │
│    - Slack (webhook)                                    │
│    - Discord (webhook)                                  │
│    - Markdown file (local / committed to repo)          │
│                                                         │
│  Digest format: ranked list with score, summary,        │
│  authors, arXiv link, and PDF link.                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Source: arXiv API

The arXiv API (`export.arxiv.org/api/query`) supports:
- Querying by category (e.g., `cat:cs.AI`, `cat:cs.MA`, `cat:cs.CL`)
- Date-range filtering via `submittedDate` field
- Returns: title, abstract, authors, categories, published date, PDF link

**Daily volume estimates** (approximate new submissions per weekday):

| Category | Daily Papers |
|----------|-------------|
| cs.AI    | ~80–120     |
| cs.CL    | ~60–100     |
| cs.LG    | ~150–250    |
| cs.MA    | ~10–20      |
| cs.CV    | ~100–150    |

Watching 3–4 categories typically yields 200–400 new papers per weekday, with significant overlap (papers often have multiple categories).

After deduplication, expect **150–300 unique papers per day** for a typical ML/agents-focused configuration.

---

## 4. Model Selection & Cost Estimation

### Why Claude Sonnet 4.6

| Consideration | Assessment |
|--------------|------------|
| **Capability** | Strong reasoning about technical content. Can reliably assess whether a paper on "hierarchical planning with tool-augmented LLMs" is relevant to "AI agent architectures" even without keyword overlap. |
| **Structured output** | Reliable JSON output for scores and rationales. |
| **Cost** | $3/MTok input, $15/MTok output — 10x cheaper than Opus, but meaningfully more capable than Haiku for nuanced relevance judgments. |
| **Context window** | 1M tokens — can easily batch 10+ papers per call. |
| **Batch API** | Anthropic offers batch pricing at $1.50/$7.50 (50% discount) for async jobs, which fits the non-real-time nature of this agent. |

### Cost Breakdown (per day, 200 papers)

**Scoring stage** (200 papers, batched 10 per call = 20 calls):
- System prompt (research profile + rubric): ~800 tokens, cached after first call
- Per paper (title + abstract): ~300 tokens × 10 = 3,000 tokens per call
- Output per call: ~500 tokens (10 JSON objects)
- Total input: ~76,000 tokens/day
- Total output: ~10,000 tokens/day

**Summary stage** (top 10 papers):
- Input per paper: ~400 tokens (abstract + instructions)
- Output per paper: ~200 tokens (summary)
- Total input: ~4,000 tokens
- Total output: ~2,000 tokens

**Daily cost estimate with standard pricing:**

| Component | Input Tokens | Output Tokens | Cost |
|-----------|-------------|---------------|------|
| Scoring   | 76,000      | 10,000        | $0.38 |
| Summaries | 4,000       | 2,000         | $0.04 |
| **Total** | **80,000**  | **12,000**    | **$0.42/day** |

**With batch API pricing (50% discount): ~$0.21/day → ~$6.30/month**

**With prompt caching** (system prompt cached across calls): slightly lower, ~$0.18/day.

### Alternative: Two-Stage with Cheaper Pre-Filter

If you want to further reduce cost or handle higher volumes:
1. Use a local sentence-transformers model (free) to compute cosine similarity between each paper's abstract embedding and a pre-computed embedding of your research profile.
2. Only send the top 50–100 to Sonnet 4.6.
3. This reduces LLM costs by 50–70% but adds a dependency on running a local embedding model.

**Recommendation**: Skip the pre-filter initially. At ~$0.21/day with batch pricing, the cost is negligible. Add the pre-filter only if you scale to many categories or want to run on free-tier infrastructure with tight cost constraints.

---

## 5. Research Profile Configuration

Your research interests are defined in a YAML configuration file. The profile is written in natural language — this is the key advantage over keyword systems.

```yaml
# paper-scout.yml

profile:
  name: "ML/AI Agents Research"
  description: |
    I am a researcher focused on autonomous AI agents, particularly:

    HIGH INTEREST:
    - Agent architectures (planning, reasoning, memory, tool use)
    - Multi-agent systems and collaboration
    - LLM-based agents for software engineering and coding
    - Agent evaluation, benchmarks, and failure modes
    - Reinforcement learning for agent training (RLHF, RLAIF, reward modeling)
    - Agent safety, alignment, and controllability

    MODERATE INTEREST:
    - Foundation model training and scaling (pre-training, post-training)
    - Retrieval-augmented generation (RAG) architectures
    - Prompt engineering and in-context learning
    - Inference optimization (KV cache, speculative decoding, quantization)

    LOW INTEREST (only if groundbreaking):
    - Applications of existing AI to specific domains (medical, legal, etc.)
    - Pure computer vision without agent/reasoning components
    - Speech and audio models
    - Robotics hardware

  scoring_rubric: |
    Score each paper 1-10 based on:
    - 9-10: Directly advances agent architectures, multi-agent systems,
            or agent training. Novel contribution I would want to read today.
    - 7-8:  Relevant to my core interests. Noteworthy technique or result
            that connects to agent research.
    - 5-6:  Tangentially related. Interesting but not directly in my lane.
    - 3-4:  Only loosely connected to my interests.
    - 1-2:  Not relevant.

arxiv:
  categories:
    - "cs.AI"    # Artificial Intelligence
    - "cs.MA"    # Multi-Agent Systems
    - "cs.CL"    # Computation and Language
    - "cs.LG"    # Machine Learning
    - "cs.SE"    # Software Engineering (for coding agents)
  max_results_per_category: 200
  lookback_hours: 28  # Slightly > 24h to avoid missing papers

scoring:
  model: "claude-sonnet-4-6"
  batch_size: 10          # Papers per LLM call
  threshold: 7            # Minimum score to include
  max_papers: 15          # Maximum papers in digest
  use_batch_api: true     # Use Anthropic batch API for 50% discount
  temperature: 0.0        # Deterministic scoring

delivery:
  channels:
    - type: "email"
      smtp_host: "smtp.gmail.com"
      smtp_port: 587
      from: "paperscout@yourdomain.com"
      to: "you@yourdomain.com"
      subject_template: "Paper Scout Digest — {date} ({count} papers)"
    # - type: "slack"
    #   webhook_url: "${SLACK_WEBHOOK_URL}"
    # - type: "discord"
    #   webhook_url: "${DISCORD_WEBHOOK_URL}"
    - type: "markdown"
      output_dir: "./digests"

schedule:
  # For GitHub Actions: cron expression (UTC)
  cron: "0 7 * * 1-5"  # 7 AM UTC, weekdays only
  # For local: use system cron or launchd
```

---

## 6. Digest Output Format

Each digest is rendered as both HTML (for email) and Markdown (for file output / Slack / Discord).

Example digest:

```markdown
# 📄 Paper Scout Digest — March 28, 2026

**12 papers scored ≥7 out of 247 reviewed**

---

### 1. [HyperAgents: Recursive Metacognitive Self-Improvement in LLM Agents](https://arxiv.org/abs/2603.19461)
**Score: 9/10** · Novelty: breakthrough
*Authors: J. Smith, A. Kumar et al.*
*Categories: cs.AI, cs.MA*

This paper introduces a framework where LLM agents recursively evaluate and
improve their own reasoning strategies through metacognitive loops. The agents
maintain a "strategy library" of successful reasoning patterns and dynamically
compose them for new tasks. Evaluated on three agent benchmarks showing 23%
improvement over static CoT approaches. Directly relevant to your interest in
agent architectures and planning — this is a novel take on self-improving
agents that goes beyond simple reflection/retry patterns.

[📄 PDF](https://arxiv.org/pdf/2603.19461) · [🔗 Abstract](https://arxiv.org/abs/2603.19461)

---

### 2. [PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost](https://arxiv.org/abs/2603.21383)
**Score: 8/10** · Novelty: notable
...
```

---

## 7. Project Structure

```
paper-scout/
├── paper_scout/
│   ├── __init__.py
│   ├── config.py          # Load and validate YAML config
│   ├── fetcher.py         # arXiv API client
│   ├── scorer.py          # LLM scoring (Anthropic client)
│   ├── summarizer.py      # Summary generation
│   ├── digest.py          # Render digest (Markdown + HTML)
│   ├── delivery/
│   │   ├── __init__.py
│   │   ├── email.py       # SMTP delivery
│   │   ├── slack.py       # Slack webhook
│   │   ├── discord.py     # Discord webhook
│   │   └── file.py        # Write to file
│   └── cli.py             # CLI entry point
├── config.example.yml     # Example configuration
├── requirements.txt       # Dependencies
├── pyproject.toml         # Package metadata
├── .github/
│   └── workflows/
│       └── daily-digest.yml  # GitHub Actions workflow
├── digests/               # Output directory for markdown digests
├── tests/
│   ├── test_fetcher.py
│   ├── test_scorer.py
│   └── test_digest.py
└── README.md
```

---

## 8. Dependencies

```
anthropic>=0.45.0      # Claude API client
httpx>=0.27.0          # HTTP client (used by anthropic)
pyyaml>=6.0            # Config parsing
jinja2>=3.1            # Digest templating
python-dateutil>=2.9   # Date handling
feedparser>=6.0        # arXiv Atom feed parsing (alternative to API)
```

Optional:
```
sentence-transformers>=3.0  # For local embedding pre-filter
```

No heavy frameworks. No database. No vector store. The agent is stateless — it fetches, scores, renders, delivers, and exits. State (what has been seen) is tracked by timestamp of last run, stored in a small JSON file or GitHub Actions artifact.

---

## 9. Scheduling & Deployment Options

### Option A: GitHub Actions (recommended for simplicity)

- Free for public repos, 2,000 minutes/month for private repos.
- Secrets stored in GitHub repository settings (API keys, SMTP credentials).
- Runs on schedule, commits digest to repo, and/or sends email.
- No server to manage.

### Option B: Local cron job

- `crontab -e` → `0 7 * * 1-5 cd /path/to/paper-scout && python -m paper_scout`
- Suitable if you want to run on a personal machine or VPS.

### Option C: Cloud function

- AWS Lambda / GCP Cloud Function triggered by CloudWatch/Cloud Scheduler.
- Overkill for this use case but viable if you want serverless.

---

## 10. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **arXiv data source** | Atom feed via `feedparser` | More reliable than the official API for daily batch queries. The OAI-PMH API has rate limits and complexity; the Atom feed is simpler and sufficient. |
| **LLM model** | Claude Sonnet 4.6 | Best balance of reasoning quality vs. cost. Haiku 4.5 is too weak for nuanced relevance; Opus 4.6 is 5x more expensive with marginal gains for scoring tasks. |
| **Batching** | 10 papers per call | Keeps each call under ~4K input tokens. Maximizes throughput without hitting output quality degradation from overlong prompts. |
| **Scoring approach** | Single-pass LLM | No embedding pre-filter initially. Cost is already low enough (~$0.21/day with batch API) that the complexity is not justified. |
| **State management** | Timestamp file | Stateless design. A small `last_run.json` tracks when the agent last ran, so the next run fetches only new papers. No database needed. |
| **Delivery** | Pluggable channels | Email for daily inbox delivery. Markdown file for archival. Slack/Discord as optional additions. |
| **Structured output** | JSON mode | Anthropic's tool-use / JSON mode ensures reliable structured responses for scoring. |
| **Configuration** | Single YAML file | Easy to edit, version-controllable, human-readable. Research profile is natural language embedded in YAML. |

---

## 11. Future Extensions (Out of Scope for v1)

- **Feedback loop**: Thumbs up/down on delivered papers to refine scoring prompt over time.
- **Citation trail following**: For high-scoring papers, fetch their references and score those too.
- **Full-text analysis**: Download PDF, extract full text, generate deeper summaries.
- **Semantic Scholar enrichment**: Pull citation counts, influence scores, and related papers.
- **Multi-profile support**: Different interest profiles for different research threads.
- **Web dashboard**: Browse and search past digests.
- **Embedding pre-filter**: Add sentence-transformers stage if cost or volume becomes an issue.

---

## 12. Open Questions for Review

1. **Categories**: The default set covers cs.AI, cs.MA, cs.CL, cs.LG, cs.SE. Do you want to add or remove any?
2. **Delivery preference**: Email, Slack, Discord, or just markdown files? Multiple?
3. **Frequency**: Daily (weekdays), daily (all days), or weekly digest?
4. **Score threshold**: Default is 7/10. Do you want it tighter (8+) or looser (6+)?
5. **Batch API vs. standard**: Batch API is 50% cheaper but results are async (up to 24h turnaround, though typically minutes). Since this is a scheduled background agent, the delay is fine. Confirm?
6. **GitHub Actions vs. local**: Where do you want to run this?