# Homelab Cron Deployment

Use `scripts/run-homelab-digest.sh` to run the same pipeline that the GitHub Action runs, without the 30-minute Actions timeout.

The runner:

- loads environment variables from `.env` by default
- uses `config.example.yml` by default, matching the GitHub Action
- auto-switches the scheduled Monday `daily` run to `weekly`, matching the workflow
- runs `python -m paper_scout --config config.example.yml -v run`
- commits and pushes KB notes from `../kb/papers`
- commits and pushes `digests/` and `last_run.json`
- writes logs to `logs/scout-YYYY-MM-DD.log`
- uses a lock file so slow runs do not overlap

## One-time setup

Clone `scout` and `kb` as sibling directories so the default `../kb/papers` path works:

```bash
mkdir -p /srv/scout
git clone git@github.com:inwaves/scout.git /srv/scout
git clone git@github.com:inwaves/kb.git /srv/kb
cd /srv/scout
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Make sure the homelab user running cron has git credentials that can pull and push both repositories.

Create `/srv/scout/.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
SMTP_USERNAME=you@example.com
SMTP_PASSWORD=...
SCOUT_SEND_TO=you@example.com
SCOUT_FEEDBACK_SIGNING_SECRET=...

# Optional overrides:
# SCOUT_CONFIG_FILE=config.example.yml
# SCOUT_KB_DIR=/srv/kb
# SCOUT_LOG_DIR=/var/log/scout
# SCOUT_PYTHON=/srv/scout/.venv/bin/python
```

The `.env` file is ignored by git. Lock it down after creating it:

```bash
chmod 600 /srv/scout/.env
```

## Test run

Run a dry run first:

```bash
cd /srv/scout
MODE=daily-dry-run ./scripts/run-homelab-digest.sh
tail -n 80 logs/scout-$(date -u +%Y-%m-%d).log
```

Then run the real job once:

```bash
cd /srv/scout
MODE=daily ./scripts/run-homelab-digest.sh
```

## Cron

Edit the crontab:

```bash
crontab -e
```

Add:

```cron
CRON_TZ=UTC
0 7 * * * cd /srv/scout && ./scripts/run-homelab-digest.sh
```

If your cron implementation does not support `CRON_TZ`, schedule the equivalent local time instead. The runner itself uses UTC for Monday weekly detection, log names, and commit dates.

## Useful knobs

Set these in `.env` or before invoking the script:

```bash
MODE=daily|weekly|daily-dry-run|weekly-dry-run
SCOUT_GIT_PULL=0      # skip pulling scout/kb before the run
SCOUT_PUSH=0          # commit locally but do not push
SCOUT_INSTALL_DEPS=1  # run pip install -r requirements.txt before Scout
SCOUT_LOG_FILE=/path/to/specific.log
SCOUT_LOCK_FILE=/tmp/scout-digest.lock
```

For normal unattended operation, leave `SCOUT_GIT_PULL=1` and `SCOUT_PUSH=1`.
