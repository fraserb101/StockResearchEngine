# Stock Portfolio & Watchlist Analyzer

## What This Project Does
A daily stock research tool. Takes Stockopedia CSV exports, uses Claude's web search to research stocks, and outputs three Markdown reports for NotebookLM.

## Quick Start (for Claude Code sessions)

When the user opens a session on this repo, help them through these steps:

1. **Check API key**: Look for `.env` file containing `ANTHROPIC_API_KEY=...`. If missing, tell the user to create it.
2. **Check for CSVs**: Ask the user to upload their portfolio and watchlist CSVs from Stockopedia.
3. **Dry run first**: Always suggest `--dry-run` before a real run to validate CSVs.
4. **Real run**: `python analyze.py --portfolio <file> --watchlist <file>`
5. **Show output**: Read the three files from `/output/` and show them to the user.

## Key Commands
```
# Dry run (no API cost)
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --dry-run

# Real run
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv

# Refresh one stock
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --refresh TICKER

# Skip trial for repeat runs
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --skip-trial
```

## File Layout
- `analyze.py` — the tool
- `.env` — API key (never committed to git)
- `data/cache/` — research cache per stock
- `data/history.json` — rank snapshots over time
- `output/` — three Markdown report files per run
