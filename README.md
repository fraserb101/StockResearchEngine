# Stock Portfolio & Watchlist Analyzer

A command-line Python tool that takes CSV exports from Stockopedia and uses AI-powered research to generate actionable hold/sell recommendations for existing holdings, and ranked buy candidates from a screened watchlist.

## System Requirements

- Python 3.10+
- An Anthropic API key with access to `claude-sonnet-4-6`

## Setup

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY='your-key-here'
```

## Usage

### Standard daily run

```bash
python analyze.py --portfolio fb_2026_folio_stockopedia.csv --watchlist super_contrarians_page_1.csv
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--portfolio` | (required) | Path to portfolio CSV from Stockopedia |
| `--watchlist` | (required) | Path to watchlist CSV from Stockopedia |
| `--top N` | 5 | Number of top QV rank improvers for Tier 1 full research |
| `--skip-trial` | off | Skip trial mode for repeat same-day runs |
| `--max-cost N` | 2.00 | Maximum cost cap in USD (hard limit, cannot be overridden) |
| `--refresh TICKER` | none | Force refresh research for a single stock |
| `--refresh-all` | off | Force refresh all stocks |
| `--delay N` | 2 | Delay between API calls in seconds |

### Examples

```bash
# Control how many stocks get full research (default 5)
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --top 3

# Skip trial mode for repeat same-day runs
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --skip-trial

# Set cost cap
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --max-cost 3.00

# Refresh a single stock
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --refresh ING

# Refresh all stocks
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --refresh-all

# Custom delay between API calls
python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv --delay 3
```

## Typical Daily Workflow

1. Download portfolio and watchlist CSVs from Stockopedia
2. Run the tool: `python analyze.py --portfolio portfolio.csv --watchlist watchlist.csv`
3. Review "Today's Rank Movers" in the summary file
4. Upload the three output files to NotebookLM
5. Consume as podcast, slides, or Q&A

## Output Files

Each run generates three Markdown files in the `/output/` directory:

- **`_summary.md`** — Morning scan file. Today's rank movers, portfolio flags, executive summary. Scannable in 60 seconds.
- **`_portfolio.md`** — Full research write-ups for every held stock with Hold/Sell recommendations.
- **`_watchlist.md`** — Tiered research: Tier 1 (full write-ups for top QV movers), Tier 2 (condensed for strong/stable), Tier 3 (one-liners from cache).

Last 10 runs are retained automatically.

## Cost Expectations

- **First run (cold cache):** $3-5 depending on portfolio and watchlist size
- **Daily runs thereafter:** typically well under $1.00 as most stocks are served from cache and only the top N get full research
- **Default cost cap:** $2.00 (configurable via `--max-cost`)

The tool runs a trial on 2 stocks before every full run to estimate actual cost and asks for confirmation before proceeding.

## Data Storage

- `/data/cache/` — One Markdown file per stock, prefixed by market flag (`gb_`, `us_`)
- `/data/history.json` — Rank snapshots per stock per run (max 10 per stock)
- `/data/errors.log` — Error log with UTC timestamps
- `/output/` — Report files, last 10 runs retained

## Three-Tier Watchlist System

- **Tier 1** — Top N stocks by QV rank improvement. Full six-element research.
- **Tier 2** — Remaining stocks with QV Rank above 85. Condensed research.
- **Tier 3** — Everything else. One-line summary from cache only. No API call.

Tiers are recalculated from scratch on every run based on the latest rank snapshot.
