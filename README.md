# Stock Research Engine

A daily stock research tool for UK investors. Takes a Stockopedia CSV export, runs three-pass AI analysis on every stock using Claude Haiku, and outputs a single Markdown report for NotebookLM.

## System Requirements

- Python 3.10+
- An Anthropic API key

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Usage

### Standard daily run

```bash
python analyze.py --stocks my_stocks.csv
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--stocks` | (required) | Path to Stockopedia CSV export |
| `--top N` | 5 | Number of Tier 1 top QV movers (default 5) |
| `--refresh TICKER` | none | Force full three-pass refresh for one stock |
| `--refresh-all` | off | Force full three-pass refresh for every stock |
| `--max-cost N` | 2.00 | Hard cost cap in USD |
| `--skip-trial` | off | Skip cost confirmation prompt |
| `--delay N` | 2 | Seconds between API calls |
| `--dry-run` | off | Generate report with mock data (no API calls, no cost) |

### Examples

```bash
# Standard run
python analyze.py --stocks my_stocks.csv

# Change Tier 1 size
python analyze.py --stocks my_stocks.csv --top 3

# Force refresh a single stock
python analyze.py --stocks my_stocks.csv --refresh ING

# Force refresh all stocks
python analyze.py --stocks my_stocks.csv --refresh-all

# Set cost cap
python analyze.py --stocks my_stocks.csv --max-cost 1.00

# Skip confirmation for repeat runs
python analyze.py --stocks my_stocks.csv --skip-trial

# Custom delay between API calls
python analyze.py --stocks my_stocks.csv --delay 3

# Dry run to validate CSV with no API cost
python analyze.py --stocks my_stocks.csv --dry-run
```

## Typical Daily Workflow

1. Export your stock list from Stockopedia as a CSV
2. Dry run to validate the CSV: `python analyze.py --stocks my_stocks.csv --dry-run`
3. Real run: `python analyze.py --stocks my_stocks.csv`
4. Review the output file in `/output/`
5. Upload to NotebookLM for podcast or Q&A

## Research Process — Three Passes Per Stock

**Pass 1 — Haiku knowledge pass (no web search)**
Uses `claude-haiku-4-5-20251001` without web search. Fills in the six-element analysis from training knowledge. Flags any sections where confidence is low.

**Pass 2 — Targeted web search (conditional)**
Only runs if Pass 1 flagged gaps in sections 1, 2, or 3 (Rank Sense-Check, Current Narrative, Multibagger Potential). Uses one targeted web search for the most recent earnings or trading update. Skipped if Pass 1 produced good coverage — saving cost.

**Pass 3 — Latest Updates (always runs)**
Always runs for every stock using one web search targeted at official announcements:
- GB stocks: RNS regulatory news, trading updates, director dealings, results
- US stocks: SEC filings, official press releases

## Output Structure

Each run generates a single file: `YYYY-MM-DD_HHmm_analysis.md` in `/output/`.

```
# Stock Analysis Report — [Date] [Time]

## Summary
[Overview of Tier 1 movers and overall read on the batch]

---

## Tier 1 — Top QV Rank Movers
[Top 5 stocks by QV rank improvement since last run]

### Company Name (TICKER — Exchange) — BUY / WATCH / SELL
#### QV: prev→current (+delta) | Quality: X | Value: X | Momentum: X | Style

[Eight-section write-up: Rank Sense-Check, Current Narrative, Multibagger Potential,
Bear Case, ESG, Financial Health Red Flags, Latest Updates, Final Assessment]

---

## Remaining Stocks — By QV Rank
[All other stocks ordered by absolute QV rank descending]
```

All recommendations are **Buy**, **Watch**, or **Sell** — no other labels.

The last 10 output files are retained automatically.

## Cache Behaviour

- **First time a stock is seen:** all three passes run, result cached in `data/cache/`
- **Within 30 days:** Pass 3 only (Latest Updates), appended as dated update block
- **Cache older than 30 days:** full three-pass refresh
- **`--refresh TICKER`:** force full refresh for one stock
- **`--refresh-all`:** force full refresh for every stock

Cache files are prefixed by market flag: `gb_TICKER.md`, `us_TICKER.md`.

## Expected CSV Columns

The CSV must contain these columns (exported directly from Stockopedia):

```
Flag, Ticker, Name, Last Price, Mkt Cap (m GBP), Stock Rank™, Quality Rank,
Value Rank, Momentum Rank, Momentum Rank Previous Day, QV Rank, VM Rank,
QM Rank, StockRank Style, Risk Rating, Sector
```

## Cost Expectations

All three passes use `claude-haiku-4-5-20251001` — the most cost-efficient Claude model.

- **Full research per stock:** ~$0.025–0.035 (three passes, up to 2 web searches)
- **Update-only per stock:** ~$0.012 (Pass 3 only, 1 web search)
- **Typical daily run (30 stocks, mostly cached):** under $0.50
- **First run (all full research):** ~$0.75–1.00 for 30 stocks
- **Default cost cap:** $2.00 (configurable via `--max-cost`)

The tool prints a cost estimate and prompts `Proceed? (y/n)` before every run unless `--skip-trial` is passed.

## Data Storage

- `data/cache/` — One Markdown file per stock, prefixed by market flag
- `data/history.json` — Rank snapshots per stock per run (max 10 per stock)
- `data/errors.log` — Error log with UTC timestamps
- `output/` — Report files, last 10 retained automatically
