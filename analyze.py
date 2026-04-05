#!/usr/bin/env python3
"""Stock Portfolio & Watchlist Analyzer.

A command-line tool that takes CSV exports from Stockopedia and uses AI-powered
research to generate actionable hold/sell recommendations for holdings and
ranked buy candidates from a screened watchlist.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd

# ─── Version & Python Check ────────────────────────────────────────────────────

MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Error: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required. "
             f"You have {sys.version_info.major}.{sys.version_info.minor}.")

# ─── Constants ──────────────────────────────────────────────────────────────────

MODEL_DEFAULT = "claude-haiku-4-5"
MODEL_SEARCH = "claude-sonnet-4-6"  # Haiku doesn't support web search
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"

# Token budgets (hard max_tokens on every API call)
TOKENS_FULL = 1500
TOKENS_CONDENSED = 600
TOKENS_UPDATE = 400
TOKENS_SUMMARY = 600

# Web search limits per API call
SEARCHES_FULL = 1
SEARCHES_CONDENSED = 1
SEARCHES_UPDATE = 1

# Pricing — Haiku default, Sonnet when using web search
PRICE_INPUT_HAIKU = 0.80 / 1_000_000        # $0.80 per 1M input tokens
PRICE_OUTPUT_HAIKU = 4.0 / 1_000_000        # $4 per 1M output tokens
PRICE_INPUT_SONNET = 3.0 / 1_000_000        # $3 per 1M input tokens
PRICE_OUTPUT_SONNET = 15.0 / 1_000_000      # $15 per 1M output tokens
PRICE_PER_SEARCH = 10.0 / 1000              # $10 per 1K searches = $0.01 each

# Cache settings
CACHE_MAX_AGE_DAYS = 30
HISTORY_MAX_SNAPSHOTS = 10
OUTPUT_MAX_RUNS = 10

# Tier thresholds
TIER2_QV_THRESHOLD = 85

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
HISTORY_FILE = DATA_DIR / "history.json"
ERROR_LOG = DATA_DIR / "errors.log"
OUTPUT_DIR = BASE_DIR / "output"

# Expected CSV columns
EXPECTED_COLUMNS = [
    "Flag", "Ticker", "Name", "Last Price", "Mkt Cap (m GBP)", "Stock Rank™",
    "Quality Rank", "Value Rank", "Momentum Rank", "Momentum Rank Previous Day",
    "QV Rank", "VM Rank", "QM Rank", "StockRank Style", "Risk Rating", "Sector"
]

# ─── Utility Functions ──────────────────────────────────────────────────────────


def ensure_dirs():
    """Create required directories if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_error(ticker, error_msg):
    """Append an error entry to the error log."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{timestamp}] {ticker}: {error_msg}\n")


def derive_screen_name(filename):
    """Derive a human-readable screen name from a watchlist filename.

    Example: super_contrarians_page_1.csv -> 'Super Contrarians'
    """
    stem = Path(filename).stem
    # Remove page number suffixes like _page_1, _page_2
    stem = re.sub(r"_page_\d+$", "", stem)
    # Replace underscores with spaces and title-case
    return stem.replace("_", " ").title()


def get_exchange_label(flag):
    """Return exchange label for display."""
    if flag == "gb":
        return "LSE"
    elif flag == "us":
        return "NASDAQ/NYSE"
    return flag.upper()


def get_search_suffix(flag):
    """Return search suffix for web queries."""
    if flag == "gb":
        return "LSE"
    elif flag == "us":
        return "NASDAQ NYSE"
    return ""


def format_rank_delta(old_val, new_val):
    """Format a rank change like '74→86 (+12)'."""
    if old_val is None:
        return str(new_val)
    delta = new_val - old_val
    sign = "+" if delta > 0 else ""
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    return f"{old_val}→{new_val} ({sign}{delta})"


def format_rank_arrow(old_val, new_val):
    """Return just the arrow character for a rank change."""
    if old_val is None:
        return "→"
    if new_val > old_val:
        return "↑"
    elif new_val < old_val:
        return "↓"
    return "→"


def local_now():
    """Return current local time."""
    return datetime.now()


def utc_now():
    """Return current UTC time."""
    return datetime.now(timezone.utc)


# ─── CSV Loading & Validation ──────────────────────────────────────────────────


def validate_csv(filepath, label):
    """Validate CSV has all expected columns. Exit on failure."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        sys.exit(f"Error: Could not read {label} CSV '{filepath}': {e}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        sys.exit(
            f"Error: {label} CSV is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
            + f"\n\nFound columns: {', '.join(df.columns)}"
        )
    return df


def load_stocks(portfolio_path, watchlist_path):
    """Load and validate both CSVs. Return portfolio_df, watchlist_df, and
    a merged unique stocks dict keyed by (flag, ticker)."""
    print(f"CSV validation: ", end="")

    portfolio_df = validate_csv(portfolio_path, "Portfolio")
    watchlist_df = validate_csv(watchlist_path, "Watchlist")
    print("OK")

    # Build unique stocks dict keyed by (flag, ticker)
    unique = {}
    for _, row in portfolio_df.iterrows():
        key = (str(row["Flag"]).strip().lower(), str(row["Ticker"]).strip())
        unique[key] = {
            "ticker": str(row["Ticker"]).strip(),
            "name": str(row["Name"]).strip(),
            "flag": str(row["Flag"]).strip().lower(),
            "last_price": row["Last Price"],
            "mkt_cap": row["Mkt Cap (m GBP)"],
            "stock_rank": int(row["Stock Rank™"]),
            "quality_rank": int(row["Quality Rank"]),
            "value_rank": int(row["Value Rank"]),
            "momentum_rank": int(row["Momentum Rank"]),
            "momentum_rank_previous_day": int(row["Momentum Rank Previous Day"]),
            "qv_rank": int(row["QV Rank"]),
            "vm_rank": int(row["VM Rank"]),
            "qm_rank": int(row["QM Rank"]),
            "stockrank_style": str(row["StockRank Style"]).strip(),
            "risk_rating": str(row["Risk Rating"]).strip(),
            "sector": str(row["Sector"]).strip(),
            "in_portfolio": True,
            "in_watchlist": False,
        }

    for _, row in watchlist_df.iterrows():
        key = (str(row["Flag"]).strip().lower(), str(row["Ticker"]).strip())
        if key in unique:
            unique[key]["in_watchlist"] = True
        else:
            unique[key] = {
                "ticker": str(row["Ticker"]).strip(),
                "name": str(row["Name"]).strip(),
                "flag": str(row["Flag"]).strip().lower(),
                "last_price": row["Last Price"],
                "mkt_cap": row["Mkt Cap (m GBP)"],
                "stock_rank": int(row["Stock Rank™"]),
                "quality_rank": int(row["Quality Rank"]),
                "value_rank": int(row["Value Rank"]),
                "momentum_rank": int(row["Momentum Rank"]),
                "momentum_rank_previous_day": int(row["Momentum Rank Previous Day"]),
                "qv_rank": int(row["QV Rank"]),
                "vm_rank": int(row["VM Rank"]),
                "qm_rank": int(row["QM Rank"]),
                "stockrank_style": str(row["StockRank Style"]).strip(),
                "risk_rating": str(row["Risk Rating"]).strip(),
                "sector": str(row["Sector"]).strip(),
                "in_portfolio": False,
                "in_watchlist": True,
            }

    portfolio_count = sum(1 for s in unique.values() if s["in_portfolio"])
    watchlist_count = sum(1 for s in unique.values() if s["in_watchlist"])
    print(f"Portfolio: {portfolio_count} stocks | Watchlist: {watchlist_count} stocks | Unique: {len(unique)} stocks")

    return portfolio_df, watchlist_df, unique


# ─── History Management ─────────────────────────────────────────────────────────


def load_history():
    """Load rank history from file. Returns dict keyed by 'flag_ticker'."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        # Convert list format to dict keyed by flag_ticker
        if isinstance(data, list):
            return {f"{s['flag']}_{s['ticker']}": s for s in data}
        return data
    except (json.JSONDecodeError, KeyError):
        return {}


def save_history(history):
    """Save rank history to file."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_snapshot(history, stock):
    """Add a new rank snapshot for a stock. Trim to max 10 snapshots."""
    key = f"{stock['flag']}_{stock['ticker']}"

    snapshot = {
        "timestamp_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stock_rank": stock["stock_rank"],
        "quality_rank": stock["quality_rank"],
        "value_rank": stock["value_rank"],
        "momentum_rank": stock["momentum_rank"],
        "momentum_rank_previous_day": stock["momentum_rank_previous_day"],
        "qv_rank": stock["qv_rank"],
        "stockrank_style": stock["stockrank_style"],
        "risk_rating": stock["risk_rating"],
    }

    if key not in history:
        history[key] = {
            "ticker": stock["ticker"],
            "name": stock["name"],
            "flag": stock["flag"],
            "snapshots": [],
        }

    history[key]["snapshots"].append(snapshot)
    # Trim to max snapshots
    if len(history[key]["snapshots"]) > HISTORY_MAX_SNAPSHOTS:
        history[key]["snapshots"] = history[key]["snapshots"][-HISTORY_MAX_SNAPSHOTS:]


def get_previous_snapshot(history, stock):
    """Get the most recent previous snapshot for a stock. Returns None if none."""
    key = f"{stock['flag']}_{stock['ticker']}"
    if key not in history or not history[key]["snapshots"]:
        return None
    return history[key]["snapshots"][-1]


def calculate_qv_delta(stock, prev_snapshot):
    """Calculate QV rank improvement since last snapshot."""
    if prev_snapshot is None:
        return None
    return stock["qv_rank"] - prev_snapshot["qv_rank"]


# ─── Tier Assignment ────────────────────────────────────────────────────────────


def assign_tiers(unique_stocks, history, top_n, is_first_run):
    """Assign tiers to all watchlist stocks. Returns dict of tier assignments."""
    watchlist_stocks = {k: v for k, v in unique_stocks.items() if v["in_watchlist"]}
    tiers = {}

    if is_first_run:
        # First run: use absolute QV rank
        sorted_stocks = sorted(watchlist_stocks.items(),
                               key=lambda x: x[1]["qv_rank"], reverse=True)
        for i, (key, stock) in enumerate(sorted_stocks):
            if i < top_n:
                tiers[key] = 1
            elif stock["qv_rank"] > TIER2_QV_THRESHOLD:
                tiers[key] = 2
            else:
                tiers[key] = 3
    else:
        # Normal run: rank by QV improvement
        deltas = {}
        for key, stock in watchlist_stocks.items():
            prev = get_previous_snapshot(history, stock)
            delta = calculate_qv_delta(stock, prev)
            deltas[key] = delta

        # Sort by delta descending (None sorts last)
        sorted_by_delta = sorted(
            watchlist_stocks.items(),
            key=lambda x: (deltas[x[0]] is not None, deltas[x[0]] or 0),
            reverse=True
        )

        # Take top N improvers for Tier 1
        tier1_count = 0
        tier1_keys = set()
        for key, stock in sorted_by_delta:
            if tier1_count >= top_n:
                break
            delta = deltas[key]
            if delta is not None and delta > 0:
                tier1_keys.add(key)
                tier1_count += 1

        # If fewer than N improvers, fill with highest absolute QV
        if tier1_count < top_n:
            remaining = [(k, s) for k, s in sorted_by_delta if k not in tier1_keys]
            remaining.sort(key=lambda x: x[1]["qv_rank"], reverse=True)
            for key, stock in remaining:
                if tier1_count >= top_n:
                    break
                tier1_keys.add(key)
                tier1_count += 1

        for key, stock in watchlist_stocks.items():
            if key in tier1_keys:
                tiers[key] = 1
            elif stock["qv_rank"] > TIER2_QV_THRESHOLD:
                tiers[key] = 2
            else:
                tiers[key] = 3

    return tiers


# ─── Cache Management ──────────────────────────────────────────────────────────


def cache_path(stock):
    """Return the cache file path for a stock."""
    return CACHE_DIR / f"{stock['flag']}_{stock['ticker']}.md"


def get_cache_age_days(stock):
    """Return cache age in days, or None if no cache exists."""
    path = cache_path(stock)
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (utc_now() - mtime).days


def read_cache(stock):
    """Read cached research for a stock. Returns content string or None."""
    path = cache_path(stock)
    if not path.exists():
        return None
    return path.read_text()


def write_cache(stock, content):
    """Write or overwrite cache for a stock."""
    path = cache_path(stock)
    path.write_text(content)


def append_cache(stock, update_block):
    """Append an update block to existing cache."""
    path = cache_path(stock)
    existing = path.read_text() if path.exists() else ""
    path.write_text(existing + "\n\n" + update_block)


def determine_research_action(stock, tier, force_refresh=False):
    """Determine what research action is needed for a stock.

    Returns one of: 'full', 'condensed', 'update', 'cache', 'none'
    """
    if force_refresh:
        return "full" if tier <= 1 else ("condensed" if tier == 2 else "none")

    cache_age = get_cache_age_days(stock)

    # Tier 3: never call API
    if tier == 3:
        return "none"

    # No cache: full or condensed based on tier
    if cache_age is None:
        return "full" if tier == 1 else "condensed"

    # Cache expired
    if cache_age > CACHE_MAX_AGE_DAYS:
        return "full" if tier == 1 else "condensed"

    # Tier 1 always gets full research (rank jump is new info)
    if tier == 1:
        return "full"

    # Tier 2 with valid cache: lightweight update
    if tier == 2:
        return "update"

    return "none"


# ─── Research Prompts ───────────────────────────────────────────────────────────


def build_full_research_prompt(stock, prev_snapshot, is_first_run, is_portfolio):
    """Build the system+user prompt for a full research pass."""
    name = stock["name"]
    flag = stock["flag"]
    exchange = get_exchange_label(flag)
    search_suffix = get_search_suffix(flag)

    rank_context = f"""Current ranks:
- Stock Rank: {stock['stock_rank']}
- Quality Rank: {stock['quality_rank']}
- Value Rank: {stock['value_rank']}
- Momentum Rank: {stock['momentum_rank']}
- QV Rank: {stock['qv_rank']}
- StockRank Style: {stock['stockrank_style']}
- Risk Rating: {stock['risk_rating']}
- Sector: {stock['sector']}
- Market Cap: {stock['mkt_cap']}m GBP
- Last Price: {stock['last_price']}"""

    if prev_snapshot and not is_first_run:
        rank_context += f"""

Previous snapshot ({prev_snapshot['timestamp_utc']}):
- Stock Rank: {prev_snapshot['stock_rank']}
- Quality Rank: {prev_snapshot['quality_rank']}
- Value Rank: {prev_snapshot['value_rank']}
- Momentum Rank: {prev_snapshot['momentum_rank']}
- QV Rank: {prev_snapshot['qv_rank']}
- StockRank Style: {prev_snapshot['stockrank_style']}"""

        qv_delta = stock["qv_rank"] - prev_snapshot["qv_rank"]
        if qv_delta > 0:
            rank_context += f"\n\nQV Rank has improved by {qv_delta} points since last snapshot. This is the key question: what has driven this improvement?"

    system_prompt = """You are a stock research analyst. Write in flowing prose, not bullet points. Always label currency (GBP or USD) on any financial figure. Be balanced — give equal weight to bull and bear cases. Be specific to this company, not generic."""

    if is_portfolio:
        action_instruction = "End with a HOLD or SELL recommendation with a one-paragraph rationale. Flag rank deterioration as a sell signal and rank improvement as strengthening the hold case."
    else:
        action_instruction = "End with a BUY, WATCH, or PASS recommendation with a one-paragraph rationale focused on whether any QV rank improvement is a credible entry signal."

    user_prompt = f"""Research {name} ({exchange}) for a UK investor.

{rank_context}

{"BASELINE RUN — no previous data available. Rank change tracking begins from next run." if is_first_run else ""}

Write a research report covering these six elements in flowing prose (400-500 words total):

1. RANK SENSE-CHECK: Do the Quality and Value ranks hold up against what is actually happening at the company? Are improving ranks justified by real fundamental change, or distorted by a one-off event?

2. CURRENT COMPANY NARRATIVE: What has happened in the last 3-6 months? Earnings, guidance, management changes, contract wins, strategic shifts. Look for the event or trend that explains any rank improvement.

3. MULTIBAGGER POTENTIAL: The bull case across three lenses — revenue growth (credible path to significantly higher sales?), margin expansion (operational leverage, pricing power?), and multiple re-rating (genuinely undervalued with room for market recognition?).

4. BEAR CASE: The 2-3 most credible specific risks right now. Specific to this company and sector, not generic market risk.

5. ESG AND SUSTAINABILITY: Material ESG factors for this company's sector. Any recent controversies. Is the ESG trajectory improving or deteriorating?

6. FINANCIAL HEALTH RED FLAGS: Brief scan — flag anything contradicting the ranks: unusual debt, cash flow concerns, recent dilution. If nothing flags, say so in one sentence.

{action_instruction}"""

    return system_prompt, user_prompt


def build_condensed_research_prompt(stock, prev_snapshot, is_first_run):
    """Build prompt for a condensed Tier 2 research pass."""
    name = stock["name"]
    exchange = get_exchange_label(stock["flag"])
    search_suffix = get_search_suffix(stock["flag"])

    rank_context = f"Quality: {stock['quality_rank']} | Value: {stock['value_rank']} | QV: {stock['qv_rank']} | Style: {stock['stockrank_style']} | Risk: {stock['risk_rating']} | Sector: {stock['sector']}"

    system_prompt = """You are a stock research analyst. Write in flowing prose, not bullet points. Always label currency (GBP or USD). Be concise."""

    user_prompt = f"""Condensed research on {name} ({exchange}) for a UK investor.

{rank_context}

{"BASELINE RUN — no previous data available." if is_first_run else ""}

Write a condensed assessment (~200 words) covering:
1. RANK SENSE-CHECK: Do the QV ranks match what's happening at the company?
2. MULTIBAGGER POTENTIAL: Brief bull case — revenue growth, margin expansion, re-rating potential.
3. FINAL ASSESSMENT: WATCH or PASS with brief rationale."""

    return system_prompt, user_prompt


def build_update_prompt(stock, existing_cache):
    """Build prompt for a lightweight cache update."""
    name = stock["name"]
    exchange = get_exchange_label(stock["flag"])
    search_suffix = get_search_suffix(stock["flag"])

    system_prompt = """You are a stock research analyst. Be extremely concise. One paragraph only."""

    user_prompt = f"""Lightweight update for {name} ({exchange}).

Current ranks: Quality {stock['quality_rank']} | Value {stock['value_rank']} | QV {stock['qv_rank']} | Style: {stock['stockrank_style']}

Write ONE concise paragraph covering only material changes since the last research. If nothing material has changed, say so in one sentence. Do not repeat background information."""

    return system_prompt, user_prompt


# ─── API Calls ──────────────────────────────────────────────────────────────────


def make_research_call(client, system_prompt, user_prompt, max_tokens, max_searches, use_search=False):
    """Make a research API call, optionally with web search. Retries on rate limit."""
    model = MODEL_SEARCH if use_search else MODEL_DEFAULT
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    if use_search and max_searches > 0:
        kwargs["tools"] = [{
            "type": WEB_SEARCH_TOOL_TYPE,
            "name": "web_search",
            "max_uses": max_searches,
        }]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)

            # Extract text from response
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            text = "\n".join(text_parts)

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "search_requests": 0,
            }
            if response.usage.server_tool_use:
                usage["search_requests"] = response.usage.server_tool_use.web_search_requests

            return text, usage

        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s
                print(f" rate limited, waiting {wait_time}s...", end="", flush=True)
                time.sleep(wait_time)
            else:
                raise


def calculate_cost(usage):
    """Calculate dollar cost from a usage dict."""
    if usage.get("search_requests", 0) > 0:
        # Sonnet pricing when web search was used
        input_price = PRICE_INPUT_SONNET
        output_price = PRICE_OUTPUT_SONNET
    else:
        # Haiku pricing
        input_price = PRICE_INPUT_HAIKU
        output_price = PRICE_OUTPUT_HAIKU
    return (
        usage["input_tokens"] * input_price
        + usage["output_tokens"] * output_price
        + usage.get("search_requests", 0) * PRICE_PER_SEARCH
    )


def research_stock(client, stock, action, prev_snapshot, is_first_run, is_portfolio, delay, use_search=False):
    """Research a single stock. Returns (research_text, usage_dict) or raises."""
    if action == "full":
        system_prompt, user_prompt = build_full_research_prompt(
            stock, prev_snapshot, is_first_run, is_portfolio
        )
        text, usage = make_research_call(client, system_prompt, user_prompt,
                                         TOKENS_FULL, SEARCHES_FULL, use_search=use_search)
        # Write to cache
        timestamp = utc_now().strftime("%Y-%m-%d")
        header = f"# {stock['name']} ({stock['ticker']} — {get_exchange_label(stock['flag'])})\n## Research — {timestamp} UTC\n\n"
        existing = read_cache(stock)
        if existing:
            # Append as new dated block
            append_cache(stock, f"\n## Full Research — {timestamp} UTC\n\n{text}")
        else:
            write_cache(stock, header + text)
        return text, usage

    elif action == "condensed":
        system_prompt, user_prompt = build_condensed_research_prompt(
            stock, prev_snapshot, is_first_run
        )
        text, usage = make_research_call(client, system_prompt, user_prompt,
                                         TOKENS_CONDENSED, SEARCHES_CONDENSED, use_search=use_search)
        timestamp = utc_now().strftime("%Y-%m-%d")
        existing = read_cache(stock)
        if existing:
            append_cache(stock, f"\n## Condensed Research — {timestamp} UTC\n\n{text}")
        else:
            header = f"# {stock['name']} ({stock['ticker']} — {get_exchange_label(stock['flag'])})\n"
            write_cache(stock, header + f"## Condensed Research — {timestamp} UTC\n\n{text}")
        return text, usage

    elif action == "update":
        existing_cache = read_cache(stock) or ""
        system_prompt, user_prompt = build_update_prompt(stock, existing_cache)
        text, usage = make_research_call(client, system_prompt, user_prompt,
                                         TOKENS_UPDATE, SEARCHES_UPDATE, use_search=use_search)
        timestamp = utc_now().strftime("%Y-%m-%d")
        append_cache(stock, f"### Update — {timestamp} UTC\n{text}")
        return text, usage

    return "", {"input_tokens": 0, "output_tokens": 0, "search_requests": 0}


# ─── Cost Estimation & Trial Mode ──────────────────────────────────────────────


def estimate_full_run_cost(trial_usage, trial_stock_count, run_plan):
    """Estimate full run cost based on trial results."""
    if trial_stock_count == 0:
        return 0.0

    avg_cost_per_full = calculate_cost({
        "input_tokens": trial_usage["input_tokens"] // trial_stock_count,
        "output_tokens": trial_usage["output_tokens"] // trial_stock_count,
        "search_requests": trial_usage["search_requests"] // trial_stock_count,
    })

    # Scale condensed and update costs relative to full
    condensed_ratio = TOKENS_CONDENSED / TOKENS_FULL
    update_ratio = TOKENS_UPDATE / TOKENS_FULL

    cost = 0.0
    cost += run_plan["full_research"] * avg_cost_per_full
    cost += run_plan["condensed"] * avg_cost_per_full * condensed_ratio
    cost += run_plan["updates"] * avg_cost_per_full * update_ratio
    # Add summary generation cost estimate
    cost += avg_cost_per_full * (TOKENS_SUMMARY / TOKENS_FULL)

    return cost


def build_run_plan(unique_stocks, tiers, history, is_first_run, refresh_ticker, refresh_all, tier1_only=False):
    """Build a plan of what research actions are needed. Returns categorised counts and action map."""
    action_map = {}  # key -> action
    counts = {
        "full_research": 0,
        "condensed": 0,
        "updates": 0,
        "cache_only": 0,
        "no_api": 0,
    }

    for key, stock in unique_stocks.items():
        is_portfolio = stock["in_portfolio"]
        is_watchlist = stock["in_watchlist"]

        force = refresh_all or (refresh_ticker and stock["ticker"] == refresh_ticker)

        if is_portfolio:
            # Portfolio always gets full research
            cache_age = get_cache_age_days(stock)
            if force or cache_age is None or cache_age > CACHE_MAX_AGE_DAYS:
                action_map[key] = "full"
                counts["full_research"] += 1
            else:
                action_map[key] = "update"
                counts["updates"] += 1

        if is_watchlist:
            tier = tiers.get(key, 3)
            # In tier1-only mode, skip API calls for Tier 2/3 watchlist-only stocks
            if tier1_only and tier > 1 and not is_portfolio:
                action = "none"
            elif force:
                action = "full" if tier <= 1 else ("condensed" if tier == 2 else "none")
            else:
                action = determine_research_action(stock, tier, force_refresh=False)

            # Don't double-count if already planned as portfolio
            if key in action_map:
                # Use the more thorough action
                existing = action_map[key]
                if _action_priority(action) > _action_priority(existing):
                    # Adjust counts
                    counts[_action_count_key(existing)] -= 1
                    action_map[key] = action
                    counts[_action_count_key(action)] += 1
            else:
                action_map[key] = action
                counts[_action_count_key(action)] += 1

    return counts, action_map


def _action_priority(action):
    """Return priority for research actions (higher = more thorough)."""
    return {"full": 3, "condensed": 2, "update": 1, "none": 0}.get(action, 0)


def _action_count_key(action):
    """Map action to count key."""
    return {
        "full": "full_research",
        "condensed": "condensed",
        "update": "updates",
        "none": "no_api",
    }.get(action, "no_api")


def run_trial(client, unique_stocks, tiers, history, is_first_run, delay, use_search=False):
    """Run trial on 2 stocks. Returns total usage dict."""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "search_requests": 0}

    # Pick one portfolio stock and one Tier 1 watchlist stock
    trial_stocks = []

    # Find a portfolio stock
    for key, stock in unique_stocks.items():
        if stock["in_portfolio"]:
            trial_stocks.append((key, stock, "full", True))
            break

    # Find a Tier 1 watchlist stock
    for key, stock in unique_stocks.items():
        if stock["in_watchlist"] and tiers.get(key) == 1:
            # Skip if already selected as portfolio stock
            if trial_stocks and trial_stocks[0][0] == key:
                continue
            trial_stocks.append((key, stock, "full", False))
            break

    # If we couldn't find both, fill with whatever we can
    if len(trial_stocks) < 2:
        for key, stock in unique_stocks.items():
            if len(trial_stocks) >= 2:
                break
            if any(t[0] == key for t in trial_stocks):
                continue
            trial_stocks.append((key, stock, "full", stock["in_portfolio"]))

    print(f"\nRunning trial on {len(trial_stocks)} stocks...")

    for i, (key, stock, action, is_portfolio) in enumerate(trial_stocks):
        ticker = stock["ticker"]
        name = stock["name"]
        exchange = get_exchange_label(stock["flag"])
        label = "full research"
        print(f"[Trial {i+1}/{len(trial_stocks)}]  {ticker:<6s}({name}, {exchange})  [{label}]")

        prev = get_previous_snapshot(history, stock)
        try:
            text, usage = research_stock(client, stock, action, prev,
                                         is_first_run, is_portfolio, delay,
                                         use_search=use_search)
            for k in total_usage:
                total_usage[k] += usage[k]
        except Exception as e:
            print(f"  Trial error: {e}")

        if i < len(trial_stocks) - 1:
            time.sleep(delay)

    return total_usage, len(trial_stocks)


# ─── Report Generation ─────────────────────────────────────────────────────────


def generate_output_filenames():
    """Generate output filenames with local time. Handle same-minute collision."""
    now = local_now()
    prefix = now.strftime("%Y-%m-%d_%H%M")

    # Check for collision
    existing = list(OUTPUT_DIR.glob(f"{prefix}_*.md"))
    if existing:
        prefix = prefix + "_2"

    return {
        "summary": OUTPUT_DIR / f"{prefix}_summary.md",
        "portfolio": OUTPUT_DIR / f"{prefix}_portfolio.md",
        "watchlist": OUTPUT_DIR / f"{prefix}_watchlist.md",
    }


def cleanup_old_outputs():
    """Keep only the last OUTPUT_MAX_RUNS sets of output files."""
    # Group files by their timestamp prefix
    all_files = sorted(OUTPUT_DIR.glob("*.md"))
    prefixes = set()
    for f in all_files:
        # Extract prefix: everything before _summary/_portfolio/_watchlist
        name = f.stem
        for suffix in ("_summary", "_portfolio", "_watchlist"):
            if name.endswith(suffix):
                prefixes.add(name[: -len(suffix)])
                break

    sorted_prefixes = sorted(prefixes)
    if len(sorted_prefixes) > OUTPUT_MAX_RUNS:
        to_delete = sorted_prefixes[: -OUTPUT_MAX_RUNS]
        for prefix in to_delete:
            for suffix in ("_summary", "_portfolio", "_watchlist"):
                path = OUTPUT_DIR / f"{prefix}{suffix}.md"
                if path.exists():
                    path.unlink()


def build_portfolio_stock_section(stock, research_text, prev_snapshot, is_first_run):
    """Build markdown section for one portfolio stock."""
    name = stock["name"]
    ticker = stock["ticker"]
    exchange = get_exchange_label(stock["flag"])

    # Determine recommendation from research text
    rec = "HOLD"
    text_upper = research_text.upper() if research_text else ""
    if "SELL" in text_upper and "HOLD" not in text_upper:
        rec = "SELL"

    lines = [f"### {name} ({ticker} — {exchange}) — {rec}"]

    # Rank deltas
    if prev_snapshot and not is_first_run:
        q_delta = format_rank_delta(prev_snapshot["quality_rank"], stock["quality_rank"])
        v_delta = format_rank_delta(prev_snapshot["value_rank"], stock["value_rank"])
        sr_delta = format_rank_delta(prev_snapshot["stock_rank"], stock["stock_rank"])
        q_arrow = format_rank_arrow(prev_snapshot["quality_rank"], stock["quality_rank"])
        v_arrow = format_rank_arrow(prev_snapshot["value_rank"], stock["value_rank"])
        sr_arrow = format_rank_arrow(prev_snapshot["stock_rank"], stock["stock_rank"])
        lines.append(f"#### Quality: {q_delta} {q_arrow} | Value: {v_delta} {v_arrow} | Stock Rank: {sr_delta} {sr_arrow}")
    else:
        lines.append(f"#### Quality: {stock['quality_rank']} | Value: {stock['value_rank']} | Stock Rank: {stock['stock_rank']}")
        if is_first_run:
            lines.append("*Baseline run — no previous data. Rank change tracking begins from next run.*")

    if research_text:
        lines.append(research_text)
    else:
        lines.append("*Research unavailable. Retry with: --refresh " + ticker + "*")

    lines.append("\n---\n")
    return "\n".join(lines)


def build_watchlist_tier1_section(stock, research_text, prev_snapshot, is_first_run):
    """Build markdown section for one Tier 1 watchlist stock."""
    name = stock["name"]
    ticker = stock["ticker"]
    exchange = get_exchange_label(stock["flag"])

    # Determine recommendation
    rec = "WATCH"
    text_upper = research_text.upper() if research_text else ""
    if "BUY" in text_upper and "PASS" not in text_upper:
        rec = "BUY"
    elif "PASS" in text_upper and "BUY" not in text_upper:
        rec = "PASS"

    lines = [f"### {name} ({ticker} — {exchange}) — {rec}"]

    if prev_snapshot and not is_first_run:
        q_delta = format_rank_delta(prev_snapshot["quality_rank"], stock["quality_rank"])
        v_delta = format_rank_delta(prev_snapshot["value_rank"], stock["value_rank"])
        qv_delta = format_rank_delta(prev_snapshot["qv_rank"], stock["qv_rank"])
        lines.append(f"#### Quality: {q_delta} | Value: {v_delta} | QV: {qv_delta} | {stock['stockrank_style']}")
    else:
        lines.append(f"#### Quality: {stock['quality_rank']} | Value: {stock['value_rank']} | QV: {stock['qv_rank']} | {stock['stockrank_style']}")
        if is_first_run:
            lines.append("*Baseline run — no previous data. Rank change tracking begins from next run.*")

    if research_text:
        lines.append(research_text)
    else:
        lines.append(f"*Research unavailable due to API error. Retry with: --refresh {ticker}*")

    lines.append("\n---\n")
    return "\n".join(lines)


def build_watchlist_tier2_section(stock, research_text, prev_snapshot, is_first_run):
    """Build markdown section for one Tier 2 watchlist stock."""
    name = stock["name"]
    ticker = stock["ticker"]
    exchange = get_exchange_label(stock["flag"])

    rec = "WATCH"
    text_upper = research_text.upper() if research_text else ""
    if "PASS" in text_upper:
        rec = "PASS"

    lines = [f"### {name} ({ticker} — {exchange}) — {rec}"]
    lines.append(f"#### Quality: {stock['quality_rank']} | Value: {stock['value_rank']} | QV: {stock['qv_rank']}")

    if is_first_run:
        lines.append("*Baseline run — no previous data. Rank change tracking begins from next run.*")

    if research_text:
        lines.append(research_text)
    else:
        cached = read_cache(stock)
        if cached:
            # Extract last update or summary
            lines.append(cached[-500:] if len(cached) > 500 else cached)
        else:
            lines.append(f"*No research available. Retry with: --refresh {ticker}*")

    lines.append("\n---\n")
    return "\n".join(lines)


def build_watchlist_tier3_line(stock):
    """Build one-line summary for Tier 3 stock from cache."""
    name = stock["name"]
    ticker = stock["ticker"]
    style = stock["stockrank_style"]
    qv = stock["qv_rank"]

    cached = read_cache(stock)
    summary = "No cached research available."
    if cached:
        # Extract first meaningful sentence from cache
        for line in cached.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("*") and len(line) > 20:
                summary = line[:150]
                if len(line) > 150:
                    summary += "..."
                break

    return f"### {name} ({ticker}) — {style} | QV: {qv}\n{summary}\n"


def generate_summary_file(filepath, unique_stocks, tiers, research_results,
                          history, is_first_run, screen_name, prev_history):
    """Generate the summary markdown file."""
    now_local = local_now()
    now_utc = utc_now()

    lines = [
        "# Stock Analysis — Daily Summary",
        f"## {now_local.strftime('%A %d %B %Y')} {now_local.strftime('%H:%M')} (UTC: {now_utc.strftime('%Y-%m-%d %H:%M')})",
        "",
    ]

    # Today's Rank Movers — Tier 1 watchlist only
    lines.append("## Today's Rank Movers — Act Now")
    tier1_stocks = []
    for key, stock in unique_stocks.items():
        if stock["in_watchlist"] and tiers.get(key) == 1:
            prev = None
            prev_key = f"{stock['flag']}_{stock['ticker']}"
            if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
                prev = prev_history[prev_key]["snapshots"][-1]
            tier1_stocks.append((key, stock, prev))

    # Sort by QV delta descending
    def qv_sort_key(item):
        _, stock, prev = item
        if prev and not is_first_run:
            return stock["qv_rank"] - prev["qv_rank"]
        return stock["qv_rank"]

    tier1_stocks.sort(key=qv_sort_key, reverse=True)

    if not tier1_stocks:
        lines.append("*No significant rank movers today.*\n")
    else:
        for key, stock, prev in tier1_stocks:
            exchange = get_exchange_label(stock["flag"])
            if prev and not is_first_run:
                qv_str = format_rank_delta(prev["qv_rank"], stock["qv_rank"])
            else:
                qv_str = str(stock["qv_rank"])
            style = stock["stockrank_style"]

            # Get one-line summary from research
            research = research_results.get(key, "")
            one_line = ""
            if research:
                for sent in research.split(". "):
                    sent = sent.strip()
                    if len(sent) > 30:
                        one_line = sent.rstrip(".") + "."
                        break
            if not one_line:
                one_line = "Research pending."

            lines.append(f"**{stock['name']}** ({stock['ticker']}, {exchange}) | QV: {qv_str} | {style}")
            lines.append(f"  {one_line}\n")

    # Portfolio Flags
    lines.append("## Portfolio Flags")
    portfolio_flags = []
    for key, stock in unique_stocks.items():
        if not stock["in_portfolio"]:
            continue
        prev_key = f"{stock['flag']}_{stock['ticker']}"
        if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
            prev = prev_history[prev_key]["snapshots"][-1]
            sr_delta = stock["stock_rank"] - prev["stock_rank"]
            qv_delta = stock["qv_rank"] - prev["qv_rank"]
            if abs(sr_delta) >= 5 or abs(qv_delta) >= 5:
                direction = "improving" if qv_delta > 0 else "deteriorating"
                portfolio_flags.append(
                    f"**{stock['name']}** ({stock['ticker']}): "
                    f"Stock Rank {format_rank_delta(prev['stock_rank'], stock['stock_rank'])}, "
                    f"QV {format_rank_delta(prev['qv_rank'], stock['qv_rank'])} — {direction}"
                )

    if portfolio_flags:
        lines.extend(portfolio_flags)
    else:
        lines.append("*No material rank moves in portfolio since last run.*")
    lines.append("")

    # Since Last Run
    lines.append("## Since Last Run")
    if is_first_run:
        lines.append("*Baseline run — all stocks recorded for the first time.*")
    else:
        changes = []
        # Check for style changes
        for key, stock in unique_stocks.items():
            prev_key = f"{stock['flag']}_{stock['ticker']}"
            if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
                prev = prev_history[prev_key]["snapshots"][-1]
                if prev.get("stockrank_style") != stock["stockrank_style"]:
                    changes.append(
                        f"**{stock['name']}** style changed: {prev.get('stockrank_style', '?')} → {stock['stockrank_style']}"
                    )
        if changes:
            lines.extend(changes)
        else:
            lines.append("*No material changes since last run.*")
    lines.append("")

    # Executive Summary placeholder
    lines.append("## Executive Summary")
    lines.append(f"This analysis covers a portfolio and the {screen_name} watchlist screen. ")
    if is_first_run:
        lines.append("This is the baseline run establishing rank history for future comparison. "
                      "All stocks have been researched and cached. From the next run onwards, "
                      "the system will track rank changes and focus research on the top movers.")
    else:
        t1_count = sum(1 for t in tiers.values() if t == 1)
        t2_count = sum(1 for t in tiers.values() if t == 2)
        t3_count = sum(1 for t in tiers.values() if t == 3)
        lines.append(f"Today's run identified {t1_count} Tier 1 stocks for full research, "
                      f"{t2_count} Tier 2 stocks for condensed review, and {t3_count} Tier 3 stocks on the radar.")
    lines.append("")

    filepath.write_text("\n".join(lines))


def generate_portfolio_file(filepath, unique_stocks, research_results,
                            history, is_first_run, prev_history):
    """Generate the portfolio markdown file."""
    now_local = local_now()
    now_utc = utc_now()

    lines = [
        "# Stock Analysis — Portfolio Holdings",
        f"## {now_local.strftime('%A %d %B %Y')} {now_local.strftime('%H:%M')} (UTC: {now_utc.strftime('%Y-%m-%d %H:%M')})",
        "",
    ]

    portfolio_stocks = [(k, s) for k, s in unique_stocks.items() if s["in_portfolio"]]
    portfolio_stocks.sort(key=lambda x: x[1]["stock_rank"], reverse=True)

    for key, stock in portfolio_stocks:
        prev = None
        prev_key = f"{stock['flag']}_{stock['ticker']}"
        if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
            prev = prev_history[prev_key]["snapshots"][-1]

        research = research_results.get(key, "")
        section = build_portfolio_stock_section(stock, research, prev, is_first_run)
        lines.append(section)

    filepath.write_text("\n".join(lines))


def generate_watchlist_file(filepath, unique_stocks, tiers, research_results,
                            history, is_first_run, screen_name, prev_history):
    """Generate the watchlist markdown file."""
    now_local = local_now()
    now_utc = utc_now()

    lines = [
        f"# Stock Analysis — Watchlist: {screen_name}",
        f"## {now_local.strftime('%A %d %B %Y')} {now_local.strftime('%H:%M')} (UTC: {now_utc.strftime('%Y-%m-%d %H:%M')})",
        "",
    ]

    watchlist_stocks = {k: s for k, s in unique_stocks.items() if s["in_watchlist"]}

    # ── Tier 1 ──
    lines.append("---")
    lines.append("## Tier 1 — Top Rank Movers (Full Research)\n")

    tier1 = [(k, s) for k, s in watchlist_stocks.items() if tiers.get(k) == 1]
    # Sort by QV delta descending
    def t1_sort(item):
        k, s = item
        prev_key = f"{s['flag']}_{s['ticker']}"
        if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
            prev = prev_history[prev_key]["snapshots"][-1]
            return s["qv_rank"] - prev["qv_rank"]
        return s["qv_rank"]

    tier1.sort(key=t1_sort, reverse=True)

    for key, stock in tier1:
        prev = None
        prev_key = f"{stock['flag']}_{stock['ticker']}"
        if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
            prev = prev_history[prev_key]["snapshots"][-1]
        research = research_results.get(key, "")
        lines.append(build_watchlist_tier1_section(stock, research, prev, is_first_run))

    if not tier1:
        lines.append("*No Tier 1 stocks this run.*\n")

    # ── Tier 2 ──
    lines.append("---")
    lines.append("## Tier 2 — Strong and Stable (Condensed)\n")

    tier2 = [(k, s) for k, s in watchlist_stocks.items() if tiers.get(k) == 2]
    tier2.sort(key=lambda x: x[1]["qv_rank"], reverse=True)

    for key, stock in tier2:
        prev = None
        prev_key = f"{stock['flag']}_{stock['ticker']}"
        if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
            prev = prev_history[prev_key]["snapshots"][-1]
        research = research_results.get(key, "")
        lines.append(build_watchlist_tier2_section(stock, research, prev, is_first_run))

    if not tier2:
        lines.append("*No Tier 2 stocks this run.*\n")

    # ── Tier 3 ──
    lines.append("---")
    lines.append("## Tier 3 — On the Radar\n")

    tier3 = [(k, s) for k, s in watchlist_stocks.items() if tiers.get(k) == 3]
    tier3.sort(key=lambda x: x[1]["stock_rank"], reverse=True)

    for key, stock in tier3:
        lines.append(build_watchlist_tier3_line(stock))

    if not tier3:
        lines.append("*No Tier 3 stocks this run.*\n")

    filepath.write_text("\n".join(lines))


# ─── CLI & Main ─────────────────────────────────────────────────────────────────


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Portfolio & Watchlist Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--portfolio", required=True, help="Path to portfolio CSV")
    parser.add_argument("--watchlist", required=True, help="Path to watchlist CSV")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top QV rank improvers for Tier 1 (default: 5)")
    parser.add_argument("--skip-trial", action="store_true",
                        help="Skip trial mode for repeat same-day runs")
    parser.add_argument("--max-cost", type=float, default=2.00,
                        help="Maximum cost cap in USD (default: $2.00)")
    parser.add_argument("--refresh", type=str, default=None,
                        help="Force refresh a single stock by ticker")
    parser.add_argument("--refresh-all", action="store_true",
                        help="Force refresh all stocks")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between API calls in seconds (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate reports with mock data (no API key needed)")
    parser.add_argument("--tier1-only", action="store_true",
                        help="Only research portfolio + Tier 1 watchlist stocks (skip Tier 2/3 API calls)")
    parser.add_argument("--with-search", action="store_true",
                        help="Enable live web search (uses Sonnet, slower and costlier)")
    return parser.parse_args()


def generate_mock_research(stock, action, is_portfolio):
    """Generate realistic mock research text for --dry-run mode."""
    name = stock["name"]
    sector = stock["sector"]
    style = stock["stockrank_style"]
    qv = stock["qv_rank"]
    quality = stock["quality_rank"]
    value = stock["value_rank"]
    risk = stock["risk_rating"]
    mkt_cap = stock["mkt_cap"]
    currency = "GBP" if stock["flag"] == "gb" else "USD"

    if action == "none":
        return f"{name} remains on the radar with a {style} profile. No material developments."

    if action == "condensed":
        return (
            f"{name} carries a QV Rank of {qv}, reflecting a Quality score of {quality} "
            f"and Value score of {value}. The {sector} company trades at a market capitalisation "
            f"of {mkt_cap}m {currency} and holds a {risk} risk rating. "
            f"The ranks appear broadly justified by the company's fundamentals, with no obvious "
            f"distortions from one-off events. The bull case rests on potential margin expansion "
            f"as the business scales, combined with what looks like genuine undervaluation relative "
            f"to quality peers in the {sector} space. However, at this stage the catalysts for a "
            f"re-rating remain uncertain, and the stock warrants monitoring rather than immediate action. "
            f"WATCH — ranks are solid but a clearer catalyst is needed before committing capital."
        )

    # Full research
    if is_portfolio:
        rec = "HOLD"
        rec_text = (
            f"HOLD. {name} continues to justify its place in the portfolio. The Quality Rank of "
            f"{quality} reflects genuine operational strength, and the Value Rank of {value} suggests "
            f"the market has not yet fully priced in the company's quality. The {style} classification "
            f"and overall QV Rank of {qv} reinforce the investment thesis. No deterioration in ranks "
            f"that would trigger a sell signal."
        )
    else:
        rec = "BUY" if qv >= 90 else "WATCH"
        if rec == "BUY":
            rec_text = (
                f"BUY. The QV Rank of {qv} represents a compelling entry signal. Quality and Value "
                f"are both elevated, Momentum has not yet caught up, and the narrative supports a "
                f"credible re-rating story. The {risk} risk profile is acceptable given the upside potential."
            )
        else:
            rec_text = (
                f"WATCH. The QV Rank of {qv} is encouraging but not yet compelling enough for immediate "
                f"action. Quality at {quality} is solid, but the Value score of {value} suggests the market "
                f"may already be partially pricing in the quality. Monitor for further rank improvement."
            )

    return (
        f"The Quality Rank of {quality} and Value Rank of {value} for {name} appear well-supported by "
        f"the company's underlying fundamentals. Operating in the {sector} sector with a market cap of "
        f"{mkt_cap}m {currency}, the company has demonstrated consistent execution. The {style} "
        f"classification from Stockopedia aligns with what we see in the financials — this is a business "
        f"where quality metrics are genuinely strong rather than artificially inflated by one-off items.\n\n"
        f"Over the past three to six months, {name} has continued to execute against its strategic plan. "
        f"Recent trading updates have been broadly in line with expectations, with management maintaining "
        f"guidance. The {sector} sector has seen mixed conditions, but {name} has navigated these "
        f"effectively, suggesting operational resilience that justifies the elevated Quality Rank.\n\n"
        f"The multibagger case for {name} rests on three pillars. First, revenue growth potential looks "
        f"credible as the company expands its addressable market. Second, there is a realistic path to "
        f"margin expansion through operational leverage as fixed costs are spread across a growing revenue "
        f"base. Third, the valuation remains undemanding relative to quality — the market appears to be "
        f"applying a discount that does not reflect the company's true earnings power.\n\n"
        f"The bear case centres on two specific risks. The {sector} sector faces potential headwinds from "
        f"regulatory changes that could compress margins. Additionally, the company's relatively modest "
        f"market capitalisation of {mkt_cap}m {currency} means liquidity risk in a downturn is real — "
        f"the bid-offer spread can widen significantly during periods of market stress.\n\n"
        f"From an ESG perspective, {name} operates within the norms for its {sector} peer group. "
        f"No material controversies have emerged recently. Governance appears adequate with no red flags "
        f"in board composition or executive compensation structures.\n\n"
        f"No financial health red flags are apparent. The balance sheet looks clean with manageable debt "
        f"levels and adequate cash flow generation. No recent dilution or unusual financing activity.\n\n"
        f"{rec_text}"
    )


def main():
    """Main entry point."""
    print("Stock Portfolio & Watchlist Analyzer")
    print("=" * 37)

    # Check Python version
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}+ check: OK")

    args = parse_args()
    dry_run = args.dry_run

    # Check API key: try environment variable first, then .env file
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not api_key and not dry_run:
        sys.exit(
            "Error: Anthropic API key not found.\n\n"
            "Option 1 — Create a .env file in the project folder:\n"
            f"  {BASE_DIR / '.env'}\n"
            "  with the line: ANTHROPIC_API_KEY=sk-ant-your-key-here\n\n"
            "Option 2 — Set an environment variable:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-your-key-here'"
        )

    if dry_run:
        print("MODE: Dry run (mock data, no API calls)")

    # Ensure directories exist
    print("Creating directories...", end=" ", flush=True)
    ensure_dirs()
    print("OK")

    # Load and validate CSVs
    print("Validating CSVs...", end=" ", flush=True)
    portfolio_df, watchlist_df, unique_stocks = load_stocks(args.portfolio, args.watchlist)

    # Derive screen name from watchlist filename
    screen_name = derive_screen_name(args.watchlist)
    print(f"Screen name: {screen_name}")

    # Load history
    print("Loading rank history...", end=" ", flush=True)
    history = load_history()
    prev_history = json.loads(json.dumps(history)) if history else None  # deep copy
    is_first_run = len(history) == 0
    print(f"{'No history found (first run)' if is_first_run else f'{len(history)} stocks in history'}")

    print(f"Timezone: Europe/London (all internal timestamps in UTC)")

    # Assign tiers
    print("Assigning watchlist tiers...", end=" ", flush=True)
    tiers = assign_tiers(unique_stocks, history, args.top, is_first_run)
    print("done")

    # Print tier breakdown
    t1_count = sum(1 for t in tiers.values() if t == 1)
    t2_count = sum(1 for t in tiers.values() if t == 2)
    t3_count = sum(1 for t in tiers.values() if t == 3)

    if is_first_run:
        print(f"\nFirst run — no history available. Using absolute QV ranks for tier assignment.")
    else:
        last_ts = None
        for entry in history.values():
            if entry["snapshots"]:
                ts = entry["snapshots"][-1]["timestamp_utc"]
                if last_ts is None or ts > last_ts:
                    last_ts = ts
        print(f"\nRank change analysis (vs last run {last_ts or 'unknown'}):")

    print(f"  Tier 1 (top {args.top} by QV improvement):    {t1_count} stocks  <- today's focus")
    print(f"  Tier 2 (QV rank > {TIER2_QV_THRESHOLD}, stable):       {t2_count} stocks")
    print(f"  Tier 3 (everything else):            {t3_count} stocks")

    # Build run plan
    print("\nBuilding run plan...", end=" ", flush=True)
    counts, action_map = build_run_plan(
        unique_stocks, tiers, history, is_first_run,
        args.refresh, args.refresh_all, tier1_only=args.tier1_only
    )
    print("done")
    print(f"  Full research:     {counts['full_research']}")
    print(f"  Condensed:         {counts['condensed']}")
    print(f"  Lightweight update:{counts['updates']}")
    print(f"  No API call:       {counts['no_api']}")

    # Initialize API client (skip for dry run)
    client = None
    if not dry_run:
        print("\nInitialising API client...", end=" ", flush=True)
        client = anthropic.Anthropic(api_key=api_key)
        print("OK")

    # Trial mode (skip for dry run)
    if not dry_run and not args.skip_trial:
        trial_usage, trial_count = run_trial(
            client, unique_stocks, tiers, history, is_first_run, args.delay
        )
        trial_cost = calculate_cost(trial_usage)

        print(f"\nTrial complete ({trial_count} stocks researched).")
        print(f"Actual cost for {trial_count} stocks: ${trial_cost:.2f}")

        estimated_cost = estimate_full_run_cost(trial_usage, trial_count, counts)

        portfolio_full = sum(1 for k, s in unique_stocks.items()
                            if s["in_portfolio"] and action_map.get(k) == "full")

        print(f"\nFull run breakdown:")
        print(f"  Portfolio full research:    {portfolio_full} stocks")
        print(f"  Tier 1 full research:        {t1_count} stocks  (top {args.top} by QV rank improvement)")
        print(f"  Tier 2 condensed:           {counts['condensed']} stocks")
        print(f"  Tier 3 no API call:         {counts['no_api']} stocks")
        print(f"  Lightweight cache updates:   {counts['updates']} stocks")
        cache_count = sum(1 for k, a in action_map.items() if a == "none")
        print(f"  Served from cache:          {cache_count} stocks")
        print(f"\nEstimated full run cost: ~${estimated_cost:.2f}")
        print(f"Cost cap: ${args.max_cost:.2f}")

        if estimated_cost > args.max_cost:
            sys.exit(f"\nEstimated cost (${estimated_cost:.2f}) exceeds cap (${args.max_cost:.2f}).\n"
                     f"Use --refresh TICKER for individual stocks or increase --max-cost.")

        response = input("\nProceed with full run? (y/n): ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # ── Full run ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN' if dry_run else 'FULL RUN'} — Processing {len(unique_stocks)} unique stocks")
    print(f"{'=' * 60}\n")

    research_results = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0, "search_requests": 0}
    succeeded = 0
    failed = 0
    failed_tickers = []

    # Build ordered list: portfolio first, then watchlist by tier
    process_order = []
    for key, stock in unique_stocks.items():
        if stock["in_portfolio"]:
            process_order.append(key)
    for tier_num in [1, 2, 3]:
        for key, stock in unique_stocks.items():
            if stock["in_watchlist"] and tiers.get(key) == tier_num:
                if key not in process_order:
                    process_order.append(key)

    total_to_process = len(process_order)

    for i, key in enumerate(process_order):
        stock = unique_stocks[key]
        action = action_map.get(key, "none")
        ticker = stock["ticker"]
        name = stock["name"]
        exchange = get_exchange_label(stock["flag"])
        tier = tiers.get(key, "-")
        is_portfolio = stock["in_portfolio"]

        # Build progress label
        if is_portfolio:
            tier_label = "portfolio — full"
        elif action == "full":
            tier_label = f"Tier {tier} — full"
            prev_key = f"{stock['flag']}_{stock['ticker']}"
            if prev_history and prev_key in prev_history and prev_history[prev_key]["snapshots"]:
                prev = prev_history[prev_key]["snapshots"][-1]
                qv_d = stock["qv_rank"] - prev["qv_rank"]
                tier_label += f"  QV: {prev['qv_rank']}->{stock['qv_rank']} ({'+' if qv_d > 0 else ''}{qv_d})"
        elif action == "condensed":
            tier_label = f"Tier {tier} — condensed"
        elif action == "update":
            tier_label = "cache update"
        else:
            tier_label = "cache"

        print(f"[{i+1}/{total_to_process}]  {ticker:<6s}({name}, {exchange})  [{tier_label}]", end="", flush=True)

        if action == "none":
            cached = read_cache(stock)
            if cached:
                research_results[key] = cached
                print("  -> served from cache")
            else:
                print("  -> no cache (Tier 3, skipped)")
            succeeded += 1
            continue

        if dry_run:
            # Generate mock research
            text = generate_mock_research(stock, action, is_portfolio)
            research_results[key] = text
            # Write to cache for dry run too
            timestamp = utc_now().strftime("%Y-%m-%d")
            header = f"# {name} ({ticker} — {exchange})\n## Research — {timestamp} UTC\n\n"
            write_cache(stock, header + text)
            print(f"  -> mock {action} research generated")
            succeeded += 1
            continue

        prev = get_previous_snapshot(history, stock)

        print(f"  -> calling API...", end="", flush=True)
        try:
            text, usage = research_stock(
                client, stock, action, prev,
                is_first_run, is_portfolio, args.delay,
                use_search=args.with_search
            )
            research_results[key] = text
            cost = calculate_cost(usage)
            for k in total_usage:
                total_usage[k] += usage[k]
            succeeded += 1
            print(f" done (${cost:.3f}, {usage['output_tokens']} tokens, {usage['search_requests']} searches)")
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            log_error(ticker, err_msg)
            print(f" FAILED: {err_msg}")
            failed += 1
            failed_tickers.append(ticker)
            research_results[key] = ""

        # Rate limiting
        if i < total_to_process - 1:
            time.sleep(args.delay)

    # Update history with new snapshots
    print(f"\nSaving rank snapshots to history...", end=" ", flush=True)
    for key, stock in unique_stocks.items():
        add_snapshot(history, stock)
    save_history(history)
    print("done")

    # Generate reports
    print("Generating reports...", end=" ", flush=True)
    output_files = generate_output_filenames()

    generate_summary_file(
        output_files["summary"], unique_stocks, tiers, research_results,
        history, is_first_run, screen_name, prev_history
    )
    generate_portfolio_file(
        output_files["portfolio"], unique_stocks, research_results,
        history, is_first_run, prev_history
    )
    generate_watchlist_file(
        output_files["watchlist"], unique_stocks, tiers, research_results,
        history, is_first_run, screen_name, prev_history
    )
    print("done")

    # Cleanup old outputs
    cleanup_old_outputs()

    # Final summary
    total_cost = calculate_cost(total_usage)
    print(f"\n{'=' * 60}")
    print(f"Run complete. {succeeded} succeeded, {failed} failed.")
    if failed_tickers:
        print(f"Failed: {', '.join(failed_tickers)} — see {ERROR_LOG}")
    if not dry_run:
        print(f"Total cost: ${total_cost:.2f}")
    else:
        print("Total cost: $0.00 (dry run)")
    print(f"\nReports saved:")
    for label, path in output_files.items():
        print(f"  {path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
