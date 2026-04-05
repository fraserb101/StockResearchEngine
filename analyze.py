#!/usr/bin/env python3
"""Stock Research Engine.

Single CSV input. Three-pass AI research using claude-haiku-4-5-20251001.
One output file per run.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd

# ── Python version check ────────────────────────────────────────────────────────
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Error: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
             f"You have {sys.version_info.major}.{sys.version_info.minor}.")

# ── Constants ───────────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"
WEB_SEARCH_TOOL = "web_search_20260209"

TOKENS_FULL = 1500    # Pass 1 and Pass 2
TOKENS_UPDATE = 400   # Pass 3 (Latest Updates)

# Haiku pricing
PRICE_INPUT = 0.80 / 1_000_000
PRICE_OUTPUT = 4.00 / 1_000_000
PRICE_PER_SEARCH = 0.01   # $10 per 1K searches = $0.01 each

CACHE_MAX_AGE_DAYS = 30
HISTORY_MAX_SNAPSHOTS = 10
OUTPUT_MAX_RUNS = 10

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
HISTORY_FILE = DATA_DIR / "history.json"
ERROR_LOG = DATA_DIR / "errors.log"
OUTPUT_DIR = BASE_DIR / "output"

EXPECTED_COLUMNS = [
    "Flag", "Ticker", "Name", "Last Price", "Mkt Cap (m GBP)", "Stock Rank™",
    "Quality Rank", "Value Rank", "Momentum Rank", "Momentum Rank Previous Day",
    "QV Rank", "VM Rank", "QM Rank", "StockRank Style", "Risk Rating", "Sector",
]


# ── Utilities ───────────────────────────────────────────────────────────────────

def utc_now():
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_dirs():
    """Create required directories if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_error(ticker, msg):
    """Append error to the error log with UTC timestamp."""
    ts = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{ts}] {ticker}: {msg}\n")


def get_exchange_label(flag):
    """Return human-readable exchange label."""
    return {"gb": "LSE", "us": "NASDAQ/NYSE"}.get(flag.lower(), flag.upper())


def add_usage(total, usage):
    """Accumulate usage counters."""
    for k in ("input_tokens", "output_tokens", "search_requests"):
        total[k] = total.get(k, 0) + usage.get(k, 0)


def calculate_cost(usage):
    """Calculate dollar cost from a usage dict."""
    return (
        usage.get("input_tokens", 0) * PRICE_INPUT
        + usage.get("output_tokens", 0) * PRICE_OUTPUT
        + usage.get("search_requests", 0) * PRICE_PER_SEARCH
    )


# ── CSV Loading & Validation ───────────────────────────────────────────────────

def load_stocks(stocks_path):
    """Load and validate CSV. Return ordered dict of stock dicts keyed by flag_ticker."""
    try:
        df = pd.read_csv(stocks_path)
    except Exception as e:
        sys.exit(f"Error: Could not read CSV '{stocks_path}': {e}")

    df.columns = df.columns.str.strip()
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(
            "Error: CSV is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + f"\n\nFound columns: {', '.join(df.columns)}"
        )

    stocks = {}
    for _, row in df.iterrows():
        flag = str(row["Flag"]).strip().lower()
        ticker = str(row["Ticker"]).strip()
        key = f"{flag}_{ticker}"
        try:
            stocks[key] = {
                "key": key,
                "ticker": ticker,
                "name": str(row["Name"]).strip(),
                "flag": flag,
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
            }
        except (ValueError, KeyError) as e:
            sys.exit(f"Error parsing row for {ticker}: {e}")

    return stocks


# ── History Management ─────────────────────────────────────────────────────────

def load_history():
    """Load rank history. Returns dict keyed by flag_ticker."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {f"{s['flag']}_{s['ticker']}": s for s in data}
        return data
    except (json.JSONDecodeError, KeyError):
        return {}


def save_history(history):
    """Persist rank history to disk."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_snapshot(history, stock):
    """Append a rank snapshot for this stock. Trim to max 10."""
    key = stock["key"]
    snap = {
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
    history[key]["snapshots"].append(snap)
    if len(history[key]["snapshots"]) > HISTORY_MAX_SNAPSHOTS:
        history[key]["snapshots"] = history[key]["snapshots"][-HISTORY_MAX_SNAPSHOTS:]


def get_prev_snapshot(prev_history, stock):
    """Return the most recent previous snapshot, or None."""
    entry = prev_history.get(stock["key"], {})
    snaps = entry.get("snapshots", [])
    return snaps[-1] if snaps else None


def get_qv_delta(stock, prev):
    """QV rank change vs previous snapshot. None if no previous."""
    if prev is None:
        return None
    return stock["qv_rank"] - prev["qv_rank"]



# ── Cache Management ──────────────────────────────────────────────────────────

def cache_path(stock):
    """Return the cache file path for a stock."""
    return CACHE_DIR / f"{stock['flag']}_{stock['ticker']}.md"


def cache_age_days(stock):
    """Return cache age in days, or None if no cache exists."""
    p = cache_path(stock)
    if not p.exists():
        return None
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return (utc_now() - mtime).days


def read_cache(stock):
    """Read cached research content, or None if absent."""
    p = cache_path(stock)
    return p.read_text() if p.exists() else None


def write_cache(stock, content):
    """Write (overwrite) cache for a stock."""
    cache_path(stock).write_text(content)


def append_cache_update(stock, update_block):
    """Append a dated update block to existing cache."""
    p = cache_path(stock)
    existing = p.read_text() if p.exists() else ""
    p.write_text(existing.rstrip() + "\n\n" + update_block + "\n")


def determine_action(stock, force_refresh):
    """Return 'full' or 'update' for this stock."""
    if force_refresh:
        return "full"
    age = cache_age_days(stock)
    if age is None or age > CACHE_MAX_AGE_DAYS:
        return "full"
    return "update"


def strip_cache_header(cached_text):
    """Strip the cache file header (# Name, ## Research — date) from cached text."""
    lines = cached_text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## Research") or stripped == "":
            start = i + 1
        else:
            break
    return "\n".join(lines[start:]).strip()


# ── Research Prompts ──────────────────────────────────────────────────────────

def build_pass1_prompt(stock, prev, is_first_run):
    """Build Pass 1 prompt — Haiku knowledge pass, no web search."""
    name = stock["name"]
    exchange = get_exchange_label(stock["flag"])
    currency = "GBP" if stock["flag"] == "gb" else "USD"

    rank_lines = (
        f"Stock Rank: {stock['stock_rank']} | Quality Rank: {stock['quality_rank']} | "
        f"Value Rank: {stock['value_rank']} | Momentum Rank: {stock['momentum_rank']}\n"
        f"QV Rank: {stock['qv_rank']} | StockRank Style: {stock['stockrank_style']} | "
        f"Risk Rating: {stock['risk_rating']}\n"
        f"Sector: {stock['sector']} | Market Cap: {stock['mkt_cap']}m GBP | "
        f"Last Price: {stock['last_price']}"
    )

    prev_block = ""
    if prev and not is_first_run:
        delta = stock["qv_rank"] - prev["qv_rank"]
        sign = "+" if delta >= 0 else ""
        prev_block = (
            f"\n\nPrevious snapshot ({prev['timestamp_utc']}):\n"
            f"QV Rank: {prev['qv_rank']} → {stock['qv_rank']} ({sign}{delta}) | "
            f"Quality: {prev['quality_rank']} | Value: {prev['value_rank']} | "
            f"Momentum: {prev['momentum_rank']} | Style: {prev['stockrank_style']}"
        )

    baseline_note = (
        "\n\nBASELINE RUN — no previous snapshot data. "
        "Note in the rank sense-check that this is the baseline."
    ) if is_first_run else ""

    system = (
        "You are a stock research analyst writing for a UK private investor. "
        "Write exclusively in flowing prose — absolutely no bullet points, numbered lists, or dashes. "
        "Every section must contain 2-4 sentences of specific, well-informed analysis. "
        "Total across all eight items: approximately 400-500 words. "
        "Always label currency as GBP or USD on any financial figure. "
        "Be balanced: give fair weight to both bull and bear considerations."
    )

    user = f"""Write a research report on {name} ({exchange}) for a UK investor.

Stockopedia data:
{rank_lines}{prev_block}{baseline_note}

Use these exact section headers with bold formatting. Write flowing prose under each — no lists, no dashes.

**1. Stockopedia Rank Sense-Check**
Does the Quality Rank and Value Rank hold up against what is actually known about this company? Are the ranks justified by genuine fundamentals or potentially distorted by one-off events?

**2. Current Company Narrative**
What has happened at this company in the last 3-6 months? Earnings, guidance changes, management moves, contract wins, strategic shifts. What explains any rank movement?

**3. Multibagger Potential**
The bull case across three lenses: credible path to significantly higher revenue, margin expansion potential through operational leverage or pricing power, and re-rating potential if the stock is genuinely undervalued.

**4. Bear Case**
Two or three specific, credible risks for this company and sector right now. Not generic market risk — company-specific and sector-specific risks only.

**5. ESG and Sustainability**
Material ESG factors relevant to this company's sector. Any recent controversies. Is the ESG trajectory improving or deteriorating?

**6. Financial Health Red Flags**
Quick scan: unusual debt structures, cash flow concerns, recent dilution, off-balance-sheet risks. If nothing flags, say so in one sentence.

**8. Final Assessment — Buy / Watch / Sell**
One paragraph. The recommendation must be exactly one of: Buy, Watch, or Sell — no other labels. State it clearly in the first sentence with a brief rationale.

Where your training knowledge is limited or confidence is low for sections 1, 2, or 3, include the exact phrase <<LOW CONFIDENCE>> somewhere in that section.

On the very last line of your response, write exactly one of:
PASS2_NEEDED: YES
PASS2_NEEDED: NO

Write YES if any of sections 1, 2, or 3 contain <<LOW CONFIDENCE>>. Write NO if those sections are well-covered from training data alone."""

    return system, user


def build_pass2_prompt(stock, pass1_clean):
    """Build Pass 2 prompt — conditional web search for sections 1-3 gaps."""
    name = stock["name"]
    exchange = get_exchange_label(stock["flag"])

    system = (
        "You are a stock research analyst supplementing an existing research report. "
        "Use your web search tool to find current company information. "
        "Be concise — 2-3 sentences of genuinely new information only. "
        "Do not repeat anything already stated in the existing analysis."
    )

    user = f"""You are supplementing a research report on {name} ({exchange}).

The initial analysis flagged gaps in sections 1, 2, or 3 (Rank Sense-Check, Current Narrative, Multibagger Potential) because training knowledge was insufficient for some details.

Use the web search tool to find the most recent earnings report, trading update, or significant company announcement for {name} on {exchange}. Always search by the full company name — never by ticker alone.

Here is the beginning of the existing Pass 1 analysis for context:

{pass1_clean[:900]}

Write 2-3 sentences of supplementary information that fills the most material gaps identified. Focus only on information that is new or that updates/corrects the existing analysis.

Label your response with this exact header:
**Web Search Supplement**

Then write your supplementary sentences below it."""

    return system, user


def build_pass3_prompt(stock):
    """Build Pass 3 prompt — always-on Latest Updates search."""
    name = stock["name"]
    flag = stock["flag"]
    exchange = get_exchange_label(flag)

    if flag == "gb":
        search_guidance = (
            f"Search for the most recent RNS (Regulatory News Service) announcements for "
            f"{name} on the London Stock Exchange. Look for trading updates, director dealings, "
            f"results announcements, or material regulatory news."
        )
    else:
        search_guidance = (
            f"Search for the most recent SEC filings (8-K, 10-Q, or 10-K) or official "
            f"press releases for {name} listed on a US exchange."
        )

    system = (
        "You are a stock research analyst. Find only the most recent official announcement. "
        "Write 1-2 sentences maximum. "
        "If nothing material is found within the last 6 months, say so clearly in one sentence."
    )

    user = f"""Find the most recent official announcement for {name} ({exchange}).

{search_guidance}

Important: always search using the full company name — never by ticker symbol alone.

Write 1-2 sentences covering only the most material recent announcement. Include the approximate date if found. If no material announcement was published in the last 6 months, state that clearly in one sentence.

This will populate the "7. Latest Updates" section of the research report."""

    return system, user



# ── API Calls ─────────────────────────────────────────────────────────────────

def call_api(client, system, user, max_tokens, use_search=False):
    """Make a Haiku API call, optionally with web search. Retries on rate limit.
    Returns (text, usage_dict).
    """
    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if use_search:
        kwargs["tools"] = [{
            "type": WEB_SEARCH_TOOL,
            "name": "web_search",
            "max_uses": 1,
            "allowed_callers": ["direct"],
        }]

    for attempt in range(5):
        try:
            resp = client.messages.create(**kwargs)
            text = "\n".join(
                block.text for block in resp.content if hasattr(block, "text")
            )
            search_reqs = 0
            if hasattr(resp.usage, "server_tool_use") and resp.usage.server_tool_use:
                search_reqs = resp.usage.server_tool_use.web_search_requests
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "search_requests": search_reqs,
            }
            return text, usage
        except anthropic.RateLimitError:
            if attempt < 4:
                wait = 60 * (attempt + 1)
                print(f" [rate limited — waiting {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                raise


# ── Three-Pass Research Orchestration ────────────────────────────────────────

def run_full_research(client, stock, prev, is_first_run, delay):
    """Run all three passes. Returns (pass1_clean, pass2_text, pass3_text, usage, pass2_triggered)."""
    total = {"input_tokens": 0, "output_tokens": 0, "search_requests": 0}

    # ── Pass 1: Haiku knowledge pass (no web search) ──────────────────────────
    print(f"       Pass 1: Haiku knowledge pass...", end="", flush=True)
    sys1, usr1 = build_pass1_prompt(stock, prev, is_first_run)
    p1_raw, u1 = call_api(client, sys1, usr1, TOKENS_FULL, use_search=False)
    add_usage(total, u1)
    cost1 = calculate_cost(u1)
    print(f" done (${cost1:.3f})", flush=True)

    # Extract PASS2_NEEDED marker and clean text
    pass2_needed = "PASS2_NEEDED: YES" in p1_raw
    pass1_clean = re.sub(r'\nPASS2_NEEDED:.*$', '', p1_raw, flags=re.MULTILINE).strip()

    time.sleep(delay)

    # ── Pass 2: Targeted web search (conditional) ─────────────────────────────
    pass2_text = ""
    if pass2_needed:
        print(f"       Pass 2: Web search triggered...", end="", flush=True)
        sys2, usr2 = build_pass2_prompt(stock, pass1_clean)
        pass2_text, u2 = call_api(client, sys2, usr2, TOKENS_FULL, use_search=True)
        add_usage(total, u2)
        cost2 = calculate_cost(u2)
        searches2 = u2.get("search_requests", 0)
        print(f" done (${cost2:.3f}, {searches2} search)", flush=True)
        time.sleep(delay)
    else:
        print(f"       Pass 2: Web search skipped (Pass 1 coverage sufficient)", flush=True)

    # ── Pass 3: Latest Updates (always runs) ──────────────────────────────────
    print(f"       Pass 3: Latest Updates search...", end="", flush=True)
    sys3, usr3 = build_pass3_prompt(stock)
    pass3_text, u3 = call_api(client, sys3, usr3, TOKENS_UPDATE, use_search=True)
    add_usage(total, u3)
    cost3 = calculate_cost(u3)
    searches3 = u3.get("search_requests", 0)
    print(f" done (${cost3:.3f}, {searches3} search)", flush=True)

    return pass1_clean, pass2_text, pass3_text, total, pass2_needed


def run_update_research(client, stock, delay):
    """Run Pass 3 only (cache is fresh). Returns (pass3_text, usage)."""
    print(f"       Pass 1: skipped (cached)", flush=True)
    print(f"       Pass 2: skipped (cached)", flush=True)
    print(f"       Pass 3: Latest Updates search...", end="", flush=True)
    sys3, usr3 = build_pass3_prompt(stock)
    pass3_text, usage = call_api(client, sys3, usr3, TOKENS_UPDATE, use_search=True)
    cost3 = calculate_cost(usage)
    searches3 = usage.get("search_requests", 0)
    print(f" done (${cost3:.3f}, {searches3} search)", flush=True)
    return pass3_text, usage


# ── Write-up Assembly ─────────────────────────────────────────────────────────

def _find_marker(text, markers):
    """Return the index of the first marker found in text, or -1."""
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            return idx
    return -1


def insert_before_section8(text, section7_block):
    """Insert section 7 block before section 8 in the text."""
    idx = _find_marker(text, ["**8.", "8. Final Assessment", "Final Assessment"])
    if idx != -1:
        return text[:idx].rstrip() + "\n\n" + section7_block + "\n\n" + text[idx:]
    # Section 8 not found — append section 7 at the end
    return text.rstrip() + "\n\n" + section7_block


def insert_after_section3(text, supplement):
    """Insert Pass 2 supplement after section 3, before section 4."""
    idx = _find_marker(text, ["**4.", "4. Bear Case", "Bear Case"])
    if idx != -1:
        return text[:idx].rstrip() + "\n\n" + supplement + "\n\n" + text[idx:]
    # Fall back: append at end of section 3 area
    idx3 = _find_marker(text, ["**3.", "3. Multibagger", "Multibagger Potential"])
    if idx3 != -1:
        # Find end of section 3 content by looking for next **X. pattern
        after3 = text[idx3 + 3:]
        next_sec = re.search(r'\*\*[456789]\.', after3)
        if next_sec:
            split_at = idx3 + 3 + next_sec.start()
            return text[:split_at].rstrip() + "\n\n" + supplement + "\n\n" + text[split_at:]
    return text.rstrip() + "\n\n" + supplement


def assemble_full_writeup(pass1_clean, pass2_text, pass3_text):
    """Combine three passes into the final ordered 8-section write-up."""
    text = pass1_clean

    # Insert Pass 2 supplement after section 3, before section 4 (if it ran)
    if pass2_text.strip():
        text = insert_after_section3(text, pass2_text.strip())

    # Build and insert section 7 (Latest Updates) before section 8
    section7 = f"**7. Latest Updates**\n\n{pass3_text.strip()}"
    text = insert_before_section8(text, section7)

    return text


def extract_recommendation(text):
    """Extract Buy / Watch / Sell from research text. Default: Watch.

    Searches the body of section 8 (after the header line) to avoid matching
    the literal 'Buy / Watch / Sell' in the section header itself.
    """
    # Try to match the body of section 8 — skip past the bold header line
    fa_body = re.search(
        r'\*\*8\.\s*Final Assessment[^*\n]*\*\*\s*\n+(.*?)(?=\n\*\*[0-9]\.|\Z)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fa_body:
        search_in = fa_body.group(1)
    else:
        # Fall back: look in the last 400 chars of text
        search_in = text[-400:] if len(text) > 400 else text

    upper = search_in.upper()
    if re.search(r'\bSELL\b', upper):
        return "SELL"
    if re.search(r'\bBUY\b', upper):
        return "BUY"
    return "WATCH"



# ── Tier / Ordering Logic ─────────────────────────────────────────────────────

def compute_tiers(stocks_list, prev_history, is_first_run, top_n):
    """Return (tier1_keys_ordered, remaining_keys_ordered).

    Tier 1: top_n by QV delta (or absolute QV on first run), largest delta first.
            Ties broken by absolute QV rank descending.
    Remaining: ordered by absolute QV rank descending.
    """
    keys = list(stocks_list.keys())

    if is_first_run:
        sorted_all = sorted(keys, key=lambda k: stocks_list[k]["qv_rank"], reverse=True)
        tier1 = sorted_all[:top_n]
        remaining = sorted_all[top_n:]
        return tier1, remaining

    # Compute QV deltas
    deltas = {}
    for k in keys:
        stock = stocks_list[k]
        prev_snaps = prev_history.get(k, {}).get("snapshots", [])
        if prev_snaps:
            deltas[k] = stock["qv_rank"] - prev_snaps[-1]["qv_rank"]
        else:
            deltas[k] = None

    # Sort: known deltas first (descending), then None; ties broken by absolute QV
    def tier1_sort(k):
        d = deltas[k]
        qv = stocks_list[k]["qv_rank"]
        return (d is not None, d if d is not None else -9999, qv)

    sorted_by_delta = sorted(keys, key=tier1_sort, reverse=True)
    tier1 = sorted_by_delta[:top_n]
    remaining = sorted(
        sorted_by_delta[top_n:],
        key=lambda k: stocks_list[k]["qv_rank"],
        reverse=True,
    )
    return tier1, remaining


# ── Output File Generation ────────────────────────────────────────────────────

def format_qv_header(stock, prev, is_first_run):
    """Format the rank header line for a stock section."""
    qv = stock["qv_rank"]
    q = stock["quality_rank"]
    v = stock["value_rank"]
    m = stock["momentum_rank"]
    style = stock["stockrank_style"]

    if prev and not is_first_run:
        delta = qv - prev["qv_rank"]
        sign = "+" if delta >= 0 else ""
        qv_str = f"QV: {prev['qv_rank']}→{qv} ({sign}{delta})"
    else:
        qv_str = f"QV: {qv} →"

    return f"#### {qv_str} | Quality: {q} | Value: {v} | Momentum: {m} | {style}"


def build_stock_section(stock, writeup, prev, is_first_run, is_update=False, pass3_update=""):
    """Build the markdown output block for one stock."""
    name = stock["name"]
    ticker = stock["ticker"]
    exchange = get_exchange_label(stock["flag"])
    rec = extract_recommendation(writeup)

    lines = []
    lines.append(f"### {name} ({ticker} — {exchange}) — {rec}")
    lines.append(format_qv_header(stock, prev if not is_first_run else None, is_first_run))
    if is_first_run:
        lines.append("")
        lines.append("*Baseline run — no previous data. Rank change tracking begins from next run.*")
    lines.append("")
    lines.append(writeup)

    # For update stocks, append the new Pass 3 update block below the cached write-up
    if is_update and pass3_update.strip():
        today = utc_now().strftime("%Y-%m-%d")
        lines.append(f"\n### Update — {today} UTC\n\n{pass3_update.strip()}")

    lines.append("\n---\n")
    return "\n".join(lines)


def build_error_section(stock):
    """Build the error placeholder section for a failed stock."""
    name = stock["name"]
    ticker = stock["ticker"]
    exchange = get_exchange_label(stock["flag"])
    lines = [
        f"### {name} ({ticker} — {exchange})",
        "",
        f"Research unavailable due to API error. Retry with: `--refresh {ticker}`",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def generate_summary_prose(stocks_list, tier1_keys, prev_history, is_first_run):
    """Generate the summary section as flowing prose without an extra API call."""
    now = utc_now()
    total = len(stocks_list)
    t1 = len(tier1_keys)

    if is_first_run:
        names = [stocks_list[k]["name"] for k in tier1_keys[:3]]
        names_str = ", ".join(names)
        if t1 > 3:
            names_str += f" and {t1 - 3} others"
        return (
            f"This is the baseline run covering {total} stocks analysed on "
            f"{now.strftime('%d %B %Y')}. Rank history begins from this run — from the next "
            f"session onwards, QV rank movements will drive the ordering of stocks in this report. "
            f"Today's top QV performers include {names_str}. "
            f"All stocks have been researched across three passes and cached. "
            f"All recommendations use Buy, Watch, or Sell throughout."
        )

    # Normal run — describe movers
    movers = []
    for k in tier1_keys:
        s = stocks_list[k]
        prev_snaps = prev_history.get(k, {}).get("snapshots", [])
        if prev_snaps:
            delta = s["qv_rank"] - prev_snaps[-1]["qv_rank"]
            movers.append((s["name"], delta, s["qv_rank"]))

    movers.sort(key=lambda x: x[1], reverse=True)

    if movers:
        top = movers[0]
        sign = "+" if top[1] >= 0 else ""
        top_str = (
            f"{top[0]} led the movers with a QV rank change of {sign}{top[1]} to "
            f"a current QV of {top[2]}"
        )
    else:
        top_str = "no material QV rank movements were detected since the last run"

    # Count any negative movers in Tier 1
    declines = [(n, d) for n, d, _ in movers if d < 0]
    decline_note = ""
    if declines:
        decline_note = (
            f" {declines[0][0]} showed a decline of {declines[0][1]} points, "
            f"warranting closer attention."
        )

    return (
        f"This report covers {total} stocks as of {now.strftime('%d %B %Y')} "
        f"{now.strftime('%H:%M')} UTC. "
        f"The top {t1} stocks by QV rank movement are highlighted in Tier 1. "
        f"{top_str.capitalize()}.{decline_note} "
        f"Full three-pass research was run for stocks requiring updates; "
        f"cached stocks received a Pass 3 Latest Updates refresh only. "
        f"All recommendations are Buy, Watch, or Sell."
    )


def generate_output_file(stocks_list, tier1_keys, remaining_keys,
                         research_results, prev_history, is_first_run):
    """Generate the single output markdown file. Returns the file path."""
    now = utc_now()
    filename = now.strftime("%Y-%m-%d_%H%M") + "_analysis.md"
    filepath = OUTPUT_DIR / filename
    # Handle same-minute collision
    if filepath.exists():
        filename = now.strftime("%Y-%m-%d_%H%M") + "_2_analysis.md"
        filepath = OUTPUT_DIR / filename

    lines = [
        f"# Stock Analysis Report — {now.strftime('%d %B %Y')} {now.strftime('%H:%M')} UTC",
        "",
        "## Summary",
        "",
        generate_summary_prose(stocks_list, tier1_keys, prev_history, is_first_run),
        "",
        "---",
        "",
        "## Tier 1 — Top QV Rank Movers",
        "",
    ]

    for key in tier1_keys:
        stock = stocks_list[key]
        result = research_results.get(key, {})
        if result.get("error"):
            lines.append(build_error_section(stock))
            continue

        writeup = result.get("writeup", "")
        is_update = result.get("action") == "update"
        pass3_update = result.get("pass3_text", "") if is_update else ""

        prev = None
        if prev_history and key in prev_history:
            snaps = prev_history[key].get("snapshots", [])
            if snaps:
                prev = snaps[-1]

        lines.append(build_stock_section(
            stock, writeup, prev, is_first_run,
            is_update=is_update, pass3_update=pass3_update,
        ))

    if not tier1_keys:
        lines.append("*No Tier 1 stocks this run.*")
        lines.append("")

    lines += ["---", "", "## Remaining Stocks — By QV Rank", ""]

    for key in remaining_keys:
        stock = stocks_list[key]
        result = research_results.get(key, {})
        if result.get("error"):
            lines.append(build_error_section(stock))
            continue

        writeup = result.get("writeup", "")
        is_update = result.get("action") == "update"
        pass3_update = result.get("pass3_text", "") if is_update else ""

        prev = None
        if prev_history and key in prev_history:
            snaps = prev_history[key].get("snapshots", [])
            if snaps:
                prev = snaps[-1]

        lines.append(build_stock_section(
            stock, writeup, prev, is_first_run,
            is_update=is_update, pass3_update=pass3_update,
        ))

    if not remaining_keys:
        lines.append("*No remaining stocks.*")

    filepath.write_text("\n".join(lines))
    return filepath


def cleanup_old_outputs():
    """Delete oldest output files, keeping only the last OUTPUT_MAX_RUNS."""
    files = sorted(OUTPUT_DIR.glob("*_analysis.md"))
    if len(files) > OUTPUT_MAX_RUNS:
        for f in files[:-OUTPUT_MAX_RUNS]:
            f.unlink()



# ── Cost Estimation ───────────────────────────────────────────────────────────

# Estimated token usage per action (conservative)
_EST_FULL_INPUT = 2200    # avg across 3 passes (pass1 ~800, pass2 ~1500, pass3 ~500)
_EST_FULL_OUTPUT = 2300   # avg across 3 passes (pass1 ~1500, pass2 ~500, pass3 ~300)
_EST_FULL_SEARCHES = 1.5  # pass3 always (1) + pass2 ~50% chance (0.5)

_EST_UPD_INPUT = 500
_EST_UPD_OUTPUT = 350
_EST_UPD_SEARCHES = 1.0   # pass3 always


def estimate_run_cost(full_count, update_count):
    """Estimate total cost for the planned run."""
    full_cost = full_count * (
        _EST_FULL_INPUT * PRICE_INPUT
        + _EST_FULL_OUTPUT * PRICE_OUTPUT
        + _EST_FULL_SEARCHES * PRICE_PER_SEARCH
    )
    update_cost = update_count * (
        _EST_UPD_INPUT * PRICE_INPUT
        + _EST_UPD_OUTPUT * PRICE_OUTPUT
        + _EST_UPD_SEARCHES * PRICE_PER_SEARCH
    )
    return full_cost + update_cost


# ── Mock Research (dry run) ───────────────────────────────────────────────────

def generate_mock_writeup(stock, is_first_run):
    """Return a realistic mock 8-section write-up for --dry-run mode."""
    name = stock["name"]
    q = stock["quality_rank"]
    v = stock["value_rank"]
    m = stock["momentum_rank"]
    qv = stock["qv_rank"]
    style = stock["stockrank_style"]
    sector = stock["sector"]
    mkt = stock["mkt_cap"]
    flag = stock["flag"]
    currency = "GBP" if flag == "gb" else "USD"
    exchange = get_exchange_label(flag)

    if qv >= 90:
        rec = "Buy"
    elif qv < 50:
        rec = "Sell"
    else:
        rec = "Watch"

    baseline = (
        "Baseline run — no previous snapshot data. "
        "Rank change tracking begins from next run. "
    ) if is_first_run else ""

    return f"""**1. Stockopedia Rank Sense-Check**

{baseline}The Quality Rank of {q} and Value Rank of {v} for {name} appear broadly consistent with the company's fundamental position in the {sector} sector. The {style} classification from Stockopedia reflects a business where quality and value metrics appear to be genuinely complementary rather than temporarily inflated by one-off items, though a deeper review of the latest accounts would be needed to confirm.

**2. Current Company Narrative**

{name} operates in the {sector} sector with a market capitalisation of {mkt}m {currency} on the {exchange}. Recent trading has been broadly in line with market expectations, with management maintaining guidance for the current financial year. No material strategic pivots or significant management changes have been announced in the period under review.

**3. Multibagger Potential**

The bull case for {name} rests on three credible pillars. Revenue growth looks achievable as the company expands its addressable market within the {sector} space, where structural tailwinds remain supportive. Margin expansion is plausible through operational leverage as fixed costs are spread across a growing revenue base. At a QV Rank of {qv}, the current valuation appears undemanding relative to quality peers, leaving room for a meaningful multiple re-rating as earnings quality becomes more widely recognised.

**4. Bear Case**

The two most credible risks for {name} are sector-specific regulatory or competitive headwinds that could compress margins without warning, and the liquidity risk inherent in a {mkt}m {currency} market capitalisation where the bid-offer spread can widen sharply in volatile conditions. A third risk is execution: the strategic plan is ambitious, and any shortfall against guidance at the next results announcement would likely be penalised disproportionately given current market sentiment.

**5. ESG and Sustainability**

{name} operates within the typical ESG norms for the {sector} peer group. No material controversies have emerged in recent periods that would represent a differentiating risk factor. Governance appears adequate with no obvious red flags in board composition or remuneration structures, and the ESG trajectory appears broadly stable to improving.

**6. Financial Health Red Flags**

No material financial health red flags are apparent from the available data. The balance sheet appears clean with no unusual debt structures or recent dilutive equity raises flagged, and cash flow generation appears adequate relative to stated obligations.

**7. Latest Updates**

No material official announcements have been identified for {name} in the last six months based on available information. [Dry run — no web search performed.]

**8. Final Assessment — Buy / Watch / Sell**

{rec}. {name} presents a {'compelling case for inclusion at current levels' if rec == 'Buy' else ('reasonable case for monitoring ahead of a potential entry' if rec == 'Watch' else 'deteriorating picture that warrants caution')}. With a QV Rank of {qv} and a {style} classification, the stock {'offers an attractive risk-reward profile for investors willing to accept the liquidity constraints of a smaller company' if rec == 'Buy' else ('warrants a watching brief pending further rank improvement or a clearer fundamental catalyst' if rec == 'Watch' else 'shows rank deterioration sufficient to justify a re-evaluation of the investment thesis')}."""


# ── CLI Argument Parsing ──────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Research Engine — single CSV, three-pass AI research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --stocks my_stocks.csv
  python analyze.py --stocks my_stocks.csv --top 3
  python analyze.py --stocks my_stocks.csv --refresh ING
  python analyze.py --stocks my_stocks.csv --refresh-all
  python analyze.py --stocks my_stocks.csv --max-cost 1.00
  python analyze.py --stocks my_stocks.csv --skip-trial
  python analyze.py --stocks my_stocks.csv --delay 3
  python analyze.py --stocks my_stocks.csv --dry-run
""",
    )
    parser.add_argument("--stocks", required=True,
                        help="Path to Stockopedia CSV export")
    parser.add_argument("--top", type=int, default=5,
                        help="Tier 1 size — number of top QV movers (default: 5)")
    parser.add_argument("--refresh", type=str, default=None, metavar="TICKER",
                        help="Force full three-pass refresh for one stock")
    parser.add_argument("--refresh-all", action="store_true",
                        help="Force full three-pass refresh for every stock")
    parser.add_argument("--max-cost", type=float, default=2.00,
                        help="Hard cost cap in USD (default: $2.00)")
    parser.add_argument("--skip-trial", action="store_true",
                        help="Skip the cost confirmation prompt")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between API calls (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate report with mock data — no API calls, no cost")
    return parser.parse_args()



# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    print("Stock Research Engine")
    print("=" * 21)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}: OK")

    args = parse_args()
    dry_run = args.dry_run

    # ── API key ───────────────────────────────────────────────────────────────
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
            f"Option 1 — Create {BASE_DIR / '.env'} containing:\n"
            "  ANTHROPIC_API_KEY=sk-ant-your-key-here\n\n"
            "Option 2 — Set environment variable:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-your-key-here'"
        )

    if dry_run:
        print("MODE: Dry run (mock data, no API calls, $0.00 cost)")

    # ── Directories ───────────────────────────────────────────────────────────
    ensure_dirs()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\nLoading {args.stocks}...", end=" ", flush=True)
    stocks_list = load_stocks(args.stocks)
    print(f"OK — {len(stocks_list)} stocks")

    if len(stocks_list) == 0:
        sys.exit("Error: CSV contains no valid stock rows.")

    # ── History ───────────────────────────────────────────────────────────────
    print("Loading rank history...", end=" ", flush=True)
    history = load_history()
    # Deep copy for comparison (before this run's snapshots are added)
    prev_history = json.loads(json.dumps(history))
    is_first_run = len(history) == 0
    if is_first_run:
        print("no history found (baseline run)")
    else:
        print(f"{len(history)} stocks in history")

    # ── Compute ordering (Tier 1 / Remaining) ─────────────────────────────────
    tier1_keys, remaining_keys = compute_tiers(
        stocks_list, prev_history, is_first_run, args.top
    )

    print(f"\nOrdering:")
    print(f"  Tier 1 — top {args.top} QV {'by absolute QV (baseline)' if is_first_run else 'movers'}:  "
          f"{len(tier1_keys)} stocks")
    print(f"  Remaining — by QV rank desc:                     {len(remaining_keys)} stocks")

    if not is_first_run:
        last_ts = max(
            (s["snapshots"][-1]["timestamp_utc"]
             for s in prev_history.values() if s.get("snapshots")),
            default="unknown",
        )
        print(f"  Comparing vs last snapshot: {last_ts}")

    # ── Build action map ──────────────────────────────────────────────────────
    action_map = {}
    for key, stock in stocks_list.items():
        force = args.refresh_all or (
            args.refresh is not None and stock["ticker"] == args.refresh
        )
        action_map[key] = determine_action(stock, force)

    full_count = sum(1 for a in action_map.values() if a == "full")
    update_count = sum(1 for a in action_map.values() if a == "update")

    print(f"\nRun plan:")
    print(f"  Full research  (3 passes — Pass 1 + conditional Pass 2 + Pass 3):  {full_count} stocks")
    print(f"  Update only    (Pass 3 — Latest Updates only):                      {update_count} stocks")

    # ── Cost estimate ─────────────────────────────────────────────────────────
    if not dry_run:
        est = estimate_run_cost(full_count, update_count)
        print(f"\nEstimated cost:  ~${est:.2f}  (cap: ${args.max_cost:.2f})")
        print(f"  Full research stocks:   {full_count} × ~${_EST_FULL_INPUT * PRICE_INPUT + _EST_FULL_OUTPUT * PRICE_OUTPUT + _EST_FULL_SEARCHES * PRICE_PER_SEARCH:.3f} each")
        print(f"  Update-only stocks:     {update_count} × ~${_EST_UPD_INPUT * PRICE_INPUT + _EST_UPD_OUTPUT * PRICE_OUTPUT + _EST_UPD_SEARCHES * PRICE_PER_SEARCH:.3f} each")

        if est > args.max_cost:
            sys.exit(
                f"\nEstimated cost (${est:.2f}) exceeds the cap (${args.max_cost:.2f}).\n"
                f"To run a subset: --refresh TICKER\n"
                f"To raise the cap: --max-cost {est + 0.50:.2f}"
            )

        if not args.skip_trial:
            try:
                resp = input("\nProceed? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return
            if resp != "y":
                print("Aborted.")
                return

    # ── Initialise API client ─────────────────────────────────────────────────
    client = None
    if not dry_run:
        client = anthropic.Anthropic(api_key=api_key)

    # ── Research loop ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN' if dry_run else 'FULL RUN'} — {len(stocks_list)} stocks")
    print(f"{'=' * 60}\n")

    research_results = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0, "search_requests": 0}
    succeeded = 0
    failed = 0
    failed_tickers = []

    all_keys = tier1_keys + remaining_keys  # process in output order
    total = len(all_keys)

    for i, key in enumerate(all_keys):
        stock = stocks_list[key]
        action = action_map[key]
        ticker = stock["ticker"]
        name = stock["name"]
        exchange = get_exchange_label(stock["flag"])
        label = "full research" if action == "full" else "update (Pass 3 only)"

        print(f"[{i+1}/{total}]  {ticker:<8s}({name}, {exchange})  [{label}]")

        # ── Dry run: generate mock data ───────────────────────────────────────
        if dry_run:
            writeup = generate_mock_writeup(stock, is_first_run)
            ts = utc_now().strftime("%Y-%m-%d")
            header = (f"# {name} ({ticker} — {exchange})\n"
                      f"## Research — {ts} UTC\n\n")
            write_cache(stock, header + writeup)
            research_results[key] = {
                "action": action,
                "writeup": writeup,
                "pass3_text": "",
                "pass2_triggered": False,
                "error": False,
            }
            print(f"       [dry run — mock write-up generated]")
            succeeded += 1
            continue

        # ── Live research ─────────────────────────────────────────────────────
        prev = get_prev_snapshot(prev_history, stock)

        try:
            if action == "full":
                p1, p2, p3, usage, p2_triggered = run_full_research(
                    client, stock, prev, is_first_run, args.delay
                )
                writeup = assemble_full_writeup(p1, p2, p3)

                # Write full write-up to cache
                ts = utc_now().strftime("%Y-%m-%d")
                header = (f"# {name} ({ticker} — {exchange})\n"
                          f"## Research — {ts} UTC\n\n")
                write_cache(stock, header + writeup)

                research_results[key] = {
                    "action": "full",
                    "writeup": writeup,
                    "pass3_text": "",
                    "pass2_triggered": p2_triggered,
                    "error": False,
                }
                add_usage(total_usage, usage)

            else:  # update
                cached = read_cache(stock)
                if not cached:
                    # Cache missing despite "update" decision — run full instead
                    print(f"       No cache found — upgrading to full research")
                    p1, p2, p3, usage, p2_triggered = run_full_research(
                        client, stock, prev, is_first_run, args.delay
                    )
                    writeup = assemble_full_writeup(p1, p2, p3)
                    ts = utc_now().strftime("%Y-%m-%d")
                    header = (f"# {name} ({ticker} — {exchange})\n"
                              f"## Research — {ts} UTC\n\n")
                    write_cache(stock, header + writeup)
                    research_results[key] = {
                        "action": "full",
                        "writeup": writeup,
                        "pass3_text": "",
                        "pass2_triggered": p2_triggered,
                        "error": False,
                    }
                else:
                    p3, usage = run_update_research(client, stock, args.delay)

                    # Append dated update block to cache
                    today = utc_now().strftime("%Y-%m-%d")
                    update_block = f"### Update — {today} UTC\n\n{p3}"
                    append_cache_update(stock, update_block)

                    # Output uses cached write-up body + new Pass 3 as update
                    cached_body = strip_cache_header(cached)
                    research_results[key] = {
                        "action": "update",
                        "writeup": cached_body,
                        "pass3_text": p3,
                        "pass2_triggered": False,
                        "error": False,
                    }

                add_usage(total_usage, usage)

            run_cost = calculate_cost(total_usage)
            succeeded += 1
            print(f"       Cost so far: ${run_cost:.2f}")

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            log_error(ticker, err_msg)
            print(f"       FAILED: {err_msg}")
            failed += 1
            failed_tickers.append(ticker)
            research_results[key] = {
                "action": action,
                "writeup": f"*Research unavailable due to API error. Retry with: `--refresh {ticker}`*",
                "pass3_text": "",
                "pass2_triggered": False,
                "error": True,
            }

        # Delay between stocks (skip after last)
        if i < total - 1 and not dry_run:
            time.sleep(args.delay)

    # ── Save history snapshots ────────────────────────────────────────────────
    print(f"\nSaving rank history...", end=" ", flush=True)
    for key, stock in stocks_list.items():
        add_snapshot(history, stock)
    save_history(history)
    print("done")

    # ── Generate output file ──────────────────────────────────────────────────
    print("Generating output file...", end=" ", flush=True)
    output_path = generate_output_file(
        stocks_list, tier1_keys, remaining_keys,
        research_results, prev_history, is_first_run,
    )
    cleanup_old_outputs()
    print("done")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_cost = calculate_cost(total_usage)
    print(f"\n{'=' * 60}")
    print(f"Run complete. {succeeded} succeeded, {failed} failed.")
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers)}")
        print(f"Errors logged to: {ERROR_LOG}")
    if dry_run:
        print("Total cost: $0.00 (dry run)")
    else:
        print(f"Total cost: ${total_cost:.2f}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
