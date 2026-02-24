# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Polymarket wallet tracker: an API client library (`polymarket.py`) and a Textual TUI dashboard (`app.py`) for discovering and following profitable Polymarket traders.

## Running

```bash
# TUI dashboard
python3 app.py

# CLI commands
python3 polymarket.py test          # smoke test (hits live APIs)
python3 polymarket.py snapshot      # save leaderboard to SQLite (~48s for 10k wallets)
python3 polymarket.py movers        # rank changes between snapshots
python3 polymarket.py interesting   # scored list of wallets worth following
python3 polymarket.py pnl <wallet> [--budget N]  # PnL history / copy-trade sim
```

## Dependencies

Python 3.12. Install with: `pip install -r requirements.txt`

No API keys needed — all Polymarket and Polygon RPC endpoints are public.

## Architecture

**`polymarket.py`** — standalone API client + CLI. All API functions return typed dataclasses. Key internals:
- `_get()` / `_paginate_all()` — shared HTTP helpers using a module-level `requests.Session`
- `_fetch_leaderboard()` — handles the v1 leaderboard's 50-per-page pagination and 10k offset cap
- SQLite DB (`polymarket.db`) stores leaderboard snapshots for historical comparison. Schema: `snapshots` + `wallet_ranks` tables, initialized lazily in `_get_db()`
- `poly_wallet_pnl_history()` merges trades + REDEEMs into a timestamped PnL curve, with optional copy-trade scaling

**`app.py`** — Textual TUI that imports from `polymarket.py`. Three-panel grid layout:
- Leaderboard table (top-left) with async trade count loading via ThreadPoolExecutor
- Positions detail pane (top-right) with per-market stats merged from PnL curve + positions API
- PnL chart (bottom, full width) using `textual-plotext`, with market highlight overlay
- Modal screens for search (`s`) and favorites (`i`). Favorites persist to `favorites.json`
- All data fetching runs in `@work(thread=True)` workers with staleness checks

## Key Domain Concepts

- Polymarket uses **proxy wallets** — all API calls use `proxyWallet`, not the user's main wallet
- Leaderboard API can return duplicates at page boundaries — always deduplicate by wallet
- The `/trades` endpoint has no offset cap for per-user queries but caps at 3000 for global queries
- Activity types beyond trades: SPLIT, MERGE, REDEEM (market resolution payouts)

## API Endpoints

- `data-api.polymarket.com` — leaderboard, trades, positions, activity, value, holders
- `gamma-api.polymarket.com` — market/event metadata
- `clob.polymarket.com` — order book, prices
- `polygon-rpc.com` — public Polygon RPC for on-chain tx lookup
