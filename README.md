# Polymarket Wallets Tracker

Discover and follow profitable Polymarket traders. Includes a Python API client library and a terminal dashboard (TUI).

## Setup

Requires Python 3.12+. Tested with [uv](https://docs.astral.sh/uv/).

```bash
uv venv p312 --python 3.12 --seed
source p312/bin/activate
pip install -r requirements.txt
```

No API keys needed — all endpoints are public.

## TUI Dashboard

```bash
python3 app.py
```

Three-panel layout: leaderboard, positions detail, and PnL chart.

**Keybindings:**

| Key | Action |
|-----|--------|
| `1` `2` `3` `4` | Switch period: All Time / Week / Month / Day |
| `5` | Active (AT) — All-time leaderboard filtered to wallets active this week |
| `a` | Trader analysis — shows quality metrics (monotonicity, drawdown, diversification, efficiency) |
| `b` | Set copy-trade budget |
| `s` | Search wallets |
| `f` | Toggle favorite |
| `i` | View favorites |
| `r` | Refresh leaderboard |
| `q` | Quit |

Select a wallet from the leaderboard to see its positions and PnL curve. Set a budget to simulate copy-trading at proportional scale.

### Trader Analysis (`a`)

Shows a modal with 4 quality metrics for the selected wallet:

- **Monotonicity** — % of days where cumulative PnL increased (consistency signal)
- **Max Drawdown** — largest peak-to-trough drop as % of peak (risk measure)
- **Diversification** — profitable markets vs total markets traded (concentration risk)
- **Efficiency** — PnL per trade (skill signal)

## CLI

```bash
python3 polymarket.py test                        # smoke test
python3 polymarket.py snapshot                    # save leaderboard snapshot to SQLite
python3 polymarket.py movers                      # who's climbing/falling the ranks
python3 polymarket.py interesting                  # scored list of wallets to follow
python3 polymarket.py pnl <wallet> --budget 1000  # PnL history with copy-trade sim
```

## Snapshot Workflow

Take daily snapshots to track leaderboard movement over time:

```bash
python3 polymarket.py snapshot      # run daily (fetches top 10k wallets, ~48s)
python3 polymarket.py movers        # compare latest vs ~7 days ago
python3 polymarket.py interesting   # combines weekly PnL + rank changes + new entrants
```

Snapshots are stored in `polymarket.db` (SQLite, created automatically).

## Library Usage

```python
from polymarket import poly_top_k, poly_user_trades, poly_user_positions

# Top 10 traders this week
for t in poly_top_k(10, period="week"):
    print(f"#{t.rank} {t.name}  PnL: ${t.pnl:,.2f}")

# Recent trades for a wallet
trades = poly_user_trades("0xabc...", limit=20)

# Active positions
positions = poly_user_positions("0xabc...", active_only=True)

# Full PnL curve with copy-trade simulation
from polymarket import poly_wallet_pnl_history
history = poly_wallet_pnl_history("0xabc...", budget=5000)
print(f"Projected MTM: ${history.mtm_pnl:,.2f}")
```

## API Functions

| Function | Description |
|----------|-------------|
| `poly_top_k(k, period)` | Top traders by PnL (max ~10,050) |
| `poly_user_trades(wallet, limit)` | Trade history for a wallet |
| `poly_user_positions(wallet, active_only)` | Current/historical positions |
| `poly_user_profile(wallet)` | Combined portfolio overview |
| `poly_market_info(condition_id_or_slug)` | Market metadata + top holders |
| `poly_transaction_info(tx_hash)` | On-chain transaction details |
| `poly_user_activity(wallet, types)` | Activity feed (trades, redeems, splits, merges) |
| `poly_wallet_pnl_history(wallet, budget)` | Realized PnL curve + copy-trade simulation |
| `poly_snapshot(k, periods)` | Save leaderboard to SQLite |
| `poly_movers(days, period)` | Rank changes between snapshots |
| `interesting_wallets(min_weekly_pnl)` | Scored list of wallets worth following |
