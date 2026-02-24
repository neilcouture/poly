"""
Polymarket API client.

Provides functions to query trader leaderboards, trades, positions,
market info, transaction details, and discover interesting wallets.

APIs used:
  - data-api.polymarket.com — leaderboard, trades, positions, activity, value, holders
  - gamma-api.polymarket.com — market/event metadata
  - clob.polymarket.com     — order book, prices
  - polygon-rpc.com         — public Polygon RPC for on-chain tx lookup

Usage:
    from polymarket import poly_top_k, poly_user_trades, interesting_wallets
    top = poly_top_k(10)
    trades = poly_user_trades("0xabc...")

    # Snapshot-based workflow:
    from polymarket import poly_snapshot, poly_movers, interesting_wallets
    poly_snapshot()           # take a snapshot (run daily)
    movers = poly_movers(7)   # who moved in the last 7 days
    picks = interesting_wallets()  # combined interesting wallet list
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
POLYGON_RPC = "https://polygon-rpc.com"

DB_PATH = Path(__file__).parent / "polymarket.db"

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})

REQUEST_TIMEOUT = 15  # seconds
LEADERBOARD_PAGE_SIZE = 50  # API max per page
LEADERBOARD_MAX_OFFSET = 10000  # API hard cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(url: str, params: Optional[dict] = None) -> dict | list:
    """GET request with basic error handling."""
    resp = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _paginate_all(url: str, params: dict, limit_per_page: int = 100) -> list:
    """Fetch all pages from a paginated endpoint."""
    results = []
    offset = params.get("offset", 0)
    while True:
        page_params = {**params, "limit": limit_per_page, "offset": offset}
        try:
            page = _get(url, page_params)
        except requests.HTTPError:
            break  # API offset cap reached — return what we have
        if not page:
            break
        results.extend(page)
        if len(page) < limit_per_page:
            break
        offset += limit_per_page
    return results


def _fetch_leaderboard(
    k: int,
    period: str = "all",
    order_by: str = "PNL",
) -> list[dict]:
    """Fetch up to k entries from the v1 leaderboard, paginating as needed.

    Args:
        k: Number of wallets to fetch (max ~10,050).
        period: Time period — "all", "day", "week", or "month".
        order_by: Sort field — "PNL" or "VOLUME".

    Returns:
        List of raw dicts from the API.
    """
    results = []
    offset = 0
    while len(results) < k and offset <= LEADERBOARD_MAX_OFFSET:
        page_size = min(LEADERBOARD_PAGE_SIZE, k - len(results))
        data = _get(
            f"{DATA_API}/v1/leaderboard",
            params={
                "orderBy": order_by,
                "timePeriod": period,
                "limit": page_size,
                "offset": offset,
            },
        )
        if not isinstance(data, list) or not data:
            break
        results.extend(data)
        if len(data) < page_size:
            break
        offset += LEADERBOARD_PAGE_SIZE
    return results[:k]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the SQLite database."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,   -- ISO 8601
            period      TEXT NOT NULL,   -- all, day, week, month
            wallet_count INTEGER NOT NULL,
            duration_sec REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS wallet_ranks (
            snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
            rank        INTEGER NOT NULL,
            wallet      TEXT NOT NULL,
            username    TEXT NOT NULL DEFAULT '',
            pnl         REAL NOT NULL,
            volume      REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (snapshot_id, wallet)
        );
        CREATE INDEX IF NOT EXISTS idx_wallet_ranks_wallet
            ON wallet_ranks(wallet);
        CREATE INDEX IF NOT EXISTS idx_wallet_ranks_snapshot_rank
            ON wallet_ranks(snapshot_id, rank);
        CREATE INDEX IF NOT EXISTS idx_snapshots_period_ts
            ON snapshots(period, timestamp);
    """)
    return conn


# ---------------------------------------------------------------------------
# 1) poly_top_k
# ---------------------------------------------------------------------------


@dataclass
class TopTrader:
    rank: int
    wallet: str
    name: str
    pnl: float
    volume: float
    verified: bool = False


def poly_top_k(
    k: int = 100,
    period: str = "all",
) -> list[TopTrader]:
    """Return the top-k traders on Polymarket ranked by profit.

    Args:
        k: Number of top traders to return (max ~10,050).
        period: Time period — "all", "day", "week", or "month".

    Returns:
        List of TopTrader objects sorted by PnL descending.
    """
    data = _fetch_leaderboard(k, period=period)
    return [
        TopTrader(
            rank=int(row.get("rank", i + 1)),
            wallet=row["proxyWallet"],
            name=row.get("userName", ""),
            pnl=row.get("pnl", 0),
            volume=row.get("vol", 0),
            verified=row.get("verifiedBadge", False),
        )
        for i, row in enumerate(data)
    ]


# ---------------------------------------------------------------------------
# 2) poly_user_trades
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    wallet: str
    timestamp: int
    side: str  # BUY or SELL
    outcome: str  # e.g. "Yes", "No", "Up", "Down"
    outcome_index: int
    size: float  # shares
    price: float  # per share
    usdc_size: float
    market_title: str
    market_slug: str
    event_slug: str
    condition_id: str
    asset: str
    tx_hash: str
    trader_name: str = ""
    trader_pseudonym: str = ""


def poly_user_trades(
    wallet: str,
    limit: int = 100,
    offset: int = 0,
) -> list[Trade]:
    """Return trades for a given wallet (proxy wallet address).

    Args:
        wallet: The proxy wallet address (0x...).
        limit: Max number of trades to return. Use 0 for all trades.
        offset: Pagination offset.

    Returns:
        List of Trade objects, most recent first.
    """
    if limit == 0:
        raw = _paginate_all(
            f"{DATA_API}/trades",
            params={"user": wallet, "offset": offset},
        )
    else:
        raw = _get(
            f"{DATA_API}/trades",
            params={"user": wallet, "limit": limit, "offset": offset},
        )

    return [
        Trade(
            wallet=row.get("proxyWallet", wallet),
            timestamp=row.get("timestamp", 0),
            side=row.get("side", ""),
            outcome=row.get("outcome", ""),
            outcome_index=row.get("outcomeIndex", 0),
            size=row.get("size", 0),
            price=row.get("price", 0),
            usdc_size=row.get("usdcSize", row.get("size", 0) * row.get("price", 0)),
            market_title=row.get("title", ""),
            market_slug=row.get("slug", ""),
            event_slug=row.get("eventSlug", ""),
            condition_id=row.get("conditionId", ""),
            asset=row.get("asset", ""),
            tx_hash=row.get("transactionHash", ""),
            trader_name=row.get("name", ""),
            trader_pseudonym=row.get("pseudonym", ""),
        )
        for row in raw
    ]


# ---------------------------------------------------------------------------
# 3) poly_user_positions
# ---------------------------------------------------------------------------


@dataclass
class Position:
    wallet: str
    market_title: str
    market_slug: str
    event_slug: str
    event_id: str
    condition_id: str
    outcome: str
    outcome_index: int
    opposite_outcome: str
    size: float
    avg_price: float
    initial_value: float
    current_value: float
    cash_pnl: float
    percent_pnl: float
    realized_pnl: float
    cur_price: float
    redeemable: bool
    end_date: str
    neg_risk: bool
    asset: str


def poly_user_positions(
    wallet: str,
    active_only: bool = True,
) -> list[Position]:
    """Return positions for a given wallet.

    Args:
        wallet: The proxy wallet address (0x...).
        active_only: If True, only return positions with size > 0 and
                     current_value > 0 (i.e. still open). If False, return all.

    Returns:
        List of Position objects.
    """
    raw = _paginate_all(
        f"{DATA_API}/positions",
        params={"user": wallet},
    )

    positions = [
        Position(
            wallet=row.get("proxyWallet", wallet),
            market_title=row.get("title", ""),
            market_slug=row.get("slug", ""),
            event_slug=row.get("eventSlug", ""),
            event_id=row.get("eventId", ""),
            condition_id=row.get("conditionId", ""),
            outcome=row.get("outcome", ""),
            outcome_index=row.get("outcomeIndex", 0),
            opposite_outcome=row.get("oppositeOutcome", ""),
            size=row.get("size", 0),
            avg_price=row.get("avgPrice", 0),
            initial_value=row.get("initialValue", 0),
            current_value=row.get("currentValue", 0),
            cash_pnl=row.get("cashPnl", 0),
            percent_pnl=row.get("percentPnl", 0),
            realized_pnl=row.get("realizedPnl", 0),
            cur_price=row.get("curPrice", 0),
            redeemable=row.get("redeemable", False),
            end_date=row.get("endDate", ""),
            neg_risk=row.get("negativeRisk", False),
            asset=row.get("asset", ""),
        )
        for row in raw
    ]

    if active_only:
        positions = [p for p in positions if p.size > 0 and p.current_value > 0]

    return positions


# ---------------------------------------------------------------------------
# 4) poly_user_profile
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    wallet: str
    portfolio_value: float
    total_positions: int
    active_positions: int
    total_pnl: float
    recent_trades: list[Trade]
    top_positions: list[Position]


def poly_user_profile(wallet: str) -> UserProfile:
    """Return a combined profile for a wallet: value, positions summary, recent trades.

    Args:
        wallet: The proxy wallet address (0x...).

    Returns:
        UserProfile with portfolio value, positions, and recent trades.
    """
    value_data = _get(f"{DATA_API}/value", params={"user": wallet})
    portfolio_value = 0.0
    if isinstance(value_data, list) and value_data:
        portfolio_value = value_data[0].get("value", 0)
    elif isinstance(value_data, dict):
        portfolio_value = value_data.get("value", 0)

    all_positions = poly_user_positions(wallet, active_only=False)
    active_positions = [p for p in all_positions if p.size > 0 and p.current_value > 0]
    total_pnl = sum(p.cash_pnl for p in all_positions)

    recent_trades = poly_user_trades(wallet, limit=20)

    top_positions = sorted(
        active_positions, key=lambda p: p.current_value, reverse=True
    )[:10]

    return UserProfile(
        wallet=wallet,
        portfolio_value=portfolio_value,
        total_positions=len(all_positions),
        active_positions=len(active_positions),
        total_pnl=total_pnl,
        recent_trades=recent_trades,
        top_positions=top_positions,
    )


# ---------------------------------------------------------------------------
# 5) poly_market_info
# ---------------------------------------------------------------------------


@dataclass
class MarketInfo:
    id: str
    question: str
    slug: str
    condition_id: str
    description: str
    outcomes: list[str]
    outcome_prices: list[str]
    volume: float
    volume_24hr: float
    liquidity: float
    open_interest: float
    start_date: str
    end_date: str
    active: bool
    closed: bool
    best_bid: float
    best_ask: float
    spread: float
    last_trade_price: float
    clob_token_ids: list[str]
    neg_risk: bool
    resolution_source: str
    event_slug: str
    event_title: str
    holders: list[dict] = field(default_factory=list)


def poly_market_info(condition_id_or_slug: str) -> MarketInfo:
    """Return detailed info for a market, identified by conditionId or slug.

    Args:
        condition_id_or_slug: Either a conditionId (0x...) or a market slug string.

    Returns:
        MarketInfo object with full market details.
    """
    if condition_id_or_slug.startswith("0x"):
        params = {"condition_ids": condition_id_or_slug}
    else:
        params = {"slug": condition_id_or_slug}

    markets = _get(f"{GAMMA_API}/markets", params={**params, "limit": 1})
    if not markets:
        raise ValueError(f"Market not found: {condition_id_or_slug}")

    m = markets[0]

    outcomes = (
        json.loads(m.get("outcomes", "[]"))
        if isinstance(m.get("outcomes"), str)
        else m.get("outcomes", [])
    )
    outcome_prices = (
        json.loads(m.get("outcomePrices", "[]"))
        if isinstance(m.get("outcomePrices"), str)
        else m.get("outcomePrices", [])
    )
    clob_token_ids = (
        json.loads(m.get("clobTokenIds", "[]"))
        if isinstance(m.get("clobTokenIds"), str)
        else m.get("clobTokenIds", [])
    )

    events = m.get("events", [])
    event_slug = events[0].get("slug", "") if events else m.get("eventSlug", "")
    event_title = events[0].get("title", "") if events else ""

    condition_id = m.get("conditionId", "")
    holders = []
    try:
        holders_raw = _get(
            f"{DATA_API}/holders",
            params={"market": condition_id, "limit": 10},
        )
        for token_group in holders_raw:
            for h in token_group.get("holders", []):
                holders.append(
                    {
                        "wallet": h.get("proxyWallet", ""),
                        "name": h.get("name", ""),
                        "pseudonym": h.get("pseudonym", ""),
                        "amount": h.get("amount", 0),
                        "outcome_index": h.get("outcomeIndex", 0),
                    }
                )
    except Exception:
        pass

    return MarketInfo(
        id=m.get("id", ""),
        question=m.get("question", ""),
        slug=m.get("slug", ""),
        condition_id=condition_id,
        description=m.get("description", ""),
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        volume=m.get("volumeNum", 0) or 0,
        volume_24hr=m.get("volume24hr", 0) or 0,
        liquidity=m.get("liquidityNum", 0) or 0,
        open_interest=m.get("openInterest", 0) or 0,
        start_date=m.get("startDate", ""),
        end_date=m.get("endDate", ""),
        active=m.get("active", False),
        closed=m.get("closed", False),
        best_bid=m.get("bestBid", 0) or 0,
        best_ask=m.get("bestAsk", 0) or 0,
        spread=m.get("spread", 0) or 0,
        last_trade_price=m.get("lastTradePrice", 0) or 0,
        clob_token_ids=clob_token_ids,
        neg_risk=m.get("negRisk", False),
        resolution_source=m.get("resolutionSource", ""),
        event_slug=event_slug,
        event_title=event_title,
        holders=holders,
    )


# ---------------------------------------------------------------------------
# 6) poly_transaction_info
# ---------------------------------------------------------------------------


@dataclass
class TransactionInfo:
    tx_hash: str
    block_number: int
    timestamp: int
    from_address: str
    to_address: str
    value: str  # in wei (MATIC)
    gas_used: int
    gas_price: str
    status: bool  # True = success
    method_id: str  # first 4 bytes of input data
    contract_address: str
    num_logs: int = 0


def _rpc_call(method: str, params: list) -> dict | None:
    """Make a JSON-RPC call to the Polygon public RPC."""
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    resp = _SESSION.post(POLYGON_RPC, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("result")


def poly_transaction_info(tx_hash: str) -> TransactionInfo:
    """Return on-chain transaction details via Polygon public RPC.

    Args:
        tx_hash: The transaction hash (0x...).

    Returns:
        TransactionInfo with block, sender, receiver, gas, status.

    Raises:
        ValueError: If the transaction is not found.
    """
    result = _rpc_call("eth_getTransactionByHash", [tx_hash])
    if not result:
        raise ValueError(f"Transaction not found: {tx_hash}")

    receipt = _rpc_call("eth_getTransactionReceipt", [tx_hash])

    input_data = result.get("input", "0x")
    method_id = input_data[:10] if len(input_data) >= 10 else input_data

    return TransactionInfo(
        tx_hash=tx_hash,
        block_number=int(result.get("blockNumber", "0x0"), 16),
        timestamp=0,
        from_address=result.get("from", ""),
        to_address=result.get("to", ""),
        value=str(int(result.get("value", "0x0"), 16)),
        gas_used=int(receipt.get("gasUsed", "0x0"), 16) if receipt else 0,
        gas_price=str(int(result.get("gasPrice", "0x0"), 16)),
        status=receipt.get("status") == "0x1" if receipt else False,
        method_id=method_id,
        contract_address=result.get("to", ""),
        num_logs=len(receipt.get("logs", [])) if receipt else 0,
    )


# ---------------------------------------------------------------------------
# 7) poly_user_activity — activity feed (trades, redeems, splits, merges)
# ---------------------------------------------------------------------------


@dataclass
class Activity:
    wallet: str
    timestamp: int
    type: str  # TRADE, SPLIT, MERGE, REDEEM
    outcome: str
    size: float
    usdc_size: float
    market_title: str
    condition_id: str
    tx_hash: str


def poly_user_activity(
    wallet: str,
    types: Optional[list[str]] = None,
) -> list[Activity]:
    """Return activity entries for a wallet from the /activity endpoint.

    This captures events like REDEEMs (market resolutions) that don't appear
    in the /trades endpoint.

    Args:
        wallet: The proxy wallet address (0x...).
        types: Activity types to fetch. Default: all types.
               Valid types: TRADE, SPLIT, MERGE, REDEEM.

    Returns:
        List of Activity objects, most recent first.
    """
    if types is None:
        types = ["TRADE", "SPLIT", "MERGE", "REDEEM"]

    all_activities = []
    for activity_type in types:
        raw = _paginate_all(
            f"{DATA_API}/activity",
            params={"user": wallet, "type": activity_type},
        )
        for row in raw:
            all_activities.append(
                Activity(
                    wallet=row.get("proxyWallet", wallet),
                    timestamp=row.get("timestamp", 0),
                    type=row.get("type", activity_type),
                    outcome=row.get("outcome", ""),
                    size=float(row.get("size", 0)),
                    usdc_size=float(row.get("usdcSize", 0)),
                    market_title=row.get("title", ""),
                    condition_id=row.get("conditionId", ""),
                    tx_hash=row.get("transactionHash", ""),
                )
            )

    all_activities.sort(key=lambda a: a.timestamp, reverse=True)
    return all_activities


# ---------------------------------------------------------------------------
# 8) poly_wallet_pnl_history — realized PnL curve + copy-trade backtest
# ---------------------------------------------------------------------------


@dataclass
class PnLPoint:
    timestamp: int
    cumulative_pnl: float  # running sum
    event_type: str  # BUY, SELL, REDEEM
    usdc_amount: float  # signed: negative for buys, positive for sells/redeems
    market_title: str


@dataclass
class PnLHistory:
    wallet: str
    budget: Optional[float]  # None = actual, float = simulated copy-trade
    scale_factor: float  # 1.0 for actual, budget/wallet_total_deployed for copy
    total_pnl: float  # final cumulative PnL (realized only)
    unrealized_pnl: float  # MTM value of open positions
    mtm_pnl: float  # total_pnl + unrealized_pnl
    total_trades: int
    total_redeems: int
    first_trade: int  # timestamp
    last_trade: int  # timestamp
    curve: list[PnLPoint]


def poly_wallet_pnl_history(
    wallet: str,
    budget: Optional[float] = None,
) -> PnLHistory:
    """Build a realized PnL timeline for a wallet, optionally simulating copy-trading.

    With no budget: returns the wallet's actual realized PnL curve.
    With a budget: simulates copy-trading at proportional scale.

    Args:
        wallet: The proxy wallet address (0x...).
        budget: If set, simulate copy-trading with this USDC budget.
                The wallet's trades are scaled by (budget / total_deployed).

    Returns:
        PnLHistory with the full PnL curve.
    """
    # 1. Fetch all trades
    trades = poly_user_trades(wallet, limit=0)

    # 2. Fetch REDEEMs
    redeems = poly_user_activity(wallet, types=["REDEEM"])

    # 3. Merge into unified event list
    events = []
    for t in trades:
        events.append({
            "timestamp": t.timestamp,
            "type": t.side,  # BUY or SELL
            "usdc_size": float(t.usdc_size),
            "market_title": t.market_title,
        })
    for r in redeems:
        events.append({
            "timestamp": r.timestamp,
            "type": "REDEEM",
            "usdc_size": r.usdc_size,
            "market_title": r.market_title,
        })

    # Sort by timestamp ascending
    events.sort(key=lambda e: e["timestamp"])

    # 3b. Fetch open positions for MTM
    positions = poly_user_positions(wallet, active_only=True)
    open_value = sum(p.current_value for p in positions)

    if not events:
        unrealized = open_value
        return PnLHistory(
            wallet=wallet,
            budget=budget,
            scale_factor=1.0 if budget is None else 0.0,
            total_pnl=0.0,
            unrealized_pnl=unrealized,
            mtm_pnl=unrealized,
            total_trades=len(trades),
            total_redeems=len(redeems),
            first_trade=0,
            last_trade=0,
            curve=[],
        )

    # 4. Compute scale factor if budget is set
    total_deployed = sum(e["usdc_size"] for e in events if e["type"] == "BUY")
    if budget is not None and total_deployed > 0:
        scale_factor = budget / total_deployed
    elif budget is not None:
        scale_factor = 0.0
    else:
        scale_factor = 1.0

    # 5. Build PnL curve
    curve = []
    cumulative = 0.0
    for e in events:
        raw_amount = e["usdc_size"]
        if e["type"] == "BUY":
            signed = -raw_amount * scale_factor
        else:  # SELL or REDEEM
            signed = raw_amount * scale_factor

        cumulative += signed
        curve.append(
            PnLPoint(
                timestamp=e["timestamp"],
                cumulative_pnl=cumulative,
                event_type=e["type"],
                usdc_amount=signed,
                market_title=e["market_title"],
            )
        )

    # 6. Mark-to-market: add open position value
    unrealized = open_value * scale_factor
    mtm = cumulative + unrealized

    # Append a MTM point at current time if there are open positions
    if unrealized != 0:
        curve.append(
            PnLPoint(
                timestamp=int(time.time()),
                cumulative_pnl=mtm,
                event_type="MTM",
                usdc_amount=unrealized,
                market_title=f"[{len(positions)} open positions]",
            )
        )

    return PnLHistory(
        wallet=wallet,
        budget=budget,
        scale_factor=scale_factor,
        total_pnl=cumulative,
        unrealized_pnl=unrealized,
        mtm_pnl=mtm,
        total_trades=len(trades),
        total_redeems=len(redeems),
        first_trade=events[0]["timestamp"],
        last_trade=events[-1]["timestamp"],
        curve=curve,
    )


# ---------------------------------------------------------------------------
# 9) poly_snapshot — save leaderboard to SQLite
# ---------------------------------------------------------------------------


def poly_snapshot(
    k: int = 10000,
    periods: Optional[list[str]] = None,
    db_path: Optional[Path] = None,
    verbose: bool = True,
) -> list[int]:
    """Take a snapshot of the leaderboard and save it to the local database.

    Fetches the top-k wallets by PnL for each time period and inserts them
    into the SQLite database with a timestamp.

    Args:
        k: Number of wallets to fetch per period (default 10,000; max ~10,050).
        periods: List of time periods to snapshot. Default: ["all", "week"].
        db_path: Path to SQLite database. Default: polymarket.db next to this file.
        verbose: Print progress to stdout.

    Returns:
        List of snapshot IDs created.
    """
    if periods is None:
        periods = ["all", "week"]

    conn = _get_db(db_path)
    snapshot_ids = []

    for period in periods:
        if verbose:
            print(f"  Fetching top {k} by PnL (period={period})...", end="", flush=True)

        start = time.time()
        data = _fetch_leaderboard(k, period=period, order_by="PNL")
        elapsed = time.time() - start

        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            "INSERT INTO snapshots (timestamp, period, wallet_count, duration_sec) VALUES (?, ?, ?, ?)",
            (now, period, len(data), elapsed),
        )
        snapshot_id = cursor.lastrowid

        # Deduplicate by wallet (API can return dupes at page boundaries)
        seen = set()
        unique_rows = []
        for i, row in enumerate(data):
            wallet = row["proxyWallet"]
            if wallet not in seen:
                seen.add(wallet)
                unique_rows.append((
                    snapshot_id,
                    int(row.get("rank", i + 1)),
                    wallet,
                    row.get("userName", ""),
                    row.get("pnl", 0),
                    row.get("vol", 0),
                ))

        conn.executemany(
            "INSERT INTO wallet_ranks (snapshot_id, rank, wallet, username, pnl, volume) VALUES (?, ?, ?, ?, ?, ?)",
            unique_rows,
        )

        conn.commit()
        snapshot_ids.append(snapshot_id)

        if verbose:
            print(f" {len(data)} wallets in {elapsed:.1f}s (snapshot #{snapshot_id})")

    conn.close()
    return snapshot_ids


# ---------------------------------------------------------------------------
# 10) poly_movers — compare snapshots
# ---------------------------------------------------------------------------


@dataclass
class Mover:
    wallet: str
    username: str
    current_rank: int
    previous_rank: int  # 0 if new to the leaderboard
    rank_change: int  # positive = moved up
    current_pnl: float
    previous_pnl: float
    pnl_change: float
    is_new: bool  # first time on the leaderboard


def poly_movers(
    days: int = 7,
    period: str = "all",
    top_n: int = 100,
    db_path: Optional[Path] = None,
) -> list[Mover]:
    """Compare two snapshots and return the biggest movers.

    Finds the most recent snapshot and the closest snapshot ~days ago for the
    given period, then computes rank changes.

    Args:
        days: How many days back to compare (finds closest available snapshot).
        period: Which leaderboard period to compare ("all", "week", etc.).
        top_n: Return the top N movers by rank improvement.
        db_path: Path to SQLite database.

    Returns:
        List of Mover objects sorted by rank_change descending (biggest climbers first).

    Raises:
        ValueError: If fewer than 2 snapshots exist for the given period.
    """
    conn = _get_db(db_path)

    # Get the most recent snapshot
    latest = conn.execute(
        "SELECT id, timestamp FROM snapshots WHERE period = ? ORDER BY timestamp DESC LIMIT 1",
        (period,),
    ).fetchone()

    if not latest:
        conn.close()
        raise ValueError(f"No snapshots found for period={period}. Run poly_snapshot() first.")

    # Find the closest snapshot to `days` ago
    target_ts = datetime.fromisoformat(latest["timestamp"]) - __import__("datetime").timedelta(days=days)
    target_str = target_ts.isoformat()

    previous = conn.execute(
        """SELECT id, timestamp FROM snapshots
           WHERE period = ? AND timestamp <= ? AND id != ?
           ORDER BY timestamp DESC LIMIT 1""",
        (period, target_str, latest["id"]),
    ).fetchone()

    if not previous:
        # Fall back: just use the oldest snapshot that isn't the latest
        previous = conn.execute(
            """SELECT id, timestamp FROM snapshots
               WHERE period = ? AND id != ?
               ORDER BY timestamp ASC LIMIT 1""",
            (period, latest["id"]),
        ).fetchone()

    if not previous:
        conn.close()
        raise ValueError(
            f"Need at least 2 snapshots for period={period}. "
            f"Only found 1 (#{latest['id']} at {latest['timestamp']}). "
            f"Run poly_snapshot() again later."
        )

    # Load both snapshots into dicts
    current_rows = conn.execute(
        "SELECT wallet, username, rank, pnl FROM wallet_ranks WHERE snapshot_id = ?",
        (latest["id"],),
    ).fetchall()

    previous_rows = conn.execute(
        "SELECT wallet, username, rank, pnl FROM wallet_ranks WHERE snapshot_id = ?",
        (previous["id"],),
    ).fetchall()

    conn.close()

    current = {r["wallet"]: r for r in current_rows}
    prev = {r["wallet"]: r for r in previous_rows}

    movers = []
    for wallet, cur in current.items():
        if wallet in prev:
            prev_rank = prev[wallet]["rank"]
            prev_pnl = prev[wallet]["pnl"]
            rank_change = prev_rank - cur["rank"]  # positive = moved up
            is_new = False
        else:
            prev_rank = 0
            prev_pnl = 0.0
            rank_change = LEADERBOARD_MAX_OFFSET  # new entrants sort high
            is_new = True

        movers.append(
            Mover(
                wallet=wallet,
                username=cur["username"],
                current_rank=cur["rank"],
                previous_rank=prev_rank,
                rank_change=rank_change,
                current_pnl=cur["pnl"],
                previous_pnl=prev_pnl,
                pnl_change=cur["pnl"] - prev_pnl,
                is_new=is_new,
            )
        )

    # Sort: biggest rank climbers first, then by PnL change
    movers.sort(key=lambda m: (m.rank_change, m.pnl_change), reverse=True)
    return movers[:top_n]


# ---------------------------------------------------------------------------
# 11) interesting_wallets — the final filtered list
# ---------------------------------------------------------------------------


@dataclass
class InterestingWallet:
    wallet: str
    username: str
    alltime_rank: int
    alltime_pnl: float
    weekly_rank: int
    weekly_pnl: float
    rank_change: int  # from all-time movers
    pnl_change: float
    is_new_to_leaderboard: bool
    score: float  # combined interestingness score


def interesting_wallets(
    min_weekly_pnl: float = 1000,
    min_alltime_pnl: float = 0,
    top_n: int = 50,
    db_path: Optional[Path] = None,
) -> list[InterestingWallet]:
    """Return wallets worth following based on snapshot data.

    Combines:
    - All-time leaderboard rank changes (who's climbing)
    - Weekly PnL (who's winning right now)
    - New entrants (fresh wallets cracking the top 10k)

    Requires at least 1 snapshot with both "all" and "week" periods.
    For movers analysis, requires 2+ snapshots of the "all" period.

    Args:
        min_weekly_pnl: Minimum weekly PnL to qualify (filters noise).
        min_alltime_pnl: Minimum all-time PnL to qualify.
        top_n: Number of interesting wallets to return.
        db_path: Path to SQLite database.

    Returns:
        List of InterestingWallet sorted by score descending.
    """
    conn = _get_db(db_path)

    # Get latest "all" snapshot
    latest_all = conn.execute(
        "SELECT id FROM snapshots WHERE period = 'all' ORDER BY timestamp DESC LIMIT 1",
    ).fetchone()

    # Get latest "week" snapshot
    latest_week = conn.execute(
        "SELECT id FROM snapshots WHERE period = 'week' ORDER BY timestamp DESC LIMIT 1",
    ).fetchone()

    if not latest_all:
        conn.close()
        raise ValueError("No 'all' period snapshot found. Run poly_snapshot() first.")

    # Load all-time data
    alltime_rows = conn.execute(
        "SELECT wallet, username, rank, pnl, volume FROM wallet_ranks WHERE snapshot_id = ?",
        (latest_all["id"],),
    ).fetchall()
    alltime = {r["wallet"]: r for r in alltime_rows}

    # Load weekly data if available
    weekly = {}
    if latest_week:
        weekly_rows = conn.execute(
            "SELECT wallet, username, rank, pnl FROM wallet_ranks WHERE snapshot_id = ?",
            (latest_week["id"],),
        ).fetchall()
        weekly = {r["wallet"]: r for r in weekly_rows}

    # Load previous "all" snapshot for movers
    movers_map = {}
    try:
        movers_list = poly_movers(days=7, period="all", top_n=99999, db_path=db_path)
        movers_map = {m.wallet: m for m in movers_list}
    except ValueError:
        pass  # Only 1 snapshot — no movers data yet

    conn.close()

    # Build the interesting wallets list
    candidates = []

    # Start from weekly leaders (currently winning)
    wallets_to_check = set()
    for wallet, w in weekly.items():
        if w["pnl"] >= min_weekly_pnl:
            wallets_to_check.add(wallet)

    # Add movers (climbing the all-time ranks)
    for wallet, m in movers_map.items():
        if m.rank_change > 0 or m.is_new:
            wallets_to_check.add(wallet)

    for wallet in wallets_to_check:
        at = alltime.get(wallet)
        wk = weekly.get(wallet)
        mv = movers_map.get(wallet)

        alltime_rank = at["rank"] if at else 99999
        alltime_pnl = at["pnl"] if at else 0
        weekly_rank = wk["rank"] if wk else 99999
        weekly_pnl = wk["pnl"] if wk else 0
        rank_change = mv.rank_change if mv else 0
        pnl_change = mv.pnl_change if mv else 0
        is_new = mv.is_new if mv else (wallet not in alltime)
        username = ""
        if wk:
            username = wk["username"]
        elif at:
            username = at["username"]
        if mv and not username:
            username = mv.username

        if alltime_pnl < min_alltime_pnl:
            continue

        # Score: weighted combination of signals
        # - Weekly PnL (most important — they're winning NOW)
        # - Rank climb (they're improving)
        # - New entrant bonus
        score = 0.0
        score += weekly_pnl * 1.0  # $1 weekly pnl = 1 point
        score += rank_change * 10.0  # each rank climbed = 10 points
        if is_new:
            score += 5000  # bonus for new entrants
        # Penalize very low all-time PnL (might be noise)
        if alltime_pnl < 0:
            score *= 0.5

        candidates.append(
            InterestingWallet(
                wallet=wallet,
                username=username,
                alltime_rank=alltime_rank,
                alltime_pnl=alltime_pnl,
                weekly_rank=weekly_rank,
                weekly_pnl=weekly_pnl,
                rank_change=rank_change,
                pnl_change=pnl_change,
                is_new_to_leaderboard=is_new,
                score=score,
            )
        )

    candidates.sort(key=lambda w: w.score, reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def _print_section(title: str):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"

    if cmd == "snapshot":
        # Usage: python polymarket.py snapshot [k]
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
        _print_section(f"Taking snapshot (k={k})")
        ids = poly_snapshot(k=k)
        print(f"\n  Snapshot IDs: {ids}")

    elif cmd == "movers":
        # Usage: python polymarket.py movers [days]
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        _print_section(f"Movers (last {days} days)")
        try:
            movers = poly_movers(days=days)
            for m in movers[:20]:
                new_tag = " [NEW]" if m.is_new else ""
                print(
                    f"  {m.username or m.wallet[:12]:20s}  "
                    f"rank: {m.previous_rank:>6d} -> {m.current_rank:>6d}  "
                    f"({m.rank_change:>+6d})  "
                    f"PnL: ${m.current_pnl:>12,.2f}  "
                    f"({m.pnl_change:>+12,.2f}){new_tag}"
                )
        except ValueError as e:
            print(f"  {e}")

    elif cmd == "interesting":
        # Usage: python polymarket.py interesting
        _print_section("Interesting Wallets")
        try:
            wallets = interesting_wallets()
            for w in wallets[:20]:
                new_tag = " [NEW]" if w.is_new_to_leaderboard else ""
                print(
                    f"  {w.username or w.wallet[:12]:20s}  "
                    f"score={w.score:>10,.0f}  "
                    f"week_pnl=${w.weekly_pnl:>10,.2f}  "
                    f"all_rank={w.alltime_rank:>6d}  "
                    f"rank_chg={w.rank_change:>+6d}  "
                    f"all_pnl=${w.alltime_pnl:>12,.2f}{new_tag}"
                )
        except ValueError as e:
            print(f"  {e}")

    elif cmd == "pnl":
        # Usage: python polymarket.py pnl <wallet> [--budget N]
        if len(sys.argv) < 3:
            print("Usage: python polymarket.py pnl <wallet> [--budget N]")
            sys.exit(1)
        pnl_wallet = sys.argv[2]
        pnl_budget = None
        if "--budget" in sys.argv:
            idx = sys.argv.index("--budget")
            if idx + 1 < len(sys.argv):
                pnl_budget = float(sys.argv[idx + 1])

        label = f"PnL History for {pnl_wallet[:16]}..."
        if pnl_budget is not None:
            label += f" (copy-trade budget=${pnl_budget:,.0f})"
        _print_section(label)

        print("  Fetching trades and redemptions...", flush=True)
        history = poly_wallet_pnl_history(pnl_wallet, budget=pnl_budget)

        print(f"  Trades: {history.total_trades}  Redeems: {history.total_redeems}")
        print(f"  Scale factor: {history.scale_factor:.4f}")
        if history.first_trade:
            first = datetime.fromtimestamp(history.first_trade, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            last = datetime.fromtimestamp(history.last_trade, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"  Period: {first} -> {last}")
        print(f"  Realized PnL:   ${history.total_pnl:>14,.2f}")
        print(f"  Unrealized PnL: ${history.unrealized_pnl:>14,.2f}")
        print(f"  MTM PnL:        ${history.mtm_pnl:>14,.2f}")

        # Print a sample of the curve (first 5, last 5)
        if history.curve:
            print(f"\n  PnL curve ({len(history.curve)} points):")
            show = history.curve[:5]
            if len(history.curve) > 10:
                show += [None]  # separator
                show += history.curve[-5:]
            elif len(history.curve) > 5:
                show += history.curve[5:]

            for pt in show:
                if pt is None:
                    print(f"  {'...':>60s}")
                    continue
                ts = datetime.fromtimestamp(pt.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                print(
                    f"  {ts}  {pt.event_type:6s}  "
                    f"${pt.usdc_amount:>+12,.2f}  "
                    f"cum=${pt.cumulative_pnl:>14,.2f}  "
                    f"{pt.market_title[:35]}"
                )

    elif cmd == "test":
        # Quick smoke test of the original 6 functions
        _print_section("Top 5 Traders by Profit (all-time)")
        for t in poly_top_k(5):
            print(f"  #{t.rank} {t.name:20s}  ${t.pnl:>14,.2f}  vol=${t.volume:>14,.2f}")

        _print_section("Top 5 Traders by Profit (this week)")
        for t in poly_top_k(5, period="week"):
            print(f"  #{t.rank} {t.name:20s}  ${t.pnl:>14,.2f}  vol=${t.volume:>14,.2f}")

        # Use an active trader for drill-down
        weekly_top = poly_top_k(1, period="week")
        if weekly_top:
            wallet = weekly_top[0].wallet
            name = weekly_top[0].name
            print(f"\n  Using wallet: {wallet} ({name})")

            _print_section(f"Recent Trades for {name}")
            for t in poly_user_trades(wallet, limit=5):
                print(
                    f"  {t.side:4s} {t.size:>10.2f} @ ${t.price:.4f}  "
                    f"{t.outcome:5s}  {t.market_title[:45]}"
                )

            _print_section(f"Active Positions for {name}")
            positions = poly_user_positions(wallet, active_only=True)
            for p in positions[:5]:
                print(
                    f"  {p.outcome:5s} {p.size:>10.2f} @ ${p.avg_price:.4f}  "
                    f"PnL: ${p.cash_pnl:>10,.2f}  {p.market_title[:40]}"
                )
            if len(positions) > 5:
                print(f"  ... and {len(positions) - 5} more")

            _print_section(f"Profile for {name}")
            profile = poly_user_profile(wallet)
            print(f"  Portfolio value:  ${profile.portfolio_value:>14,.2f}")
            print(f"  Total positions:  {profile.total_positions}")
            print(f"  Active positions: {profile.active_positions}")
            print(f"  Total PnL:        ${profile.total_pnl:>14,.2f}")

    else:
        print("Usage: python polymarket.py [test|snapshot|movers|interesting|pnl]")
        print()
        print("  test                          Smoke test the basic API functions")
        print("  snapshot [k]                  Take a leaderboard snapshot (default k=10000)")
        print("  movers [days]                 Show who's climbing/falling (default 7 days)")
        print("  interesting                   Show interesting wallets to follow")
        print("  pnl <wallet> [--budget N]     PnL history (add --budget for copy-trade sim)")
