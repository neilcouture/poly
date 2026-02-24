"""Polymarket Wallet Tracker — Textual TUI Dashboard.

Run:  python3 app.py
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Select, Static

from textual_plotext import PlotextPlot

from polymarket import (
    DATA_API,
    PnLHistory,
    Position,
    TopTrader,
    _get,
    poly_top_k,
    poly_user_positions,
    poly_wallet_pnl_history,
)

PERIODS = [("All Time", "all"), ("Week", "week"), ("Month", "month"), ("Day", "day")]
PERIOD_SECONDS = {"day": 86400, "week": 7 * 86400, "month": 30 * 86400, "all": 0}
FAVORITES_PATH = Path(__file__).parent / "favorites.json"


def _load_favorites() -> dict:
    if FAVORITES_PATH.exists():
        return json.loads(FAVORITES_PATH.read_text())
    return {}


def _save_favorites(favs: dict) -> None:
    FAVORITES_PATH.write_text(json.dumps(favs, indent=2))


def _count_trades(wallet: str, period: str) -> int:
    """Count trades for a wallet in the given period by paginating."""
    import time as _time

    params: dict = {"user": wallet}
    if period != "all":
        secs = PERIOD_SECONDS.get(period, 0)
        if secs:
            params["after"] = int(_time.time()) - secs

    count = 0
    offset = 0
    page_size = 100
    while True:
        try:
            data = _get(
                f"{DATA_API}/trades",
                {**params, "limit": page_size, "offset": offset},
            )
        except Exception:
            break
        if not data:
            break
        count += len(data)
        if len(data) < page_size:
            break
        offset += page_size
    return count


class FavoritesModal(ModalScreen[str | None]):
    """Modal showing favorite wallets."""

    DEFAULT_CSS = """
    FavoritesModal {
        align: center middle;
    }

    #fav-container {
        width: 70;
        max-height: 80%;
        background: #1a1a2e;
        border: round #d946ef;
        border-title-color: #d946ef;
        border-title-style: bold;
        padding: 1 2;
    }

    #fav-table {
        height: 1fr;
        max-height: 20;
        scrollbar-size: 1 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._favs = _load_favorites()
        self._row_wallets: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="fav-container") as v:
            v.border_title = " Favorites "
            if not self._favs:
                yield Static("No favorites yet. Press [bold]f[/bold] on a wallet to add one.")
            else:
                yield DataTable(id="fav-table", cursor_type="row")

    def on_mount(self) -> None:
        if not self._favs:
            return
        table = self.query_one("#fav-table", DataTable)
        table.add_columns("Name", "Wallet", "Added")
        for i, (wallet, info) in enumerate(self._favs.items()):
            table.add_row(
                info.get("name", ""),
                wallet[:16] + "...",
                info.get("added", ""),
            )
            self._row_wallets[i] = wallet

    @on(DataTable.RowSelected, "#fav-table")
    def _on_fav_selected(self, event: DataTable.RowSelected) -> None:
        wallet = self._row_wallets.get(event.cursor_row)
        self.dismiss(wallet)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SearchModal(ModalScreen[str | None]):
    """Modal for searching wallets by username."""

    DEFAULT_CSS = """
    SearchModal {
        align: center middle;
    }

    #search-container {
        width: 70;
        max-height: 80%;
        background: #1a1a2e;
        border: round #6cb4ee;
        border-title-color: #6cb4ee;
        border-title-style: bold;
        padding: 1 2;
    }

    #search-input {
        margin-bottom: 1;
        border: round #444;
    }

    #search-results {
        height: 1fr;
        max-height: 20;
        scrollbar-size: 1 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def __init__(self, wallets: list[TopTrader]) -> None:
        super().__init__()
        self._wallets = wallets
        self._filtered: list[TopTrader] = wallets[:]
        self._result_by_row: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="search-container") as v:
            v.border_title = "Search Wallets"
            yield Input(placeholder="Type to filter...", id="search-input")
            yield DataTable(id="search-results", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#search-results", DataTable)
        table.add_columns("Rank", "Name", "PnL")
        self._update_results("")
        self.query_one("#search-input", Input).focus()

    @on(Input.Changed, "#search-input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._update_results(event.value)

    @on(Input.Submitted, "#search-input")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        # Select the first result
        if self._filtered:
            self.dismiss(self._filtered[0].wallet)
        else:
            self.dismiss(None)

    @on(DataTable.RowSelected, "#search-results")
    def _on_result_selected(self, event: DataTable.RowSelected) -> None:
        wallet = self._result_by_row.get(event.cursor_row)
        self.dismiss(wallet)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _update_results(self, query: str) -> None:
        q = query.strip().lower()
        if q:
            self._filtered = [
                t for t in self._wallets
                if q in (t.name or "").lower() or q in t.wallet.lower()
            ]
        else:
            self._filtered = self._wallets[:]

        table = self.query_one("#search-results", DataTable)
        table.clear()
        self._result_by_row = {}
        for i, t in enumerate(self._filtered[:50]):
            table.add_row(str(t.rank), t.name or t.wallet[:12], _fmt_usd(t.pnl))
            self._result_by_row[i] = t.wallet


def _fmt_usd(v: float) -> str:
    """Format a dollar value compactly: $1.2M, $340K, $52, -$3K."""
    neg = v < 0
    v = abs(v)
    if v >= 1_000_000:
        s = f"${v / 1_000_000:,.1f}M"
    elif v >= 1_000:
        s = f"${v / 1_000:,.1f}K"
    else:
        s = f"${v:,.0f}"
    return f"-{s}" if neg else s


class PolyApp(App):
    TITLE = "Polymarket Tracker"
    SUB_TITLE = "Wallet PnL Dashboard"

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 1fr 1fr auto;
    }

    #wallet-table-pane {
        row-span: 1;
        column-span: 1;
        border: round #444;
        border-title-color: #6cb4ee;
        border-title-style: bold;
    }

    #detail-pane {
        row-span: 1;
        column-span: 1;
        border: round #444;
        border-title-color: #6cb4ee;
        border-title-style: bold;
    }

    #chart-pane {
        row-span: 1;
        column-span: 2;
        border: round #444;
        border-title-color: #6cb4ee;
        border-title-style: bold;
        height: 100%;
    }

    #controls {
        height: 3;
        dock: top;
    }

    #period-select {
        width: 20;
    }

    #budget-input {
        width: 20;
    }

    #wallet-count {
        width: auto;
        padding: 1 2 0 2;
        color: $text-muted;
    }

    #detail-placeholder {
        padding: 0 2;
        color: $text-muted;
        text-style: italic;
    }

    #detail-stats {
        padding: 0 2;
        height: auto;
        color: #50c878;
    }

    #budget-stats {
        padding: 0 2;
        height: auto;
        color: #50c878;
    }

    #budget-stats.has-budget {
        color: #d946ef;
    }

    #position-info {
        padding: 0 2;
        height: auto;
        color: #e8a838;
    }

    .hidden {
        display: none;
    }

    #positions-table {
        height: 1fr;
    }

    DataTable {
        scrollbar-size: 1 1;
    }

    DataTable > .datatable--cursor {
        background: #264f78;
    }

    #pnl-chart {
        height: 100%;
    }

    Footer {
        background: #1a1a2e;
    }

    Header {
        background: #16213e;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "set_period('all')", "All Time"),
        Binding("2", "set_period('week')", "Week"),
        Binding("3", "set_period('month')", "Month"),
        Binding("4", "set_period('day')", "Day"),
        Binding("b", "focus_budget", "Budget"),
        Binding("r", "refresh_leaderboard", "Refresh"),
        Binding("s", "search", "Search"),
        Binding("f", "toggle_favorite", "Fav"),
        Binding("i", "show_favorites", "Favs"),
    ]

    period: reactive[str] = reactive("all")
    budget: reactive[float | None] = reactive(None)
    selected_wallet: reactive[str | None] = reactive(None)

    def __init__(self) -> None:
        super().__init__()
        self._wallets: list[TopTrader] = []
        self._wallet_by_row: dict[int, str] = {}
        self._current_pnl: PnLHistory | None = None
        self._position_by_row: dict[int, str] = {}  # row -> market_title
        self._market_info: dict[str, dict] = {}  # market_title -> detail dict
        self._open_markets: set[str] = set()  # market_titles of OPEN positions

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="wallet-table-pane") as pane:
            pane.border_title = " Leaderboard "
            with Horizontal(id="controls"):
                yield Select(PERIODS, value="all", id="period-select", allow_blank=False)
                yield Input(placeholder="Budget $", id="budget-input", type="number")
                yield Static("Loading...", id="wallet-count")
            yield DataTable(id="wallet-table", cursor_type="row")
        with VerticalScroll(id="detail-pane") as dp:
            dp.border_title = " Positions "
            yield Static("Select a wallet from the table...", id="detail-placeholder")
            yield Static("", id="detail-stats")
            yield Static("", id="budget-stats")
            yield Static("", id="position-info")
            yield DataTable(id="positions-table", cursor_type="row")
        with Vertical(id="chart-pane") as cp:
            cp.border_title = " PnL Chart "
            yield PlotextPlot(id="pnl-chart")
        yield Footer()

    def on_mount(self) -> None:
        # Set up wallet table columns
        table = self.query_one("#wallet-table", DataTable)
        table.add_columns("Rank", "Name", "PnL", "Volume", ("# Trades", "trades"))

        # Set up positions table columns
        pos_table = self.query_one("#positions-table", DataTable)
        pos_table.add_column("", width=4)
        pos_table.add_column("Market", width=40)
        pos_table.add_column("Side")
        pos_table.add_column("Size", width=10)
        pos_table.add_column("Avg", width=7)
        pos_table.add_column("Cur", width=7)
        pos_table.add_column("PnL", width=10)

        # Clear chart
        chart_widget = self.query_one("#pnl-chart", PlotextPlot)
        plt = chart_widget.plt
        plt.clear_data()
        plt.theme("dark")
        plt.title("Select a wallet to view PnL")
        chart_widget.refresh()

        self._load_leaderboard()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "period-select" and event.value is not None:
            self.period = str(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "budget-input":
            val = event.value.strip()
            self.budget = float(val) if val else None
            # Return focus to the table
            self.query_one("#wallet-table", DataTable).focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "wallet-table":
            return
        if event.cursor_row is not None and event.cursor_row in self._wallet_by_row:
            self.selected_wallet = self._wallet_by_row[event.cursor_row]

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "positions-table":
            return
        if self._current_pnl is None:
            return
        market = self._position_by_row.get(event.cursor_row)
        self._populate_chart(self._current_pnl, highlight_market=market)
        self._show_position_info(market)

    def _show_position_info(self, market: str | None) -> None:
        widget = self.query_one("#position-info", Static)
        if not market or market not in self._market_info:
            widget.add_class("hidden")
            return
        widget.remove_class("hidden")

        info = self._market_info[market]
        lines = [f"[bold]{info['market']}[/bold]  ({info['status']})"]

        # Trade activity
        parts = []
        if info["buys"]:
            parts.append(f"{info['buys']} buys ({_fmt_usd(info['buy_total'])})")
        if info["sells"]:
            parts.append(f"{info['sells']} sells ({_fmt_usd(info['sell_total'])})")
        if info["redeems"]:
            parts.append(f"{info['redeems']} redeems ({_fmt_usd(info['redeem_total'])})")
        lines.append("  ".join(parts))

        # Position details if available from API
        if "size" in info:
            lines.append(
                f"{info['outcome']}  size: {info['size']:,.1f}  "
                f"avg: ${info['avg_price']:.3f}  cur: ${info['cur_price']:.3f}  "
                f"value: {_fmt_usd(info['current_value'])}"
            )
            lines.append(
                f"PnL: {_fmt_usd(info['cash_pnl'])}  "
                f"({info['percent_pnl']:+.1f}%)  "
                f"realized: {_fmt_usd(info['realized_pnl'])}"
            )

        lines.append(f"Net flow: {_fmt_usd(info['net_flow'])}  |  {info['first']} to {info['last']}")

        widget.update("\n".join(lines))

    # ------------------------------------------------------------------
    # Actions (keybindings)
    # ------------------------------------------------------------------

    def action_set_period(self, period: str) -> None:
        self.period = period
        self.query_one("#period-select", Select).value = period

    def action_focus_budget(self) -> None:
        self.query_one("#budget-input", Input).focus()

    def action_refresh_leaderboard(self) -> None:
        self._load_leaderboard()

    def action_search(self) -> None:
        if not self._wallets:
            return

        def _on_dismiss(wallet: str | None) -> None:
            if wallet:
                self.selected_wallet = wallet
                table = self.query_one("#wallet-table", DataTable)
                for row_idx, w in self._wallet_by_row.items():
                    if w == wallet:
                        table.move_cursor(row=row_idx)
                        break

        self.push_screen(SearchModal(self._wallets), _on_dismiss)

    def action_toggle_favorite(self) -> None:
        wallet = self.selected_wallet
        if not wallet:
            # Use highlighted row from leaderboard
            table = self.query_one("#wallet-table", DataTable)
            row = table.cursor_row
            wallet = self._wallet_by_row.get(row)
        if not wallet:
            return

        name = wallet[:12]
        for t in self._wallets:
            if t.wallet == wallet:
                name = t.name or wallet[:12]
                break

        favs = _load_favorites()
        if wallet in favs:
            del favs[wallet]
            self.notify(f"Removed {name} from favorites")
        else:
            favs[wallet] = {
                "name": name,
                "wallet": wallet,
                "added": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
            self.notify(f"Added {name} to favorites")
        _save_favorites(favs)

        # Refresh the table to show/hide star
        if self._wallets:
            self._populate_wallet_table(self._wallets, self.period)

    def action_show_favorites(self) -> None:
        def _on_dismiss(wallet: str | None) -> None:
            if wallet:
                self.selected_wallet = wallet
                table = self.query_one("#wallet-table", DataTable)
                for row_idx, w in self._wallet_by_row.items():
                    if w == wallet:
                        table.move_cursor(row=row_idx)
                        break

        self.push_screen(FavoritesModal(), _on_dismiss)

    # ------------------------------------------------------------------
    # Watchers
    # ------------------------------------------------------------------

    def watch_period(self, new_period: str) -> None:
        self._load_leaderboard()

    def watch_selected_wallet(self, wallet: str | None) -> None:
        if wallet:
            self._load_wallet_detail()

    def watch_budget(self, new_budget: float | None) -> None:
        if self.selected_wallet:
            self._load_wallet_detail()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=True, group="leaderboard")
    def _load_leaderboard(self) -> None:
        period = self.period
        self.call_from_thread(self._set_wallet_count, "Loading...")

        traders = poly_top_k(100, period=period)

        # Check staleness
        if self.period != period:
            return

        self.call_from_thread(self._populate_wallet_table, traders, period)

    def _set_wallet_count(self, text: str) -> None:
        self.query_one("#wallet-count", Static).update(text)

    def _populate_wallet_table(self, traders: list[TopTrader], period: str) -> None:
        self._wallets = traders
        table = self.query_one("#wallet-table", DataTable)
        table.clear()
        self._wallet_by_row = {}
        self._row_keys: dict[int, str] = {}

        favs = _load_favorites()
        for i, t in enumerate(traders):
            name = t.name or t.wallet[:10]
            if t.wallet in favs:
                name = f"* {name}"
            row_key = table.add_row(
                str(t.rank),
                name,
                _fmt_usd(t.pnl),
                _fmt_usd(t.volume),
                "...",
            )
            self._wallet_by_row[i] = t.wallet
            self._row_keys[i] = row_key

        period_label = dict(PERIODS).get(period, period)
        self.query_one("#wallet-count", Static).update(
            f"{len(traders)} wallets ({period_label})"
        )

        # Re-select first row if we had a selection
        if traders:
            table.move_cursor(row=0)

        # Kick off async trade count loading
        self._load_trade_counts(period)

    @work(thread=True, exclusive=True, group="trade-counts")
    def _load_trade_counts(self, period: str) -> None:
        """Fetch trade counts for each wallet in the background."""
        table_wallets = list(self._wallet_by_row.items())

        def _fetch(item: tuple[int, str]) -> tuple[int, int]:
            row_idx, wallet = item
            return row_idx, _count_trades(wallet, period)

        with ThreadPoolExecutor(max_workers=10) as pool:
            for row_idx, count in pool.map(_fetch, table_wallets):
                if self.period != period:
                    return
                self.call_from_thread(self._update_trade_count, row_idx, str(count))

    def _update_trade_count(self, row_idx: int, count_label: str) -> None:
        row_key = self._row_keys.get(row_idx)
        if row_key is None:
            return
        table = self.query_one("#wallet-table", DataTable)
        try:
            table.update_cell(row_key, "trades", count_label)
        except Exception:
            pass  # row may have been cleared

    @work(thread=True, exclusive=True, group="wallet-detail")
    def _load_wallet_detail(self) -> None:
        wallet = self.selected_wallet
        budget = self.budget

        if not wallet:
            return

        self.call_from_thread(self._show_detail_loading, wallet)

        # Fetch PnL history and positions in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            pnl_future = pool.submit(poly_wallet_pnl_history, wallet, budget)
            pos_future = pool.submit(poly_user_positions, wallet, False)

            pnl_history: PnLHistory = pnl_future.result()
            positions: list[Position] = pos_future.result()

        # Check staleness
        if self.selected_wallet != wallet:
            return

        self.call_from_thread(self._populate_detail, wallet, pnl_history, positions)
        self.call_from_thread(self._populate_chart, pnl_history, None)

    def _show_detail_loading(self, wallet: str) -> None:
        # Find wallet name
        name = wallet[:12]
        for t in self._wallets:
            if t.wallet == wallet:
                name = t.name or wallet[:12]
                break
        self.query_one("#detail-pane").border_title = f" Positions — loading {name}... "
        self.query_one("#detail-placeholder", Static).add_class("hidden")
        self.query_one("#detail-stats", Static).update("Fetching data...")
        self.query_one("#budget-stats", Static).add_class("hidden")
        self.query_one("#position-info", Static).add_class("hidden")
        self.query_one("#positions-table", DataTable).clear()

    def _populate_detail(
        self,
        wallet: str,
        pnl: PnLHistory,
        positions: list[Position],
    ) -> None:
        # Find wallet name
        name = wallet[:12]
        for t in self._wallets:
            if t.wallet == wallet:
                name = t.name or wallet[:12]
                break

        self.query_one("#detail-pane").border_title = f" Positions — {name} "
        self.query_one("#detail-placeholder", Static).add_class("hidden")

        # Build wallet stats (green)
        period_str = ""
        if pnl.first_trade:
            first = datetime.fromtimestamp(pnl.first_trade, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
            last = datetime.fromtimestamp(pnl.last_trade, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
            period_str = f"  |  {first} to {last}"

        stats = (
            f"Trades: {pnl.total_trades}  Redeems: {pnl.total_redeems}"
            f"{period_str}"
        )
        self.query_one("#detail-stats", Static).update(stats)

        # Build PnL section: green for actuals, magenta for budget projection
        budget_widget = self.query_one("#budget-stats", Static)
        if pnl.budget:
            budget_widget.add_class("has-budget")
            budget_widget.update(
                f"[bold]Copy-trade ${pnl.budget:,.0f}[/bold]  "
                f"(scale: {pnl.scale_factor:.4f})\n"
                f"[bold]Projected MTM:[/bold] {_fmt_usd(pnl.mtm_pnl)}  "
                f"[bold]Realized:[/bold] {_fmt_usd(pnl.total_pnl)}  "
                f"[bold]Unrealized:[/bold] {_fmt_usd(pnl.unrealized_pnl)}"
            )
        else:
            budget_widget.remove_class("has-budget")
            budget_widget.update(
                f"[bold]MTM PnL:[/bold] {_fmt_usd(pnl.mtm_pnl)}  "
                f"[bold]Realized:[/bold] {_fmt_usd(pnl.total_pnl)}  "
                f"[bold]Unrealized:[/bold] {_fmt_usd(pnl.unrealized_pnl)}"
            )
        budget_widget.remove_class("hidden")

        # Store PnL for chart highlighting
        self._current_pnl = pnl

        # Build position lookup from API data
        pos_by_market: dict[str, Position] = {}
        for p in positions:
            if p.market_title:
                if p.market_title not in pos_by_market or abs(p.cash_pnl) > abs(
                    pos_by_market[p.market_title].cash_pnl
                ):
                    pos_by_market[p.market_title] = p

        # Collect per-market stats from the PnL curve
        curve_stats: dict[str, dict] = {}
        for pt in pnl.curve:
            if pt.event_type == "MTM":
                continue
            if pt.market_title not in curve_stats:
                curve_stats[pt.market_title] = {
                    "net_flow": 0.0, "buys": 0, "sells": 0, "redeems": 0,
                    "buy_total": 0.0, "sell_total": 0.0, "redeem_total": 0.0,
                    "first_ts": pt.timestamp, "last_ts": pt.timestamp,
                }
            s = curve_stats[pt.market_title]
            s["net_flow"] += pt.usdc_amount
            s["last_ts"] = pt.timestamp
            if pt.event_type == "BUY":
                s["buys"] += 1
                s["buy_total"] += abs(pt.usdc_amount)
            elif pt.event_type == "SELL":
                s["sells"] += 1
                s["sell_total"] += pt.usdc_amount
            elif pt.event_type == "REDEEM":
                s["redeems"] += 1
                s["redeem_total"] += pt.usdc_amount

        # Merge: every curve market gets a row + detailed info
        self._open_markets = set()
        self._market_info = {}
        rows = []
        for market_title, cs in curve_stats.items():
            p = pos_by_market.get(market_title)
            is_open = p is not None and p.size > 0 and p.current_value > 0
            status = "OPEN" if is_open else "----"
            if is_open:
                self._open_markets.add(market_title)

            pnl_val = p.cash_pnl if p else cs["net_flow"]

            info = {
                "status": status,
                "market": market_title,
                "buys": cs["buys"], "buy_total": cs["buy_total"],
                "sells": cs["sells"], "sell_total": cs["sell_total"],
                "redeems": cs["redeems"], "redeem_total": cs["redeem_total"],
                "net_flow": cs["net_flow"],
                "first": datetime.fromtimestamp(cs["first_ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "last": datetime.fromtimestamp(cs["last_ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            }
            if p:
                info.update({
                    "outcome": p.outcome, "size": p.size,
                    "avg_price": p.avg_price, "cur_price": p.cur_price,
                    "current_value": p.current_value,
                    "cash_pnl": p.cash_pnl, "percent_pnl": p.percent_pnl,
                    "realized_pnl": p.realized_pnl,
                })
            self._market_info[market_title] = info

            side = p.outcome if p else ""
            size = f"{p.size:,.1f}" if p else ""
            avg = f"${p.avg_price:.3f}" if p else ""
            cur = f"${p.cur_price:.3f}" if p else ""
            rows.append((
                status, market_title, side, size, avg, cur,
                _fmt_usd(pnl_val), 0 if is_open else 1, abs(pnl_val),
            ))

        # Sort: OPEN first, then by |PnL| desc
        rows.sort(key=lambda r: (r[7], -r[8]))

        pos_table = self.query_one("#positions-table", DataTable)
        pos_table.clear()
        self._position_by_row = {}
        for i, row in enumerate(rows):
            pos_table.add_row(*row[:7])
            self._position_by_row[i] = row[1]

    def _populate_chart(
        self, pnl: PnLHistory, highlight_market: str | None = None
    ) -> None:
        chart_widget = self.query_one("#pnl-chart", PlotextPlot)
        plt = chart_widget.plt
        plt.clear_data()
        plt.clear_figure()

        if not pnl.curve:
            plt.title("No trade history")
            chart_widget.refresh()
            return

        curve = pnl.curve
        timestamps = [pt.timestamp for pt in curve]
        values = [pt.cumulative_pnl for pt in curve]

        # Compute highlight timestamp range from full curve
        hl_ts_lo: int | None = None
        hl_ts_hi: int | None = None
        if highlight_market:
            for pt in curve:
                if pt.market_title == highlight_market:
                    if hl_ts_lo is None:
                        hl_ts_lo = pt.timestamp
                    hl_ts_hi = pt.timestamp
            if hl_ts_lo is not None and highlight_market in self._open_markets:
                hl_ts_hi = timestamps[-1]

        # Downsample if too many points (keep timestamps aligned)
        if len(values) > 200:
            step = len(values) // 200
            timestamps = timestamps[::step]
            values = values[::step]

        # Chart theme
        plt.theme("dark")

        # Plot using timestamps as x-values (time-proportional axis)
        base_color = "gray+" if highlight_market else "cyan+"
        plt.plot(timestamps, values, color=base_color, marker="hd")

        # Overlay highlight using timestamp range
        if hl_ts_lo is not None:
            seg_ts = [t for t, v in zip(timestamps, values) if hl_ts_lo <= t <= hl_ts_hi]
            seg_vs = [v for t, v in zip(timestamps, values) if hl_ts_lo <= t <= hl_ts_hi]
            if seg_ts:
                plt.plot(seg_ts, seg_vs, color="yellow+", marker="hd")

        title = f"Cash Flow + MTM — {_fmt_usd(pnl.mtm_pnl)}"
        if highlight_market:
            title += f"  |  {highlight_market[:40]}"
        plt.title(title)

        # Time-based x-axis labels
        if len(timestamps) > 1:
            t_min, t_max = timestamps[0], timestamps[-1]
            n_ticks = min(8, len(timestamps))
            tick_ts = [t_min + i * (t_max - t_min) // n_ticks for i in range(n_ticks + 1)]
            tick_labels = [
                datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
                for t in tick_ts
            ]
            plt.xticks(tick_ts, tick_labels)

        chart_widget.refresh()


if __name__ == "__main__":
    app = PolyApp()
    app.run()
