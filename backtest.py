"""Point-in-time backtest for the crypto strategy.

Signals are calculated after a completed daily candle. A simulated entry is
allowed only at the next day's open. Incomplete forward periods are excluded.
This module never places real orders.
"""

from pathlib import Path

import pandas as pd

from strategy.indicators import add_relative_strength
from strategy.scanner import scan_market
from strategy.signals import BUY_CANDIDATE, generate_signals


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
HISTORY_DIRECTORY = DATA_DIRECTORY / "history"
APPROVED_FILE = DATA_DIRECTORY / "approved_assets.txt"

TRADES_FILE = DATA_DIRECTORY / "backtest_trades.csv"
SUMMARY_FILE = DATA_DIRECTORY / "backtest_summary.csv"

HOLDING_PERIODS = (1, 3, 7, 30)
MAX_CANDIDATES = 3

# 0.20 % ostossa ja 0.20 % myynnissä:
# kaupankäyntikulu sekä varovainen arvio slippagesta.
COST_PER_SIDE = 0.002


def load_approved_symbols() -> list[str]:
    """Load only the explicitly approved asset universe."""

    if not APPROVED_FILE.exists():
        raise FileNotFoundError(
            "Approved asset list is missing. Run data/market_data.py first."
        )

    symbols = [
        line.strip()
        for line in APPROVED_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if "BTC" not in symbols:
        raise ValueError("BTC must be included in approved_assets.txt")

    return symbols


def load_market_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load validated daily OHLCV histories."""

    datasets = {}

    for symbol in symbols:
        path = HISTORY_DIRECTORY / f"{symbol.lower()}_usd_1d.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing approved history file: {path}"
            )

        data = pd.read_csv(path)
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            utc=True,
            errors="raise",
        )

        datasets[symbol] = data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    return datasets


def build_indicator_data(
    raw_datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Calculate indicators using only current and earlier candles."""

    bitcoin = raw_datasets["BTC"]

    return {
        symbol: add_relative_strength(data, bitcoin)
        for symbol, data in raw_datasets.items()
    }


def net_return(entry_price: float, exit_price: float) -> float:
    """Calculate return after estimated costs on both sides."""

    paid = float(entry_price) * (1.0 + COST_PER_SIDE)
    received = float(exit_price) * (1.0 - COST_PER_SIDE)

    return (received / paid) - 1.0


def next_row_position(
    data: pd.DataFrame,
    signal_time: pd.Timestamp,
) -> int | None:
    """Find the first candle strictly after the signal candle."""

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(data["timestamp"], utc=True)
    )

    position = int(timestamps.searchsorted(signal_time, side="right"))

    if position >= len(data):
        return None

    return position


def run_backtest(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run a non-overlapping point-in-time event backtest."""

    bitcoin_dates = pd.DatetimeIndex(
        datasets["BTC"]["timestamp"]
    )

    trades = []

    # Prevent repeated daily signals from creating overlapping trades
    # for the same asset and holding period.
    blocked_until: dict[tuple[str, int], pd.Timestamp] = {}

    for signal_time in bitcoin_dates:
        scan = scan_market(datasets, signal_time)
        signals = generate_signals(
            scan,
            max_candidates=MAX_CANDIDATES,
        )

        candidates = signals[
            (signals["signal"] == BUY_CANDIDATE)
            & (signals["selected_candidate"])
        ]

        for _, candidate in candidates.iterrows():
            symbol = str(candidate["symbol"])
            asset = datasets[symbol]

            entry_position = next_row_position(
                asset,
                signal_time,
            )

            if entry_position is None:
                continue

            entry_row = asset.iloc[entry_position]
            entry_time = pd.Timestamp(entry_row["timestamp"])
            entry_price = float(entry_row["open"])

            for holding_days in HOLDING_PERIODS:
                key = (symbol, holding_days)
                previous_exit = blocked_until.get(key)

                if (
                    previous_exit is not None
                    and entry_time <= previous_exit
                ):
                    continue

                exit_position = (
                    entry_position + holding_days - 1
                )

                # Never calculate an unfinished future horizon.
                if exit_position >= len(asset):
                    continue

                exit_row = asset.iloc[exit_position]
                exit_time = pd.Timestamp(exit_row["timestamp"])
                exit_price = float(exit_row["close"])
                result = net_return(entry_price, exit_price)

                trades.append(
                    {
                        "symbol": symbol,
                        "signal_time": signal_time,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "holding_days": holding_days,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": (
                            exit_price / entry_price
                        ) - 1.0,
                        "net_return": result,
                        "win": result > 0,
                        "score": float(candidate["score"]),
                        "market_regime": candidate[
                            "market_regime"
                        ],
                    }
                )

                blocked_until[key] = exit_time

    return pd.DataFrame(trades)


def create_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Create honest per-horizon performance statistics."""

    columns = [
        "holding_days",
        "trades",
        "win_rate_pct",
        "average_return_pct",
        "median_return_pct",
        "best_return_pct",
        "worst_return_pct",
    ]

    if trades.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        trades.groupby("holding_days")
        .agg(
            trades=("net_return", "size"),
            win_rate_pct=("win", "mean"),
            average_return_pct=("net_return", "mean"),
            median_return_pct=("net_return", "median"),
            best_return_pct=("net_return", "max"),
            worst_return_pct=("net_return", "min"),
        )
        .reset_index()
    )

    percentage_columns = [
        "win_rate_pct",
        "average_return_pct",
        "median_return_pct",
        "best_return_pct",
        "worst_return_pct",
    ]

    summary[percentage_columns] = (
        summary[percentage_columns] * 100.0
    ).round(2)

    return summary[columns]


def main() -> None:
    """Load data, run the backtest and save reproducible results."""

    symbols = load_approved_symbols()
    raw_datasets = load_market_data(symbols)
    datasets = build_indicator_data(raw_datasets)

    trades = run_backtest(datasets)
    summary = create_summary(trades)

    trades.to_csv(TRADES_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("CRYPTO SALKKUBOTTI - POINT-IN-TIME BACKTEST")
    print("=" * 55)
    print(f"Approved assets: {len(symbols)}")
    print(f"Completed simulated trades: {len(trades)}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved trades: {TRADES_FILE}")
    print(f"Saved summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
