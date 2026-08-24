"""Download and strictly validate crypto data for the backtester.

Only approved assets are downloaded. The module does not create trading
signals or place real trades.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Asset:
    symbol: str
    ticker: str
    expected_start: date


# Ensimmäisen backtestin tarkastetut ja hyväksytyt kryptot.
APPROVED_ASSETS = (
    Asset("BTC", "BTC-USD", date(2020, 1, 1)),
    Asset("ETH", "ETH-USD", date(2020, 1, 1)),
    Asset("BNB", "BNB-USD", date(2020, 1, 1)),
    Asset("XRP", "XRP-USD", date(2020, 1, 1)),
    Asset("DOGE", "DOGE-USD", date(2020, 1, 1)),
    Asset("TRX", "TRX-USD", date(2020, 1, 1)),
    Asset("LINK", "LINK-USD", date(2020, 1, 1)),
    Asset("XMR", "XMR-USD", date(2020, 1, 1)),
    Asset("ZEC", "ZEC-USD", date(2020, 1, 1)),
    Asset("ADA", "ADA-USD", date(2020, 1, 1)),
    Asset("SUI", "SUI20947-USD", date(2023, 5, 3)),
    Asset("HBAR", "HBAR-USD", date(2020, 1, 1)),
    Asset("NEAR", "NEAR-USD", date(2020, 10, 14)),
    Asset("INJ", "INJ-USD", date(2020, 10, 21)),
    Asset("ONDO", "ONDO-USD", date(2024, 1, 18)),
)

DATA_DIRECTORY = Path(__file__).resolve().parent / "history"
AUDIT_FILE = Path(__file__).resolve().parent / "market_data_audit.csv"
APPROVED_FILE = Path(__file__).resolve().parent / "approved_assets.txt"

DOWNLOAD_START = "2020-01-01"
MINIMUM_ROWS = 200
MAX_START_DELAY_DAYS = 3
MAX_DATA_AGE_DAYS = 3
MAX_DAILY_CHANGE = 10.0

PRICE_COLUMNS = ["open", "high", "low", "close"]
REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def download_asset(asset: Asset) -> pd.DataFrame:
    """Download daily candles from the explicitly defined Yahoo ticker."""

    data = yf.download(
        asset.ticker,
        start=DOWNLOAD_START,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("No data returned by Yahoo Finance")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    wanted = ["open", "high", "low", "close", "volume"]
    missing = [column for column in wanted if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    data = data[wanted].copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data.index.name = "timestamp"
    data = data.reset_index()

    for column in wanted:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Tämän päivän keskeneräinen päiväkynttilä ei kuulu aineistoon.
    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    data = data[data["timestamp"] < today_utc].copy()

    return data.reset_index(drop=True)


def validate_asset(asset: Asset, data: pd.DataFrame) -> dict:
    """Reject incomplete, contaminated or logically impossible data."""

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if len(data) < MINIMUM_ROWS:
        raise ValueError(
            f"Only {len(data)} completed candles; minimum is {MINIMUM_ROWS}"
        )

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing or non-numeric OHLCV values detected")

    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not in chronological order")

    first_date = data["timestamp"].iloc[0].date()
    start_delay = (first_date - asset.expected_start).days

    if start_delay < 0:
        raise ValueError(
            f"History starts too early: {first_date}; "
            f"expected {asset.expected_start}"
        )

    if start_delay > MAX_START_DELAY_DAYS:
        raise ValueError(
            f"History starts {start_delay} days late: {first_date}; "
            f"expected {asset.expected_start}"
        )

    last_day = data["timestamp"].iloc[-1].normalize()
    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    data_age = (today_utc - last_day).days

    if data_age > MAX_DATA_AGE_DAYS:
        raise ValueError(f"Stale data: newest candle is {data_age} days old")

    if (data[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Zero or negative price detected")

    # Nollavolyymi voi paljastaa väärän tai migraatiolla sekoittuneen sarjan.
    if (data["volume"] <= 0).any():
        zero_days = int((data["volume"] <= 0).sum())
        raise ValueError(f"Zero or negative volume on {zero_days} days")

    invalid_high = data["high"] < data[
        ["open", "close", "low"]
    ].max(axis=1)

    invalid_low = data["low"] > data[
        ["open", "close", "high"]
    ].min(axis=1)

    if invalid_high.any() or invalid_low.any():
        raise ValueError("Logically impossible OHLC candle detected")

    expected_dates = pd.date_range(
        start=data["timestamp"].iloc[0],
        end=data["timestamp"].iloc[-1],
        freq="D",
    )

    actual_dates = pd.DatetimeIndex(data["timestamp"])
    missing_dates = expected_dates.difference(actual_dates)

    if len(missing_dates) > 0:
        raise ValueError(
            f"{len(missing_dates)} missing calendar days detected"
        )

    daily_change = data["close"].pct_change(fill_method=None)
    maximum_change = float(daily_change.abs().max())

    if maximum_change > MAX_DAILY_CHANGE:
        raise ValueError(
            f"Suspicious daily price change: {maximum_change:.2%}"
        )

    return {
        "symbol": asset.symbol,
        "ticker": asset.ticker,
        "status": "APPROVED",
        "rows": len(data),
        "first_date": first_date.isoformat(),
        "last_date": data["timestamp"].iloc[-1].date().isoformat(),
        "missing_days": 0,
        "duplicate_days": 0,
        "maximum_daily_change_pct": round(maximum_change * 100, 2),
        "error": "",
    }


def save_asset(asset: Asset, data: pd.DataFrame) -> Path:
    """Save an approved dataset without partially overwriting the old file."""

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    destination = (
        DATA_DIRECTORY / f"{asset.symbol.lower()}_usd_1d.csv"
    )
    temporary = destination.with_suffix(".csv.tmp")

    data.to_csv(temporary, index=False)
    temporary.replace(destination)

    return destination


def save_audit_report(results: list[dict]) -> None:
    """Save the result of every validation attempt."""

    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(AUDIT_FILE, index=False)


def save_approved_list(symbols: list[str]) -> None:
    """Create the only symbol list that the backtester may use."""

    temporary = APPROVED_FILE.with_suffix(".txt.tmp")
    temporary.write_text(
        "\n".join(symbols) + "\n",
        encoding="utf-8",
    )
    temporary.replace(APPROVED_FILE)


def main() -> None:
    print("CRYPTO SALKKUBOTTI - STRICT MARKET DATA")
    print("=" * 55)

    results = []
    approved_symbols = []

    for asset in APPROVED_ASSETS:
        try:
            data = download_asset(asset)
            audit = validate_asset(asset, data)
            destination = save_asset(asset, data)

            results.append(audit)
            approved_symbols.append(asset.symbol)

            print(
                f"{asset.symbol:<6} APPROVED: "
                f"{len(data)} candles -> {destination}"
            )

        except Exception as error:
            results.append(
                {
                    "symbol": asset.symbol,
                    "ticker": asset.ticker,
                    "status": "REJECTED",
                    "rows": 0,
                    "first_date": "",
                    "last_date": "",
                    "missing_days": "",
                    "duplicate_days": "",
                    "maximum_daily_change_pct": "",
                    "error": str(error),
                }
            )

            print(f"{asset.symbol:<6} REJECTED: {error}")

    save_audit_report(results)

    if len(approved_symbols) != len(APPROVED_ASSETS):
        print("=" * 55)
        print(
            f"FAILED: {len(approved_symbols)}/"
            f"{len(APPROVED_ASSETS)} assets approved"
        )
        raise SystemExit(
            "One or more required datasets failed strict validation"
        )

    save_approved_list(approved_symbols)

    print("=" * 55)
    print(f"SUCCESS: {len(approved_symbols)}/15 assets approved")
    print(f"Audit report: {AUDIT_FILE}")
    print(f"Backtest whitelist: {APPROVED_FILE}")


if __name__ == "__main__":
    main()
