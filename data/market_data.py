"""Download and validate daily market data for the crypto backtester.

This module only collects completed daily candles. It does not create
trading signals or place real trades.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yfinance as yf


CRYPTO_SYMBOLS: Tuple[str, ...] = (
    "BTC",
    "ETH",
    "BNB",
    "SOL",
    "XRP",
    "DOGE",
    "TRX",
    "HYPE",
    "LINK",
    "XMR",
    "ZEC",
    "ADA",
    "AVAX",
    "SUI",
    "HBAR",
    "AAVE",
    "UNI",
    "NEAR",
    "RENDER",
    "INJ",
    "TAO",
    "ONDO",
    "SEI",
    "ARB",
    "OP",
)

YAHOO_TICKERS: Dict[str, str] = {
    symbol: f"{symbol}-USD" for symbol in CRYPTO_SYMBOLS
}

DATA_DIRECTORY = Path(__file__).resolve().parent / "history"
START_DATE = "2020-01-01"
MINIMUM_ROWS = 200

REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert downloaded data into a consistent OHLCV format."""

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

    wanted_columns = ["open", "high", "low", "close", "volume"]
    missing = [
        column for column in wanted_columns if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    data = data[wanted_columns].copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data.index.name = "timestamp"
    data = data.reset_index()

    for column in wanted_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["open", "high", "low", "close"])
    data = data.drop_duplicates(subset=["timestamp"], keep="last")
    data = data.sort_values("timestamp").reset_index(drop=True)

    return data


def _remove_incomplete_candles(data: pd.DataFrame) -> pd.DataFrame:
    """Remove today's candle because it is not finished yet."""

    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    return data[data["timestamp"] < today_utc].copy()


def _validate_market_data(
    symbol: str,
    data: pd.DataFrame,
) -> None:
    """Reject incomplete or logically impossible market data."""

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: missing columns: {', '.join(missing)}"
        )

    if len(data) < MINIMUM_ROWS:
        raise ValueError(
            f"{symbol}: only {len(data)} completed candles available"
        )

    if data["timestamp"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate timestamps detected")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{symbol}: timestamps are not ordered")

    price_columns = ["open", "high", "low", "close"]

    if (data[price_columns] <= 0).any().any():
        raise ValueError(f"{symbol}: zero or negative price detected")

    invalid_high = data["high"] < data[
        ["open", "low", "close"]
    ].max(axis=1)

    invalid_low = data["low"] > data[
        ["open", "high", "close"]
    ].min(axis=1)

    if invalid_high.any() or invalid_low.any():
        raise ValueError(f"{symbol}: invalid OHLC candle detected")

    if (data["volume"].dropna() < 0).any():
        raise ValueError(f"{symbol}: negative volume detected")


def download_market_data(
    symbol: str,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """Download and validate one cryptocurrency's daily candles."""

    symbol = symbol.upper()

    if symbol not in YAHOO_TICKERS:
        raise ValueError(f"Unsupported symbol: {symbol}")

    ticker = YAHOO_TICKERS[symbol]

    raw_data = yf.download(
        ticker,
        start=start_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw_data.empty:
        raise ValueError(f"{symbol}: no market data returned")

    data = _normalise_columns(raw_data)
    data = _remove_incomplete_candles(data)
    _validate_market_data(symbol, data)

    return data


def save_market_data(
    symbol: str,
    data: pd.DataFrame,
) -> Path:
    """Save validated data without leaving a half-written CSV file."""

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    destination = DATA_DIRECTORY / f"{symbol.lower()}_usd_1d.csv"
    temporary_file = destination.with_suffix(".tmp")

    data.to_csv(temporary_file, index=False)
    temporary_file.replace(destination)

    return destination


def update_all_market_data() -> Dict[str, str]:
    """Download every configured asset and report individual failures."""

    results: Dict[str, str] = {}

    for symbol in CRYPTO_SYMBOLS:
        try:
            data = download_market_data(symbol)
            destination = save_market_data(symbol, data)

            results[symbol] = (
                f"OK: {len(data)} candles -> {destination}"
            )
        except Exception as error:
            results[symbol] = f"ERROR: {error}"

    return results


def main() -> None:
    """Run the complete market-data update."""

    print("CRYPTO SALKKUBOTTI - MARKET DATA")
    print("=" * 40)

    results = update_all_market_data()
    successful = 0

    for symbol, result in results.items():
        print(f"{symbol:<7} {result}")

        if result.startswith("OK"):
            successful += 1

    print("=" * 40)
    print(
        f"Completed: {successful}/{len(CRYPTO_SYMBOLS)} assets"
    )

    if successful == 0:
        raise SystemExit("No usable market data was downloaded")


if __name__ == "__main__":
    main()
