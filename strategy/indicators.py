"""Technical indicators for the crypto portfolio backtester.

Every indicator uses only the current and earlier candles. Missing warm-up
values remain empty and must never be replaced with future information.
"""

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

EMA_PERIODS = (20, 50, 200)
MOMENTUM_PERIODS = (7, 30, 90)
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_PERIOD = 20
VOLATILITY_PERIOD = 30
ANNUALIZATION_DAYS = 365


def _validate_period(period: int) -> None:
    """Ensure an indicator period is a positive integer."""

    if not isinstance(period, int) or period < 1:
        raise ValueError("Indicator period must be a positive integer")


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy without silently repairing invalid data."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Market data must be a pandas DataFrame")

    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Market data is empty")

    clean = data.copy()
    clean["timestamp"] = pd.to_datetime(
        clean["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = ["open", "high", "low", "close", "volume"]

    for column in numeric_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

    if clean[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Market data contains missing or invalid values")

    if clean["timestamp"].duplicated().any():
        raise ValueError("Market data contains duplicate timestamps")

    clean = clean.sort_values("timestamp").reset_index(drop=True)

    if (clean[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Market data contains zero or negative prices")

    if (clean["volume"] < 0).any():
        raise ValueError("Market data contains negative volume")

    invalid_high = clean["high"] < clean[
        ["open", "close", "low"]
    ].max(axis=1)

    invalid_low = clean["low"] > clean[
        ["open", "close", "high"]
    ].min(axis=1)

    if invalid_high.any() or invalid_low.any():
        raise ValueError("Market data contains invalid OHLC candles")

    return clean


def exponential_moving_average(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate an exponential moving average without future data."""

    _validate_period(period)

    return close.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def relative_strength_index(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """Calculate Wilder-style RSI."""

    _validate_period(period)

    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0,
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50.0,
    )

    return rsi


def average_true_range(
    data: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """Calculate Wilder-style Average True Range."""

    _validate_period(period)

    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def rate_of_change(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate percentage price change over a fixed period."""

    _validate_period(period)
    return close.pct_change(periods=period, fill_method=None)


def historical_volatility(
    close: pd.Series,
    period: int = VOLATILITY_PERIOD,
) -> pd.Series:
    """Calculate annualized rolling volatility from daily returns."""

    _validate_period(period)

    daily_returns = close.pct_change(fill_method=None)

    return (
        daily_returns.rolling(
            window=period,
            min_periods=period,
        ).std()
        * (ANNUALIZATION_DAYS ** 0.5)
    )


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add trend, momentum, volume and risk indicators."""

    result = validate_market_data(data)

    result["return_1d"] = result["close"].pct_change(
        fill_method=None
    )

    for period in EMA_PERIODS:
        result[f"ema_{period}"] = exponential_moving_average(
            result["close"],
            period,
        )

    result["rsi_14"] = relative_strength_index(
        result["close"],
        RSI_PERIOD,
    )

    for period in MOMENTUM_PERIODS:
        result[f"roc_{period}"] = rate_of_change(
            result["close"],
            period,
        )

    result["atr_14"] = average_true_range(
        result,
        ATR_PERIOD,
    )

    result["atr_pct"] = result["atr_14"] / result["close"]

    # Käytetään vain edeltävien päivien keskiarvoa vertailutasona.
    previous_average_volume = (
        result["volume"]
        .shift(1)
        .rolling(
            window=VOLUME_PERIOD,
            min_periods=VOLUME_PERIOD,
        )
        .mean()
    )

    result["volume_ratio_20"] = (
        result["volume"] / previous_average_volume
    )

    result["volatility_30"] = historical_volatility(
        result["close"],
        VOLATILITY_PERIOD,
    )

    result["above_ema_20"] = (
        result["close"] > result["ema_20"]
    )

    result["above_ema_50"] = (
        result["close"] > result["ema_50"]
    )

    result["above_ema_200"] = (
        result["close"] > result["ema_200"]
    )

    result["ema_bullish_order"] = (
        (result["ema_20"] > result["ema_50"])
        & (result["ema_50"] > result["ema_200"])
    )

    return result


def add_relative_strength(
    asset_data: pd.DataFrame,
    bitcoin_data: pd.DataFrame,
    periods: Iterable[int] = MOMENTUM_PERIODS,
) -> pd.DataFrame:
    """Compare an asset's returns with Bitcoin on matching dates."""

    asset = add_indicators(asset_data)
    bitcoin = validate_market_data(bitcoin_data)[
        ["timestamp", "close"]
    ].rename(columns={"close": "btc_close"})

    result = asset.merge(
        bitcoin,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    for period in periods:
        _validate_period(period)

        asset_return = result["close"].pct_change(
            periods=period,
            fill_method=None,
        )

        bitcoin_return = result["btc_close"].pct_change(
            periods=period,
            fill_method=None,
        )

        result[f"relative_strength_{period}"] = (
            asset_return - bitcoin_return
        )

    return result


def indicator_columns() -> tuple[str, ...]:
    """Return the indicator fields required by later strategy modules."""

    return (
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "roc_7",
        "roc_30",
        "roc_90",
        "atr_14",
        "atr_pct",
        "volume_ratio_20",
        "volatility_30",
        "above_ema_20",
        "above_ema_50",
        "above_ema_200",
        "ema_bullish_order",
        "relative_strength_7",
        "relative_strength_30",
        "relative_strength_90",
    )
