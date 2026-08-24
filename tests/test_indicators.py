"""Automatic tests for technical indicators and future-data leakage."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.indicators import (  # noqa: E402
    add_indicators,
    add_relative_strength,
    indicator_columns,
    validate_market_data,
)


def create_market_data(
    daily_growth: float = 0.5,
    rows: int = 260,
) -> pd.DataFrame:
    """Create deterministic daily OHLCV data for testing."""

    timestamps = pd.date_range(
        start="2020-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    close = pd.Series(
        [
            100.0 + (index * daily_growth)
            for index in range(rows)
        ],
        dtype="float64",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.20,
            "high": close + 1.00,
            "low": close - 1.00,
            "close": close,
            "volume": [
                1_000_000 + (index * 1_000)
                for index in range(rows)
            ],
        }
    )


def test_indicator_columns_and_values() -> None:
    """Indicators must exist and produce sensible completed values."""

    result = add_indicators(create_market_data())

    required = {
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
        "ema_bullish_order",
    }

    missing = required.difference(result.columns)

    assert not missing, f"Missing indicator columns: {sorted(missing)}"
    assert len(result) == 260
    assert result["timestamp"].is_monotonic_increasing
    assert result["ema_200"].iloc[:199].isna().all()
    assert pd.notna(result["ema_200"].iloc[199])
    assert 0 <= result["rsi_14"].iloc[-1] <= 100
    assert result["atr_14"].iloc[-1] > 0
    assert result["atr_pct"].iloc[-1] > 0
    assert result["volume_ratio_20"].iloc[-1] > 0


def test_no_future_data_leakage() -> None:
    """Changing the final candle must not alter earlier indicators."""

    original_data = create_market_data()
    original_result = add_indicators(original_data)

    changed_data = original_data.copy()
    last_row = changed_data.index[-1]

    changed_data.loc[last_row, "open"] = 400.0
    changed_data.loc[last_row, "high"] = 510.0
    changed_data.loc[last_row, "low"] = 390.0
    changed_data.loc[last_row, "close"] = 500.0
    changed_data.loc[last_row, "volume"] = 9_000_000

    changed_result = add_indicators(changed_data)

    pd.testing.assert_frame_equal(
        original_result.iloc[:-1].reset_index(drop=True),
        changed_result.iloc[:-1].reset_index(drop=True),
        check_exact=True,
    )


def test_relative_strength_against_bitcoin() -> None:
    """A faster-rising asset must have positive relative strength."""

    asset_data = create_market_data(daily_growth=0.8)
    bitcoin_data = create_market_data(daily_growth=0.2)

    result = add_relative_strength(
        asset_data,
        bitcoin_data,
    )

    for column in (
        "relative_strength_7",
        "relative_strength_30",
        "relative_strength_90",
    ):
        assert column in result.columns
        assert pd.notna(result[column].iloc[-1])
        assert result[column].iloc[-1] > 0


def test_invalid_data_is_rejected() -> None:
    """Duplicate timestamps must be rejected instead of repaired."""

    invalid_data = create_market_data()
    invalid_data.loc[1, "timestamp"] = invalid_data.loc[0, "timestamp"]

    try:
        validate_market_data(invalid_data)

    except ValueError as error:
        assert "duplicate" in str(error).lower()

    else:
        raise AssertionError("Duplicate timestamp was not rejected")


def test_indicator_contract() -> None:
    """The public indicator list must not contain duplicate names."""

    columns = indicator_columns()

    assert len(columns) == len(set(columns))
    assert "ema_200" in columns
    assert "relative_strength_30" in columns


def main() -> None:
    tests = (
        test_indicator_columns_and_values,
        test_no_future_data_leakage,
        test_relative_strength_against_bitcoin,
        test_invalid_data_is_rejected,
        test_indicator_contract,
    )

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 55)
    print(f"SUCCESS: {len(tests)}/{len(tests)} indicator tests passed")


if __name__ == "__main__":
    main()
