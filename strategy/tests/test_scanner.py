"""Automatic tests for the point-in-time market scanner."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.scanner import (  # noqa: E402
    calculate_score,
    market_regime,
    scan_market,
)


def strong_row(timestamp: str) -> dict:
    """Create a strong but realistic bullish setup."""

    return {
        "timestamp": pd.Timestamp(timestamp, tz="UTC"),
        "close": 120.0,
        "ema_20": 110.0,
        "ema_50": 100.0,
        "ema_200": 90.0,
        "rsi_14": 55.0,
        "roc_30": 0.20,
        "roc_90": 0.40,
        "atr_pct": 0.04,
        "volume_ratio_20": 1.50,
        "volatility_30": 0.80,
        "relative_strength_30": 0.15,
        "relative_strength_90": 0.30,
    }


def weak_row(timestamp: str) -> dict:
    """Create a weak bearish setup."""

    return {
        "timestamp": pd.Timestamp(timestamp, tz="UTC"),
        "close": 80.0,
        "ema_20": 90.0,
        "ema_50": 100.0,
        "ema_200": 110.0,
        "rsi_14": 25.0,
        "roc_30": -0.20,
        "roc_90": -0.35,
        "atr_pct": 0.12,
        "volume_ratio_20": 0.60,
        "volatility_30": 1.80,
        "relative_strength_30": -0.25,
        "relative_strength_90": -0.40,
    }


def test_strong_setup_scores_higher() -> None:
    """A strong setup must outrank a weak setup."""

    strong = calculate_score(
        pd.Series(strong_row("2026-01-01"))
    )

    weak = calculate_score(
        pd.Series(weak_row("2026-01-01"))
    )

    assert 0 <= weak.total <= 100
    assert 0 <= strong.total <= 100
    assert strong.total > weak.total
    assert strong.trend == 30.0


def test_market_regime_detection() -> None:
    """Bitcoin EMA structure must control the market regime."""

    bull_data = pd.DataFrame(
        [strong_row("2026-01-01")]
    )

    bear_data = pd.DataFrame(
        [weak_row("2026-01-01")]
    )

    assert (
        market_regime(
            bull_data,
            pd.Timestamp("2026-01-01", tz="UTC"),
        )
        == "BULL"
    )

    assert (
        market_regime(
            bear_data,
            pd.Timestamp("2026-01-01", tz="UTC"),
        )
        == "BEAR"
    )


def test_market_is_sorted_by_score() -> None:
    """The highest-scoring asset must appear first."""

    bitcoin = strong_row("2026-01-01")
    bitcoin["relative_strength_30"] = 0.0
    bitcoin["relative_strength_90"] = 0.0

    ethereum = strong_row("2026-01-01")
    ethereum["relative_strength_30"] = 0.25
    ethereum["relative_strength_90"] = 0.60

    datasets = {
        "BTC": pd.DataFrame([bitcoin]),
        "ETH": pd.DataFrame([ethereum]),
        "DOGE": pd.DataFrame(
            [weak_row("2026-01-01")]
        ),
    }

    result = scan_market(
        datasets,
        pd.Timestamp("2026-01-01", tz="UTC"),
    )

    assert result.iloc[0]["symbol"] == "ETH"
    assert result.iloc[-1]["symbol"] == "DOGE"
    assert result.iloc[0]["score"] > result.iloc[-1]["score"]
    assert (result["market_regime"] == "BULL").all()


def test_future_rows_are_not_used() -> None:
    """A scan must ignore every candle after its cutoff date."""

    bitcoin_now = strong_row("2026-01-01")
    bitcoin_future = weak_row("2026-01-03")

    asset_now = strong_row("2026-01-01")
    asset_future = weak_row("2026-01-03")

    datasets = {
        "BTC": pd.DataFrame(
            [bitcoin_now, bitcoin_future]
        ),
        "ETH": pd.DataFrame(
            [asset_now, asset_future]
        ),
    }

    cutoff = pd.Timestamp("2026-01-02", tz="UTC")
    result = scan_market(datasets, cutoff)

    eth_result = result[result["symbol"] == "ETH"].iloc[0]
    expected = calculate_score(pd.Series(asset_now))

    assert eth_result["timestamp"] == asset_now["timestamp"]
    assert eth_result["score"] == expected.total
    assert eth_result["market_regime"] == "BULL"


def test_insufficient_data_is_not_scored() -> None:
    """Missing indicators must never generate artificial points."""

    bitcoin = strong_row("2026-01-01")
    incomplete = strong_row("2026-01-01")
    del incomplete["relative_strength_90"]

    datasets = {
        "BTC": pd.DataFrame([bitcoin]),
        "INCOMPLETE": pd.DataFrame([incomplete]),
    }

    result = scan_market(
        datasets,
        pd.Timestamp("2026-01-01", tz="UTC"),
    )

    row = result[
        result["symbol"] == "INCOMPLETE"
    ].iloc[0]

    assert row["status"] == "INSUFFICIENT_DATA"
    assert pd.isna(row["score"])


def main() -> None:
    tests = (
        test_strong_setup_scores_higher,
        test_market_regime_detection,
        test_market_is_sorted_by_score,
        test_future_rows_are_not_used,
        test_insufficient_data_is_not_scored,
    )

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 55)
    print(f"SUCCESS: {len(tests)}/{len(tests)} scanner tests passed")


if __name__ == "__main__":
    main()
