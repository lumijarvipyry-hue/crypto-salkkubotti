"""Automatic tests for rule-based strategy signals."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.signals import (  # noqa: E402
    BUY_CANDIDATE,
    NO_TRADE,
    WATCH,
    decide_signal,
    generate_signals,
)


def scan_row(
    symbol: str,
    score: float = 85.0,
    regime: str = "BULL",
    status: str = "SCORED",
) -> dict:
    """Create one valid scanner result for signal testing."""

    return {
        "symbol": symbol,
        "timestamp": pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        ),
        "market_regime": regime,
        "status": status,
        "score": score,
        "trend_score": 25.0,
        "momentum_score": 18.0,
        "relative_strength_score": 17.0,
        "volume_score": 8.0,
        "risk_score": 5.0,
    }


def test_strong_bull_setup_is_candidate() -> None:
    """A complete strong setup may become a buy candidate."""

    decision = decide_signal(
        pd.Series(scan_row("ETH"))
    )

    assert decision.signal == BUY_CANDIDATE
    assert decision.eligible is True


def test_bear_regime_disables_long_entry() -> None:
    """A high score must not create a long entry in a bear regime."""

    decision = decide_signal(
        pd.Series(
            scan_row(
                "ETH",
                score=95.0,
                regime="BEAR",
            )
        )
    )

    assert decision.signal == NO_TRADE
    assert decision.eligible is False
    assert "bear" in decision.reason.lower()


def test_neutral_regime_creates_watch_only() -> None:
    """A strong neutral-regime setup must remain on watch."""

    decision = decide_signal(
        pd.Series(
            scan_row(
                "ETH",
                score=90.0,
                regime="NEUTRAL",
            )
        )
    )

    assert decision.signal == WATCH
    assert decision.eligible is False


def test_insufficient_data_never_creates_signal() -> None:
    """Unverified or incomplete data must produce NO_TRADE."""

    row = scan_row(
        "ETH",
        status="INSUFFICIENT_DATA",
    )
    row["score"] = pd.NA

    decision = decide_signal(
        pd.Series(row)
    )

    assert decision.signal == NO_TRADE
    assert decision.eligible is False
    assert "data" in decision.reason.lower()


def test_candidate_limit_keeps_best_assets() -> None:
    """Only the highest-ranked assets may remain candidates."""

    scan_results = pd.DataFrame(
        [
            scan_row("ETH", score=95.0),
            scan_row("LINK", score=90.0),
            scan_row("SUI", score=85.0),
            scan_row("DOGE", score=55.0),
        ]
    )

    result = generate_signals(
        scan_results,
        max_candidates=2,
    )

    candidates = result[
        result["signal"] == BUY_CANDIDATE
    ]

    selected_symbols = set(
        candidates["symbol"]
    )

    assert len(candidates) == 2
    assert selected_symbols == {"ETH", "LINK"}

    sui = result[
        result["symbol"] == "SUI"
    ].iloc[0]

    doge = result[
        result["symbol"] == "DOGE"
    ].iloc[0]

    assert sui["signal"] == WATCH
    assert sui["eligible"] == False
    assert doge["signal"] == NO_TRADE


def main() -> None:
    tests = (
        test_strong_bull_setup_is_candidate,
        test_bear_regime_disables_long_entry,
        test_neutral_regime_creates_watch_only,
        test_insufficient_data_never_creates_signal,
        test_candidate_limit_keeps_best_assets,
    )

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 55)
    print(f"SUCCESS: {len(tests)}/{len(tests)} signal tests passed")


if __name__ == "__main__":
    main()
