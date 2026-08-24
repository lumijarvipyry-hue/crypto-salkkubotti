"""Point-in-time crypto market scanner.

The scanner compares approved assets using only information available on the
requested date. It ranks opportunities but does not create orders or trades.
"""

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


REQUIRED_INDICATORS = {
    "timestamp",
    "close",
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "roc_30",
    "roc_90",
    "atr_pct",
    "volume_ratio_20",
    "volatility_30",
    "relative_strength_30",
    "relative_strength_90",
}

MAX_TREND_SCORE = 30.0
MAX_MOMENTUM_SCORE = 25.0
MAX_RELATIVE_STRENGTH_SCORE = 25.0
MAX_VOLUME_SCORE = 10.0
MAX_RISK_SCORE = 10.0
MAX_TOTAL_SCORE = 100.0


@dataclass(frozen=True)
class ScoreBreakdown:
    """Individual components of an asset's total scanner score."""

    trend: float
    momentum: float
    relative_strength: float
    volume: float
    risk: float

    @property
    def total(self) -> float:
        return round(
            min(
                MAX_TOTAL_SCORE,
                self.trend
                + self.momentum
                + self.relative_strength
                + self.volume
                + self.risk,
            ),
            2,
        )


def _linear_score(
    value: float,
    minimum: float,
    maximum: float,
    points: float,
) -> float:
    """Convert a value into a bounded linear score."""

    if pd.isna(value):
        return 0.0

    if maximum <= minimum:
        raise ValueError("Maximum must be greater than minimum")

    normalized = (float(value) - minimum) / (maximum - minimum)
    bounded = max(0.0, min(1.0, normalized))

    return round(bounded * points, 4)


def _trend_score(row: pd.Series) -> float:
    """Score price position and EMA alignment."""

    score = 0.0

    if row["close"] > row["ema_20"]:
        score += 5.0

    if row["close"] > row["ema_50"]:
        score += 7.0

    if row["close"] > row["ema_200"]:
        score += 8.0

    if row["ema_20"] > row["ema_50"] > row["ema_200"]:
        score += 10.0

    return min(score, MAX_TREND_SCORE)


def _momentum_score(row: pd.Series) -> float:
    """Score RSI and medium-term price momentum."""

    rsi = float(row["rsi_14"])

    if 45.0 <= rsi <= 65.0:
        rsi_score = 8.0

    elif 35.0 <= rsi <= 75.0:
        rsi_score = 4.0

    else:
        rsi_score = 0.0

    roc_30_score = _linear_score(
        row["roc_30"],
        minimum=-0.10,
        maximum=0.30,
        points=8.0,
    )

    roc_90_score = _linear_score(
        row["roc_90"],
        minimum=-0.20,
        maximum=0.60,
        points=9.0,
    )

    return min(
        rsi_score + roc_30_score + roc_90_score,
        MAX_MOMENTUM_SCORE,
    )


def _relative_strength_score(row: pd.Series) -> float:
    """Score outperformance against Bitcoin."""

    score_30 = _linear_score(
        row["relative_strength_30"],
        minimum=-0.15,
        maximum=0.25,
        points=12.0,
    )

    score_90 = _linear_score(
        row["relative_strength_90"],
        minimum=-0.25,
        maximum=0.60,
        points=13.0,
    )

    return min(
        score_30 + score_90,
        MAX_RELATIVE_STRENGTH_SCORE,
    )


def _volume_score(row: pd.Series) -> float:
    """Reward confirmed moves with above-average volume."""

    return min(
        _linear_score(
            row["volume_ratio_20"],
            minimum=0.80,
            maximum=2.00,
            points=MAX_VOLUME_SCORE,
        ),
        MAX_VOLUME_SCORE,
    )


def _risk_score(row: pd.Series) -> float:
    """Reward manageable volatility without treating risk as zero."""

    atr_percentage = float(row["atr_pct"])
    volatility = float(row["volatility_30"])

    if atr_percentage <= 0.03:
        atr_score = 5.0

    elif atr_percentage <= 0.06:
        atr_score = 3.0

    elif atr_percentage <= 0.10:
        atr_score = 1.0

    else:
        atr_score = 0.0

    if volatility <= 0.60:
        volatility_score = 5.0

    elif volatility <= 1.00:
        volatility_score = 3.0

    elif volatility <= 1.50:
        volatility_score = 1.0

    else:
        volatility_score = 0.0

    return min(
        atr_score + volatility_score,
        MAX_RISK_SCORE,
    )


def calculate_score(row: pd.Series) -> ScoreBreakdown:
    """Calculate a complete score for one asset on one date."""

    missing = [
        column
        for column in REQUIRED_INDICATORS
        if column not in row.index
    ]

    if missing:
        raise ValueError(
            f"Missing scanner fields: {', '.join(sorted(missing))}"
        )

    if row[list(REQUIRED_INDICATORS)].isna().any():
        raise ValueError("Insufficient indicator history for scoring")

    return ScoreBreakdown(
        trend=_trend_score(row),
        momentum=_momentum_score(row),
        relative_strength=_relative_strength_score(row),
        volume=_volume_score(row),
        risk=_risk_score(row),
    )


def market_regime(
    bitcoin_data: pd.DataFrame,
    as_of: pd.Timestamp,
) -> str:
    """Classify the market as BULL, NEUTRAL or BEAR using Bitcoin."""

    required = {
        "timestamp",
        "close",
        "ema_50",
        "ema_200",
    }

    missing = required.difference(bitcoin_data.columns)

    if missing:
        raise ValueError(
            f"Bitcoin data missing: {', '.join(sorted(missing))}"
        )

    cutoff = pd.Timestamp(as_of)

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")

    else:
        cutoff = cutoff.tz_convert("UTC")

    available = bitcoin_data[
        pd.to_datetime(bitcoin_data["timestamp"], utc=True) <= cutoff
    ]

    if available.empty:
        return "INSUFFICIENT_DATA"

    row = available.iloc[-1]

    if row[["close", "ema_50", "ema_200"]].isna().any():
        return "INSUFFICIENT_DATA"

    if (
        row["close"] > row["ema_200"]
        and row["ema_50"] > row["ema_200"]
    ):
        return "BULL"

    if (
        row["close"] < row["ema_200"]
        and row["ema_50"] < row["ema_200"]
    ):
        return "BEAR"

    return "NEUTRAL"


def scan_market(
    datasets: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Rank every asset using data available at or before the cutoff."""

    if "BTC" not in datasets:
        raise ValueError("BTC data is required for market regime detection")

    cutoff = pd.Timestamp(as_of)

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")

    else:
        cutoff = cutoff.tz_convert("UTC")

    regime = market_regime(
        datasets["BTC"],
        cutoff,
    )

    results = []

    for symbol, data in datasets.items():
        if data.empty:
            results.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.NaT,
                    "market_regime": regime,
                    "status": "INSUFFICIENT_DATA",
                    "score": pd.NA,
                    "trend_score": pd.NA,
                    "momentum_score": pd.NA,
                    "relative_strength_score": pd.NA,
                    "volume_score": pd.NA,
                    "risk_score": pd.NA,
                }
            )
            continue

        timestamps = pd.to_datetime(
            data["timestamp"],
            utc=True,
            errors="coerce",
        )

        available = data[timestamps <= cutoff]

        if available.empty:
            results.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.NaT,
                    "market_regime": regime,
                    "status": "INSUFFICIENT_DATA",
                    "score": pd.NA,
                    "trend_score": pd.NA,
                    "momentum_score": pd.NA,
                    "relative_strength_score": pd.NA,
                    "volume_score": pd.NA,
                    "risk_score": pd.NA,
                }
            )
            continue

        row = available.iloc[-1]

        try:
            breakdown = calculate_score(row)

            results.append(
                {
                    "symbol": symbol,
                    "timestamp": row["timestamp"],
                    "market_regime": regime,
                    "status": "SCORED",
                    "score": breakdown.total,
                    "trend_score": breakdown.trend,
                    "momentum_score": breakdown.momentum,
                    "relative_strength_score": (
                        breakdown.relative_strength
                    ),
                    "volume_score": breakdown.volume,
                    "risk_score": breakdown.risk,
                }
            )

        except ValueError:
            results.append(
                {
                    "symbol": symbol,
                    "timestamp": row.get("timestamp", pd.NaT),
                    "market_regime": regime,
                    "status": "INSUFFICIENT_DATA",
                    "score": pd.NA,
                    "trend_score": pd.NA,
                    "momentum_score": pd.NA,
                    "relative_strength_score": pd.NA,
                    "volume_score": pd.NA,
                    "risk_score": pd.NA,
                }
            )

    result = pd.DataFrame(results)

    return result.sort_values(
        by=["score", "symbol"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
