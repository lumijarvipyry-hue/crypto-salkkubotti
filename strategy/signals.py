"""Convert scanner rankings into rule-based strategy signals.

Signals are research labels only. This module does not place orders, allocate
real money or communicate with an exchange.
"""

from dataclasses import dataclass

import pandas as pd


NO_TRADE = "NO_TRADE"
WATCH = "WATCH"
BUY_CANDIDATE = "BUY_CANDIDATE"

BUY_SCORE_THRESHOLD = 75.0
WATCH_SCORE_THRESHOLD = 60.0

MINIMUM_TREND_SCORE = 20.0
MINIMUM_MOMENTUM_SCORE = 12.0
MINIMUM_RELATIVE_STRENGTH_SCORE = 12.0
MINIMUM_RISK_SCORE = 1.0

DEFAULT_MAX_CANDIDATES = 3

REQUIRED_COLUMNS = {
    "symbol",
    "timestamp",
    "market_regime",
    "status",
    "score",
    "trend_score",
    "momentum_score",
    "relative_strength_score",
    "volume_score",
    "risk_score",
}


@dataclass(frozen=True)
class SignalDecision:
    """A transparent signal and the reason it was assigned."""

    signal: str
    eligible: bool
    reason: str


def _validate_scan_results(scan_results: pd.DataFrame) -> pd.DataFrame:
    """Validate scanner output without inventing missing information."""

    if not isinstance(scan_results, pd.DataFrame):
        raise TypeError("Scanner results must be a pandas DataFrame")

    missing = sorted(
        REQUIRED_COLUMNS.difference(scan_results.columns)
    )

    if missing:
        raise ValueError(
            f"Scanner results missing columns: {', '.join(missing)}"
        )

    if scan_results.empty:
        return scan_results.copy()

    result = scan_results.copy()

    if result["symbol"].isna().any():
        raise ValueError("Scanner results contain a missing symbol")

    if result["symbol"].duplicated().any():
        raise ValueError("Scanner results contain duplicate symbols")

    return result


def decide_signal(row: pd.Series) -> SignalDecision:
    """Assign one signal using a single point-in-time scanner row."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in row.index
    ]

    if missing:
        raise ValueError(
            f"Signal row missing fields: {', '.join(sorted(missing))}"
        )

    if row["status"] != "SCORED" or pd.isna(row["score"]):
        return SignalDecision(
            signal=NO_TRADE,
            eligible=False,
            reason="Insufficient verified data",
        )

    regime = str(row["market_regime"])
    score = float(row["score"])

    if regime == "INSUFFICIENT_DATA":
        return SignalDecision(
            signal=NO_TRADE,
            eligible=False,
            reason="Market regime unavailable",
        )

    if regime == "BEAR":
        return SignalDecision(
            signal=NO_TRADE,
            eligible=False,
            reason="Long entries disabled in bear regime",
        )

    component_columns = [
        "trend_score",
        "momentum_score",
        "relative_strength_score",
        "risk_score",
    ]

    if row[component_columns].isna().any():
        return SignalDecision(
            signal=NO_TRADE,
            eligible=False,
            reason="Required score component is missing",
        )

    trend_score = float(row["trend_score"])
    momentum_score = float(row["momentum_score"])
    relative_strength_score = float(
        row["relative_strength_score"]
    )
    risk_score = float(row["risk_score"])

    buy_requirements_met = (
        regime == "BULL"
        and score >= BUY_SCORE_THRESHOLD
        and trend_score >= MINIMUM_TREND_SCORE
        and momentum_score >= MINIMUM_MOMENTUM_SCORE
        and relative_strength_score
        >= MINIMUM_RELATIVE_STRENGTH_SCORE
        and risk_score >= MINIMUM_RISK_SCORE
    )

    if buy_requirements_met:
        return SignalDecision(
            signal=BUY_CANDIDATE,
            eligible=True,
            reason="Bull regime and all entry requirements passed",
        )

    if score >= WATCH_SCORE_THRESHOLD:
        if regime == "NEUTRAL":
            reason = "Strong score but market regime is neutral"

        elif trend_score < MINIMUM_TREND_SCORE:
            reason = "Score passed watch level but trend is too weak"

        elif momentum_score < MINIMUM_MOMENTUM_SCORE:
            reason = "Score passed watch level but momentum is too weak"

        elif (
            relative_strength_score
            < MINIMUM_RELATIVE_STRENGTH_SCORE
        ):
            reason = (
                "Score passed watch level but relative strength is too weak"
            )

        elif risk_score < MINIMUM_RISK_SCORE:
            reason = "Score passed watch level but risk is too high"

        else:
            reason = "Setup is close to entry requirements"

        return SignalDecision(
            signal=WATCH,
            eligible=False,
            reason=reason,
        )

    return SignalDecision(
        signal=NO_TRADE,
        eligible=False,
        reason="Total score is below the watch threshold",
    )


def generate_signals(
    scan_results: pd.DataFrame,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> pd.DataFrame:
    """Create signals and retain only the best limited candidates."""

    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")

    result = _validate_scan_results(scan_results)

    if result.empty:
        return result.assign(
            signal=pd.Series(dtype="object"),
            eligible=pd.Series(dtype="bool"),
            reason=pd.Series(dtype="object"),
        )

    decisions = [
        decide_signal(row)
        for _, row in result.iterrows()
    ]

    result["signal"] = [
        decision.signal
        for decision in decisions
    ]

    result["eligible"] = [
        decision.eligible
        for decision in decisions
    ]

    result["reason"] = [
        decision.reason
        for decision in decisions
    ]

    candidate_indexes = (
        result[result["signal"] == BUY_CANDIDATE]
        .sort_values(
            by=["score", "symbol"],
            ascending=[False, True],
        )
        .index
        .tolist()
    )

    allowed_indexes = set(
        candidate_indexes[:max_candidates]
    )

    for index in candidate_indexes[max_candidates:]:
        result.at[index, "signal"] = WATCH
        result.at[index, "eligible"] = False
        result.at[index, "reason"] = (
            "Candidate limit reached; higher-ranked assets selected"
        )

    result["rank"] = (
        result["score"]
        .rank(
            method="first",
            ascending=False,
            na_option="bottom",
        )
        .astype("Int64")
    )

    result["selected_candidate"] = result.index.isin(
        allowed_indexes
    )

    return result.sort_values(
        by=["rank", "symbol"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
