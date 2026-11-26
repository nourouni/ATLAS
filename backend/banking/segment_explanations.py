from __future__ import annotations

from typing import Any


def customer_segment_explanation(segment: dict[str, Any] | None) -> dict[str, Any]:
    """Translate stored segment observations into respectful customer language."""
    segment = segment or {}
    transaction = str(segment.get("transaction_behavior") or "").lower()
    balance = str(segment.get("balance_and_cashflow_behavior") or "").lower()
    digital = str(segment.get("digital_channel_behavior") or "").lower()
    description = str(segment.get("description") or "").lower()

    strengths: list[str] = []
    neutral: list[str] = []
    if "highest transaction" in description or "high-activity" in description or "high frequency" in description:
        strengths.append("frequent banking activity")
    elif "lowest transaction" in description or "lower-activity" in description or "low-activity" in description:
        neutral.append("less frequent banking activity")
    elif "moderate transaction" in description or "moderate activity" in description:
        strengths.append("regular banking activity")
    elif "moderate-activity" in description:
        strengths.append("regular banking activity")
    if "moderate transactional value" in transaction:
        strengths.append("moderate transactional value relative to other retail profiles")
    if "highly channel-diverse" in description or "broad product" in description or "multi-product" in description:
        strengths.append("use of a broad range of banking services or channels")
    if "positive net cash flow" in description or "cash-flow-positive" in description:
        strengths.append("a positive cash-flow pattern during the analysis period")
    if "digital" in digital and "lowest" not in digital and "low" not in digital:
        strengths.append("meaningful engagement with digital banking channels")
    elif "digital" in digital and ("lowest" in digital or "0.0" in digital or "0.1" in digital):
        neutral.append("more limited use of digital channels")
    if "atm-dominant" in description or "atm-reliant" in description:
        neutral.append("activity is more concentrated around ATM services")
    if "low_balance_week_share 0.00" not in balance and ("low_balance_week_share" in balance or "low balance" in description or "low-balance" in description):
        neutral.append("balances are more modest during a substantial part of the analysis period")
    if "variable balance" in description or "volatile" in description or "erratic" in digital or "highest" in balance and "deposit_cv" in balance:
        neutral.append("deposit timing or cash-flow patterns vary over time")
    if "negative net cash flow" in transaction and not any("cash-flow" in item for item in neutral):
        neutral.append("cash flow is negative during the observed period")
    if not strengths and transaction:
        strengths.append("a distinct pattern of banking activity")
    if not neutral and not strengths:
        neutral.append("the available data does not support a more detailed interpretation")

    summary = "Your profile reflects " + ", ".join(strengths[:2]) + "." if strengths else "Your profile reflects the banking patterns observed during the analysis period."
    if neutral:
        summary += " It also shows " + ", ".join(neutral[:2]) + "."
    return {
        "technical_name": segment.get("segment_name") or segment.get("segment_label"),
        "customer_friendly_summary": summary,
        "strengths": strengths,
        "neutral_characteristics": neutral,
        "evolution_message": "This describes observed banking patterns during the analysis period; it is not a permanent label.",
    }
