"""Calculate dashboard metrics and funnel tables."""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)

from src.classification import DROP_OFF_MAP, FUNNEL_STAGES, STAGE_ORDER

FUNNEL_LABELS = {
    "0_no_dialog": "No Dialog",
    "1_contact_greeting": "Contact / Greeting",
    "2_consent_permission": "Consent",
    "3_offer_delivered": "Offer Delivered",
    "4_relevance_signal_detected": "Relevance Signal",
    "5_next_step_proposed": "Next Step Proposed",
    "6_next_step_accepted": "Next Step Accepted",
    "7_context_collected_or_correctly_finished": "Context / Finished",
}

DROP_OFF_LABELS = {
    "before_contact": "Before Contact",
    "contact_to_consent": "Contact → Consent",
    "consent_to_offer": "Consent → Offer",
    "offer_to_relevance_signal": "Offer → Relevance Signal",
    "relevance_to_next_step_proposed": "Relevance → Next Step",
    "proposal_to_acceptance": "Proposal → Acceptance",
    "acceptance_to_context_or_finish": "Acceptance → Context",
    "completed": "Completed",
}


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 2)


def build_funnel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build funnel stage counts and step conversions."""
    total = len(df)
    logger.info("Building funnel table | rows=%s", total)
    rows = []

    for i, stage in enumerate(FUNNEL_STAGES):
        count = int((df["funnel_stage_reached"] == stage).sum())
        reached_at_least = int(
            df["funnel_stage_reached"].map(lambda s: STAGE_ORDER[s] >= STAGE_ORDER[stage]).sum()
        )
        prev_reached = total if i == 0 else int(
            df["funnel_stage_reached"].map(
                lambda s: STAGE_ORDER[s] >= STAGE_ORDER[FUNNEL_STAGES[i - 1]]
            ).sum()
        )
        conversion = _rate(reached_at_least, prev_reached) if i > 0 else 100.0
        drop_off = prev_reached - reached_at_least if i > 0 else 0
        drop_off_pct = _rate(drop_off, prev_reached) if i > 0 and prev_reached else 0.0

        rows.append(
            {
                "stage": stage,
                "stage_label": FUNNEL_LABELS[stage],
                "calls_at_stage": count,
                "reached_at_least": reached_at_least,
                "step_conversion_pct": conversion,
                "drop_off_count": drop_off,
                "drop_off_pct": drop_off_pct,
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Funnel table completed | stages=%s", len(result))
    return result


def _weakest_transition(funnel_table: pd.DataFrame) -> str:
    subset = funnel_table[funnel_table["stage"] != "0_no_dialog"].copy()
    if subset.empty:
        return "before_contact"
    worst = subset.loc[subset["drop_off_pct"].idxmax()]
    stage = worst["stage"]
    return DROP_OFF_MAP.get(stage, "before_contact")


def calculate_metrics(df: pd.DataFrame, funnel_table: pd.DataFrame | None = None) -> dict:
    """Compute all internal metrics."""
    logger.info("Metrics calculation started | rows=%s", len(df))
    if funnel_table is None:
        funnel_table = build_funnel_table(df)

    total = len(df)
    with_tx = int(df["has_transcript"].sum())
    without_tx = total - with_tx

    qualified = int(df["is_qualified_next_step"].sum())
    with_conversation = with_tx if with_tx else 1

    offer_reached = int(
        df["funnel_stage_reached"].map(lambda s: STAGE_ORDER[s] >= STAGE_ORDER["3_offer_delivered"]).sum()
    )
    relevance = int(df["has_relevance_signal"].sum())
    proposed = int(df["next_step_proposed"].sum())
    accepted = int(df["next_step_accepted"].sum())

    offer_to_rel_num = int(
        (
            df["has_relevance_signal"]
            & df["funnel_stage_reached"].map(
                lambda s: STAGE_ORDER[s] >= STAGE_ORDER["3_offer_delivered"]
            )
        ).sum()
    )
    proposal_to_acc_num = int((df["next_step_proposed"] & df["next_step_accepted"]).sum())

    weakest = _weakest_transition(funnel_table)

    return {
        "Total Calls": total,
        "Calls With Transcript": with_tx,
        "Calls Without Transcript": without_tx,
        "No Transcript Rate": _rate(without_tx, total),
        "Calls With Transcript Rate": _rate(with_tx, total),
        "Early No-dialog Calls": int(df["early_no_dialog"].sum()),
        "Suspicious No-transcript Calls": int(df["technical_suspect_no_transcript"].sum()),
        "Fixed Timeout Without Transcript": int(df["fixed_timeout_without_transcript"].sum()),
        "Bot Hangup Without Transcript": int(df["bot_hangup_without_transcript"].sum()),
        "Conversation Start Rate": _rate(
            int(
                df["funnel_stage_reached"].map(
                    lambda s: STAGE_ORDER[s] >= STAGE_ORDER["1_contact_greeting"]
                ).sum()
            ),
            with_conversation if with_tx else total,
        ),
        "Offer Reach Rate": _rate(offer_reached, with_conversation if with_tx else total),
        "Relevance Signal Rate": _rate(relevance, with_conversation if with_tx else total),
        "Offer → Relevance Signal Conversion": _rate(offer_to_rel_num, offer_reached if offer_reached else 1),
        "Next Step Proposal Rate": _rate(proposed, with_conversation if with_tx else total),
        "Next Step Acceptance Rate": _rate(accepted, proposed if proposed else 1),
        "Proposal → Acceptance Conversion": _rate(proposal_to_acc_num, proposed if proposed else 1),
        "Qualified Next Step Rate": _rate(qualified, total),
        "Qualified Next Step Rate From Conversations": _rate(qualified, with_conversation if with_tx else total),
        "Meeting Without Relevance Signal Rate": _rate(
            int(df["is_meeting_without_relevance_signal"].sum()),
            total,
        ),
        "Weakest Funnel Stage": weakest,
        "Weakest Funnel Stage Label": DROP_OFF_LABELS.get(weakest, weakest),
    }


def get_adaptive_metrics(metrics: dict, recommendation_type: str) -> list[tuple[str, str]]:
    """Return supporting metrics for the adaptive overview."""
    base = [
        ("Total Calls", f"{metrics['Total Calls']:,}"),
        ("Calls With Transcript", f"{metrics['Calls With Transcript']:,}"),
        (
            "Calls Without Transcript / No Transcript Rate",
            f"{metrics['Calls Without Transcript']:,} / {metrics['No Transcript Rate']}%",
        ),
    ]

    if recommendation_type == "technical":
        base.extend(
            [
                ("Suspicious No-transcript Calls", f"{metrics['Suspicious No-transcript Calls']:,}"),
                ("Fixed Timeout Without Transcript", f"{metrics['Fixed Timeout Without Transcript']:,}"),
                ("Bot Hangup Without Transcript", f"{metrics['Bot Hangup Without Transcript']:,}"),
            ]
        )
    elif recommendation_type in ("offer", "qualification", "risk"):
        base.extend(
            [
                ("Offer Reach Rate", f"{metrics['Offer Reach Rate']}%"),
                ("Offer → Relevance Signal Conversion", f"{metrics['Offer → Relevance Signal Conversion']}%"),
                ("Relevance Signal Rate", f"{metrics['Relevance Signal Rate']}%"),
            ]
        )
    elif recommendation_type == "script":
        base.extend(
            [
                ("Conversation Start Rate", f"{metrics['Conversation Start Rate']}%"),
                ("Offer Reach Rate", f"{metrics['Offer Reach Rate']}%"),
                ("Early No-dialog Calls", f"{metrics['Early No-dialog Calls']:,}"),
            ]
        )
    elif recommendation_type == "next_step":
        base.extend(
            [
                ("Next Step Proposal Rate", f"{metrics['Next Step Proposal Rate']}%"),
                ("Next Step Acceptance Rate", f"{metrics['Next Step Acceptance Rate']}%"),
                ("Proposal → Acceptance Conversion", f"{metrics['Proposal → Acceptance Conversion']}%"),
            ]
        )
    else:
        base.extend(
            [
                ("Qualified Next Step Rate", f"{metrics['Qualified Next Step Rate']}%"),
                ("Relevance Signal Rate", f"{metrics['Relevance Signal Rate']}%"),
            ]
        )

    return base
