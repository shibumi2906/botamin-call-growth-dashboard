"""CSV export utilities."""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)

EXPORT_COLUMNS = [
    "phone_id",
    "timestamp",
    "duration_seconds",
    "status",
    "termination_reason",
    "record_url",
    "raw_transcript",
    "transcript_clean",
    "has_transcript",
    "transcript_length_chars",
    "funnel_stage_reached",
    "drop_off_stage",
    "has_relevance_signal",
    "relevance_signal_type",
    "next_step_proposed",
    "next_step_accepted",
    "next_step_type",
    "is_qualified_next_step",
    "is_meeting_without_relevance_signal",
    "no_transcript_type",
    "technical_suspect_no_transcript",
    "fixed_timeout_without_transcript",
    "bot_hangup_without_transcript",
    "early_no_dialog",
    "possible_system_anomaly",
]


def prepare_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order export columns."""
    available = [c for c in EXPORT_COLUMNS if c in df.columns]
    return df[available].copy()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return CSV as UTF-8 bytes with BOM for Excel compatibility."""
    export_df = prepare_export_df(df)
    logger.info("Preparing CSV export | rows=%s | columns=%s", len(export_df), len(export_df.columns))
    return export_df.to_csv(index=False).encode("utf-8-sig")
