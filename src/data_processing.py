"""Load, validate, normalize and preprocess call data from Excel."""

from __future__ import annotations

import re
import logging
from datetime import datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "телефон",
    "дата и время",
    "длительность мин:сек",
    "статус",
    "запись аудио",
    "причина завершения",
    "история диалога юзер-бот",
]

COLUMN_MAP = {
    "телефон": "phone_id",
    "дата и время": "timestamp",
    "длительность мин:сек": "duration_raw",
    "статус": "status",
    "запись аудио": "record_url",
    "причина завершения": "termination_reason",
    "история диалога юзер-бот": "raw_transcript",
}

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "calls_week_anon.xlsx"


def validate_columns(df: pd.DataFrame) -> tuple[bool, str | None]:
    """Return (ok, error_message)."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            return False, f"Ошибка загрузки: в файле отсутствует обязательная колонка: {col}"
    return True, None


def load_excel(source: Union[str, Path, BinaryIO, BytesIO]) -> pd.DataFrame:
    """Load Excel file into a DataFrame."""
    logger.info("Reading Excel source")
    df = pd.read_excel(source, engine="openpyxl")
    logger.info("Excel loaded | rows=%s | columns=%s", len(df), len(df.columns))
    return df


def parse_duration_seconds(value) -> int:
    """Parse duration to seconds. Returns 0 on failure."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0

    if isinstance(value, timedelta):
        return max(0, int(value.total_seconds()))

    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second

    if isinstance(value, datetime):
        return value.hour * 3600 + value.minute * 60 + value.second

    if isinstance(value, (int, float)):
        if isinstance(value, float) and 0 < value < 1:
            return max(0, int(round(value * 86400)))
        return max(0, int(value))

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return 0

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(float(p)) for p in parts)
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = (int(float(p)) for p in parts)
            return minutes * 60 + seconds
        if len(parts) == 1:
            return max(0, int(float(parts[0])))
    except (ValueError, TypeError):
        pass

    return 0


def clean_transcript(text) -> str:
    """Normalize transcript text for keyword matching."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    cleaned = str(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\[.*?\]", " ", cleaned)
    cleaned = re.sub(r"<.*?>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().lower()


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns to normalized fields."""
    normalized = pd.DataFrame()
    for src, dst in COLUMN_MAP.items():
        normalized[dst] = df[src]

    normalized["phone_id"] = normalized["phone_id"].astype(str)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized["duration_seconds"] = normalized["duration_raw"].apply(parse_duration_seconds)
    normalized["status"] = normalized["status"].fillna("").astype(str)
    normalized["record_url"] = normalized["record_url"].fillna("").astype(str)
    normalized["termination_reason"] = (
        normalized["termination_reason"].fillna("").astype(str).str.strip().str.lower()
    )
    normalized["raw_transcript"] = normalized["raw_transcript"].fillna("").astype(str)
    normalized["transcript_clean"] = normalized["raw_transcript"].apply(clean_transcript)
    normalized["has_transcript"] = normalized["transcript_clean"].str.len() > 0
    normalized["transcript_length_chars"] = normalized["transcript_clean"].str.len()

    normalized = normalized.drop(columns=["duration_raw"])
    return normalized


def preprocess_calls(source: Union[str, Path, BinaryIO, BytesIO]) -> tuple[pd.DataFrame | None, str | None]:
    """Full pipeline: load → validate → normalize."""
    try:
        df = load_excel(source)
    except Exception:
        logger.exception("Excel reading failed")
        return None, "Ошибка загрузки: не удалось прочитать файл Excel."

    if df.empty:
        logger.warning("Excel file is empty")
        return None, "Ошибка загрузки: файл Excel пуст."

    ok, err = validate_columns(df)
    if not ok:
        logger.error("Column validation failed | error=%s", err)
        return None, err

    normalized = normalize_dataframe(df)
    invalid_duration = int((normalized["duration_seconds"] == 0).sum())
    logger.info("Preprocessing completed | rows=%s | with_transcript=%s | zero_duration=%s", len(normalized), int(normalized["has_transcript"].sum()), invalid_duration)
    return normalized, None
