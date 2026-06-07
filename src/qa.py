"""Intent-based Ask the Dashboard Q&A."""

from __future__ import annotations

import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

from src.classification import DROP_OFF_MAP
from src.metrics import DROP_OFF_LABELS


SUGGESTED_QUESTIONS = [
    "Где самый большой drop-off?",
    "Почему Qualified Next Step Rate низкий?",
    "Сколько звонков без текста?",
    "Сколько подозрительных звонков без транскрипта?",
    "Какой A/B-тест запустить первым?",
]


def _sample_calls(df: pd.DataFrame, condition, n: int = 5) -> pd.DataFrame | None:
    subset = df[condition].head(n)
    if subset.empty:
        return None
    cols = [
        "phone_id", "timestamp", "duration_seconds", "termination_reason",
        "funnel_stage_reached", "drop_off_stage", "has_relevance_signal",
        "next_step_proposed", "next_step_accepted", "transcript_clean",
    ]
    available = [c for c in cols if c in subset.columns]
    return subset[available].copy()


def _weakest_drop_off(funnel_table: pd.DataFrame) -> tuple[str, float, int]:
    subset = funnel_table[funnel_table["stage"] != "7_context_collected_or_correctly_finished"]
    if subset.empty:
        return "before_contact", 0.0, 0
    row = subset.loc[subset["drop_off_pct"].idxmax()]
    stage = DROP_OFF_MAP.get(row["stage"], "before_contact")
    return stage, row["drop_off_pct"], int(row["drop_off_count"])


def answer_analyst_question(
    question: str,
    metrics: dict,
    funnel_table: pd.DataFrame,
    enriched_df: pd.DataFrame,
    recommendation: dict,
    ab_test: dict,
    previous_context: dict | None = None,
) -> dict:
    """Answer analyst questions using intent matching."""
    logger.info("Analytical question received | length=%s | has_previous_context=%s", len(question or ""), bool(previous_context))
    q = (question or "").strip().lower()
    ctx = previous_context or {}
    last_topic = ctx.get("last_topic", "")
    last_stage = ctx.get("last_stage", "")
    last_filter = ctx.get("last_filter", "")

    follow_up_map = {
        "покажи примеры": "examples",
        "покажи подробнее": "examples",
        "почему": "why",
        "что делать": "action",
        "какой тест": "ab_test",
        "сколько таких": "count",
    }

    intent = None
    for phrase, mapped in follow_up_map.items():
        if phrase in q:
            intent = mapped
            break

    if intent and last_topic:
        if intent == "examples":
            if last_filter == "suspicious_no_transcript":
                examples = _sample_calls(enriched_df, enriched_df["technical_suspect_no_transcript"])
            elif last_filter == "meeting_no_relevance":
                examples = _sample_calls(enriched_df, enriched_df["is_meeting_without_relevance_signal"])
            elif last_stage:
                examples = _sample_calls(enriched_df, enriched_df["drop_off_stage"] == last_stage)
            else:
                examples = None
            return {
                "answer": "Примеры звонков по предыдущему контексту:",
                "supporting_metric": ctx.get("last_metric", ""),
                "related_stage": last_stage,
                "recommended_action": recommendation.get("recommended_focus", ""),
                "example_calls": examples,
                "context": ctx,
            }
        if intent == "why":
            return {
                "answer": recommendation.get("why_it_matters", "Недостаточно данных для объяснения."),
                "supporting_metric": recommendation.get("supporting_metric", ""),
                "related_stage": last_stage or recommendation.get("affected_stage", ""),
                "recommended_action": recommendation.get("recommended_focus", ""),
                "example_calls": None,
                "context": ctx,
            }
        if intent == "action":
            return {
                "answer": f"Рекомендуемый фокус: {recommendation.get('recommended_focus', '')}. "
                f"Первый A/B-тест: {ab_test.get('hypothesis', '')}",
                "supporting_metric": ab_test.get("main_metric", ""),
                "related_stage": recommendation.get("affected_stage", ""),
                "recommended_action": ab_test.get("variant_b", ""),
                "example_calls": None,
                "context": ctx,
            }
        if intent == "ab_test":
            return {
                "answer": (
                    f"Гипотеза: {ab_test.get('hypothesis', '')}\n\n"
                    f"Variant A: {ab_test.get('variant_a', '')}\n\n"
                    f"Variant B: {ab_test.get('variant_b', '')}"
                ),
                "supporting_metric": ab_test.get("main_metric", ""),
                "related_stage": recommendation.get("affected_stage", ""),
                "recommended_action": ab_test.get("expected_effect", ""),
                "example_calls": None,
                "context": ctx,
            }
        if intent == "count" and last_filter == "suspicious_no_transcript":
            count = metrics["Suspicious No-transcript Calls"]
            return {
                "answer": f"Подозрительных звонков без транскрипта: {count:,} "
                f"({metrics['No Transcript Rate']}% от всех звонков без текста — см. метрики).",
                "supporting_metric": f"Suspicious No-transcript Calls: {count}",
                "related_stage": "before_contact",
                "recommended_action": recommendation.get("recommended_focus", ""),
                "example_calls": None,
                "context": ctx,
            }

    new_ctx = {
        "last_topic": "",
        "last_stage": "",
        "last_filter": "",
        "last_recommendation_type": recommendation.get("recommendation_type", ""),
        "last_metric": "",
    }

    if re.search(r"drop.?off|самый большой|где теря", q):
        stage, pct, count = _weakest_drop_off(funnel_table)
        label = DROP_OFF_LABELS.get(stage, stage)
        new_ctx.update({"last_topic": "drop_off", "last_stage": stage, "last_metric": f"{pct}%"})
        examples = _sample_calls(enriched_df, enriched_df["drop_off_stage"] == stage)
        return {
            "answer": f"Самый большой drop-off — «{label}»: {count:,} звонков ({pct:.1f}%).",
            "supporting_metric": f"Drop-off {label}: {pct:.1f}%",
            "related_stage": stage,
            "recommended_action": recommendation.get("recommended_focus", ""),
            "example_calls": examples,
            "context": new_ctx,
        }

    if re.search(r"qualified next step.*низк|почему.*qualified", q):
        rate = metrics["Qualified Next Step Rate"]
        conv_rate = metrics["Qualified Next Step Rate From Conversations"]
        new_ctx.update({"last_topic": "qualified_rate", "last_metric": f"{rate}%"})
        return {
            "answer": (
                f"Qualified Next Step Rate = {rate}% ({conv_rate}% от звонков с транскриптом). "
                f"Главная причина: {recommendation.get('main_problem', '')}. "
                f"{recommendation.get('why_it_matters', '')}"
            ),
            "supporting_metric": f"Qualified Next Step Rate: {rate}%",
            "related_stage": recommendation.get("affected_stage", ""),
            "recommended_action": recommendation.get("recommended_focus", ""),
            "example_calls": _sample_calls(enriched_df, ~enriched_df["is_qualified_next_step"] & enriched_df["has_transcript"]),
            "context": new_ctx,
        }

    if re.search(r"без текста|no.?transcript|нет транскрипт", q) and "подозр" not in q:
        count = metrics["Calls Without Transcript"]
        new_ctx.update({"last_topic": "no_transcript", "last_filter": "no_transcript"})
        return {
            "answer": f"Звонков без текста: {count:,} ({metrics['No Transcript Rate']}% от всех). "
            f"Early no-dialog: {metrics['Early No-dialog Calls']:,}.",
            "supporting_metric": f"No Transcript Rate: {metrics['No Transcript Rate']}%",
            "related_stage": "before_contact",
            "recommended_action": "Проверить технические звонки без транскрипта отдельно от ранних отказов.",
            "example_calls": _sample_calls(enriched_df, ~enriched_df["has_transcript"]),
            "context": new_ctx,
        }

    if re.search(r"подозрительн|suspicious", q):
        count = metrics["Suspicious No-transcript Calls"]
        new_ctx.update({"last_topic": "suspicious", "last_filter": "suspicious_no_transcript", "last_stage": "before_contact"})
        examples = _sample_calls(enriched_df, enriched_df["technical_suspect_no_transcript"]) if "пример" in q else None
        if "пример" in q:
            examples = _sample_calls(enriched_df, enriched_df["technical_suspect_no_transcript"])
        return {
            "answer": f"Подозрительных звонков без транскрипта (≥30 сек): {count:,}. "
            f"Fixed timeout: {metrics['Fixed Timeout Without Transcript']:,}, "
            f"bot_hangup: {metrics['Bot Hangup Without Transcript']:,}.",
            "supporting_metric": f"Suspicious No-transcript Calls: {count}",
            "related_stage": "before_contact",
            "recommended_action": recommendation.get("recommended_focus", ""),
            "example_calls": examples,
            "context": new_ctx,
        }

    if re.search(r"длинн.*без текста|long.*no.?transcript", q):
        long_no_tx = enriched_df[(~enriched_df["has_transcript"]) & (enriched_df["duration_seconds"] >= 60)]
        count = len(long_no_tx)
        new_ctx.update({"last_topic": "long_no_transcript", "last_filter": "suspicious_no_transcript"})
        return {
            "answer": f"Длинных звонков (≥60 сек) без транскрипта: {count:,}. Это сильный признак системной проблемы.",
            "supporting_metric": f"Long no-transcript calls: {count}",
            "related_stage": "before_contact",
            "recommended_action": "Проверить audio/text logging и fixed timeout.",
            "example_calls": _sample_calls(
                enriched_df,
                (~enriched_df["has_transcript"]) & (enriched_df["duration_seconds"] >= 60),
            ),
            "context": new_ctx,
        }

    if re.search(r"фиксированн|fixed timeout|таймаут", q):
        count = metrics["Fixed Timeout Without Transcript"]
        new_ctx.update({"last_topic": "fixed_timeout", "last_filter": "fixed_timeout"})
        return {
            "answer": f"Звонков на фиксированном таймауте (180/240 сек) без транскрипта: {count:,}.",
            "supporting_metric": f"Fixed Timeout Without Transcript: {count}",
            "related_stage": "before_contact",
            "recommended_action": "Проверить логику fixed timeout и сохранение транскрипта.",
            "example_calls": _sample_calls(enriched_df, enriched_df["fixed_timeout_without_transcript"]),
            "context": new_ctx,
        }

    if re.search(r"next step.*relevance|relevance.*next step|без relevance", q):
        count = int(enriched_df["is_meeting_without_relevance_signal"].sum())
        new_ctx.update({"last_topic": "meeting_no_rel", "last_filter": "meeting_no_relevance"})
        return {
            "answer": f"Звонков с next step без relevance signal: {count:,} "
            f"(Meeting Without Relevance Signal Rate: {metrics['Meeting Without Relevance Signal Rate']}%).",
            "supporting_metric": f"Meeting Without Relevance Signal Rate: {metrics['Meeting Without Relevance Signal Rate']}%",
            "related_stage": "offer_to_relevance_signal",
            "recommended_action": "Добавить квалификационный вопрос до предложения встречи.",
            "example_calls": _sample_calls(enriched_df, enriched_df["is_meeting_without_relevance_signal"]),
            "context": new_ctx,
        }

    if re.search(r"a/b.?тест|ab.?test|какой тест|первым", q):
        new_ctx.update({"last_topic": "ab_test"})
        return {
            "answer": (
                f"Рекомендуемый первый A/B-тест:\n\n"
                f"**Гипотеза:** {ab_test.get('hypothesis', '')}\n\n"
                f"**Variant A:** {ab_test.get('variant_a', '')}\n\n"
                f"**Variant B:** {ab_test.get('variant_b', '')}\n\n"
                f"**Main metric:** {ab_test.get('main_metric', '')}"
            ),
            "supporting_metric": ab_test.get("main_metric", ""),
            "related_stage": recommendation.get("affected_stage", ""),
            "recommended_action": ab_test.get("expected_effect", ""),
            "example_calls": None,
            "context": new_ctx,
        }

    if re.search(r"почему.*гипотез|почему.*выбран|почему.*тест", q):
        new_ctx.update({"last_topic": "why_hypothesis"})
        return {
            "answer": (
                f"Гипотеза выбрана на основе главной проблемы: {recommendation.get('main_problem', '')}. "
                f"{recommendation.get('why_it_matters', '')} "
                f"Supporting metric: {recommendation.get('supporting_metric', '')}."
            ),
            "supporting_metric": recommendation.get("supporting_metric", ""),
            "related_stage": recommendation.get("affected_stage", ""),
            "recommended_action": ab_test.get("variant_b", ""),
            "example_calls": None,
            "context": new_ctx,
        }

    logger.warning("Analytical question was not recognized")
    suggestions = "\n".join(f"• {s}" for s in SUGGESTED_QUESTIONS)
    return {
        "answer": f"Не удалось точно определить вопрос. Попробуйте один из вариантов:\n\n{suggestions}",
        "supporting_metric": "",
        "related_stage": "",
        "recommended_action": "",
        "example_calls": None,
        "context": ctx,
    }
