"""Main Growth Opportunity and A/B test recommendation logic."""

from __future__ import annotations

import logging

from src.classification import DROP_OFF_MAP

logger = logging.getLogger(__name__)


def generate_main_growth_opportunity(metrics: dict, funnel_table) -> dict:
    """Determine the primary growth bottleneck."""
    logger.info("Main growth opportunity calculation started")
    total = metrics["Total Calls"]
    no_tx_rate = metrics["No Transcript Rate"]
    suspicious = metrics["Suspicious No-transcript Calls"]
    fixed_timeout = metrics["Fixed Timeout Without Transcript"]
    bot_hangup = metrics["Bot Hangup Without Transcript"]

    suspicious_pct = (suspicious / total * 100) if total else 0

    if suspicious_pct >= 5 or no_tx_rate >= 15 or fixed_timeout >= 10 or bot_hangup >= 20:
        return {
            "main_problem": "Техническая потеря: значительная доля звонков без транскрипта",
            "why_it_matters": (
                "Без текста диалога невозможно оптимизировать скрипт. "
                "Часть потерь может быть системной — таймаут, bot_hangup, сбой сохранения транскрипта."
            ),
            "supporting_metric": f"Suspicious No-transcript: {suspicious} ({suspicious_pct:.1f}%)",
            "affected_stage": "before_contact",
            "recommended_focus": "Проверить логирование транскриптов, таймауты и причины bot_hangup",
            "recommendation_type": "technical",
        }

    transitions = [
        ("contact_to_consent", "script", "Contact → Consent", "Открывающая фраза и запрос разрешения"),
        ("consent_to_offer", "script", "Consent → Offer", "Переход к офферу после согласия"),
        ("offer_to_relevance_signal", "offer", "Offer → Relevance Signal", "Конкретика оффера и квалификационный вопрос"),
        ("relevance_to_next_step_proposed", "next_step", "Relevance → Next Step", "Переход от интереса к действию"),
        ("proposal_to_acceptance", "next_step", "Proposal → Acceptance", "Формулировка следующего шага"),
        ("acceptance_to_context_or_finish", "qualification", "Acceptance → Context", "Сбор контекста после согласия"),
    ]

    drop_map = {}
    for _, row in funnel_table.iterrows():
        drop = DROP_OFF_MAP.get(row["stage"], "")
        if drop and drop != "completed":
            drop_map[drop] = row["drop_off_pct"]

    meeting_no_rel_rate = metrics["Meeting Without Relevance Signal Rate"]
    if meeting_no_rel_rate >= 1.0:
        return {
            "main_problem": "Встречи без сигнала релевантности",
            "why_it_matters": (
                "Бот согласовывает next step с клиентами, у которых нет признаков релевантности. "
                "Meeting Rate завышается, а Qualified Next Step Rate остаётся низким."
            ),
            "supporting_metric": f"Meeting Without Relevance Signal Rate: {meeting_no_rel_rate}%",
            "affected_stage": "offer_to_relevance_signal",
            "recommended_focus": "Добавить квалификационный вопрос до предложения встречи",
            "recommendation_type": "risk",
        }

    weakest = metrics.get("Weakest Funnel Stage", "offer_to_relevance_signal")
    worst_pct = drop_map.get(weakest, 0)

    for stage_key, rec_type, label, focus in transitions:
        if stage_key == weakest:
            return {
                "main_problem": f"Основной drop-off: {label}",
                "why_it_matters": (
                    f"На переходе «{label}» теряется {worst_pct:.1f}% звонков — "
                    "это главное узкое место воронки по данным недели."
                ),
                "supporting_metric": f"Drop-off на {label}: {worst_pct:.1f}%",
                "affected_stage": stage_key,
                "recommended_focus": focus,
                "recommendation_type": rec_type,
            }

    return {
        "main_problem": "Низкий Qualified Next Step Rate",
        "why_it_matters": "Мало звонков доходят до осмысленного next step с сигналом релевантности.",
        "supporting_metric": f"Qualified Next Step Rate: {metrics['Qualified Next Step Rate']}%",
        "affected_stage": weakest,
        "recommended_focus": "Усилить квалификацию и переход к next step",
        "recommendation_type": "qualification",
    }


def generate_ab_test(recommendation: dict, metrics: dict) -> dict:
    """Generate first A/B test based on recommendation."""
    logger.info("A/B test generation started | recommendation_type=%s | stage=%s", recommendation.get("recommendation_type"), recommendation.get("affected_stage"))
    rec_type = recommendation.get("recommendation_type", "qualification")
    stage = recommendation.get("affected_stage", "offer_to_relevance_signal")

    if rec_type == "technical":
        return {
            "hypothesis": (
                "Значительная часть потерь происходит до оптимизации скрипта: "
                "звонки без транскрипта при осмысленной длительности или фиксированном таймауте."
            ),
            "variant_a": "Текущая логика звонков и сохранения транскриптов.",
            "variant_b": (
                "Проверить и исправить обработку no-transcript: fixed timeout, bot_hangup, "
                "сохранение транскрипта, audio/text logging и статусы завершения. "
                "Затем сравнить Calls With Transcript Rate и объём suspicious no-transcript."
            ),
            "main_metric": "Calls With Transcript Rate",
            "guardrail_metrics": [
                "Bot Hangup Without Transcript",
                "Fixed Timeout Without Transcript",
                "Average Duration of No-Transcript Calls",
            ],
            "expected_effect": "Рост Calls With Transcript Rate на 5–15 п.п. после устранения системных сбоев.",
        }

    if stage == "offer_to_relevance_signal" or rec_type == "offer":
        return {
            "hypothesis": "Клиенты не видят связь оффера со своей задачей — нет relevance signal после оффера.",
            "variant_a": "Текущий скрипт оффера.",
            "variant_b": (
                "Мы запускаем AI-продавца в первую линию: он сам прозванивает базу, "
                "задаёт первые вопросы и передаёт менеджерам только потенциально релевантных клиентов. "
                "Подскажите, у вас сейчас есть база или заявки, которые менеджеры регулярно обзванивают?"
            ),
            "main_metric": "Offer → Relevance Signal Conversion",
            "guardrail_metrics": ["Offer Reach Rate", "Conversation Start Rate"],
            "expected_effect": "Рост Offer → Relevance Signal Conversion на 3–8 п.п.",
        }

    if stage == "proposal_to_acceptance" or rec_type == "next_step":
        return {
            "hypothesis": "Клиенты соглашаются реже, когда next step звучит как большая встреча.",
            "variant_a": "Текущая формулировка полноценной встречи/демо.",
            "variant_b": (
                "Давайте сделаем не полноценную встречу, а короткий 15-минутный созвон с экспертом. "
                "Он покажет кейс по вашей отрасли и прикинет, есть ли экономика под ваш объём."
            ),
            "main_metric": "Proposal → Acceptance Conversion",
            "guardrail_metrics": ["Next Step Proposal Rate", "Qualified Next Step Rate"],
            "expected_effect": "Рост Proposal → Acceptance Conversion на 5–10 п.п.",
        }

    if rec_type == "risk" or stage == "offer_to_relevance_signal":
        return {
            "hypothesis": "Бот предлагает встречу до появления relevance signal.",
            "variant_a": "Текущий порядок: оффер → встреча без квалификации.",
            "variant_b": (
                "Подскажите, у вас есть база клиентов или заявок, "
                "которую менеджеры регулярно обзванивают?"
            ),
            "main_metric": "Qualified Next Step Rate",
            "guardrail_metrics": ["Meeting Without Relevance Signal Rate", "Next Step Proposal Rate"],
            "expected_effect": "Снижение Meeting Without Relevance Signal Rate и рост Qualified Next Step Rate.",
        }

    stage_tests = {
        "contact_to_consent": {
            "hypothesis": "Клиенты чаще дают согласие на разговор при более коротком и конкретном opening.",
            "variant_a": "Текущее приветствие и запрос разрешения.",
            "variant_b": "Здравствуйте! Буквально 30 секунд — удобно? Я коротко расскажу, как AI-агент помогает отделам продаж.",
            "main_metric": "Conversation Start Rate",
            "guardrail_metrics": ["Early No-dialog Calls", "Consent → Offer drop-off"],
            "expected_effect": "Рост Conversation Start Rate на 3–5 п.п.",
        },
        "consent_to_offer": {
            "hypothesis": "После согласия бот слишком долго переходит к офферу.",
            "variant_a": "Текущий переход consent → offer.",
            "variant_b": "Спасибо! Сразу к делу: мы автоматизируем прозвон базы AI-агентом и передаём менеджерам только горячих клиентов.",
            "main_metric": "Offer Reach Rate",
            "guardrail_metrics": ["Conversation Start Rate", "Offer → Relevance Signal Conversion"],
            "expected_effect": "Рост Offer Reach Rate на 3–7 п.п.",
        },
    }

    if stage in stage_tests:
        return stage_tests[stage]

    return {
        "hypothesis": "Усиление квалификации перед next step повысит качество конверсии.",
        "variant_a": "Текущий скрипт.",
        "variant_b": (
            "Подскажите, у вас есть база клиентов или заявок, "
            "которую менеджеры регулярно обзванивают?"
        ),
        "main_metric": "Qualified Next Step Rate",
        "guardrail_metrics": ["Relevance Signal Rate", "Next Step Proposal Rate"],
        "expected_effect": "Рост Qualified Next Step Rate на 2–5 п.п.",
    }
