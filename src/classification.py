"""Rule-based funnel classification and no-transcript diagnostics."""

from __future__ import annotations

import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

FUNNEL_STAGES = [
    "0_no_dialog",
    "1_contact_greeting",
    "2_consent_permission",
    "3_offer_delivered",
    "4_relevance_signal_detected",
    "5_next_step_proposed",
    "6_next_step_accepted",
    "7_context_collected_or_correctly_finished",
]

DROP_OFF_MAP = {
    "0_no_dialog": "before_contact",
    "1_contact_greeting": "contact_to_consent",
    "2_consent_permission": "consent_to_offer",
    "3_offer_delivered": "offer_to_relevance_signal",
    "4_relevance_signal_detected": "relevance_to_next_step_proposed",
    "5_next_step_proposed": "proposal_to_acceptance",
    "6_next_step_accepted": "acceptance_to_context_or_finish",
    "7_context_collected_or_correctly_finished": "completed",
}

STAGE_ORDER = {stage: idx for idx, stage in enumerate(FUNNEL_STAGES)}

GREETING_SIGNALS = [
    "добрый день", "здравствуйте", "алло", "слушаю", "меня зовут",
]
CONSENT_SIGNALS = [
    "удобно говорить", "можно коротко", "буквально пару минут", "30 секунд",
    "я коротко", "в двух словах", "тридцать секунд", "ладно", "давайте",
]
OFFER_SIGNALS = [
    "ai-агент", "ии-агент", "искусственный интеллект", "бот", "автоматизируем",
    "прозванивает", "звонки", "продажи", "лиды", "заявки", "менеджерам",
    "кейс", "по вашей отрасли", "горячих клиентов",
]

RELEVANCE_ROLE = [
    "я руководитель", "я директор", "я собственник", "я владелец",
    "я принимаю решения", "это ко мне", "я занимаюсь продажами",
    "я отвечаю за продажи", "коммерческий директор", "передайте коммерческому",
    "соединю с руководителем", "могу передать директору",
]
RELEVANCE_SALES = [
    "отдел продаж", "менеджеры по продажам", "у нас есть менеджеры",
    "команда продаж", "есть база", "база клиентов", "лиды", "заявки",
    "входящие заявки", "исходящий прозвон", "поток клиентов", "много заявок",
    "объём базы", "crm", "битрикс", "amocrm", "воронка продаж",
]
RELEVANCE_INTEREST = [
    "сколько стоит", "какая стоимость", "как это работает", "какие результаты",
    "какая конверсия", "есть кейсы", "можете показать кейс", "как внедряется",
    "с чем интегрируется", "интересно", "можно посмотреть", "пришлите информацию",
    "отправьте материалы", "давайте обсудим", "можно подробнее",
    "пусть специалист перезвонит",
]

NEXT_STEP_PROPOSAL = [
    "давайте назначим", "созвон", "встреча", "демо", "презентация",
    "передам специалисту", "передам эксперту", "можем показать",
    "отправить материалы", "перезвонить позже", "подберём время",
    "эксперт созвонится", "онлайн-звонок",
]
NEXT_STEP_ACCEPTANCE = [
    "давайте", "да", "можно", "хорошо", "отправьте", "пусть перезвонят",
    "свяжитесь", "запишите", "завтра", "после обеда", "на почту",
    "в телеграм", "в понедельник", "готов", "подходит",
]

CONTEXT_FINISH = [
    "на почту", "в телеграм", "завтра", "после обеда", "в понедельник",
    "перезвоните", "не звоните", "не актуально", "не интересно",
    "отправьте материалы", "передам директору", "соединю с",
    "запишите", "готов", "подходит", "эксперт", "созвон", "демо",
]

NOT_RELEVANCE = [
    "алло", "да", "слушаю", "ну", "понятно", "угу", "хорошо",
    "говорите", "быстрее", "мне некогда", "не интересно", "не актуально",
    "перезвоните потом",
]

MEETING_KEYWORDS = ["встреча", "созвон", "демо", "презентация", "онлайн-звонок", "назначим"]


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)


def _find_first_group(text: str, groups: dict[str, list[str]]) -> str | None:
    for name, phrases in groups.items():
        if _contains_any(text, phrases):
            return name
    return None


def _bot_lines(text: str) -> str:
    lines = re.split(r"[\n;|]+", text)
    bot_parts = []
    for line in lines:
        low = line.strip().lower()
        if low.startswith(("bot:", "бот:", "assistant:", "ассистент:", "ai:")):
            bot_parts.append(low)
    return " ".join(bot_parts) if bot_parts else text


def _user_lines(text: str) -> str:
    lines = re.split(r"[\n;|]+", text)
    user_parts = []
    for line in lines:
        low = line.strip().lower()
        if low.startswith(("user:", "юзер:", "клиент:", "client:", "абонент:")):
            user_parts.append(low)
    return " ".join(user_parts) if user_parts else text


def _detect_relevance(text: str) -> tuple[bool, str | None]:
    if not text:
        return False, None

    user_text = _user_lines(text)
    check_text = user_text if user_text.strip() else text

    if _contains_any(check_text, NOT_RELEVANCE) and not any(
        _contains_any(check_text, g)
        for g in (RELEVANCE_ROLE, RELEVANCE_SALES, RELEVANCE_INTEREST)
    ):
        return False, None

    groups = {
        "role_decision_maker": RELEVANCE_ROLE,
        "sales_team_leads": RELEVANCE_SALES,
        "meaningful_interest": RELEVANCE_INTEREST,
    }
    found = _find_first_group(check_text, groups)
    if found:
        return True, found
    return False, None


def _detect_next_step_proposed(text: str) -> tuple[bool, str | None]:
    bot_text = _bot_lines(text)
    if _contains_any(bot_text, NEXT_STEP_PROPOSAL):
        for kw in MEETING_KEYWORDS:
            if kw in bot_text:
                return True, "meeting" if kw in ("встреча", "назначим") else kw
        if "материал" in bot_text:
            return True, "materials"
        if "перезвон" in bot_text:
            return True, "callback"
        if "эксперт" in bot_text or "специалист" in bot_text:
            return True, "expert_call"
        return True, "other"
    return False, None


def _detect_next_step_accepted(text: str, proposed: bool) -> bool:
    if not proposed:
        return False
    user_text = _user_lines(text)
    check_text = user_text if user_text.strip() else text

    proposal_idx = -1
    bot_text = _bot_lines(text)
    for phrase in NEXT_STEP_PROPOSAL:
        idx = bot_text.rfind(phrase)
        if idx > proposal_idx:
            proposal_idx = idx

    after_proposal = check_text[proposal_idx:] if proposal_idx >= 0 else check_text
    acceptance_hits = sum(1 for p in NEXT_STEP_ACCEPTANCE if p in after_proposal)
    strong = ["завтра", "после обеда", "на почту", "в телеграм", "в понедельник", "запишите", "готов", "подходит", "отправьте"]
    if any(s in after_proposal for s in strong):
        return True
    return acceptance_hits >= 1 and len(after_proposal.split()) > 2


def _detect_context_finish(text: str, accepted: bool) -> bool:
    if not text:
        return False
    if accepted and _contains_any(text, CONTEXT_FINISH):
        return True
    if _contains_any(text, ["не звоните", "не актуально", "не интересно", "не подходит"]):
        return True
    if _contains_any(text, ["передам директору", "соединю с", "коммерческому директору"]):
        return True
    if accepted:
        return True
    return False


def _max_funnel_stage(row: pd.Series) -> str:
    if not row["has_transcript"]:
        if row["duration_seconds"] <= 10:
            return "0_no_dialog"
        return "0_no_dialog"

    text = row["transcript_clean"]
    if len(text) < 5:
        return "0_no_dialog"

    stage = "0_no_dialog"

    if _contains_any(text, GREETING_SIGNALS):
        stage = "1_contact_greeting"

    if _contains_any(text, CONSENT_SIGNALS):
        stage = max(stage, "2_consent_permission", key=lambda s: STAGE_ORDER[s])

    if _contains_any(text, OFFER_SIGNALS):
        stage = max(stage, "3_offer_delivered", key=lambda s: STAGE_ORDER[s])

    has_rel, _ = _detect_relevance(text)
    if has_rel:
        stage = max(stage, "4_relevance_signal_detected", key=lambda s: STAGE_ORDER[s])

    proposed, _ = _detect_next_step_proposed(text)
    if proposed:
        stage = max(stage, "5_next_step_proposed", key=lambda s: STAGE_ORDER[s])

    if _detect_next_step_accepted(text, proposed):
        stage = max(stage, "6_next_step_accepted", key=lambda s: STAGE_ORDER[s])

    if _detect_context_finish(text, STAGE_ORDER[stage] >= STAGE_ORDER["6_next_step_accepted"]):
        stage = max(stage, "7_context_collected_or_correctly_finished", key=lambda s: STAGE_ORDER[s])

    return stage


def _classify_no_transcript(row: pd.Series) -> dict:
    has_transcript = row["has_transcript"]
    duration = row["duration_seconds"]
    termination = str(row["termination_reason"]).strip().lower()

    result = {
        "no_transcript_type": "",
        "technical_suspect_no_transcript": False,
        "fixed_timeout_without_transcript": False,
        "bot_hangup_without_transcript": False,
        "early_no_dialog": False,
        "possible_system_anomaly": False,
    }

    if has_transcript:
        return result

    types: list[str] = []

    if duration <= 10:
        result["early_no_dialog"] = True
        types.append("early_no_dialog")

    if duration >= 30:
        result["technical_suspect_no_transcript"] = True
        types.append("technical_suspect_no_transcript")

    if duration in (180, 240):
        result["fixed_timeout_without_transcript"] = True
        types.append("fixed_timeout_without_transcript")

    if termination == "bot_hangup":
        result["bot_hangup_without_transcript"] = True
        types.append("bot_hangup_without_transcript")

    if duration >= 60 or result["fixed_timeout_without_transcript"] or result["bot_hangup_without_transcript"]:
        result["possible_system_anomaly"] = True
        types.append("possible_system_anomaly")

    result["no_transcript_type"] = "|".join(types) if types else "no_transcript"
    return result


def classify_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Add funnel and diagnostic columns."""
    logger.info("Classification started | rows=%s", len(df))
    enriched = df.copy()

    enriched["funnel_stage_reached"] = enriched.apply(_max_funnel_stage, axis=1)
    enriched["drop_off_stage"] = enriched["funnel_stage_reached"].map(DROP_OFF_MAP)

    relevance = enriched["transcript_clean"].apply(_detect_relevance)
    enriched["has_relevance_signal"] = relevance.apply(lambda x: x[0])
    enriched["relevance_signal_type"] = relevance.apply(lambda x: x[1] or "")

    proposals = enriched["transcript_clean"].apply(_detect_next_step_proposed)
    enriched["next_step_proposed"] = proposals.apply(lambda x: x[0])
    enriched["next_step_type"] = proposals.apply(lambda x: x[1] or "")

    enriched["next_step_accepted"] = enriched.apply(
        lambda r: _detect_next_step_accepted(r["transcript_clean"], r["next_step_proposed"]),
        axis=1,
    )

    enriched["is_qualified_next_step"] = (
        enriched["has_relevance_signal"]
        & enriched["next_step_proposed"]
        & enriched["next_step_accepted"]
    )

    enriched["is_meeting_without_relevance_signal"] = (
        enriched["next_step_proposed"]
        & enriched["next_step_accepted"]
        & ~enriched["has_relevance_signal"]
        & enriched["next_step_type"].isin(["meeting", "созвон", "демо", "expert_call"])
    )

    no_tx = enriched.apply(_classify_no_transcript, axis=1, result_type="expand")
    for col in no_tx.columns:
        enriched[col] = no_tx[col]

    logger.info("Classification completed | rows=%s | no_dialog=%s | qualified=%s | suspicious_no_transcript=%s", len(enriched), int((enriched["funnel_stage_reached"] == "0_no_dialog").sum()), int(enriched["is_qualified_next_step"].sum()), int(enriched["technical_suspect_no_transcript"].sum()))
    return enriched
