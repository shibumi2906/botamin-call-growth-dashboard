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
    "7_context_collected",
]

DROP_OFF_MAP = {
    "0_no_dialog": "before_contact",
    "1_contact_greeting": "contact_to_consent",
    "2_consent_permission": "consent_to_offer",
    "3_offer_delivered": "offer_to_relevance_signal",
    "4_relevance_signal_detected": "relevance_to_next_step_proposed",
    "5_next_step_proposed": "proposal_to_acceptance",
    "6_next_step_accepted": "acceptance_to_context_or_finish",
    "7_context_collected": "completed",
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
    "эксперт созвонится", "онлайн-звонок", "созвониться", "встречу", "перезвонить",
]
NEXT_STEP_ACCEPTANCE = [
    "давайте", "да", "можно", "хорошо", "отправьте", "пусть перезвонят",
    "свяжитесь", "запишите", "завтра", "после обеда", "на почту",
    "в телеграм", "в понедельник", "готов", "подходит",
]

CONTEXT_FINISH = [
    "на почту", "в телеграм", "завтра", "после обеда", "в понедельник",
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


ROLE_MARKER = re.compile(
    r"(?<!\w)(bot|бот|assistant|ассистент|ai|user|юзер|клиент|client|абонент)\s*:",
    re.IGNORECASE,
)
BOT_ROLES = {"bot", "бот", "assistant", "ассистент", "ai"}


def parse_turns(text: str) -> list[tuple[str, str]]:
    """Keep speaker order, including inline markers and multiline utterances.

    Unlabelled text is intentionally not attributed to either participant.
    """
    markers = list(ROLE_MARKER.finditer(text))
    turns = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        utterance = text[marker.end():end].strip(" \n\r;|")
        if utterance:
            role = "bot" if marker.group(1).lower() in BOT_ROLES else "user"
            turns.append((role, utterance.lower()))
    return turns


def _bot_lines(text: str) -> str:
    return " ".join(t for role, t in parse_turns(text) if role == "bot")


def _user_lines(text: str) -> str:
    return " ".join(t for role, t in parse_turns(text) if role == "user")


def _phrase_present(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def _detect_relevance(text: str) -> tuple[bool, str | None]:
    groups = {
        "role_decision_maker": RELEVANCE_ROLE,
        "sales_team_leads": RELEVANCE_SALES,
        "meaningful_interest": RELEVANCE_INTEREST,
    }
    for role, utterance in parse_turns(text):
        if role != "user":
            continue
        for sentence in re.split(r"[.!?;]+", utterance):
            if re.search(r"\b(не|нет|без)\b", sentence):
                continue
            found = _find_first_group(sentence, groups)
            if found:
                return True, found
    return False, None


def _proposal_type(utterance: str) -> str | None:
    if re.search(r"отмечаю встречу|зафиксировала|зафиксировал", utterance):
        return None
    if not _contains_any(utterance, NEXT_STEP_PROPOSAL):
        return None
    if re.search(r"\b(не|нет)\b.{0,20}(встреч|демо|созвон)", utterance):
        return None
    for kw in MEETING_KEYWORDS:
        if kw in utterance:
            return "meeting" if kw in ("встреча", "назначим") else kw
    if "материал" in utterance:
        return "materials"
    if "перезвон" in utterance:
        return "callback"
    if "эксперт" in utterance or "специалист" in utterance:
        return "expert_call"
    return "other"


def _accepts(utterance: str) -> bool:
    if re.search(r"\b(не|нет)\b|подумаю|подумать|неудобно", utterance):
        return False
    # Bare assent must answer the proposal, not occur inside another conversation.
    if re.fullmatch(r"[\s,!.]*(ну[, ]*)?(да|давайте|можно|хорошо|готов|подходит|угу)[\s,!.]*", utterance):
        return True
    strong = ["завтра", "после обеда", "на почту", "в телеграм", "в понедельник",
              "запишите", "подходит", "отправьте", "пусть перезвонят", "свяжитесь"]
    return any(_phrase_present(utterance, phrase) for phrase in strong)


def _refusal_outcome(utterance: str) -> str | None:
    """Explicit refusal can revoke acceptance even after a different bot turn."""
    if re.search(r"\bне\s+звоните\b|\bбольше\s+не\s+звон", utterance):
        return "do_not_call"
    if re.search(r"\b(отмен\w*|передум\w*)\b|\bне\s*(интересно|актуально)\b|"
                 r"\bне\s+(подходит|надо|нужно|хочу|буду)\b", utterance):
        return "refused"
    return None


def _next_step_analysis(text: str) -> dict:
    result = {"next_step_proposed": False, "next_step_type": "",
              "next_step_accepted": False, "proposal_evidence": "",
              "acceptance_evidence": "", "acceptance_proposal_evidence": "",
              "call_outcome": "no_next_step", "outcome_evidence": ""}
    pending = ""
    for role, utterance in parse_turns(text):
        if role == "bot":
            proposal_type = _proposal_type(utterance)
            if proposal_type:
                pending = utterance
                result.update(next_step_proposed=True, next_step_type=proposal_type,
                              proposal_evidence="bot: " + utterance,
                              next_step_accepted=False, acceptance_evidence="",
                              acceptance_proposal_evidence="")
                if result["call_outcome"] not in ("refused", "do_not_call"):
                    result.update(call_outcome="proposed", outcome_evidence="")
            else:
                # A short yes after a different bot question is not proposal acceptance.
                pending = ""
        else:
            refusal = _refusal_outcome(utterance)
            if refusal:
                result.update(next_step_accepted=False, acceptance_evidence="",
                              acceptance_proposal_evidence="", call_outcome=refusal,
                              outcome_evidence="user: " + utterance)
                pending = ""
            elif pending and _accepts(utterance):
                result.update(next_step_accepted=True,
                              acceptance_evidence="user: " + utterance,
                              acceptance_proposal_evidence="bot: " + pending,
                              call_outcome="accepted", outcome_evidence="user: " + utterance)
            elif pending and re.search(r"\b(нет|не)\b|подумаю|подумать", utterance):
                result.update(next_step_accepted=False, acceptance_evidence="",
                              acceptance_proposal_evidence="", call_outcome="not_agreed",
                              outcome_evidence="user: " + utterance)
    return result


def _detect_next_step_proposed(text: str) -> tuple[bool, str | None]:
    result = _next_step_analysis(text)
    return result["next_step_proposed"], result["next_step_type"] or None


def _detect_next_step_accepted(text: str, proposed: bool) -> bool:
    return bool(proposed and _next_step_analysis(text)["next_step_accepted"])


def _detect_context_finish(text: str, accepted: bool) -> bool:
    """Only context in the currently valid acceptance advances the funnel."""
    return accepted and _contains_any(text, CONTEXT_FINISH)


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

    next_step = _next_step_analysis(text)
    if next_step["next_step_proposed"]:
        stage = max(stage, "5_next_step_proposed", key=lambda s: STAGE_ORDER[s])

    if next_step["next_step_accepted"]:
        stage = max(stage, "6_next_step_accepted", key=lambda s: STAGE_ORDER[s])

    if _detect_context_finish(next_step["acceptance_evidence"], next_step["next_step_accepted"]):
        stage = "7_context_collected"

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

    analyses = enriched["transcript_clean"].apply(_next_step_analysis)
    for column in ("next_step_proposed", "next_step_type", "next_step_accepted",
                   "proposal_evidence", "acceptance_evidence", "acceptance_proposal_evidence",
                   "call_outcome", "outcome_evidence"):
        enriched[column] = analyses.apply(lambda result: result[column])
    enriched["relevance_evidence"] = enriched["transcript_clean"].apply(
        lambda text: next(("user: " + utterance for role, utterance in parse_turns(text)
                           if role == "user" and _detect_relevance("user: " + utterance)[0]), "")
    )
    enriched["classification_needs_review"] = enriched["transcript_clean"].apply(
        lambda text: bool(text) and not any(role == "user" for role, _ in parse_turns(text))
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
