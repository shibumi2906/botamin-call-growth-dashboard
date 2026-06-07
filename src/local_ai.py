"""Optional local Gemma helper via Ollama."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

__all__ = [
    "ask_local_gemma",
    "ask_local_llm",
    "build_ai_context",
    "build_compact_summary",
    "build_technical_diagnostics",
    "call_gemma_helper",
    "check_local_ai_health",
    "check_ollama_status",
    "get_gemma_config",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env from project root (Streamlit cwd may differ from package path).
load_dotenv(ENV_PATH, override=False)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e2b"
DEFAULT_TIMEOUT = 120

LIMITATIONS = (
    "Анализ rule-based по ключевым словам; возможны false positive/negative. "
    "Relevance signal консервативен. Gemma не классифицировал звонки — "
    "использует только агрегированные метрики дашборда."
)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, str(default)).strip().lower()
    return val in {"1", "true", "yes", "on"}


def get_gemma_config() -> dict:
    return {
        "enabled": _env_bool("ENABLE_LOCAL_AI", False),
        "provider": os.getenv("LOCAL_LLM_PROVIDER", "ollama"),
        "base_url": os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "model": os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL),
        "timeout": int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT))),
        "env_path": str(ENV_PATH),
        "env_exists": ENV_PATH.exists(),
    }


def _list_ollama_models(base_url: str) -> tuple[list[str], str | None]:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if not resp.ok:
            return [], f"Ollama ответил с кодом {resp.status_code}"
        models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
        return models, None
    except requests.ConnectionError:
        logger.error("Local LLM connection failed")
        return [], "Ollama не запущен или недоступен по адресу " + base_url
    except requests.Timeout:
        logger.warning("Local LLM request timed out")
        return [], "Таймаут подключения к Ollama"
    except requests.RequestException as exc:
        logger.exception("Local LLM HTTP request failed")
        return [], f"Ошибка подключения к Ollama: {exc}"


def _resolve_model(configured: str, available: list[str]) -> tuple[str, bool]:
    if configured in available:
        return configured, True

    configured_base = configured.split(":")[0].lower()
    for name in available:
        if name.split(":")[0].lower() == configured_base:
            return name, True

    gemma_models = sorted(m for m in available if "gemma" in m.lower())
    if gemma_models:
        return gemma_models[0], True

    return configured, False


def check_local_ai_health(config: dict | None = None) -> dict[str, Any]:
    """Lightweight health check: config flag + Ollama reachability + model availability."""
    cfg = config or get_gemma_config()
    result: dict[str, Any] = {
        "enabled": cfg["enabled"],
        "env_exists": cfg.get("env_exists", ENV_PATH.exists()),
        "env_path": cfg.get("env_path", str(ENV_PATH)),
        "ollama_reachable": False,
        "model_configured": cfg["model"],
        "model_resolved": cfg["model"],
        "model_available": False,
        "available_models": [],
        "chat_ready": False,
        "status_label": "unavailable",
        "reason": None,
    }

    if not result["env_exists"]:
        result["reason"] = (
            "Файл .env не найден. Скопируйте .env.example в .env, "
            "установите ENABLE_LOCAL_AI=true и перезапустите Streamlit."
        )
    elif not result["enabled"]:
        result["reason"] = "ENABLE_LOCAL_AI=false в .env. Установите true и перезапустите Streamlit."

    models, conn_error = _list_ollama_models(cfg["base_url"])
    result["available_models"] = models

    if conn_error:
        if result["reason"]:
            result["reason"] += f" ({conn_error})"
        else:
            result["reason"] = conn_error
        return result

    result["ollama_reachable"] = True
    resolved, model_ok = _resolve_model(cfg["model"], models)
    result["model_resolved"] = resolved
    result["model_available"] = model_ok

    if not model_ok:
        model_hint = ", ".join(models[:5]) if models else "нет моделей"
        result["reason"] = (
            f"Модель '{cfg['model']}' не найдена в Ollama. "
            f"Доступные: {model_hint}. "
            f"Обновите LOCAL_LLM_MODEL в .env и перезапустите Streamlit."
        )
        return result

    if not result["enabled"]:
        result["reason"] = (
            f"ENABLE_LOCAL_AI=false, но Ollama доступен и модель '{resolved}' найдена. "
            "Установите ENABLE_LOCAL_AI=true в .env и перезапустите Streamlit."
        )
        return result

    result["chat_ready"] = True
    result["status_label"] = "available"
    result["reason"] = None
    return result


def check_ollama_status(config: dict | None = None) -> tuple[bool, str]:
    """Backward-compatible status helper."""
    health = check_local_ai_health(config)
    if health["chat_ready"]:
        return True, f"available — {health['model_resolved']}"
    return False, health["reason"] or "Local Gemma is unavailable."


def build_technical_diagnostics(metrics: dict) -> dict[str, Any]:
    return {
        "no_transcript_rate_pct": metrics.get("No Transcript Rate"),
        "early_no_dialog": metrics.get("Early No-dialog Calls"),
        "suspicious_no_transcript": metrics.get("Suspicious No-transcript Calls"),
        "fixed_timeout_without_transcript": metrics.get("Fixed Timeout Without Transcript"),
        "bot_hangup_without_transcript": metrics.get("Bot Hangup Without Transcript"),
    }


def build_ai_context(
    metrics: dict,
    funnel_data: pd.DataFrame,
    technical_diagnostics: dict,
    recommendations: dict,
    ab_test: dict | None = None,
) -> str:
    """Compact analytical context for Gemma before every user question."""
    funnel_lines = []
    for _, row in funnel_data.iterrows():
        funnel_lines.append(
            f"- {row['stage_label']}: reached≥stage={row['reached_at_least']}, "
            f"drop-off={row['drop_off_count']} ({row['drop_off_pct']}%)"
        )

    ab = ab_test or {}
    rec = recommendations

    return (
        "=== BOTAMIN CALL GROWTH DASHBOARD CONTEXT ===\n"
        f"Total Calls: {metrics.get('Total Calls')}\n"
        f"Calls With Transcript: {metrics.get('Calls With Transcript')}\n"
        f"Calls Without Transcript: {metrics.get('Calls Without Transcript')} "
        f"({metrics.get('No Transcript Rate')}%)\n"
        f"Qualified Next Step Rate: {metrics.get('Qualified Next Step Rate')}%\n"
        f"Qualified Next Step Rate From Conversations: "
        f"{metrics.get('Qualified Next Step Rate From Conversations')}%\n"
        f"Weakest Funnel Stage: {metrics.get('Weakest Funnel Stage Label')} "
        f"({metrics.get('Weakest Funnel Stage')})\n"
        f"Offer Reach Rate: {metrics.get('Offer Reach Rate')}%\n"
        f"Relevance Signal Rate: {metrics.get('Relevance Signal Rate')}%\n"
        f"Offer → Relevance Signal Conversion: {metrics.get('Offer → Relevance Signal Conversion')}%\n"
        f"Next Step Proposal Rate: {metrics.get('Next Step Proposal Rate')}%\n"
        f"Next Step Acceptance Rate: {metrics.get('Next Step Acceptance Rate')}%\n"
        f"Proposal → Acceptance Conversion: {metrics.get('Proposal → Acceptance Conversion')}%\n"
        f"Meeting Without Relevance Signal Rate: {metrics.get('Meeting Without Relevance Signal Rate')}%\n"
        "\n=== TECHNICAL / NO-TRANSCRIPT ===\n"
        f"Early No-dialog Calls: {technical_diagnostics.get('early_no_dialog')}\n"
        f"Suspicious No-transcript Calls: {technical_diagnostics.get('suspicious_no_transcript')}\n"
        f"Fixed Timeout Without Transcript: {technical_diagnostics.get('fixed_timeout_without_transcript')}\n"
        f"Bot Hangup Without Transcript: {technical_diagnostics.get('bot_hangup_without_transcript')}\n"
        f"No Transcript Rate: {technical_diagnostics.get('no_transcript_rate_pct')}%\n"
        "\n=== FUNNEL ===\n"
        + "\n".join(funnel_lines)
        + "\n\n=== MAIN GROWTH OPPORTUNITY ===\n"
        f"Problem: {rec.get('main_problem')}\n"
        f"Why it matters: {rec.get('why_it_matters')}\n"
        f"Supporting metric: {rec.get('supporting_metric')}\n"
        f"Affected stage: {rec.get('affected_stage')}\n"
        f"Recommended focus: {rec.get('recommended_focus')}\n"
        f"Type: {rec.get('recommendation_type')}\n"
        "\n=== RECOMMENDED FIRST A/B TEST ===\n"
        f"Hypothesis: {ab.get('hypothesis')}\n"
        f"Variant A: {ab.get('variant_a')}\n"
        f"Variant B: {ab.get('variant_b')}\n"
        f"Main metric: {ab.get('main_metric')}\n"
        f"Guardrails: {', '.join(ab.get('guardrail_metrics', []))}\n"
        f"Expected effect: {ab.get('expected_effect')}\n"
        "\n=== LIMITATIONS ===\n"
        f"{LIMITATIONS}\n"
    )


def ask_local_llm(
    user_question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
    config: dict | None = None,
) -> str:
    """Internal Ollama chat call with optional history."""
    cfg = config or get_gemma_config()
    health = check_local_ai_health(cfg)

    if not health["chat_ready"]:
        return (
            "Local Gemma is disabled or unavailable. "
            "Enable ENABLE_LOCAL_AI=true and check Ollama/Gemma model availability."
            + (f"\n\nПричина: {health['reason']}" if health.get("reason") else "")
        )

    system_prompt = (
        "Ты аналитический ассистент Botamin Call Growth Dashboard. "
        "Отвечай на русском, опираясь ТОЛЬКО на переданный контекст дашборда. "
        "Ссылайся на конкретные метрики, гипотезу и A/B-тест из контекста. "
        "Не выдумывай цифры. Если данных недостаточно — скажи об этом.\n\n"
        f"КОНТЕКСТ ДАШБОРДА:\n{context}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in history or []:
        role = msg.get("role", "user")
        if role in ("user", "assistant") and msg.get("content"):
            messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_question})

    model = health["model_resolved"]
    try:
        resp = requests.post(
            f"{cfg['base_url']}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=cfg["timeout"],
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return content.strip() or "Gemma вернула пустой ответ."
    except requests.Timeout:
        logger.warning("Local LLM request timed out")
        return "Gemma timeout: запрос превысил лимит времени."
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("error", "")
        except Exception:
            pass
        return (
            "Local Gemma is disabled or unavailable. "
            "Enable ENABLE_LOCAL_AI=true and check Ollama/Gemma model availability."
            + (f"\n\nОшибка Ollama: {detail or exc}" if detail or exc else "")
        )
    except requests.RequestException as exc:
        logger.exception("Local LLM HTTP request failed")
        return (
            "Local Gemma is disabled or unavailable. "
            "Enable ENABLE_LOCAL_AI=true and check Ollama/Gemma model availability."
            f"\n\nПричина: {exc}"
        )


def ask_local_gemma(user_question: str, context: str) -> str:
    """
    Public function used by app.py.
    Sends the user question and dashboard context to local Ollama/Gemma.
    """
    return ask_local_llm(user_question=user_question, context=context)


def build_compact_summary(
    recommendation: dict,
    metrics: dict,
    ab_test: dict,
    example_calls: pd.DataFrame | None,
    technical_summary: str = "",
) -> str:
    """Compact context for one-shot Gemma hypothesis helper."""
    tech = build_technical_diagnostics(metrics)
    base = build_ai_context(metrics, pd.DataFrame(), tech, recommendation, ab_test)
    examples_text = ""
    if example_calls is not None and not example_calls.empty:
        for _, row in example_calls.head(5).iterrows():
            snippet = str(row.get("transcript_clean", ""))[:200]
            examples_text += (
                f"- phone={row.get('phone_id')}, duration={row.get('duration_seconds')}s, "
                f"stage={row.get('funnel_stage_reached')}, snippet={snippet}\n"
            )
    return base + f"\n=== EXAMPLE CALLS ===\n{examples_text or 'none'}\n" + (
        f"technical_diagnostics_note: {technical_summary}\n" if technical_summary else ""
    )


def call_gemma_helper(summary: str, config: dict | None = None) -> dict[str, Any]:
    """One-shot hypothesis refinement via Gemma (button-triggered only)."""
    logger.info("Gemma hypothesis helper started | summary_length=%s", len(summary or ""))
    prompt = (
        "На основе контекста дашборда верни ТОЛЬКО valid JSON с ключами: "
        "refined_hypothesis, improved_variant_b, main_metric, guardrail_metrics (array), "
        "why_this_test, manual_review_suggestion. Значения на русском.\n\n"
        f"{summary}"
    )
    answer = ask_local_llm(prompt, summary, config=config)
    if answer.startswith("Local Gemma is disabled") or answer.startswith("Gemma timeout"):
        return {"error": answer}

    start = answer.find("{")
    end = answer.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(answer[start:end])
        except json.JSONDecodeError:
            pass
    return {
        "refined_hypothesis": answer[:500],
        "improved_variant_b": "",
        "main_metric": "",
        "guardrail_metrics": [],
        "why_this_test": "",
        "manual_review_suggestion": "",
    }
