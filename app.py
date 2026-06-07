"""Botamin Call Growth Dashboard — Streamlit app."""

from __future__ import annotations

from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.classification import classify_calls
from src.data_processing import DEFAULT_DATA_PATH, preprocess_calls
from src.export import to_csv_bytes
from src.local_ai import (
    ask_local_gemma,
    ask_local_llm,
    build_ai_context,
    build_compact_summary,
    build_technical_diagnostics,
    call_gemma_helper,
    check_local_ai_health,
    get_gemma_config,
)
from src.metrics import build_funnel_table, calculate_metrics, get_adaptive_metrics
from src.qa import SUGGESTED_QUESTIONS, answer_analyst_question
from src.recommendations import generate_ab_test, generate_main_growth_opportunity



def configure_logging() -> logging.Logger:
    """Configure console and rotating file logging once per process."""
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        file_handler = RotatingFileHandler(
            logs_dir / "app.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(console)
        root.addHandler(file_handler)
    return logging.getLogger(__name__)


logger = configure_logging()
logger.info("Application startup")

st.set_page_config(
    page_title="Botamin Call Growth Dashboard",
    page_icon="📞",
    layout="wide",
)

st.markdown(
    """
    # Botamin Call Growth Dashboard
    **AI sales bot call analytics:** funnel, drop-off, growth opportunity and first A/B test.
    """
)


@st.cache_data(show_spinner="Загрузка и анализ данных...")
def run_pipeline(source_key: str, file_bytes: bytes | None) -> tuple:
    """Cached processing pipeline."""
    logger.info("Pipeline started | source=%s | uploaded=%s", source_key, file_bytes is not None)
    if file_bytes is not None:
        source = BytesIO(file_bytes)
    else:
        source = DEFAULT_DATA_PATH

    df, err = preprocess_calls(source)
    if err:
        logger.error("Preprocessing failed | source=%s | error=%s", source_key, err)
        return None, None, None, None, None, err

    enriched = classify_calls(df)
    funnel_table = build_funnel_table(enriched)
    metrics = calculate_metrics(enriched, funnel_table)
    recommendation = generate_main_growth_opportunity(metrics, funnel_table)
    ab_test = generate_ab_test(recommendation, metrics)
    logger.info("Pipeline completed | rows=%s | recommendation_type=%s", len(enriched), recommendation.get("recommendation_type"))
    return enriched, funnel_table, metrics, recommendation, ab_test, None


def _load_data():
    uploaded = st.session_state.get("uploaded_file")
    file_bytes = None
    file_name = DEFAULT_DATA_PATH.name

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name
        source_key = f"upload:{file_name}:{len(file_bytes)}"
        logger.info("Uploaded file selected | name=%s | bytes=%s", file_name, len(file_bytes))
    else:
        if not DEFAULT_DATA_PATH.exists():
            logger.error("Default data file not found | path=%s", DEFAULT_DATA_PATH)
            return None, None, None, None, None, "Файл data/calls_week_anon.xlsx не найден. Загрузите Excel вручную.", "—"
        source_key = f"default:{DEFAULT_DATA_PATH.stat().st_mtime}"
        logger.info("Default data file selected | path=%s", DEFAULT_DATA_PATH)

    return (*run_pipeline(source_key, file_bytes), file_name)


# --- Sidebar ---
with st.sidebar:
    st.header("Data Loading")
    uploaded_file = st.file_uploader("Загрузить Excel", type=["xlsx", "xls"])
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
    else:
        st.session_state.pop("uploaded_file", None)

    ai_health = check_local_ai_health()
    if ai_health["status_label"] == "available":
        st.success(f"Local Gemma: available ({ai_health['model_resolved']})")
    else:
        st.warning("Local Gemma: unavailable")
        if ai_health.get("reason"):
            st.caption(ai_health["reason"])
    if not ai_health["env_exists"]:
        st.caption("Подсказка: скопируйте `.env.example` → `.env` и перезапустите Streamlit.")
    elif ai_health["ollama_reachable"] and not ai_health["enabled"]:
        st.caption("Ollama работает — включите ENABLE_LOCAL_AI=true в `.env` и перезапустите Streamlit.")


try:
    enriched_df, funnel_table, metrics, recommendation, ab_test, load_error, loaded_name = _load_data()
except Exception:
    logger.exception("Unexpected application error during data loading")
    st.error("Ошибка загрузки: произошла непредвиденная ошибка при обработке файла.")
    st.stop()

if load_error:
    st.error(load_error)
    st.stop()

with st.sidebar:
    st.markdown(f"**Файл:** `{loaded_name}`")
    st.markdown(f"**Всего строк:** {metrics['Total Calls']:,}")
    st.markdown(f"**С транскриптом:** {metrics['Calls With Transcript']:,}")
    st.markdown(f"**Без транскрипта:** {metrics['Calls Without Transcript']:,}")

rec_type = recommendation.get("recommendation_type", "qualification")
adaptive = get_adaptive_metrics(metrics, rec_type if rec_type != "risk" else "offer")

# --- Adaptive Executive Summary ---
st.header("Adaptive Executive Summary")
cols = st.columns(min(len(adaptive), 3))
for i, (label, value) in enumerate(adaptive):
    cols[i % len(cols)].metric(label, value)

st.info(
    f"**Вывод:** {recommendation['main_problem']}. "
    f"{recommendation['why_it_matters']} "
    f"Ключевая метрика: {recommendation['supporting_metric']}."
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Main Growth Opportunity")
    st.markdown(f"**Проблема:** {recommendation['main_problem']}")
    st.markdown(f"**Почему важно:** {recommendation['why_it_matters']}")
    st.markdown(f"**Метрика:** {recommendation['supporting_metric']}")
    st.markdown(f"**Этап воронки:** {recommendation['affected_stage']}")
    st.markdown(f"**Фокус:** {recommendation['recommended_focus']}")

with c2:
    st.subheader("Recommended First Action / A/B Test")
    st.markdown(f"**Гипотеза:** {ab_test['hypothesis']}")
    st.markdown(f"**Variant A:** {ab_test['variant_a']}")
    st.markdown(f"**Variant B:** {ab_test['variant_b']}")
    st.markdown(f"**Main metric:** {ab_test['main_metric']}")
    st.markdown(f"**Guardrails:** {', '.join(ab_test['guardrail_metrics'])}")
    st.markdown(f"**Expected effect:** {ab_test['expected_effect']}")

st.divider()

# --- Tabs for detailed sections ---
tab_funnel, tab_technical, tab_growth, tab_ab, tab_ask, tab_gemma, tab_examples, tab_preview, tab_export = st.tabs(
    [
        "Funnel & Drop-off",
        "Technical Diagnostics",
        "Growth Opportunity",
        "A/B Test",
        "Ask the Dashboard",
        "Gemma Helper",
        "Manual Review",
        "Data Preview",
        "CSV Export",
    ]
)

with tab_funnel:
    st.subheader("Funnel & Drop-off")

    fig = go.Figure(
        go.Funnel(
            y=funnel_table["stage_label"],
            x=funnel_table["reached_at_least"],
            textinfo="value+percent initial",
        )
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        funnel_table[
            ["stage_label", "calls_at_stage", "reached_at_least", "step_conversion_pct", "drop_off_count", "drop_off_pct"]
        ].rename(
            columns={
                "stage_label": "Stage",
                "calls_at_stage": "At Stage",
                "reached_at_least": "Reached ≥ Stage",
                "step_conversion_pct": "Step Conv %",
                "drop_off_count": "Drop-off",
                "drop_off_pct": "Drop-off %",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"**Weakest transition:** {metrics['Weakest Funnel Stage Label']} "
        f"| **Qualified Next Step Rate:** {metrics['Qualified Next Step Rate']}% "
        f"| **From Conversations:** {metrics['Qualified Next Step Rate From Conversations']}%"
    )

    with st.expander("All Metrics"):
        metric_rows = [{"Metric": k, "Value": v} for k, v in metrics.items()]
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

with tab_technical:
    st.subheader("Technical / No-Transcript Diagnostics")

    tcols = st.columns(4)
    tcols[0].metric("Without Transcript", metrics["Calls Without Transcript"])
    tcols[1].metric("Early No-dialog", metrics["Early No-dialog Calls"])
    tcols[2].metric("Suspicious (≥30s)", metrics["Suspicious No-transcript Calls"])
    tcols[3].metric("Fixed Timeout", metrics["Fixed Timeout Without Transcript"])

    st.metric("Bot Hangup Without Transcript", metrics["Bot Hangup Without Transcript"])

    no_tx = enriched_df[~enriched_df["has_transcript"]]
    suspicious = enriched_df[enriched_df["technical_suspect_no_transcript"]]

    st.markdown(f"**All calls without transcript:** {len(no_tx):,}")
    st.dataframe(
        suspicious[
            ["phone_id", "timestamp", "duration_seconds", "termination_reason", "no_transcript_type", "status"]
        ].head(100),
        use_container_width=True,
        hide_index=True,
    )

with tab_growth:
    st.subheader("Main Growth Opportunity")
    st.json(recommendation)

with tab_ab:
    st.subheader("First A/B Test Recommendation")
    st.json(ab_test)

with tab_ask:
    st.subheader("Ask the Dashboard")

    if "qa_context" not in st.session_state:
        st.session_state.qa_context = {}

    st.caption("Примеры вопросов: " + " | ".join(SUGGESTED_QUESTIONS[:4]))

    question = st.text_input("Ваш вопрос", key="analyst_question")
    if st.button("Спросить") and question:
        result = answer_analyst_question(
            question,
            metrics,
            funnel_table,
            enriched_df,
            recommendation,
            ab_test,
            st.session_state.qa_context,
        )
        st.session_state.qa_context = result.get("context", {})
        st.session_state.qa_last_answer = result

    if "qa_last_answer" in st.session_state:
        ans = st.session_state.qa_last_answer
        st.markdown(f"**Ответ:** {ans['answer']}")
        if ans.get("supporting_metric"):
            st.markdown(f"**Метрика:** {ans['supporting_metric']}")
        if ans.get("related_stage"):
            st.markdown(f"**Этап:** {ans['related_stage']}")
        if ans.get("recommended_action"):
            st.markdown(f"**Действие:** {ans['recommended_action']}")
        if ans.get("example_calls") is not None and not ans["example_calls"].empty:
            st.markdown("**Примеры звонков:**")
            st.dataframe(ans["example_calls"], use_container_width=True, hide_index=True)

with tab_gemma:
    st.subheader("Gemma Analytical Chat")
    st.caption(
        "Диалог с локальной Gemma на основе уже рассчитанных метрик дашборда. "
        "Запросы отправляются только после нажатия «Отправить». "
        "После изменения `.env` перезапустите Streamlit."
    )

    ai_health = check_local_ai_health()
    ai_context = build_ai_context(
        metrics,
        funnel_table,
        build_technical_diagnostics(metrics),
        recommendation,
        ab_test,
    )

    if "gemma_chat_history" not in st.session_state:
        st.session_state.gemma_chat_history = []

    if not ai_health["chat_ready"]:
        st.warning(
            "Local Gemma is disabled or unavailable. "
            "Enable ENABLE_LOCAL_AI=true and check Ollama/Gemma model availability."
        )
        if ai_health.get("reason"):
            st.info(ai_health["reason"])
        if ai_health.get("available_models"):
            st.caption(f"Модели в Ollama: {', '.join(ai_health['available_models'][:6])}")
    else:
        for msg in st.session_state.gemma_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_chat_input = st.chat_input("Задайте вопрос по результатам дашборда...")
        if user_chat_input:
            with st.spinner("Gemma думает..."):
                answer = ask_local_llm(
                    user_chat_input,
                    ai_context,
                    st.session_state.gemma_chat_history,
                )
            st.session_state.gemma_chat_history.append({"role": "user", "content": user_chat_input})
            st.session_state.gemma_chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.button("Очистить диалог"):
            st.session_state.gemma_chat_history = []
            st.rerun()

    st.divider()
    st.subheader("One-shot Hypothesis Helper")
    st.caption("Опционально: уточнить гипотезу и variant B одним запросом (только по кнопке).")

    if st.button("Запустить Gemma Helper"):
        cfg = get_gemma_config()
        health = check_local_ai_health(cfg)
        if not health["chat_ready"]:
            st.warning(
                "Local Gemma is disabled or unavailable. "
                "Enable ENABLE_LOCAL_AI=true and check Ollama/Gemma model availability."
            )
            if health.get("reason"):
                st.info(health["reason"])
        else:
            examples = enriched_df[
                enriched_df["drop_off_stage"] == recommendation.get("affected_stage", "")
            ].head(5)
            if examples.empty:
                examples = enriched_df[enriched_df["technical_suspect_no_transcript"]].head(5)

            tech_summary = (
                f"Suspicious={metrics['Suspicious No-transcript Calls']}, "
                f"Fixed timeout={metrics['Fixed Timeout Without Transcript']}, "
                f"Bot hangup={metrics['Bot Hangup Without Transcript']}"
            )
            summary = build_compact_summary(recommendation, metrics, ab_test, examples, tech_summary)
            with st.spinner("Gemma думает..."):
                result = call_gemma_helper(summary, cfg)

            if "error" in result:
                st.warning(result["error"])
            else:
                st.json(result)

with tab_examples:
    st.subheader("Example Calls For Manual Review")

    weakest = metrics["Weakest Funnel Stage"]
    sections = [
        ("Weakest drop-off stage", enriched_df["drop_off_stage"] == weakest),
        ("Suspicious no-transcript", enriched_df["technical_suspect_no_transcript"]),
        ("Meeting without relevance signal", enriched_df["is_meeting_without_relevance_signal"]),
        ("Proposal → Acceptance", enriched_df["drop_off_stage"] == "proposal_to_acceptance"),
        ("Offer → Relevance Signal", enriched_df["drop_off_stage"] == "offer_to_relevance_signal"),
    ]

    preview_cols = [
        "phone_id", "timestamp", "duration_seconds", "funnel_stage_reached",
        "drop_off_stage", "has_relevance_signal", "next_step_proposed",
        "next_step_accepted", "termination_reason",
    ]

    for title, condition in sections:
        subset = enriched_df[condition].head(5)
        st.markdown(f"**{title}** ({len(enriched_df[condition]):,} total)")
        if subset.empty:
            st.caption("Нет примеров.")
        else:
            st.dataframe(subset[preview_cols], use_container_width=True, hide_index=True)

with tab_preview:
    st.subheader("Processed Data Preview")
    st.dataframe(enriched_df.head(200), use_container_width=True, hide_index=True)

with tab_export:
    st.subheader("CSV Export")
    csv_data = to_csv_bytes(enriched_df)
    logger.info("CSV export prepared | rows=%s", len(enriched_df))
    st.download_button(
        label="Скачать enriched CSV",
        data=csv_data,
        file_name="botamin_calls_enriched.csv",
        mime="text/csv",
    )
