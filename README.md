# Botamin Call Growth Dashboard

Streamlit-продукт для аналитика Botamin: превращает Excel-выгрузку звонков sales-бота в продуктовое решение — где теряются клиенты, техническая это проблема или скриптовая, какая метрика это доказывает, какие звонки проверить вручную и какой A/B-тест запустить первым.

**Не требует внешних AI API ключей.** Опциональный локальный Gemma через Ollama.

## Что делает продукт

1. Загружает Excel с колонками звонков (телефон, дата, длительность, статус, аудио, причина завершения, транскрипт).
2. Валидирует структуру, нормализует поля, парсит длительность, очищает транскрипты.
3. Классифицирует каждый звонок по этапам B2B-воронки (rule-based, без LLM).
4. Выделяет технические риски звонков без транскрипта.
5. Строит воронку, drop-off и адаптивный executive summary.
6. Генерирует Main Growth Opportunity и первый A/B-тест.
7. Отвечает на вопросы аналитика (Ask the Dashboard) без внешнего AI.
8. Опционально уточняет гипотезу через локальный Gemma (Ollama).
9. Экспортирует обогащённый CSV.

## Запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

Откройте: **http://localhost:8501**

## Данные

По умолчанию используется файл:

```text
data/calls_week_anon.xlsx
```

Можно загрузить свой Excel через sidebar — загруженный файл имеет приоритет над дефолтным.

Обязательные колонки:

```text
телефон
дата и время
длительность мин:сек
статус
запись аудио
причина завершения
история диалога юзер-бот
```

## Главная метрика

**Qualified Next Step Rate** — доля звонков, где одновременно:

- есть relevance signal (клиент показал признак релевантности);
- бот предложил next step;
- клиент принял next step.

Формула: `qualified calls / total calls`.

Также считается **Qualified Next Step Rate From Conversations** — только по звонкам с транскриптом.

### Почему не Meeting Rate

Meeting Rate может завышаться: бот может «назначить встречу» с нерелевантным клиентом. Qualified Next Step Rate измеряет осмысленную бизнес-конверсию — next step после сигнала релевантности.

## Этапы воронки

| Stage | Описание |
|-------|----------|
| 0_no_dialog | Нет диалога / пустой транскрипт |
| 1_contact_greeting | Контакт, приветствие |
| 2_consent_permission | Согласие говорить |
| 3_offer_delivered | Оффер озвучен |
| 4_relevance_signal_detected | Есть признак релевантности |
| 5_next_step_proposed | Предложен next step |
| 6_next_step_accepted | Клиент принял next step |
| 7_context_collected_or_correctly_finished | Контекст собран / корректное завершение |

## Drop-off

Каждый звонок получает `drop_off_stage` — переход, на котором он «остановился»:

- before_contact
- contact_to_consent
- consent_to_offer
- offer_to_relevance_signal
- relevance_to_next_step_proposed
- proposal_to_acceptance
- acceptance_to_context_or_finish
- completed

## Технические риски без транскрипта

Не все звонки без текста — ранние отказы:

- **Early no-dialog** — ≤10 сек без транскрипта
- **Suspicious no-transcript** — ≥30 сек без транскрипта
- **Fixed timeout** — 180 или 240 сек без транскрипта
- **Bot hangup** — termination_reason = bot_hangup без транскрипта

Длинные звонки без текста — сильный сигнал системной проблемы (логирование, таймаут, сохранение транскрипта).

## Адаптивный overview

Первый экран показывает только:

- Total Calls, Calls With Transcript, Calls Without Transcript / No Transcript Rate
- 2–3 supporting metrics под текущую Main Growth Opportunity

Если главная проблема техническая — suspicious/fixed timeout/bot hangup.  
Если offer drop-off — Offer Reach Rate, Offer → Relevance Signal Conversion.  
Если next step — Proposal/Acceptance metrics.

Остальные метрики — во вкладках и Ask the Dashboard.

## Main Growth Opportunity

Приоритет определения проблемы:

1. Suspicious no-transcript / technical loss
2. Contact → Consent
3. Consent → Offer
4. Offer → Relevance Signal
5. Relevance → Next Step Proposed
6. Proposal → Acceptance
7. Acceptance → Context
8. Meeting without relevance signal

## A/B test generation

На основе recommendation автоматически предлагается первый A/B-тест: гипотеза, variant A/B, main metric, guardrails, expected effect.

## Ask the Dashboard

Intent-based Q&A по рассчитанным данным. Поддерживает follow-up через session state: «покажи примеры», «почему», «что делать», «какой тест», «сколько таких».

## Optional Local Gemma Helper

Скопируйте `.env.example` в `.env` и при необходимости установите `ENABLE_LOCAL_AI=true`.

Gemma запускается **только по кнопке**, получает компактное summary (не весь датасет) и возвращает уточнённую гипотезу и variant B.

**Gemma не обязателен** — дашборд полностью работает без Ollama.

В production helper можно заменить на OpenAI, Gemini или Claude API.

## Ограничения rule-based анализа

- Классификация по ключевым словам — не NLP; возможны false positive/negative.
- Relevance signal намеренно консервативен (меньше ложных успехов).
- Bot/user реплики определяются по префиксам, если они есть в транскрипте.
- Gemma опционален и не участвует в массовой классификации.

## Структура проекта

```text
botamin-call-growth-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   └── calls_week_anon.xlsx
└── src/
    ├── data_processing.py
    ├── classification.py
    ├── metrics.py
    ├── recommendations.py
    ├── qa.py
    ├── local_ai.py
    └── export.py
```
