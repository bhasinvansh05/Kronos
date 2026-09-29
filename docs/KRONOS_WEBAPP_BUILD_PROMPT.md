# Kronos Webapp — Build Prompt (locked)

Product vision and implementation brief produced by the design subagent using
Apple Design Skill (HIG) + Kronos model constraints.

## Vision
**Kronos** is a focused market foresight instrument: search any ticker, see live
OHLCV as the primary canvas, dial forecast controls, and watch Kronos paint a
labeled predicted region for the next 24 or 48 hours.

## Design lock
- Light cool-mist instrument · Space Grotesk + IBM Plex Sans/Mono
- Teal accent `#0F766E` · chart-as-canvas · no purple AI dashboard
- Apple HIG craft (purpose, simplicity, loading/feedback, generative disclosure)

## Must-haves (shipped)
1. yfinance ticker search + OHLCV for any symbol
2. Horizon 24h/48h × interval 15m/30m/1h → derived `pred_len`
3. Kronos-mini / small / base + T / top_p / sample_count
4. Distinct forecast overlay (region + legend + “Forecast start”)
5. `/api/defaults` from historical backtest best params

## Layout
Desktop: frosted top bar (KRONOS + search + model status) · main chart canvas (~68%) · forecast rail (~32%).
Mobile: chart-first stack · sticky Predict CTA · advanced settings accordion.

Full wireframe, tokens, API contract, and acceptance criteria were used to drive
`webapp/` implementation. See PR description and `webapp/README.md`.
