# Kronos Market Foresight

**Owner:** [Vansh Bhasin](https://github.com/bhasinvansh05) · York University  
**Repo:** [bhasinvansh05/Kronos](https://github.com/bhasinvansh05/Kronos)  
**Live:** [https://kronos.vanshbhasin.dev](https://kronos.vanshbhasin.dev)

Live market charts for any ticker, plus **24–48 hour** forecasts powered by the
open-source Kronos foundation model. Built as a focused foresight instrument —
not a purple AI dashboard.

> Model-generated forecasts. Not financial advice.

---

## What’s in this repo

| Path | Purpose |
|------|---------|
| **`webapp/`** | **Primary product** — Flask UI + yfinance + Kronos predict API |
| `model/` | Kronos model / tokenizer / predictor (upstream architecture) |
| `webapp/backtest_tune.py` | Walk-forward parameter search → tuned defaults |
| `examples/`, `finetune/`, `finetune_csv/` | Upstream-style scripts kept for research |
| `webui/` | Legacy CSV demo UI (superseded by `webapp/`) |
| `docs/` | Design prompt, Heroku phone deploy guide, Apple HIG notes |

---

## Quick start

```bash
# Python 3.12 recommended (required on Heroku)
pip install -r requirements.txt
pip install -r webapp/requirements.txt

cd webapp
python run.py
# → http://localhost:7070
```

### Live demo

Production site: **[kronos.vanshbhasin.dev](https://kronos.vanshbhasin.dev)**  
(Heroku + custom domain on Cloudflare DNS)

Phone-friendly deploy guide (GitHub Student Pack → Heroku):
[`docs/MOBILE_DEPLOY_HEROKU.md`](./docs/MOBILE_DEPLOY_HEROKU.md).

Deploy **`master`** (it includes the webapp, Python 3.12 pin, Dockerfile / Procfile).
Enable Heroku **Automatic Deploys** on `master` after connecting GitHub.

### Try it

1. Search a ticker (`AAPL`, `MSFT`, `SPY`, `BTC-USD`, …)
2. Pick a display range: **1D · 5D · 1W · 1M · 3M · 6M · 1Y · 5Y**
3. Set forecast horizon (24h / 48h) and bar size (15m / 30m / 1h)
4. Toggle **dark / light** theme if you like
5. Click **Predict Next Hours** (Expand the chart for a fuller view)

---

## Features

- **Any Yahoo Finance symbol** via `yfinance` (equities, ETFs, crypto)
- **Display ranges** from one day to five years with sensible bar sizes
- **Continuous packed candles** + close line + volume (no calendar gaps)
- **Kronos-mini / small / base** with T, top_p, sample_count controls
- **Tuned defaults** from historical backtests (`GET /api/defaults`)
- **Dark / light theme** (persisted; respects system preference)
- **Mobile-ready chart UX** — denser viewport, sticky Predict CTA, Reset / Expand
- Cool-mist UI (Space Grotesk + IBM Plex, teal accent)
- Heroku-ready: `.python-version` 3.12, CPU PyTorch wheels, `Procfile` / `Dockerfile`

### Backtest / tune

```bash
cd webapp
python backtest_tune.py
# broader:
python backtest_tune.py --tickers AAPL,MSFT,SPY,BTC-USD --cutoffs 6 --max-configs 30
```

Writes `webapp/backtest_results/best_params.json` (used by the UI).

---

## API (webapp)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness + model lib status |
| GET | `/api/defaults` | Tuned or hardcoded forecast defaults |
| GET | `/api/ranges` | Display range catalog (1D…5Y) |
| GET | `/api/models` | Available Kronos sizes |
| POST | `/api/model/load` | Load tokenizer + model |
| GET | `/api/tickers/search?q=` | Ticker search |
| GET | `/api/ohlcv?symbol=&range=1M` | OHLCV for a display range |
| POST | `/api/predict` | 24–48h forecast |

---

## Models

Pretrained checkpoints are loaded from Hugging Face (upstream NeoQuasar):

| Key | Params | Context |
|-----|--------|---------|
| `kronos-mini` | 4.1M | 2048 |
| `kronos-small` | 24.7M | 512 |
| `kronos-base` | 102.3M | 512 |

---

## Project ownership

This repository is maintained by **Vansh Bhasin**. See [`NOTICE`](./NOTICE) and
[`LICENSE`](./LICENSE).

Live product: [kronos.vanshbhasin.dev](https://kronos.vanshbhasin.dev)

The Kronos model architecture and published weights are open-source work by
ShiYu / NeoQuasar (MIT). Application code, UI, data plumbing, and tooling in
this repo are part of Vansh’s Kronos Market Foresight product.

---

## Disclaimer

Forecasts are probabilistic model outputs and can be wrong. Nothing here is
investment, trading, or financial advice.
