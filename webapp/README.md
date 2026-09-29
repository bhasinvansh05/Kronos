# Kronos Market Foresight Webapp

Live ticker charts + Kronos 24–48h forecasts.

## Run

```bash
cd webapp
pip install -r requirements.txt
# also need project root deps / model package
pip install -r ../requirements.txt
python run.py
```

Open http://localhost:7070

## Features

- Search any Yahoo Finance symbol (equities, ETFs, crypto like `BTC-USD`)
- OHLCV candlestick + volume chart
- Forecast horizon 24h / 48h at 15m, 30m, or 1h intervals
- Kronos-mini / small / base with sampling controls (T, top_p, samples)
- Predictions logged under `prediction_results/` for backtesting
- Tuned sampling defaults from historical backtests (`GET /api/defaults`)

## Backtest / parameter tuner

`backtest_tune.py` fetches historical OHLCV, simulates Kronos forecasts at past
cutoffs, scores MAPE / MAE / directional accuracy on close, and writes the best
sampling params for the UI.

```bash
cd webapp
# Reduced CPU run (default): AAPL + BTC-USD, 4 cutoffs, staged ~20 configs, 24h@1h
python backtest_tune.py

# Broader search
python backtest_tune.py --tickers AAPL,MSFT,SPY,BTC-USD --cutoffs 6 --max-configs 30

# Random sample of the full grid
python backtest_tune.py --mode random --n-random 16
```

Outputs:

- `backtest_results/best_params.json` — best config (committed; used by `/api/defaults`)
- `backtest_results/latest.json` — full leaderboard (gitignored)

## API

- `GET /api/health`
- `GET /api/defaults` — tuned or hardcoded forecast defaults
- `GET /api/models`
- `POST /api/model/load`
- `GET /api/tickers/search?q=`
- `GET /api/ohlcv?symbol=&interval=&lookback=`
- `POST /api/predict`
