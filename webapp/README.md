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

## API

- `GET /api/health`
- `GET /api/models`
- `POST /api/model/load`
- `GET /api/tickers/search?q=`
- `GET /api/ohlcv?symbol=&interval=&lookback=`
- `POST /api/predict`
