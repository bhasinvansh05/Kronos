# Kronos Live

Production UI for **live ticker search → market bars → 24h Kronos forecast**.

## Run

```bash
cd live
pip install -r requirements.txt
python run.py
```

Open [http://localhost:7070](http://localhost:7070).

Optional env:

- `KRONOS_LIVE_PORT` (default `7070`)
- `KRONOS_LIVE_HOST` (default `0.0.0.0`)
- `KRONOS_LIVE_PRELOAD=0` to skip eager model load

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health + model status |
| GET | `/api/search?q=` | Ticker search |
| GET | `/api/quote/<symbol>` | Live quote |
| GET | `/api/history/<symbol>` | OHLCV history |
| POST | `/api/predict` | Live history + 24h forecast |
| POST | `/api/models/load` | Load/switch Kronos model |

Predict body example:

```json
{
  "symbol": "VFV.TO",
  "model": "kronos-mini",
  "temperature": 1.0,
  "top_p": 0.9,
  "lookback": 400,
  "horizon_hours": 24
}
```

## Notes

- Equities/ETFs forecast the **next regular session** (~24h horizon).
- Crypto (`*-USD`) forecasts a continuous **24h** path.
- Charts use TradingView Lightweight Charts (candles + volume).
