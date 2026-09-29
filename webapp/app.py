"""
Kronos Market Foresight — Flask backend.

Pulls live OHLCV via yfinance and forecasts the next 24–48 hours with Kronos.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from model import Kronos, KronosPredictor, KronosTokenizer

    MODEL_AVAILABLE = True
except ImportError as exc:
    MODEL_AVAILABLE = False
    print(f"Warning: Kronos model unavailable ({exc})")

app = Flask(__name__)
CORS(app)

tokenizer = None
model = None
predictor = None
loaded_model_key = None
loaded_device = None

AVAILABLE_MODELS = {
    "kronos-mini": {
        "name": "Kronos-mini",
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "context_length": 2048,
        "params": "4.1M",
        "description": "Lightweight — fast forecasts",
    },
    "kronos-small": {
        "name": "Kronos-small",
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "24.7M",
        "description": "Balanced speed and quality (default)",
    },
    "kronos-base": {
        "name": "Kronos-base",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "102.3M",
        "description": "Higher quality, slower",
    },
}

# Hours per bar — used for forecast pred_len. Display-only intervals included.
INTERVAL_HOURS = {
    "1m": 1 / 60,
    "5m": 5 / 60,
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1.0,
    "1d": 24.0,
    "1wk": 24.0 * 7,
}

# Default fetch window when only interval is supplied (no display range).
YF_PERIOD = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "1d": "5y",
    "1wk": "10y",
}

# Chart display ranges (Apple Stocks–style). Forecast still uses 15m/30m/1h.
DISPLAY_RANGES = {
    "1D": {"label": "1D", "period": "1d", "interval": "5m", "description": "1 day · 5m bars"},
    "5D": {"label": "5D", "period": "5d", "interval": "15m", "description": "5 days · 15m bars"},
    "1W": {"label": "1W", "period": "5d", "interval": "30m", "description": "1 week · 30m bars"},
    "1M": {"label": "1M", "period": "1mo", "interval": "1h", "description": "1 month · 1h bars"},
    "3M": {"label": "3M", "period": "3mo", "interval": "1d", "description": "3 months · daily"},
    "6M": {"label": "6M", "period": "6mo", "interval": "1d", "description": "6 months · daily"},
    "1Y": {"label": "1Y", "period": "1y", "interval": "1d", "description": "1 year · daily"},
    "5Y": {"label": "5Y", "period": "5y", "interval": "1wk", "description": "5 years · weekly"},
}

FORECAST_INTERVALS = ("15m", "30m", "1h")

# Hardcoded fallbacks when backtest_results/best_params.json is absent.
HARDCODED_DEFAULTS = {
    "model_key": "kronos-mini",
    "device": "cpu",
    "interval": "1h",
    "horizon_hours": 24,
    "lookback": 168,
    "temperature": 0.8,
    "top_p": 0.9,
    "sample_count": 1,
    "display_range": "1M",
}

BEST_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results", "best_params.json")


def _bar_to_dict(ts: pd.Timestamp, row: pd.Series) -> dict[str, Any]:
    return {
        "timestamp": pd.Timestamp(ts).isoformat(),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else 0.0,
    }


def fetch_ohlcv(
    symbol: str,
    interval: str,
    lookback: int | None = None,
    period: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    if interval not in INTERVAL_HOURS:
        raise ValueError(f"Unsupported interval: {interval}")

    ticker = yf.Ticker(symbol)
    fetch_period = period or YF_PERIOD.get(interval, "1y")
    hist = ticker.history(period=fetch_period, interval=interval, auto_adjust=True)
    if hist is None or hist.empty:
        raise LookupError(f"No market data for {symbol}")

    hist = hist.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    hist = hist[["open", "high", "low", "close", "volume"]].dropna()
    hist.index = pd.to_datetime(hist.index)

    if lookback is not None and lookback > 0:
        hist = hist.iloc[-lookback:]

    meta = {
        "currency": "USD",
        "timezone": str(hist.index.tz) if hist.index.tz else "UTC",
        "count": int(len(hist)),
        "period": fetch_period,
        "interval": interval,
    }
    try:
        info = ticker.fast_info
        currency = getattr(info, "currency", None)
        if currency:
            meta["currency"] = currency
        name = getattr(info, "shortName", None) or getattr(info, "longName", None)
        if name:
            meta["name"] = name
        last = getattr(info, "last_price", None)
        if last is not None:
            meta["last_price"] = float(last)
        prev = getattr(info, "previous_close", None)
        if prev is not None:
            meta["previous_close"] = float(prev)
    except Exception:
        pass

    return hist, meta


def build_future_timestamps(last_ts: pd.Timestamp, interval: str, pred_len: int) -> pd.Series:
    hours = INTERVAL_HOURS[interval]
    delta = pd.Timedelta(hours=hours)
    stamps = [last_ts + delta * (i + 1) for i in range(pred_len)]
    return pd.Series(stamps, name="timestamps")


def save_prediction_log(payload: dict) -> str | None:
    try:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_results")
        os.makedirs(results_dir, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(results_dir, f"prediction_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return path
    except Exception as exc:
        print(f"Failed to save prediction log: {exc}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


def load_best_params() -> dict[str, Any] | None:
    """Load tuned defaults from backtest_results/best_params.json if present."""
    try:
        if not os.path.isfile(BEST_PARAMS_PATH):
            return None
        with open(BEST_PARAMS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        print(f"Failed to read best_params.json: {exc}")
        return None


@app.route("/api/health")
def health():
    yf_ok = True
    try:
        import yfinance  # noqa: F401
    except Exception:
        yf_ok = False
    return jsonify(
        {
            "ok": True,
            "model_lib": MODEL_AVAILABLE,
            "yfinance": yf_ok,
            "version": "1.0.0",
            "model_loaded": predictor is not None,
            "model_key": loaded_model_key,
        }
    )


@app.route("/api/defaults")
def defaults():
    """Return sampling/forecast defaults (tuned best_params.json or hardcoded)."""
    tuned = load_best_params()
    out = dict(HARDCODED_DEFAULTS)
    source = "hardcoded"
    if tuned:
        source = "best_params"
        for src_key, dst_key in (
            ("model_key", "model_key"),
            ("device", "device"),
            ("interval", "interval"),
            ("horizon_hours", "horizon_hours"),
            ("lookback", "lookback"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("sample_count", "sample_count"),
        ):
            if src_key in tuned and tuned[src_key] is not None:
                out[dst_key] = tuned[src_key]
        if "metrics" in tuned:
            out["metrics"] = tuned["metrics"]
        if "generated_at" in tuned:
            out["generated_at"] = tuned["generated_at"]
    out["source"] = source
    return jsonify(out)


@app.route("/api/models")
def models():
    return jsonify(
        {
            "models": AVAILABLE_MODELS,
            "model_available": MODEL_AVAILABLE,
            "loaded": loaded_model_key,
            "device": loaded_device,
        }
    )


@app.route("/api/model/status")
def model_status():
    ctx = AVAILABLE_MODELS.get(loaded_model_key or "", {}).get("context_length")
    return jsonify(
        {
            "available": MODEL_AVAILABLE,
            "loaded": predictor is not None,
            "model_key": loaded_model_key,
            "device": loaded_device,
            "context_length": ctx,
            "message": (
                f"{AVAILABLE_MODELS[loaded_model_key]['name']} ready"
                if predictor is not None and loaded_model_key
                else "No model loaded"
            ),
        }
    )


@app.route("/api/model/load", methods=["POST"])
def load_model():
    data = request.get_json() or {}
    model_key = data.get("model_key", "kronos-small")
    device = data.get("device") or ("cuda" if _cuda_available() else "cpu")

    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    ok, err = ensure_model_loaded(model_key, device)
    if not ok:
        code = 502 if err and err.get("code") == "MODEL_DOWNLOAD_FAILED" else 500
        return jsonify(err), code

    cfg = AVAILABLE_MODELS[model_key]
    return jsonify(
        {
            "success": True,
            "model_info": {
                "key": model_key,
                "name": cfg["name"],
                "params": cfg["params"],
                "context_length": cfg["context_length"],
                "description": cfg["description"],
                "device": device,
            },
        }
    )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def ensure_model_loaded(model_key: str, device: str) -> tuple[bool, dict | None]:
    """Load model if needed. Returns (ok, error_payload)."""
    global tokenizer, model, predictor, loaded_model_key, loaded_device

    if predictor is not None and loaded_model_key == model_key and loaded_device == device:
        return True, None

    if not MODEL_AVAILABLE:
        return False, {"error": "Kronos model library not installed", "code": "MODEL_UNAVAILABLE"}

    cfg = AVAILABLE_MODELS[model_key]
    try:
        print(f"Loading {cfg['name']} on {device}…")
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        model = Kronos.from_pretrained(cfg["model_id"])
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg["context_length"])
        loaded_model_key = model_key
        loaded_device = device
        return True, None
    except Exception as exc:
        traceback.print_exc()
        tokenizer = model = predictor = None
        loaded_model_key = loaded_device = None
        return False, {"error": f"Model loading failed: {exc}", "code": "MODEL_DOWNLOAD_FAILED"}


@app.route("/api/tickers/search")
def search_tickers():
    q = (request.args.get("q") or "").strip()
    limit = min(int(request.args.get("limit", 8)), 15)
    if not q:
        return jsonify({"results": []})

    results = []
    try:
        search = yf.Search(q, max_results=limit)
        quotes = getattr(search, "quotes", None) or []
        for item in quotes[:limit]:
            symbol = item.get("symbol")
            if not symbol:
                continue
            qtype = (item.get("quoteType") or item.get("typeDisp") or "equity").lower()
            if qtype == "equity":
                qtype = "equity"
            elif "etf" in qtype:
                qtype = "etf"
            elif "crypto" in qtype:
                qtype = "crypto"
            results.append(
                {
                    "symbol": symbol,
                    "name": item.get("longname") or item.get("shortname") or symbol,
                    "type": qtype,
                    "exchange": item.get("exchDisp") or item.get("exchange") or "",
                }
            )
    except Exception as exc:
        print(f"Search error: {exc}")

    # Direct symbol fallback
    if not results:
        try:
            hist, meta = fetch_ohlcv(q.upper(), "1h", lookback=5)
            if len(hist) > 0:
                results.append(
                    {
                        "symbol": q.upper(),
                        "name": meta.get("name", q.upper()),
                        "type": "equity",
                        "exchange": "",
                    }
                )
        except Exception:
            pass

    return jsonify({"results": results})


@app.route("/api/tickers/validate")
def validate_ticker():
    symbol = (request.args.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"valid": False, "error": "Symbol required"}), 400
    try:
        hist, meta = fetch_ohlcv(symbol, "1h", lookback=5)
        if hist.empty:
            return jsonify({"valid": False, "symbol": symbol})
        return jsonify(
            {
                "valid": True,
                "symbol": symbol,
                "name": meta.get("name", symbol),
                "currency": meta.get("currency", "USD"),
                "type": "equity",
                "last_price": meta.get("last_price"),
            }
        )
    except Exception as exc:
        return jsonify({"valid": False, "symbol": symbol, "error": str(exc)})


@app.route("/api/ranges")
def ranges():
    """List chart display ranges (1D … 5Y)."""
    return jsonify(
        {
            "ranges": [
                {"key": key, **cfg}
                for key, cfg in DISPLAY_RANGES.items()
            ],
            "forecast_intervals": list(FORECAST_INTERVALS),
            "default": HARDCODED_DEFAULTS.get("display_range", "1M"),
        }
    )


@app.route("/api/ohlcv")
def ohlcv():
    symbol = (request.args.get("symbol") or "").strip().upper()
    display_range = (request.args.get("range") or "").strip().upper()
    interval = (request.args.get("interval") or "").strip()
    lookback = request.args.get("lookback")
    period = request.args.get("period")

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    range_key = None
    if display_range:
        if display_range not in DISPLAY_RANGES:
            return jsonify({"error": f"range must be one of {list(DISPLAY_RANGES)}"}), 400
        cfg = DISPLAY_RANGES[display_range]
        interval = cfg["interval"]
        period = cfg["period"]
        range_key = display_range
        lookback_n = None
    else:
        interval = interval or "1h"
        if interval not in INTERVAL_HOURS:
            return jsonify({"error": f"interval must be one of {list(INTERVAL_HOURS)}"}), 400
        lookback_n = int(lookback) if lookback else 256

    try:
        if lookback_n is not None:
            fetch_n = max(lookback_n, 64)
            hist, meta = fetch_ohlcv(symbol, interval, lookback=fetch_n, period=period)
        else:
            hist, meta = fetch_ohlcv(symbol, interval, lookback=None, period=period)

        bars = [_bar_to_dict(ts, row) for ts, row in hist.iterrows()]
        change_pct = None
        if len(bars) >= 2:
            first, last = bars[0]["close"], bars[-1]["close"]
            if first:
                change_pct = ((last - first) / first) * 100.0
        return jsonify(
            {
                "symbol": symbol,
                "interval": interval,
                "range": range_key,
                "period": period or meta.get("period"),
                "bars": bars,
                "meta": {
                    **meta,
                    "change_pct": change_pct,
                    "range": range_key,
                    "range_label": DISPLAY_RANGES[range_key]["label"] if range_key else None,
                },
            }
        )
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Upstream market data failed: {exc}"}), 502


@app.route("/api/predict", methods=["POST"])
def predict():
    global predictor, loaded_model_key, loaded_device

    data = request.get_json() or {}
    tuned = load_best_params() or {}
    defaults = {**HARDCODED_DEFAULTS, **{k: tuned[k] for k in HARDCODED_DEFAULTS if k in tuned}}

    symbol = (data.get("symbol") or "").strip().upper()
    interval = (data.get("interval") or defaults["interval"]).strip()
    horizon_hours = int(data.get("horizon_hours", defaults["horizon_hours"]))
    lookback = int(data.get("lookback", defaults["lookback"]))
    model_key = data.get("model_key", defaults["model_key"])
    temperature = float(data.get("temperature", defaults["temperature"]))
    top_p = float(data.get("top_p", defaults["top_p"]))
    sample_count = int(data.get("sample_count", defaults["sample_count"]))
    device = data.get("device") or loaded_device or defaults.get("device") or (
        "cuda" if _cuda_available() else "cpu"
    )

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    if interval not in FORECAST_INTERVALS:
        return jsonify(
            {
                "error": f"Forecast interval must be one of {list(FORECAST_INTERVALS)} "
                f"(display ranges like 1Y use daily/weekly for charting only)"
            }
        ), 400
    if horizon_hours not in (24, 48):
        return jsonify({"error": "horizon_hours must be 24 or 48"}), 400

    hours_per_bar = INTERVAL_HOURS[interval]
    pred_len = int(round(horizon_hours / hours_per_bar))
    if pred_len < 1:
        return jsonify({"error": "pred_len computed to 0"}), 400

    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    ctx = AVAILABLE_MODELS[model_key]["context_length"]
    if lookback > ctx:
        return jsonify({"error": f"lookback {lookback} exceeds model context {ctx}"}), 400
    if lookback < 32:
        return jsonify({"error": "lookback must be at least 32"}), 400

    ok, err = ensure_model_loaded(model_key, device)
    if not ok:
        code = 502 if err.get("code") == "MODEL_DOWNLOAD_FAILED" else 500
        return jsonify(err), code

    try:
        need = lookback + 8
        hist, meta = fetch_ohlcv(symbol, interval, lookback=max(need, lookback))
        if len(hist) < lookback:
            return (
                jsonify(
                    {
                        "error": f"Insufficient bars for lookback {lookback}; only {len(hist)} available. Try a shorter lookback or coarser interval."
                    }
                ),
                400,
            )

        hist = hist.iloc[-lookback:]
        required_cols = ["open", "high", "low", "close", "volume"]
        x_df = hist[required_cols].copy()
        x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)

        x_timestamp = pd.Series(hist.index.to_list(), name="timestamps")
        last_ts = x_timestamp.iloc[-1]
        y_timestamp = build_future_timestamps(pd.Timestamp(last_ts), interval, pred_len)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )

        history_bars = [_bar_to_dict(ts, row) for ts, row in hist.iterrows()]
        prediction_bars = []
        for ts, row in pred_df.iterrows():
            prediction_bars.append(
                {
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]) if "volume" in row else 0.0,
                }
            )

        payload = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "horizon_hours": horizon_hours,
            "pred_len": pred_len,
            "lookback": lookback,
            "model_key": model_key,
            "params": {"T": temperature, "top_p": top_p, "sample_count": sample_count},
            "meta": meta,
            "history": history_bars,
            "prediction": prediction_bars,
            "disclaimer": "Model-generated forecast. Not financial advice.",
        }
        save_prediction_log(payload)
        return jsonify(payload)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7070))
    app.run(host="0.0.0.0", port=port, debug=False)
