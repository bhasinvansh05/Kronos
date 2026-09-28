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

INTERVAL_HOURS = {"15m": 0.25, "30m": 0.5, "1h": 1.0}
YF_PERIOD = {"15m": "60d", "30m": "60d", "1h": "730d"}


def _bar_to_dict(ts: pd.Timestamp, row: pd.Series) -> dict[str, Any]:
    return {
        "timestamp": pd.Timestamp(ts).isoformat(),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else 0.0,
    }


def fetch_ohlcv(symbol: str, interval: str, lookback: int | None = None) -> tuple[pd.DataFrame, dict]:
    if interval not in INTERVAL_HOURS:
        raise ValueError(f"Unsupported interval: {interval}")

    ticker = yf.Ticker(symbol)
    period = YF_PERIOD[interval]
    hist = ticker.history(period=period, interval=interval, auto_adjust=True)
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
    if hist.index.tz is not None:
        # Keep timezone-aware stamps; Kronos time features use local components.
        pass

    if lookback is not None and lookback > 0:
        hist = hist.iloc[-lookback:]

    meta = {
        "currency": "USD",
        "timezone": str(hist.index.tz) if hist.index.tz else "UTC",
        "count": int(len(hist)),
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


@app.route("/api/ohlcv")
def ohlcv():
    symbol = (request.args.get("symbol") or "").strip().upper()
    interval = (request.args.get("interval") or "1h").strip()
    lookback = request.args.get("lookback")
    lookback_n = int(lookback) if lookback else 256

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    if interval not in INTERVAL_HOURS:
        return jsonify({"error": f"interval must be one of {list(INTERVAL_HOURS)}"}), 400

    try:
        # Fetch extra bars so chart has context beyond lookback window used for predict.
        fetch_n = max(lookback_n, 256)
        hist, meta = fetch_ohlcv(symbol, interval, lookback=fetch_n)
        bars = [_bar_to_dict(ts, row) for ts, row in hist.iterrows()]
        change_pct = None
        if len(bars) >= 2:
            prev, last = bars[-2]["close"], bars[-1]["close"]
            if prev:
                change_pct = ((last - prev) / prev) * 100.0
        return jsonify(
            {
                "symbol": symbol,
                "interval": interval,
                "bars": bars,
                "meta": {**meta, "change_pct": change_pct},
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
    symbol = (data.get("symbol") or "").strip().upper()
    interval = (data.get("interval") or "1h").strip()
    horizon_hours = int(data.get("horizon_hours", 24))
    lookback = int(data.get("lookback", 168))
    model_key = data.get("model_key", "kronos-small")
    temperature = float(data.get("temperature", 0.8))
    top_p = float(data.get("top_p", 0.9))
    sample_count = int(data.get("sample_count", 1))
    device = data.get("device") or loaded_device or ("cuda" if _cuda_available() else "cpu")

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    if interval not in INTERVAL_HOURS:
        return jsonify({"error": f"Unsupported interval: {interval}"}), 400
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
