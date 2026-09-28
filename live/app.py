"""Kronos Live — production live-market predictor UI."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from backend.market import fetch_history, fetch_quote, normalize_symbol, search_tickers
from backend.predictor_service import service

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "kronos-live", "predictor": service.info()})


@app.get("/api/models")
def models():
    return jsonify(service.info())


@app.post("/api/models/load")
def load_model():
    data = request.get_json(silent=True) or {}
    model_key = data.get("model", "kronos-mini")
    device = data.get("device", "auto")
    try:
        info = service.load(model_key=model_key, device=device)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/search")
def search():
    q = request.args.get("q", "")
    try:
        return jsonify({"query": q, "results": search_tickers(q)})
    except Exception as e:
        return jsonify({"error": str(e), "results": []}), 400


@app.get("/api/quote/<path:symbol>")
def quote(symbol: str):
    try:
        return jsonify(fetch_quote(symbol))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/history/<path:symbol>")
def history(symbol: str):
    try:
        interval = request.args.get("interval")
        df, meta = fetch_history(symbol, interval=interval)
        # Cap payload for chart bootstrap
        tail = df.tail(500)
        bars = [
            {
                "time": ts.isoformat(),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
            for ts, o, h, l, c, v in tail[
                ["timestamps", "open", "high", "low", "close", "volume"]
            ].itertuples(index=False)
        ]
        return jsonify({"meta": meta, "bars": bars})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    symbol = data.get("symbol") or data.get("ticker")
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    try:
        result = service.predict_symbol(
            symbol=symbol,
            lookback=int(data.get("lookback", 400)),
            temperature=float(data.get("temperature", 1.0)),
            top_p=float(data.get("top_p", 0.9)),
            sample_count=int(data.get("sample_count", 1)),
            horizon_hours=int(data.get("horizon_hours", 24)),
            model_key=data.get("model", "kronos-mini"),
        )
        # Attach live quote snapshot
        try:
            result["quote"] = fetch_quote(normalize_symbol(symbol))
        except Exception:
            result["quote"] = None
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def main():
    host = os.environ.get("KRONOS_LIVE_HOST", "0.0.0.0")
    port = int(os.environ.get("KRONOS_LIVE_PORT", "7070"))
    # Eager-load mini model for snappy first prediction
    if os.environ.get("KRONOS_LIVE_PRELOAD", "1") == "1":
        try:
            print("Preloading Kronos-mini...")
            service.load("kronos-mini")
            print("Model ready:", service.info())
        except Exception as e:
            print("Preload failed (will load on demand):", e)
    # Production WSGI server when available; Flask dev server as fallback.
    try:
        from waitress import serve

        print(f"Kronos Live serving on http://{host}:{port}")
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
