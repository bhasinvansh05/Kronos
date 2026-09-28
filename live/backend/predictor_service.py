"""Kronos prediction service for Live UI."""

from __future__ import annotations

import threading
from typing import Any

import pandas as pd
import torch

from .market import fetch_history, generate_future_timestamps, normalize_symbol


class PredictorService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tokenizer = None
        self.model = None
        self.predictor = None
        self.model_key = None
        self.device = "cpu"
        self.status = "idle"
        self.error = None

    def available_models(self) -> dict[str, dict[str, Any]]:
        return {
            "kronos-mini": {
                "name": "Kronos-mini",
                "model_id": "NeoQuasar/Kronos-mini",
                "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
                "context_length": 2048,
                "params": "4.1M",
            },
            "kronos-small": {
                "name": "Kronos-small",
                "model_id": "NeoQuasar/Kronos-small",
                "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
                "context_length": 512,
                "params": "24.7M",
            },
            "kronos-base": {
                "name": "Kronos-base",
                "model_id": "NeoQuasar/Kronos-base",
                "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
                "context_length": 512,
                "params": "102.3M",
            },
        }

    def resolve_device(self, preferred: str | None = None) -> str:
        if preferred and preferred != "auto":
            return preferred
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, model_key: str = "kronos-mini", device: str | None = None) -> dict[str, Any]:
        models = self.available_models()
        if model_key not in models:
            raise ValueError(f"Unknown model: {model_key}")
        with self._lock:
            if self.predictor is not None and self.model_key == model_key and self.status == "ready":
                return self.info()
            self.status = "loading"
            self.error = None
            try:
                import sys
                from pathlib import Path

                root = Path(__file__).resolve().parents[2]
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                from model import Kronos, KronosPredictor, KronosTokenizer

                cfg = models[model_key]
                self.device = self.resolve_device(device)
                self.tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
                self.model = Kronos.from_pretrained(cfg["model_id"])
                self.predictor = KronosPredictor(
                    self.model,
                    self.tokenizer,
                    device=self.device,
                    max_context=cfg["context_length"],
                )
                self.model_key = model_key
                self.status = "ready"
            except Exception as e:
                self.status = "error"
                self.error = str(e)
                self.predictor = None
                raise
            return self.info()

    def info(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model_key,
            "device": self.device,
            "error": self.error,
            "models": self.available_models(),
        }

    def predict_symbol(
        self,
        symbol: str,
        lookback: int = 400,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        horizon_hours: int = 24,
        model_key: str = "kronos-mini",
    ) -> dict[str, Any]:
        if self.predictor is None or self.model_key != model_key or self.status != "ready":
            self.load(model_key=model_key)

        sym = normalize_symbol(symbol)
        df, meta = fetch_history(sym)
        lookback = min(max(32, lookback), len(df), self.available_models()[self.model_key]["context_length"])
        hist = df.iloc[-lookback:].copy().reset_index(drop=True)
        future_idx = generate_future_timestamps(
            hist,
            interval=meta["interval"],
            is_crypto=meta["isCrypto"],
            horizon_hours=horizon_hours,
        )
        pred_len = len(future_idx)
        if pred_len < 1:
            raise RuntimeError("Could not build a 24h prediction window for this symbol")

        x_df = hist[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = hist["timestamps"]
        y_timestamp = pd.Series(future_idx)

        with self._lock:
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
            )

        pred_df = pred_df.copy()
        pred_df.index = future_idx
        last_close = float(hist["close"].iloc[-1])
        pred_close = float(pred_df["close"].iloc[-1])

        history_payload = [
            {
                "time": ts.isoformat(),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
            for ts, o, h, l, c, v in hist[
                ["timestamps", "open", "high", "low", "close", "volume"]
            ].itertuples(index=False)
        ]
        forecast_payload = [
            {
                "time": ts.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume) if hasattr(row, "volume") else 0.0,
            }
            for ts, row in pred_df.iterrows()
        ]

        return {
            "symbol": sym,
            "meta": meta,
            "lookback": lookback,
            "predLen": pred_len,
            "horizonHours": horizon_hours,
            "model": self.model_key,
            "device": self.device,
            "params": {"temperature": temperature, "top_p": top_p, "sample_count": sample_count},
            "summary": {
                "lastClose": last_close,
                "forecastClose": pred_close,
                "changeAbs": pred_close - last_close,
                "changePct": ((pred_close / last_close) - 1.0) * 100 if last_close else None,
                "forecastHigh": float(pred_df["high"].max()),
                "forecastLow": float(pred_df["low"].min()),
                "forecastStart": future_idx[0].isoformat(),
                "forecastEnd": future_idx[-1].isoformat(),
            },
            "history": history_payload,
            "forecast": forecast_payload,
        }


service = PredictorService()
