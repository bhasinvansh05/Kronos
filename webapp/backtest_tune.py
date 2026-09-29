#!/usr/bin/env python3
"""
Historical backtesting + sampling-parameter search for the Kronos webapp.

Fetches OHLCV via yfinance, simulates predictions at past cutoffs, scores
against realized prices, and writes the best config for the webapp defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
AVAILABLE_MODELS = {
    "kronos-mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "context_length": 2048,
    },
    "kronos-small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
    },
    "kronos-base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
    },
}

INTERVAL_HOURS = {"15m": 0.25, "30m": 0.5, "1h": 1.0}
YF_PERIOD = {"15m": "60d", "30m": "60d", "1h": "730d"}

_predictor: KronosPredictor | None = None
_loaded_key: str | None = None
_loaded_device: str | None = None


@dataclass(frozen=True)
class ParamConfig:
    T: float
    top_p: float
    sample_count: int
    lookback: int
    horizon_hours: int = 24

    def key(self) -> str:
        return (
            f"T={self.T}|top_p={self.top_p}|samples={self.sample_count}"
            f"|lb={self.lookback}|h={self.horizon_hours}"
        )


def ensure_predictor(model_key: str, device: str) -> KronosPredictor:
    global _predictor, _loaded_key, _loaded_device
    if _predictor is not None and _loaded_key == model_key and _loaded_device == device:
        return _predictor

    cfg = AVAILABLE_MODELS[model_key]
    print(f"Loading {model_key} on {device}…")
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    model = Kronos.from_pretrained(cfg["model_id"])
    _predictor = KronosPredictor(
        model, tokenizer, device=device, max_context=cfg["context_length"]
    )
    _loaded_key = model_key
    _loaded_device = device
    print(f"Model ready (context={cfg['context_length']})")
    return _predictor


def fetch_ohlcv(symbol: str, interval: str = "1h") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=YF_PERIOD[interval], interval=interval, auto_adjust=True)
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
    return hist


def build_future_timestamps(last_ts: pd.Timestamp, interval: str, pred_len: int) -> pd.Series:
    hours = INTERVAL_HOURS[interval]
    delta = pd.Timedelta(hours=hours)
    stamps = [last_ts + delta * (i + 1) for i in range(pred_len)]
    return pd.Series(stamps, name="timestamps")


def cutoff_indices(n_bars: int, lookback_max: int, pred_len: int, n_cutoffs: int) -> list[int]:
    """End-exclusive indices: lookback is hist[cutoff-lookback:cutoff], actual is hist[cutoff:cutoff+pred_len]."""
    first = lookback_max
    last = n_bars - pred_len
    if last <= first:
        raise ValueError(
            f"Not enough bars ({n_bars}) for lookback={lookback_max} and pred_len={pred_len}"
        )
    if n_cutoffs == 1:
        return [last]
    positions = np.linspace(first, last, n_cutoffs)
    return sorted({int(round(p)) for p in positions})


def mape_close(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)


def mae_close(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def directional_accuracy(actual: np.ndarray, pred: np.ndarray, prev_close: float) -> float:
    true_path = np.concatenate([[prev_close], actual])
    pred_path = np.concatenate([[prev_close], pred])
    true_chg = np.diff(true_path)
    pred_chg = np.diff(pred_path)
    return float(np.mean(np.sign(true_chg) == np.sign(pred_chg)))


def prepare_windows(
    series: dict[str, pd.DataFrame],
    lookback: int,
    pred_len: int,
    interval: str,
    n_cutoffs: int,
    lookback_max: int,
) -> list[dict[str, Any]]:
    """Build aligned windows for batch prediction (same lookback / pred_len)."""
    windows: list[dict[str, Any]] = []
    for symbol, hist in series.items():
        try:
            cuts = cutoff_indices(len(hist), lookback_max, pred_len, n_cutoffs)
        except ValueError as exc:
            print(f"  skip {symbol}: {exc}")
            continue
        for cutoff in cuts:
            start = cutoff - lookback
            end = cutoff + pred_len
            if start < 0 or end > len(hist):
                continue
            window = hist.iloc[start:cutoff]
            actual = hist.iloc[cutoff:end]
            if len(window) < lookback or len(actual) < pred_len:
                continue
            x_df = window[["open", "high", "low", "close", "volume"]].copy()
            x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
            x_timestamp = pd.Series(window.index.to_list(), name="timestamps")
            last_ts = pd.Timestamp(x_timestamp.iloc[-1])
            y_timestamp = build_future_timestamps(last_ts, interval, pred_len)
            windows.append(
                {
                    "symbol": symbol,
                    "cutoff_index": cutoff,
                    "cutoff_time": str(hist.index[cutoff - 1]),
                    "x_df": x_df,
                    "x_timestamp": x_timestamp,
                    "y_timestamp": y_timestamp,
                    "actual_close": actual["close"].to_numpy(dtype=float),
                    "prev_close": float(window["close"].iloc[-1]),
                }
            )
    return windows


def evaluate_config(
    predictor: KronosPredictor,
    series: dict[str, pd.DataFrame],
    cfg: ParamConfig,
    interval: str,
    n_cutoffs: int,
    lookback_max: int,
) -> dict[str, Any]:
    pred_len = int(round(cfg.horizon_hours / INTERVAL_HOURS[interval]))
    windows = prepare_windows(series, cfg.lookback, pred_len, interval, n_cutoffs, lookback_max)
    if not windows:
        return {
            "config": asdict(cfg),
            "mape": None,
            "mae": None,
            "directional_accuracy": None,
            "n_evals": 0,
            "per_ticker": {},
            "error": "no valid windows",
        }

    # Batch all ticker×cutoff windows (same lookback / pred_len).
    pred_dfs = predictor.predict_batch(
        df_list=[w["x_df"] for w in windows],
        x_timestamp_list=[w["x_timestamp"] for w in windows],
        y_timestamp_list=[w["y_timestamp"] for w in windows],
        pred_len=pred_len,
        T=cfg.T,
        top_p=cfg.top_p,
        sample_count=cfg.sample_count,
        verbose=False,
    )

    per_ticker: dict[str, list[dict[str, Any]]] = {}
    mapes, maes, dirs = [], [], []

    for w, pred_df in zip(windows, pred_dfs):
        actual_close = w["actual_close"]
        pred_close = pred_df["close"].to_numpy(dtype=float)[:pred_len]
        row = {
            "cutoff_index": w["cutoff_index"],
            "cutoff_time": w["cutoff_time"],
            "mape": mape_close(actual_close, pred_close),
            "mae": mae_close(actual_close, pred_close),
            "directional_accuracy": directional_accuracy(
                actual_close, pred_close, w["prev_close"]
            ),
        }
        per_ticker.setdefault(w["symbol"], []).append(row)
        mapes.append(row["mape"])
        maes.append(row["mae"])
        dirs.append(row["directional_accuracy"])

    ticker_summary = {}
    for sym, rows in per_ticker.items():
        ticker_summary[sym] = {
            "mape": float(np.mean([r["mape"] for r in rows])),
            "mae": float(np.mean([r["mae"] for r in rows])),
            "directional_accuracy": float(np.mean([r["directional_accuracy"] for r in rows])),
            "n_windows": len(rows),
        }

    return {
        "config": asdict(cfg),
        "mape": float(np.mean(mapes)) if mapes else None,
        "mae": float(np.mean(maes)) if maes else None,
        "directional_accuracy": float(np.mean(dirs)) if dirs else None,
        "n_evals": len(mapes),
        "per_ticker": ticker_summary,
        "detail": per_ticker,
    }


def score_row(row: dict[str, Any]) -> tuple:
    """Lower is better: primary MAPE, then -dir_acc, then MAE."""
    mape = row.get("mape")
    if mape is None or (isinstance(mape, float) and np.isnan(mape)):
        return (float("inf"), 0.0, float("inf"))
    dir_acc = row.get("directional_accuracy") or 0.0
    mae = row.get("mae") if row.get("mae") is not None else float("inf")
    return (mape, -dir_acc, mae)


def stage_a_configs(horizons: list[int], max_lookback: int) -> list[ParamConfig]:
    """Coarse T × top_p with fixed lookback=168 and sample_count=1."""
    lookback = min(168, max_lookback)
    temps = [0.4, 0.6, 0.8, 1.0]
    top_ps = [0.85, 0.9, 0.95]
    out: list[ParamConfig] = []
    seen: set[str] = set()
    for t, p, h in product(temps, top_ps, horizons):
        c = ParamConfig(T=t, top_p=p, sample_count=1, lookback=lookback, horizon_hours=h)
        if c.key() not in seen:
            seen.add(c.key())
            out.append(c)
    return out


def random_configs(
    horizons: list[int], max_lookback: int, n: int, seed: int
) -> list[ParamConfig]:
    temps = [0.4, 0.6, 0.8, 1.0, 1.2]
    top_ps = [0.7, 0.85, 0.9, 0.95]
    samples = [1, 2, 3]
    lookbacks = [lb for lb in (96, 128, 168, 256) if lb <= max_lookback]
    pool = [
        ParamConfig(T=t, top_p=p, sample_count=s, lookback=lb, horizon_hours=h)
        for t, p, s, lb, h in product(temps, top_ps, samples, lookbacks, horizons)
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def refine_configs(best: ParamConfig, max_lookback: int, horizons: list[int]) -> list[ParamConfig]:
    lookbacks = [lb for lb in (96, 128, 168, 256) if lb <= max_lookback]
    samples = [1, 2]
    refined: list[ParamConfig] = []
    for lb, s, h in product(lookbacks, samples, horizons):
        refined.append(
            ParamConfig(T=best.T, top_p=best.top_p, sample_count=s, lookback=lb, horizon_hours=h)
        )
    for t_delta in (-0.2, 0.2):
        t = round(best.T + t_delta, 2)
        if 0.4 <= t <= 1.2:
            refined.append(
                ParamConfig(
                    T=t,
                    top_p=best.top_p,
                    sample_count=1,
                    lookback=best.lookback,
                    horizon_hours=best.horizon_hours,
                )
            )
    seen = {best.key()}
    out = []
    for c in refined:
        if c.key() not in seen:
            seen.add(c.key())
            out.append(c)
    return out


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kronos webapp backtest + parameter tuner")
    parser.add_argument("--tickers", default="AAPL,BTC-USD", help="Comma-separated Yahoo symbols")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_HOURS))
    parser.add_argument("--cutoffs", type=int, default=4, help="Historical cutoff windows")
    parser.add_argument("--horizons", default="24", help="Comma-separated horizons in hours")
    parser.add_argument("--model", default="kronos-mini", choices=list(AVAILABLE_MODELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("staged", "random"), default="staged")
    parser.add_argument("--n-random", type=int, default=16)
    parser.add_argument("--max-configs", type=int, default=16, help="Hard cap on evaluated configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-refine", action="store_true")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    model_cfg = AVAILABLE_MODELS[args.model]
    max_lookback = model_cfg["context_length"]
    global_max_lb = min(256, max_lookback)

    print("=" * 60)
    print("Kronos backtest tuner")
    print(f"  tickers={tickers}  interval={args.interval}  cutoffs={args.cutoffs}")
    print(f"  horizons={horizons}  model={args.model}  device={args.device}  mode={args.mode}")
    print("=" * 60)

    series: dict[str, pd.DataFrame] = {}
    for sym in tickers:
        print(f"Fetching {sym} ({args.interval})…")
        df = fetch_ohlcv(sym, args.interval)
        series[sym] = df
        print(f"  {sym}: {len(df)} bars  [{df.index.min()} → {df.index.max()}]")

    predictor = ensure_predictor(args.model, args.device)

    if args.mode == "random":
        configs = random_configs(horizons, max_lookback, args.n_random, args.seed)
    else:
        configs = stage_a_configs(horizons, max_lookback)

    leaderboard: list[dict[str, Any]] = []
    evaluated: set[str] = set()

    def run_batch(batch: list[ParamConfig], label: str) -> None:
        for cfg in batch:
            if cfg.key() in evaluated:
                continue
            if len(evaluated) >= args.max_configs:
                print(f"Reached max-configs={args.max_configs}, stopping.")
                return
            evaluated.add(cfg.key())
            print(f"[{label} {len(evaluated)}/{args.max_configs}] {cfg.key()}")
            t0 = time.time()
            try:
                row = evaluate_config(
                    predictor, series, cfg, args.interval, args.cutoffs, global_max_lb
                )
                row["elapsed_sec"] = round(time.time() - t0, 2)
                leaderboard.append(row)
                if row["mape"] is not None:
                    print(
                        f"  → MAPE={row['mape']:.4f}%  MAE={row['mae']:.4f}  "
                        f"dir={row['directional_accuracy']:.3f}  "
                        f"({row['elapsed_sec']}s, n={row['n_evals']})"
                    )
                else:
                    print(f"  → no valid evals ({row['elapsed_sec']}s)")
            except Exception:
                traceback.print_exc()
                leaderboard.append(
                    {
                        "config": asdict(cfg),
                        "mape": None,
                        "mae": None,
                        "directional_accuracy": None,
                        "n_evals": 0,
                        "error": "evaluation failed",
                    }
                )

    run_batch(configs, "search")

    if args.mode == "staged" and not args.skip_refine and leaderboard:
        ranked = sorted(leaderboard, key=score_row)
        best_so_far = ranked[0]
        if best_so_far.get("mape") is not None:
            best_cfg = ParamConfig(**best_so_far["config"])
            refine = refine_configs(best_cfg, max_lookback, horizons)
            remaining = args.max_configs - len(evaluated)
            refine = refine[: max(0, remaining)]
            if refine:
                print(f"\n--- Refine around {best_cfg.key()} ({len(refine)} configs) ---")
                run_batch(refine, "refine")

    leaderboard_sorted = sorted(leaderboard, key=score_row)
    slim_board = []
    for row in leaderboard_sorted:
        slim = {
            "config": row.get("config"),
            "mape": row.get("mape"),
            "mae": row.get("mae"),
            "directional_accuracy": row.get("directional_accuracy"),
            "n_evals": row.get("n_evals"),
            "elapsed_sec": row.get("elapsed_sec"),
            "per_ticker": row.get("per_ticker"),
        }
        if row.get("error"):
            slim["error"] = row["error"]
        slim_board.append(slim)

    best = slim_board[0] if slim_board else None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model,
        "device": args.device,
        "interval": args.interval,
        "tickers": tickers,
        "n_cutoffs": args.cutoffs,
        "horizons": horizons,
        "mode": args.mode,
        "n_configs_evaluated": len(evaluated),
        "leaderboard": slim_board,
        "best": best,
    }

    latest_path = os.path.join(RESULTS_DIR, "latest.json")
    best_path = os.path.join(RESULTS_DIR, "best_params.json")
    write_json(latest_path, payload)

    if best and best.get("mape") is not None:
        cfg = best["config"]
        best_payload = {
            "generated_at": payload["generated_at"],
            "model_key": args.model,
            "device": args.device,
            "interval": args.interval,
            "horizon_hours": cfg["horizon_hours"],
            "lookback": cfg["lookback"],
            "temperature": cfg["T"],
            "top_p": cfg["top_p"],
            "sample_count": cfg["sample_count"],
            "metrics": {
                "mape": best["mape"],
                "mae": best["mae"],
                "directional_accuracy": best["directional_accuracy"],
                "n_evals": best["n_evals"],
            },
            "tickers": tickers,
            "source": "backtest_tune.py",
        }
        write_json(best_path, best_payload)

        print("\n" + "=" * 60)
        print("BEST PARAMETERS")
        print(f"  T={cfg['T']}  top_p={cfg['top_p']}  sample_count={cfg['sample_count']}")
        print(f"  lookback={cfg['lookback']}  horizon_hours={cfg['horizon_hours']}")
        print(f"  model={args.model}  interval={args.interval}")
        print(
            f"  MAPE={best['mape']:.4f}%  MAE={best['mae']:.4f}  "
            f"directional_accuracy={best['directional_accuracy']:.4f}"
        )
        print(f"  wrote {best_path}")
        print(f"  wrote {latest_path}")
        print("=" * 60)
    else:
        print("No successful evaluations; best_params.json not updated.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
