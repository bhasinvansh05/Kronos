#!/usr/bin/env python3
"""Compare saved VFV.TO Kronos forecast vs realized Yahoo 5-min bars."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "pandas"])
    import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
FORECAST_CSV = ROOT / "outputs" / "forecast_job" / "VFV_TO_forecast_24h_5min.csv"
SUMMARY_JSON = ROOT / "outputs" / "forecast_job" / "VFV_TO_forecast_24h_summary.json"
REPORT_JSON = ROOT / "outputs" / "forecast_job" / "VFV_TO_forecast_accuracy_report.json"
REPORT_MD = ROOT / "outputs" / "forecast_job" / "VFV_TO_forecast_accuracy_report.md"


def load_forecast() -> pd.DataFrame:
    fc = pd.read_csv(FORECAST_CSV, index_col=0, parse_dates=True)
    fc.index = pd.to_datetime(fc.index).tz_localize(None)
    return fc


def fetch_actual() -> pd.DataFrame:
    raw = yf.download("VFV.TO", period="5d", interval="5m", auto_adjust=False, progress=False)
    raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in raw.columns]
    raw = raw.reset_index()
    ts_col = [c for c in raw.columns if "date" in c.lower() or "time" in c.lower()][0]
    raw = raw.rename(columns={ts_col: "timestamps"})
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw.dropna(subset=["open", "high", "low", "close"]).set_index("timestamps")


def direction_match(pred_last: float, pred_first_ref: float, act_last: float, act_first_ref: float) -> bool:
    pred_dir = pred_last - pred_first_ref
    act_dir = act_last - act_first_ref
    if pred_dir == 0 or act_dir == 0:
        return abs(pred_dir) < 1e-9 and abs(act_dir) < 1e-9
    return (pred_dir > 0) == (act_dir > 0)


def main() -> int:
    if not FORECAST_CSV.exists():
        print(f"Missing forecast file: {FORECAST_CSV}", file=sys.stderr)
        return 1

    forecast = load_forecast()
    actual_all = fetch_actual()
    # Align on forecast timestamps present in actuals
    joined = forecast[["open", "high", "low", "close"]].join(
        actual_all[["open", "high", "low", "close"]],
        how="inner",
        lsuffix="_pred",
        rsuffix="_act",
    )

    summary = {}
    if SUMMARY_JSON.exists():
        summary = json.loads(SUMMARY_JSON.read_text())

    n = len(joined)
    if n == 0:
        report = {
            "status": "no_overlap",
            "message": "No overlapping 5-min bars between forecast and realized VFV.TO data yet.",
            "forecast_bars": len(forecast),
            "actual_bars_downloaded": len(actual_all),
            "forecast_window": {
                "start": str(forecast.index.min()),
                "end": str(forecast.index.max()),
            },
        }
        REPORT_JSON.write_text(json.dumps(report, indent=2))
        REPORT_MD.write_text(
            "# VFV.TO forecast check\n\nNo overlapping bars yet — market data for the forecast window is not available.\n"
        )
        print(json.dumps(report, indent=2))
        return 2

    close_err = joined["close_pred"] - joined["close_act"]
    mae = float(close_err.abs().mean())
    rmse = float((close_err**2).mean() ** 0.5)
    mape = float((close_err.abs() / joined["close_act"].abs()).mean() * 100)
    last_pred = float(joined["close_pred"].iloc[-1])
    last_act = float(joined["close_act"].iloc[-1])
    first_act = float(joined["close_act"].iloc[0])
    hist_ref = float(summary.get("history_last_close", first_act))
    pred_end = float(summary.get("forecast_close_last", last_pred))
    pred_change = pred_end - hist_ref
    act_change = last_act - hist_ref
    dir_ok = direction_match(pred_end, hist_ref, last_act, hist_ref)

    # "Match" thresholds: direction + session close within 0.25% or MAE < 0.35 CAD
    close_pct_err = abs(last_pred - last_act) / abs(last_act) * 100
    matched = bool(dir_ok and (close_pct_err <= 0.25 or mae <= 0.35))

    report = {
        "status": "ok",
        "matched": matched,
        "match_criteria": "same session direction vs prior close AND (end-close |err| <= 0.25% OR MAE <= 0.35 CAD)",
        "symbol": "VFV.TO",
        "currency": "CAD",
        "overlap_bars": n,
        "forecast_bars": len(forecast),
        "history_last_close": hist_ref,
        "forecast_session_close": pred_end,
        "actual_session_close": last_act,
        "forecast_change_abs": pred_change,
        "forecast_change_pct": pred_change / hist_ref * 100,
        "actual_change_abs": act_change,
        "actual_change_pct": act_change / hist_ref * 100,
        "direction_matched": dir_ok,
        "session_close_abs_error": abs(last_pred - last_act),
        "session_close_pct_error": close_pct_err,
        "close_mae": mae,
        "close_rmse": rmse,
        "close_mape_pct": mape,
        "actual_session_high": float(joined["high_act"].max()),
        "actual_session_low": float(joined["low_act"].min()),
        "forecast_session_high": float(summary.get("forecast_high_max", joined["high_pred"].max())),
        "forecast_session_low": float(summary.get("forecast_low_min", joined["low_pred"].min())),
        "first_overlap": str(joined.index.min()),
        "last_overlap": str(joined.index.max()),
    }

    verdict = "MATCHED" if matched else "DID NOT MATCH"
    md = f"""# VFV.TO forecast accuracy — {verdict}

- Overlap bars: **{n}** ({report['first_overlap']} → {report['last_overlap']})
- Prior close: **{hist_ref:.4f} CAD**
- Forecast close: **{pred_end:.4f}** ({pred_change:+.4f}, {report['forecast_change_pct']:+.3f}%)
- Actual close: **{last_act:.4f}** ({act_change:+.4f}, {report['actual_change_pct']:+.3f}%)
- Direction matched: **{dir_ok}**
- Session close |error|: **{report['session_close_abs_error']:.4f}** ({close_pct_err:.3f}%)
- Path MAE / RMSE / MAPE: **{mae:.4f}** / **{rmse:.4f}** / **{mape:.3f}%**

Match rule: same direction vs prior close AND (end-close error ≤ 0.25% OR MAE ≤ 0.35 CAD).
"""
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    REPORT_MD.write_text(md)
    print(json.dumps(report, indent=2))
    print(md)
    return 0 if matched else 3


if __name__ == "__main__":
    raise SystemExit(main())
