"""Live market data helpers for Kronos Live."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

CRYPTO_HINTS = ("-USD", "-USDT", "-EUR", "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA")


def is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return any(h in s for h in ("-USD", "-USDT", "-EUR")) or s.endswith("USD") and "-" in s


def normalize_symbol(query: str) -> str:
    q = (query or "").strip().upper()
    if not q:
        raise ValueError("Ticker is required")
    # Common aliases
    aliases = {
        "VFV": "VFV.TO",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SPX": "^GSPC",
        "SP500": "^GSPC",
    }
    return aliases.get(q, q)


def search_tickers(query: str, limit: int = 8) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if len(q) < 1:
        return []
    results: list[dict[str, Any]] = []
    try:
        # yfinance Search API (newer) with fallback
        search = getattr(yf, "Search", None)
        if search is not None:
            s = search(q, max_results=limit)
            quotes = getattr(s, "quotes", None) or []
            for item in quotes[:limit]:
                sym = item.get("symbol") or item.get("ticker")
                if not sym:
                    continue
                results.append(
                    {
                        "symbol": sym,
                        "name": item.get("shortname") or item.get("longname") or item.get("name") or sym,
                        "exchange": item.get("exchange") or item.get("exchDisp") or "",
                        "type": item.get("quoteType") or item.get("typeDisp") or "",
                    }
                )
    except Exception:
        results = []

    if not results:
        # Fallback: try the query itself as a ticker
        try:
            sym = normalize_symbol(q)
            info = yf.Ticker(sym).fast_info
            last = getattr(info, "last_price", None) or getattr(info, "lastPrice", None)
            if last is not None:
                results.append(
                    {
                        "symbol": sym,
                        "name": sym,
                        "exchange": getattr(info, "exchange", "") or "",
                        "type": "EQUITY",
                    }
                )
        except Exception:
            pass
    return results[:limit]


def fetch_quote(symbol: str) -> dict[str, Any]:
    sym = normalize_symbol(symbol)
    t = yf.Ticker(sym)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    fi = {}
    try:
        fi = dict(t.fast_info) if hasattr(t.fast_info, "items") else {}
        if not fi:
            # Namespace-like
            for k in ("last_price", "previous_close", "open", "day_high", "day_low", "currency", "exchange"):
                if hasattr(t.fast_info, k):
                    fi[k] = getattr(t.fast_info, k)
    except Exception:
        fi = {}

    price = fi.get("last_price") or info.get("regularMarketPrice") or info.get("currentPrice")
    prev = fi.get("previous_close") or info.get("regularMarketPreviousClose") or info.get("previousClose")
    change = None
    change_pct = None
    if price is not None and prev not in (None, 0):
        change = float(price) - float(prev)
        change_pct = (change / float(prev)) * 100

    return {
        "symbol": sym,
        "name": info.get("shortName") or info.get("longName") or sym,
        "price": float(price) if price is not None else None,
        "previousClose": float(prev) if prev is not None else None,
        "change": change,
        "changePct": change_pct,
        "open": _f(fi.get("open") or info.get("regularMarketOpen") or info.get("open")),
        "dayHigh": _f(fi.get("day_high") or info.get("dayHigh") or info.get("regularMarketDayHigh")),
        "dayLow": _f(fi.get("day_low") or info.get("dayLow") or info.get("regularMarketDayLow")),
        "volume": _f(fi.get("last_volume") or info.get("volume") or info.get("regularMarketVolume")),
        "currency": fi.get("currency") or info.get("currency") or "USD",
        "exchange": fi.get("exchange") or info.get("exchange") or info.get("fullExchangeName") or "",
        "marketState": info.get("marketState") or "",
        "isCrypto": is_crypto_symbol(sym),
        "asOf": datetime.utcnow().isoformat() + "Z",
    }


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _flatten_ohlcv(raw: pd.DataFrame, *, market_tz: str = "America/New_York") -> pd.DataFrame:
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]
    raw = raw.reset_index()
    ts_col = [c for c in raw.columns if "date" in c or "time" in c][0]
    raw = raw.rename(columns={ts_col: "timestamps"})
    ts = pd.to_datetime(raw["timestamps"], utc=True)
    # Keep exchange-local wall clock so session windows (09:30–16:00) are correct.
    try:
        ts = ts.dt.tz_convert(market_tz)
    except Exception:
        pass
    raw["timestamps"] = ts.dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "volume"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["amount"] = raw["close"] * raw["volume"].fillna(0)
    out = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna(
        subset=["open", "high", "low", "close"]
    )
    return out.sort_values("timestamps").reset_index(drop=True)


def fetch_history(symbol: str, interval: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    sym = normalize_symbol(symbol)
    crypto = is_crypto_symbol(sym)

    # Prefer dense bars for a meaningful 24h path
    candidates = []
    if interval:
        candidates = [(interval, "7d" if interval in ("1m", "2m", "5m", "15m") else "60d")]
    elif crypto:
        candidates = [("5m", "7d"), ("15m", "60d"), ("1h", "60d")]
    else:
        candidates = [("5m", "7d"), ("15m", "60d"), ("1h", "60d"), ("1d", "2y")]

    last_err = None
    for iv, period in candidates:
        try:
            raw = yf.download(sym, period=period, interval=iv, auto_adjust=False, progress=False, threads=False)
            tz = "UTC" if crypto else "America/New_York"
            df = _flatten_ohlcv(raw, market_tz=tz)
            if len(df) >= 50:
                meta = {
                    "symbol": sym,
                    "interval": iv,
                    "period": period,
                    "bars": len(df),
                    "isCrypto": crypto,
                    "start": df["timestamps"].iloc[0].isoformat(),
                    "end": df["timestamps"].iloc[-1].isoformat(),
                }
                return df, meta
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load history for {sym}: {last_err}")


def generate_future_timestamps(
    history: pd.DataFrame,
    interval: str,
    is_crypto: bool,
    horizon_hours: int = 24,
) -> pd.DatetimeIndex:
    """Build prediction timestamps covering ~24h of trading activity."""
    last = pd.Timestamp(history["timestamps"].iloc[-1])
    freq_map = {
        "1m": "1min",
        "2m": "2min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "1h",
        "1h": "1h",
        "1d": "1D",
    }
    freq = freq_map.get(interval, "5min")

    # Bars needed for calendar ~24h
    minutes = {"1min": 1, "2min": 2, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "1D": 1440}.get(freq, 5)
    if is_crypto or interval == "1d":
        n = max(1, int((horizon_hours * 60) / minutes))
        return pd.date_range(start=last + pd.Timedelta(minutes=minutes), periods=min(n, 288), freq=freq)

    # Equities / ETFs: next regular session (and spill into following if needed to approach 24h)
    # Approximate US/CA session 09:30–16:00 local wall clock on the series timezone-naive stamps.
    bars: list[pd.Timestamp] = []
    cursor = last + pd.Timedelta(minutes=minutes)
    # jump to next session open if after close or weekend
    def session_open(d: pd.Timestamp) -> pd.Timestamp:
        return d.normalize() + pd.Timedelta(hours=9, minutes=30)

    def session_close(d: pd.Timestamp) -> pd.Timestamp:
        return d.normalize() + pd.Timedelta(hours=16)

    # If we're past close, move to next weekday open
    if cursor.time() >= session_close(cursor).time() or cursor.weekday() >= 5:
        cursor = session_open(cursor + pd.Timedelta(days=1))
        while cursor.weekday() >= 5:
            cursor = session_open(cursor + pd.Timedelta(days=1))
    elif cursor.time() < session_open(cursor).time():
        cursor = session_open(cursor)

    target = int((6.5 * 60) / minutes)  # one full cash session ≈ next-day 24h horizon for ETFs
    target = max(target, int((horizon_hours * 60) / minutes) if minutes >= 60 else target)
    target = min(target, 120)

    guard = 0
    while len(bars) < target and guard < 5000:
        guard += 1
        if cursor.weekday() >= 5:
            cursor = session_open(cursor + pd.Timedelta(days=1))
            continue
        if cursor.time() < session_open(cursor).time():
            cursor = session_open(cursor)
            continue
        if cursor.time() >= session_close(cursor).time():
            cursor = session_open(cursor + pd.Timedelta(days=1))
            continue
        bars.append(cursor)
        cursor = cursor + pd.Timedelta(minutes=minutes)

    return pd.DatetimeIndex(bars)
