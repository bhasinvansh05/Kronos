(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    symbol: null,
    chart: null,
    candleSeries: null,
    forecastSeries: null,
    volumeSeries: null,
    suggestTimer: null,
    activeSuggest: -1,
    results: [],
  };

  function toast(msg, ms = 3200) {
    const el = $("toast");
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
      el.hidden = true;
    }, ms);
  }

  function setModelChip(status, label) {
    const chip = $("modelChip");
    chip.classList.remove("ready", "loading", "error");
    if (status) chip.classList.add(status);
    $("modelLabel").textContent = label;
  }

  function fmt(n, digits = 2) {
    if (n == null || Number.isNaN(n)) return "—";
    return Number(n).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function toChartTime(iso) {
    // Lightweight Charts wants UTC seconds for intraday
    return Math.floor(new Date(iso).getTime() / 1000);
  }

  function initChart() {
    const el = $("chart");
    state.chart = LightweightCharts.createChart(el, {
      layout: {
        background: { type: "solid", color: "transparent" },
        textColor: "#9aa7b5",
        fontFamily: "IBM Plex Mono, monospace",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      rightPriceScale: { borderColor: "rgba(232,214,180,0.12)" },
      timeScale: {
        borderColor: "rgba(232,214,180,0.12)",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        vertLine: { color: "rgba(224,163,91,0.35)", labelBackgroundColor: "#e0a35b" },
        horzLine: { color: "rgba(224,163,91,0.35)", labelBackgroundColor: "#e0a35b" },
      },
    });

    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: "#5ec4b2",
      downColor: "#ef6b5a",
      borderUpColor: "#5ec4b2",
      borderDownColor: "#ef6b5a",
      wickUpColor: "#5ec4b2",
      wickDownColor: "#ef6b5a",
    });

    state.forecastSeries = state.chart.addCandlestickSeries({
      upColor: "#e0a35b",
      downColor: "#c9843a",
      borderUpColor: "#f0c48a",
      borderDownColor: "#c9843a",
      wickUpColor: "#e0a35b",
      wickDownColor: "#c9843a",
    });

    state.volumeSeries = state.chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      scaleMargins: { top: 0.82, bottom: 0 },
    });

    const ro = new ResizeObserver(() => {
      state.chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    });
    ro.observe(el);
  }

  async function refreshModelStatus() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      if (data.status === "ready") {
        setModelChip("ready", `${data.model} · ${data.device}`);
      } else if (data.status === "loading") {
        setModelChip("loading", "Loading model…");
      } else if (data.status === "error") {
        setModelChip("error", data.error || "Model error");
      } else {
        setModelChip("", "Model idle — loads on first predict");
      }
    } catch {
      setModelChip("error", "API unreachable");
    }
  }

  async function search(q) {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    return data.results || [];
  }

  function renderSuggest(results) {
    const list = $("suggestList");
    state.results = results;
    state.activeSuggest = -1;
    if (!results.length) {
      list.hidden = true;
      list.innerHTML = "";
      return;
    }
    list.innerHTML = results
      .map(
        (r, i) => `
      <li data-i="${i}" data-symbol="${r.symbol}">
        <span class="s-sym">${r.symbol}</span>
        <span class="s-name">${r.name || ""}</span>
        <span class="s-ex">${r.exchange || r.type || ""}</span>
      </li>`
      )
      .join("");
    list.hidden = false;
  }

  function pickSymbol(symbol) {
    $("tickerInput").value = symbol;
    $("suggestList").hidden = true;
    state.symbol = symbol;
  }

  function paintQuote(q) {
    if (!q) return;
    $("statusRow").hidden = false;
    $("qSymbol").textContent = q.symbol;
    $("qName").textContent = q.name || "";
    $("qPrice").textContent = fmt(q.price, q.price != null && q.price < 10 ? 4 : 2);
    const chg = $("qChg");
    if (q.change == null) {
      chg.textContent = "—";
      chg.className = "chg";
    } else {
      const sign = q.change >= 0 ? "+" : "";
      chg.textContent = `${sign}${fmt(q.change)} (${sign}${fmt(q.changePct)}%)`;
      chg.className = `chg ${q.change >= 0 ? "up" : "down"}`;
    }
  }

  function paintSummary(s, meta) {
    $("mClose").textContent = fmt(s.forecastClose, s.forecastClose < 10 ? 4 : 2);
    const sign = s.changeAbs >= 0 ? "+" : "";
    $("mDelta").textContent = `${sign}${fmt(s.changeAbs)} (${sign}${fmt(s.changePct)}%)`;
    $("mDelta").style.color = s.changeAbs >= 0 ? "var(--signal)" : "var(--danger)";
    $("mRange").textContent = `${fmt(s.forecastLow)} / ${fmt(s.forecastHigh)}`;
    $("mHorizon").textContent = `${meta.interval} · ${s.forecastStart?.slice(0, 16)} → ${s.forecastEnd?.slice(11, 16) || ""}`;
  }

  function paintChart(history, forecast) {
    const histCandles = history.map((b) => ({
      time: toChartTime(b.time),
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));
    const histVol = history.map((b) => ({
      time: toChartTime(b.time),
      value: b.volume || 0,
      color: b.close >= b.open ? "rgba(94,196,178,0.35)" : "rgba(239,107,90,0.35)",
    }));
    const predCandles = forecast.map((b) => ({
      time: toChartTime(b.time),
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));

    state.candleSeries.setData(histCandles);
    state.volumeSeries.setData(histVol);
    state.forecastSeries.setData(predCandles);
    state.chart.timeScale().fitContent();
  }

  async function runPredict(symbol) {
    const btn = $("predictBtn");
    btn.disabled = true;
    btn.textContent = "Predicting…";
    setModelChip("loading", "Running Kronos…");
    $("hint").textContent = "Pulling live bars and sampling the next 24h path…";
    $("chartTitle").textContent = `${symbol} · live + forecast`;

    try {
      // Quote first for immediate feedback
      const qRes = await fetch(`/api/quote/${encodeURIComponent(symbol)}`);
      const quote = await qRes.json();
      if (!qRes.ok) throw new Error(quote.error || "Quote failed");
      paintQuote(quote);

      const body = {
        symbol,
        model: $("modelSelect").value,
        temperature: Number($("tempInput").value || 1),
        top_p: 0.9,
        sample_count: 1,
        lookback: 400,
        horizon_hours: 24,
      };
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      if (data.quote) paintQuote(data.quote);
      paintSummary(data.summary, data.meta);
      paintChart(data.history, data.forecast);
      setModelChip("ready", `${data.model} · ${data.device}`);
      $("footMeta").textContent = `${data.symbol} · ${data.meta.bars} src bars · lookback ${data.lookback} · pred ${data.predLen}`;
      $("hint").textContent = data.meta.isCrypto
        ? "Crypto path: continuous 24h forecast from the latest live bar."
        : "Equity/ETF path: next regular session (~24h horizon) from the latest live bar.";
      toast(`${data.symbol} forecast ready`);
    } catch (err) {
      setModelChip("error", "Predict failed");
      $("hint").textContent = err.message || String(err);
      toast(err.message || "Prediction failed");
    } finally {
      btn.disabled = false;
      btn.textContent = "Predict 24h";
    }
  }

  function wire() {
    initChart();
    refreshModelStatus();
    setInterval(refreshModelStatus, 15000);

    const input = $("tickerInput");
    input.addEventListener("input", () => {
      const q = input.value.trim();
      clearTimeout(state.suggestTimer);
      if (q.length < 1) {
        renderSuggest([]);
        return;
      }
      state.suggestTimer = setTimeout(async () => {
        try {
          renderSuggest(await search(q));
        } catch {
          renderSuggest([]);
        }
      }, 180);
    });

    $("suggestList").addEventListener("click", (e) => {
      const li = e.target.closest("li");
      if (!li) return;
      pickSymbol(li.dataset.symbol);
      runPredict(li.dataset.symbol);
    });

    input.addEventListener("keydown", (e) => {
      const list = $("suggestList");
      if (list.hidden || !state.results.length) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        state.activeSuggest = Math.min(state.activeSuggest + 1, state.results.length - 1);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        state.activeSuggest = Math.max(state.activeSuggest - 1, 0);
      } else if (e.key === "Enter" && state.activeSuggest >= 0) {
        e.preventDefault();
        pickSymbol(state.results[state.activeSuggest].symbol);
        return;
      } else {
        return;
      }
      [...list.children].forEach((li, i) => li.classList.toggle("active", i === state.activeSuggest));
    });

    $("searchForm").addEventListener("submit", (e) => {
      e.preventDefault();
      const symbol = input.value.trim().toUpperCase();
      if (!symbol) return;
      $("suggestList").hidden = true;
      state.symbol = symbol;
      runPredict(symbol);
    });

    // Deep-link ?symbol=VFV.TO
    const params = new URLSearchParams(location.search);
    const boot = params.get("symbol") || params.get("ticker");
    if (boot) {
      pickSymbol(boot.toUpperCase());
      runPredict(boot.toUpperCase());
    }
  }

  wire();
})();
