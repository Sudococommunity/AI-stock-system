import fs from "fs";
import path from "path";
import { NSE } from "nse-bse-api";

const NSE_UNIVERSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv";

function parseArgs(argv) {
  const args = {
    outputDir: "nse_api_universe_data",
    startDate: "1994-01-01",
    endDate: new Date().toISOString().slice(0, 10),
    workers: 1,
    retries: 2,
    refresh: false,
    stateFile: null,
    symbolsFile: null,
  };

  for (let i = 2; i < argv.length; i++) {
    const key = argv[i];
    const val = argv[i + 1];
    if (key === "--output-dir") args.outputDir = val;
    else if (key === "--start-date") args.startDate = val;
    else if (key === "--end-date") args.endDate = val;
    else if (key === "--workers") args.workers = Number(val);
    else if (key === "--retries") args.retries = Number(val);
    else if (key === "--refresh") args.refresh = true;
    else if (key === "--state-file") args.stateFile = val;
    else if (key === "--symbols-file") args.symbolsFile = val;
  }
  return args;
}

async function fetchNseUniverse() {
  const res = await fetch(NSE_UNIVERSE_URL, {
    headers: {
      "user-agent": "Mozilla/5.0",
      "accept": "text/csv,application/octet-stream,*/*",
      "referer": "https://www.nseindia.com/",
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch NSE universe: HTTP ${res.status}`);
  }
  const text = await res.text();
  const lines = text.split(/\r?\n/).filter(Boolean);
  const header = parseCsvLine(lines[0]);
  const idxSymbol = header.indexOf("SYMBOL");
  const idxName = header.indexOf("NAME OF COMPANY");
  const idxSeries = header.indexOf("SERIES");
  if (idxSymbol === -1) {
    throw new Error("NSE universe CSV missing SYMBOL column");
  }

  const symbols = [];
  for (let i = 1; i < lines.length; i++) {
    const row = parseCsvLine(lines[i]);
    const symbol = String(row[idxSymbol] || "").trim().toUpperCase();
    const series = String(row[idxSeries] || "EQ").trim().toUpperCase();
    if (!symbol) continue;
    if (!["EQ", "BE", "BZ", "SM"].includes(series)) continue;
    symbols.push({
      symbol,
      companyName: String(row[idxName] || "").trim(),
      series,
    });
  }
  return symbols;
}

function loadSymbolsFromCsv(symbolsFile) {
  const text = fs.readFileSync(symbolsFile, "utf8");
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const header = parseCsvLine(lines[0]).map((x) => String(x).trim().toLowerCase());
  const symbolIdx = header.findIndex((x) => ["symbol", "ticker"].includes(x));
  if (symbolIdx === -1) {
    throw new Error("symbols file must contain a Symbol or ticker column");
  }
  const out = [];
  for (let i = 1; i < lines.length; i++) {
    const row = parseCsvLine(lines[i]);
    const symbol = String(row[symbolIdx] || "").trim().toUpperCase();
    if (!symbol) continue;
    out.push({
      symbol,
      companyName: symbol,
      series: "EQ",
    });
  }
  return out;
}

function parseCsvLine(line) {
  const out = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  out.push(current);
  return out;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function loadState(statePath) {
  if (!fs.existsSync(statePath)) return {};
  return JSON.parse(fs.readFileSync(statePath, "utf8"));
}

function saveState(statePath, state) {
  fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
}

function writeCsv(filePath, rows) {
  const header = ["date", "open", "high", "low", "close", "volume"];
  const lines = [header.join(",")];
  for (const row of rows) {
    lines.push([
      row.date,
      row.open,
      row.high,
      row.low,
      row.close,
      row.volume,
    ].join(","));
  }
  fs.writeFileSync(filePath, lines.join("\n") + "\n", "utf8");
}

function normalizeHistorical(rows) {
  return rows
    .map((item) => ({
      date: formatNseDate(item.mtimestamp),
      open: item.chOpeningPrice,
      high: item.chTradeHighPrice,
      low: item.chTradeLowPrice,
      close: item.chClosingPrice ?? item.chLastTradedPrice,
      volume: item.chTotTradedQty,
    }))
    .filter((row) => row.date && row.open != null && row.high != null && row.low != null && row.close != null && row.volume != null)
    .sort((a, b) => a.date.localeCompare(b.date));
}

function formatNseDate(text) {
  if (!text) return null;
  const [dd, mon, yyyy] = String(text).split("-");
  const months = {
    Jan: "01", Feb: "02", Mar: "03", Apr: "04", May: "05", Jun: "06",
    Jul: "07", Aug: "08", Sep: "09", Oct: "10", Nov: "11", Dec: "12",
  };
  if (!dd || !mon || !yyyy || !months[mon]) return null;
  return `${yyyy}-${months[mon]}-${dd.padStart(2, "0")}`;
}

async function main() {
  const args = parseArgs(process.argv);
  ensureDir(args.outputDir);
  const statePath = args.stateFile || path.join(args.outputDir, "_nse_api_state.json");
  const state = loadState(statePath);

  console.log("========================================================================");
  console.log("NSE API UNIVERSE DOWNLOAD STARTED");
  console.log(`Output directory     : ${args.outputDir}`);
  console.log(`Start date           : ${args.startDate}`);
  console.log(`End date             : ${args.endDate}`);
  console.log(`Workers              : ${args.workers}`);
  console.log("========================================================================");

  const universe = args.symbolsFile ? loadSymbolsFromCsv(args.symbolsFile) : await fetchNseUniverse();
  console.log(`Universe size        : ${universe.length}`);

  const queue = [];
  for (const item of universe) {
    const filePath = path.join(args.outputDir, `NSE_${item.symbol}.csv`);
    if (fs.existsSync(filePath) && !args.refresh) {
      state[item.symbol] = { status: "skipped", path: filePath };
      continue;
    }
    queue.push({ ...item, filePath });
  }

  console.log(`To download          : ${queue.length}`);

  let completed = 0;
  let ok = 0;
  let failed = 0;
  let apiCalls = 0;

  function renderProgress(currentSymbol = "") {
    const total = queue.length + completed;
    const done = ok + failed;
    const width = 32;
    const ratio = total > 0 ? done / total : 0;
    const filled = Math.round(ratio * width);
    const bar = `${"#".repeat(filled)}${"-".repeat(width - filled)}`;
    const line =
      `[${bar}] ${done}/${total} | ok=${ok} fail=${failed} | api_calls=${apiCalls}` +
      (currentSymbol ? ` | ${currentSymbol}` : "");
    console.log(line);
  }

  async function workerLoop(workerId) {
    const nse = new NSE(path.join(args.outputDir, "_tmp_downloads"));
    try {
      while (queue.length > 0) {
        const item = queue.shift();
        if (!item) break;
        completed += 1;
        const label = `[${completed}/${queue.length + completed}]`;
        console.log(`${label} [START] ${item.symbol}`);
        renderProgress(item.symbol);
        let success = false;
        let lastError = "unknown";

        for (let attempt = 0; attempt <= args.retries; attempt++) {
          try {
            apiCalls += 1;
            const rows = await nse.historical.fetchEquityHistoricalData({
              symbol: item.symbol,
              from_date: new Date(args.startDate),
              to_date: new Date(args.endDate),
              series: [item.series || "EQ"],
            });
            const normalized = normalizeHistorical(rows);
            if (!normalized.length) {
              throw new Error("No historical rows returned");
            }
            writeCsv(item.filePath, normalized);
            state[item.symbol] = {
              status: "downloaded",
              rows: normalized.length,
              path: item.filePath,
            };
            saveState(statePath, state);
            console.log(`${label} [OK] ${item.symbol} -> ${normalized.length} rows`);
            ok += 1;
            renderProgress(item.symbol);
            success = true;
            break;
          } catch (err) {
            lastError = String(err?.message || err);
            if (attempt < args.retries) {
              console.log(`${label} [RETRY ${attempt + 1}] ${item.symbol} -> ${lastError}`);
              renderProgress(item.symbol);
            }
          }
        }

        if (!success) {
          failed += 1;
          state[item.symbol] = { status: "failed", error: lastError };
          saveState(statePath, state);
          console.log(`${label} [FAIL] ${item.symbol} -> ${lastError}`);
          renderProgress(item.symbol);
        }

        if ((ok + failed) % 25 === 0) {
          console.log(`[SUMMARY] done=${ok + failed} ok=${ok} failed=${failed} remaining=${queue.length} api_calls=${apiCalls}`);
        }
      }
    } finally {
      if (nse.exit) await nse.exit();
    }
  }

  const workers = [];
  for (let i = 0; i < Math.max(1, args.workers); i++) {
    workers.push(workerLoop(i + 1));
  }
  await Promise.all(workers);

  console.log("========================================================================");
  console.log("NSE API DOWNLOAD COMPLETE");
  console.log(`Downloaded            : ${ok}`);
  console.log(`Failed                : ${failed}`);
  console.log(`API calls             : ${apiCalls}`);
  console.log(`State file            : ${statePath}`);
  console.log("========================================================================");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
