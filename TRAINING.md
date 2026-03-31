# Training Setup

This repo has two distinct modes:

- Full recommender demo: synthetic data plus ranking/recommendation flow.
- Forecast training on Indian market history: transformer population training plus walk-forward evaluation.

For the Indian-market training path, use the local scaffold added here.

## Prerequisites

- Virtualenv available at `.venv`
- Python dependencies from `requirements.txt`
- **PostgreSQL** running and accessible (SQLite is no longer supported).
- Local CSV history already present in `india_universe_data/`

### Rebuild the local environment

If `.venv` is stale or was created on another machine, rebuild it from the
local Python install instead of reusing old interpreter paths:

```powershell
.\scripts\rebuild_env.ps1
```

That script:

1. Recreates `.venv` from the Python found on `PATH`
2. Installs `requirements.txt` into the new environment
3. Uses a repo-local `.tmp/` directory so temp-file permissions do not depend on the host profile

Optional flags:

```powershell
.\scripts\rebuild_env.ps1 -InstallNodeDeps
.\scripts\rebuild_env.ps1 -PythonExe py
.\scripts\rebuild_env.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu124
```

### Database setup

The system uses PostgreSQL.  Set the connection DSN via the environment variable
`STOCK_RECOMMENDER_DB_URL` before running any command:

```powershell
$env:STOCK_RECOMMENDER_DB_URL = "postgresql://postgres:postgres@localhost:5432/stock_recommender"
```

Or pass it explicitly with `--db` on every command.  The default DSN is
`postgresql://postgres:postgres@localhost:5432/stock_recommender`.

## Fast local run

Run a bounded training job first so you can verify the stack without committing to the full universe:

```powershell
.\scripts\train_india_local.ps1 -MaxStocks 25 -PopulationSize 1 -PopulationEpochs 1 -MaxWindows 1
```

The script uses `STOCK_RECOMMENDER_DB_URL` automatically.  To override:

```powershell
.\scripts\train_india_local.ps1 -DbUrl "postgresql://user:pass@host/db" -MaxStocks 25 -PopulationSize 1 -PopulationEpochs 1 -MaxWindows 1
```

That command will:

1. Ingest CSV data into PostgreSQL (`-Reingest` flag forces re-ingestion).
2. Train population candidates with `train_population` (out-of-sample eval, no leakage).
3. Backtest the latest transformer checkpoint with `backtest4`.

## Useful direct commands

Ingest CSV history:

```powershell
.\.venv\Scripts\python.exe main.py ingest_india --config configs/india_local_training.json --source csv --data-dir india_universe_data --metadata india_universe_data/_metadata.json
```

Train forecasting candidates:

```powershell
.\.venv\Scripts\python.exe main.py train_population --config configs/india_local_training.json --population-size 3 --population-epochs 2 --max-stocks 200 --max-windows 3 --top-k 1
```

Evaluate the latest transformer checkpoint:

```powershell
.\.venv\Scripts\python.exe main.py backtest4 --config configs/india_local_training.json --max-stocks 200 --max-windows 3
```

## Notes

- `--db` now accepts a **PostgreSQL DSN**, not a file path.  Passing a plain filename will raise `ProgrammingError: invalid dsn`.
- `STOCK_RECOMMENDER_DB_URL` env var is read automatically — no `--db` needed if set.
- `backtest4` loads the latest transformer checkpoint automatically unless `--checkpoint` is supplied.
- `--max-stocks` is the main knob for controlling runtime on local machines.
- The recommendation commands (`seed`, `train`, `recommend`, `analyze`) require user interaction data.  The forecasting transformer trains on OHLCV only.
- Population training uses an 80/20 stock split — candidates are trained on 80% and evaluated on the held-out 20%, giving a clean out-of-sample leaderboard.
