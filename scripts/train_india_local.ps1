param(
    # PostgreSQL DSN.  Leave empty to use STOCK_RECOMMENDER_DB_URL env var or config default.
    [string]$DbUrl = "",
    [string]$ConfigPath = "configs/india_local_training.json",
    [string]$DataDir = "india_universe_data",
    [string]$MetadataPath = "india_universe_data/_metadata.json",
    [int]$MaxStocks = 200,
    [int]$PopulationSize = 3,
    [int]$PopulationEpochs = 2,
    [int]$MaxWindows = 3,
    [switch]$Reingest
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python virtualenv not found at .venv\Scripts\python.exe"
}

# Build --db flag only when a DSN is explicitly supplied.
$dbFlag = if ($DbUrl) { @("--db", $DbUrl) } else { @() }

if ($Reingest) {
    & $python main.py ingest_india `
        @dbFlag `
        --config $ConfigPath `
        --source csv `
        --data-dir $DataDir `
        --metadata $MetadataPath
}

& $python main.py train_population `
    @dbFlag `
    --config $ConfigPath `
    --population-size $PopulationSize `
    --population-epochs $PopulationEpochs `
    --max-windows $MaxWindows `
    --max-stocks $MaxStocks `
    --top-k 1

& $python main.py backtest4 `
    @dbFlag `
    --config $ConfigPath `
    --max-stocks $MaxStocks `
    --max-windows $MaxWindows
