param(
    [string]$PythonExe = "python",
    [switch]$InstallNodeDeps,
    [string]$TorchIndexUrl = ""
)

$ErrorActionPreference = "Stop"

function Get-ResolvedCommandPath {
    param([string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Required command not found: $Name"
    }
    return $cmd.Source
}

$pythonPath = Get-ResolvedCommandPath -Name $PythonExe
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPath = Join-Path $projectRoot ".venv"
$tmpPath = Join-Path $projectRoot ".tmp"
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$venvPython = Join-Path $venvPath "Scripts\\python.exe"

Push-Location $projectRoot
try {
New-Item -ItemType Directory -Force $tmpPath | Out-Null
$env:TMP = $tmpPath
$env:TEMP = $tmpPath

if (Test-Path $venvPath) {
    Remove-Item -Recurse -Force $venvPath
}

& $pythonPath -m venv $venvPath

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation failed: $venvPython not found"
}

$installArgs = @(
    "-m", "pip",
    "--python", $venvPython,
    "install",
    "--upgrade",
    "pip",
    "setuptools",
    "wheel",
    "-r", $requirementsPath
)

if ($TorchIndexUrl) {
    $installArgs += @("--index-url", $TorchIndexUrl)
}

& $pythonPath @installArgs

if ($InstallNodeDeps -and (Test-Path (Join-Path $projectRoot "package.json"))) {
    $npmPath = Get-ResolvedCommandPath -Name "npm"
    & $npmPath "ci" "--ignore-scripts"
}
}
finally {
    Pop-Location
}

Write-Host "Environment rebuilt successfully."
Write-Host "Python: $venvPython"
if ($InstallNodeDeps) {
    Write-Host "Node dependencies installed with npm ci."
}
