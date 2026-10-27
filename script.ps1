
$ErrorActionPreference = "Stop"

Write-Host "Running tests..."

Set-Location $PSScriptRoot

# Activate your Conda environment
conda activate agents_env

# Run tests
python -m pytest -v

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
