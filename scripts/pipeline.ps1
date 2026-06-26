<#
.SYNOPSIS
KickCast Orchestrator Pipeline

.DESCRIPTION
This script acts as the main entry point for the automation pipeline.
It handles setting up the Python environment and running the glue script
which fetches API data, generates ML predictions, and saves to Azure Table Storage.

This script can be executed locally on a schedule or hosted inside an
Azure Function App (PowerShell worker).
#>

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonScript = Join-Path $ScriptDir "run_pipeline.py"

Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] [INFO] Starting KickCast PowerShell Orchestrator..."

try {
    # 1. Locate and activate the Python virtual environment
    $VenvPath = Join-Path $ProjectRoot ".venv"
    if (-Not (Test-Path $VenvPath)) {
        throw "Virtual environment not found at $VenvPath. Please run setup first."
    }

    # Determine activation script based on OS
    if ($IsWindows) {
        $PythonExecutable = Join-Path $VenvPath "Scripts" "python.exe"
    } else {
        $PythonExecutable = Join-Path $VenvPath "bin" "python"
    }

    if (-Not (Test-Path $PythonExecutable)) {
        throw "Python executable not found at $PythonExecutable."
    }

    Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] [INFO] Using Python at $PythonExecutable"

    # 2. Run the pipeline
    Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] [INFO] Executing Python Pipeline Script..."
    
    # Run python script and pipe output to host so we can see it
    & $PythonExecutable $PythonScript
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "Python script failed with exit code $exitCode"
    }

    Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] [INFO] Pipeline executed successfully."
}
catch {
    Write-Error "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] [ERROR] Pipeline failed: $_"
    exit 1
}
