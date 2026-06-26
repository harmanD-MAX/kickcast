param($Timer)

$TriggerTime = Get-Date
Write-Output "PowerShell Timer trigger function started at: $TriggerTime"

if ($Timer.IsPastDue) {
    Write-Output "PowerShell Timer is running late!"
}

# Resolve the path to our main pipeline orchestrator
$FuncAppDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$ProjectRoot = Split-Path -Parent $FuncAppDir
$PipelineScript = Join-Path $ProjectRoot "scripts" "pipeline.ps1"

Write-Output "Invoking Pipeline: $PipelineScript"

try {
    # Call the pipeline
    & $PipelineScript
    Write-Output "Pipeline execution finished successfully."
} catch {
    Write-Error "Pipeline execution failed: $_"
    throw
}
