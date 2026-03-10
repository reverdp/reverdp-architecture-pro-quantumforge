$ErrorActionPreference = "Stop"

$taskName   = "QuantumForgeDailyUpdateIndex6AM"
$workDir    = "D:\yandex_p\architecture-pro-quantumforge\bot"
$pythonExe  = "D:\yandex_p\architecture-pro-quantumforge\bot\venv\Scripts\python.exe"
$scriptPath = "D:\yandex_p\architecture-pro-quantumforge\bot\update_index.py"
$logPath    = "D:\yandex_p\architecture-pro-quantumforge\bot\scheduled_task.log"

$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$cmdArgs = "/c `"`"$pythonExe`" `"$scriptPath`" >> `"$logPath`" 2>&1`""

$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument $cmdArgs `
    -WorkingDirectory $workDir

$trigger = New-ScheduledTaskTrigger -Daily -At 6:00AM

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Daily execution of update_index.py at 06:00"

Write-Host "Task '$taskName' successfully registered to run daily at 6:00 AM."
Write-Host "Python: $pythonExe"
Write-Host "Script: $scriptPath"
Write-Host "Log: $logPath"