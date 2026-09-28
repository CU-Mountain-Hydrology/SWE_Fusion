$PythonExe   = "C:\Users\olaf\anaconda3\python.exe"
$ScriptPath  = "C:\Users\olaf\PycharmProjects\SWE_Fusion_Auto\run\scheduled_run.py"
$TaskName    = "SWE_Fusion_Daily_Run"

schtasks /create `
  /tn $TaskName `
  /tr "`"$PythonExe`" `"$ScriptPath`"" `
  /sc daily `
  /st 09:00 `
  /rl highest `
  /f

Write-Host "Registered scheduled task '$TaskName' to run daily at 09:00."
Write-Host "View/edit it anytime in Task Scheduler, or remove with:"
Write-Host "  schtasks /delete /tn `"$TaskName`" /f"