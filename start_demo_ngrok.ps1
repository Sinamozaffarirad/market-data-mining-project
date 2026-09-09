param(
    [string]$Domain = "composed-underline-panama.ngrok-free.dev"
)

# Serves the site on a fixed public address through ngrok.
#
# Unlike the Cloudflare quick tunnel, the address here does not change between
# runs, so the link can be sent out before the session. It needs an ngrok
# account: sign in once with the authtoken, reserve the domain in the ngrok
# dashboard, then pass it to this script or set NGROK_DOMAIN.

$ErrorActionPreference = 'Stop'
$root    = 'C:\UNI\BA Final Project'
$project = Join-Path $root 'Project_Clean\Website\market'
$python  = 'C:\Users\sinam\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$ngrok   = Join-Path $root 'tools\ngrok.exe'
# The credential lives beside the script rather than under LOCALAPPDATA, whose
# contents are not visible from every shell on this machine. Kept out of git by
# .gitignore.
$ngrokConfig = Join-Path $root 'tools\ngrok.yml'
$port    = 8000

if (-not $Domain) { $Domain = $env:NGROK_DOMAIN }
if (-not $Domain) {
    Write-Host 'No domain given.' -ForegroundColor Red
    Write-Host 'Run:  .\start_demo_ngrok.ps1 -Domain your-name.ngrok-free.app'
    exit 1
}

# Read through .NET rather than Test-Path and Select-String, which a profile is
# free to shadow with its own aliases.
$configExists = [System.IO.File]::Exists($ngrokConfig)
$configHasToken = $false
if ($configExists) {
    $configHasToken = [System.IO.File]::ReadAllText($ngrokConfig).Contains('authtoken')
}
if (-not $configHasToken) {
    Write-Host "No ngrok credential found." -ForegroundColor Red
    Write-Host "  path   : $ngrokConfig"
    Write-Host "  exists : $configExists"
    Write-Host "  token  : $configHasToken"
    Write-Host 'Add it once with:' -ForegroundColor Yellow
    Write-Host '  .\tools\ngrok.exe config add-authtoken YOUR_TOKEN'
    exit 1
}

$env:DJANGO_DEBUG                = '0'
$env:DJANGO_ALLOWED_HOSTS        = "localhost,127.0.0.1,[::1],$Domain"
$env:DJANGO_CSRF_TRUSTED_ORIGINS = "https://$Domain"

$keyFile = Join-Path $root '.demo_secret_key'
if (-not (Test-Path $keyFile)) {
    & $python -c "from django.core.management.utils import get_random_secret_key as k; print(k())" |
        Out-File -FilePath $keyFile -Encoding ascii -NoNewline
}
$env:DJANGO_SECRET_KEY = (Get-Content $keyFile -Raw).Trim()

# Anything left from a previous run is cleared first, so the same command works
# every time without tidying up by hand. The free plan allows one agent, and a
# server still holding port 8000 would stop the new one binding.
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

Write-Host ''
Write-Host '=== starting the site ===' -ForegroundColor Cyan
$server = Start-Process -FilePath $python `
    -ArgumentList "`"$(Join-Path $root 'tools\serve.py')`"","`"$project`"","$port" `
    -WorkingDirectory $project -PassThru -WindowStyle Minimized

Write-Host 'waiting for it to load (this takes up to a minute)...' -NoNewline
$ready = $false
foreach ($i in 1..60) {
    Start-Sleep -Seconds 2
    Write-Host '.' -NoNewline
    if ($server.HasExited) { break }
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', $port)
        $c.Close()
        $ready = $true
        break
    } catch { }
}
Write-Host ''
if (-not $ready) {
    Write-Host 'the site did not start. Check that SQL Server is running.' -ForegroundColor Red
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    exit 1
}
Write-Host "the site is up on http://127.0.0.1:$port" -ForegroundColor Green

Write-Host ''
Write-Host '=== opening the public address ===' -ForegroundColor Cyan
$errLog = Join-Path $root 'tools\ngrok.err'
$outLog = Join-Path $root 'tools\ngrok.log'

# The config path is given outright. ngrok looks under LOCALAPPDATA otherwise,
# which is not the same folder for an elevated shell as for an ordinary one,
# and the agent then starts with no credential at all.
$tunnel = Start-Process -FilePath $ngrok `
    -ArgumentList 'http',"127.0.0.1:$port",'--url',$Domain,'--config',"`"$ngrokConfig`"",'--log','stdout' `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 8

# ngrok exits rather than waiting when it cannot claim the address. The free
# plan allows one agent at a time, so a copy left running from an earlier
# attempt is the usual cause; its own log says which.
if ($tunnel.HasExited) {
    Write-Host ''
    Write-Host 'The tunnel closed straight away. ngrok reported:' -ForegroundColor Red
    $log = $errLog
    if (Test-Path $log) {
        Get-Content $log | Where-Object { $_ -match 'ERR_NGROK|ERROR' } |
            Select-Object -First 4 | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }
    Write-Host ''
    Write-Host 'If it mentions a session limit, close the other ngrok first:' -ForegroundColor Yellow
    Write-Host '  Stop-Process -Name ngrok -Force'
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    exit 1
}

Write-Host ''
Write-Host '======================================================='
Write-Host "  https://$Domain" -ForegroundColor Yellow
Write-Host '======================================================='
Write-Host ''
Write-Host 'This address stays the same every time you run this script.'
Write-Host 'Add /analysis/ for the main page.'
Write-Host ''
Write-Host 'Leave this window open. Press Ctrl+C here to stop both.' -ForegroundColor DarkGray

try {
    Wait-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
} finally {
    foreach ($p in @($tunnel, $server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Write-Host 'stopped.'
}
