# ---------------------------------------------------------------------------
# Runs the site and opens a public HTTPS address for it, for a live demonstration.
#
# The address changes every run, so both Django settings that depend on it are
# given wildcards rather than a fixed host: ALLOWED_HOSTS accepts any
# trycloudflare.com subdomain, and CSRF_TRUSTED_ORIGINS accepts it over HTTPS.
# Without the second one every analysis request fails as a CSRF error while
# ordinary pages keep loading, which is a confusing way to fail mid-demonstration.
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'
$root       = 'C:\UNI\BA Final Project'
$project    = Join-Path $root 'Project_Clean\Website\market'
$python     = 'C:\Users\sinam\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$cloudflared= Join-Path $root 'tools\cloudflared.exe'
$port       = 8000

# DEBUG stays off so a stack trace cannot reach the browser; --insecure keeps
# the static files served anyway, which runserver otherwise stops doing.
$env:DJANGO_DEBUG                 = '0'
$env:DJANGO_ALLOWED_HOSTS         = 'localhost,127.0.0.1,[::1],.trycloudflare.com'
$env:DJANGO_CSRF_TRUSTED_ORIGINS  = 'https://*.trycloudflare.com'

# A fixed key keeps anyone logged in if the server is restarted mid-session.
$keyFile = Join-Path $root '.demo_secret_key'
if (-not (Test-Path $keyFile)) {
    & $python -c "from django.core.management.utils import get_random_secret_key as k; print(k())" | Out-File -FilePath $keyFile -Encoding ascii -NoNewline
}
$env:DJANGO_SECRET_KEY = (Get-Content $keyFile -Raw).Trim()

Write-Host ''
Write-Host '=== starting the site ===' -ForegroundColor Cyan
$server = Start-Process -FilePath $python `
    -ArgumentList 'manage.py','runserver',"$port",'--insecure','--noreload' `
    -WorkingDirectory $project -PassThru -WindowStyle Minimized

# Loading pandas, scikit-learn and xgboost takes the best part of a minute on a
# cold start, so this waits for the port to answer rather than assuming a fixed
# delay. A plain TCP connect is the test: any HTTP status means Django is up,
# and a redirect to the login page is the normal answer here.
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
# cloudflared prints the address on its error stream and this subcommand has no
# --logfile option, so the stream is redirected to a file and read from there.
$log = Join-Path $root 'tools\tunnel.log'
if (Test-Path $log) { Remove-Item $log -Force }
$tunnel = Start-Process -FilePath $cloudflared `
    -ArgumentList 'tunnel','--url',"http://127.0.0.1:$port" `
    -RedirectStandardError $log -RedirectStandardOutput (Join-Path $root 'tools\tunnel.out') `
    -PassThru -WindowStyle Minimized

$url = $null
foreach ($i in 1..40) {
    Start-Sleep -Seconds 2
    if (Test-Path $log) {
        $m = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' |
             Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value; break }
    }
}

Write-Host ''
if ($url) {
    Write-Host '======================================================='
    Write-Host "  $url" -ForegroundColor Yellow
    Write-Host '======================================================='
    Write-Host ''
    Write-Host 'Open that address on any computer. Add /analysis/ for the main page.'
    Set-Content -Path (Join-Path $root 'tools\current_url.txt') -Value $url -Encoding ascii
} else {
    Write-Host 'the address did not appear; see tools\tunnel.log' -ForegroundColor Red
}

Write-Host ''
Write-Host 'Leave this window open. Press Ctrl+C here to stop both.' -ForegroundColor DarkGray
try {
    Wait-Process -Id $tunnel.Id
} finally {
    foreach ($p in @($tunnel, $server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Write-Host 'stopped.'
}
