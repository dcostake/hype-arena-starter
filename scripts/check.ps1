# scripts/check.sh for Windows (PowerShell 5.1 or 7, Docker Desktop): builds the image from the
# Dockerfile and runs it as the arena will - offline, as user 10001, on a read-only root, with
# 2 CPUs and 6 GiB, until it exits or the minutes pass - then checks its table.
#
#   powershell -ExecutionPolicy Bypass -File scripts\check.ps1             # build the image and run it
#   powershell -ExecutionPolicy Bypass -File scripts\check.ps1 my-image    # run an image you built
#   $env:MINUTES = 5; powershell -ExecutionPolicy Bypass -File scripts\check.ps1   # a shorter run
#
# It does the same as check.sh, which needs sh and a working python3: on Windows, python3 is often
# the Microsoft Store alias. This one runs Python as `py`, or as `python` where there is no `py`.
param([string]$Image = "")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$minutes = 30
if ($env:MINUTES) { $minutes = [int]$env:MINUTES }
$name = "hype-arena-check-$PID"
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("hype-arena-" + [guid]::NewGuid().ToString("N"))

$python = "python"
if (Get-Command py -ErrorAction SilentlyContinue) { $python = "py" }

function Remove-Run {
    docker rm -f $name 2>$null | Out-Null
    if (Test-Path $work) { Remove-Item -Recurse -Force $work }
}

try {
    if (-not $Image) {
        $Image = "hype-arena-check"
        docker build --platform linux/amd64 -t $Image $root
        if ($LASTEXITCODE -ne 0) { throw "the image did not build" }
    }
    $user = (docker image inspect $Image --format "{{.Config.User}}").Trim()
    $uid = ($user -split ":")[0]
    if ($uid -eq "" -or $uid -eq "root" -or $uid -eq "0") {
        Write-Host "refused: the image's USER is '$user'; end the Dockerfile with USER 10001"
        exit 1
    }

    New-Item -ItemType Directory -Force (Join-Path $work "universe") | Out-Null
    New-Item -ItemType Directory -Force (Join-Path $work "output") | Out-Null
    & $python (Join-Path $root "scripts\input.py") (Join-Path $root "data\practice.csv") (Join-Path $work "universe\universe.csv")
    if ($LASTEXITCODE -ne 0) { throw "scripts\input.py failed" }
    Copy-Item (Join-Path $root "data\lexicon.json") (Join-Path $work "universe\lexicon.json")

    $deadline = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + $minutes * 60
    docker run -d --name $name --platform linux/amd64 `
        --network none --read-only --user 10001:10001 `
        --cpus 2 --memory 6g --memory-swap 6g --pids-limit 256 `
        --cap-drop ALL --security-opt no-new-privileges `
        --tmpfs /tmp:rw,size=128m,mode=1777 `
        -v "$(Join-Path $work 'universe'):/universe:ro" -v "$(Join-Path $work 'output'):/output" `
        -e ARENA_DEADLINE=$deadline `
        $Image | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the container did not start" }
    Write-Host "running $Image for at most $minutes minutes"

    while ((docker inspect -f "{{.State.Running}}" $name) -eq "true") {
        if ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() -ge $deadline) {
            Write-Host "deadline: SIGTERM, and a kill 10 seconds later"
            docker stop -t 10 $name | Out-Null
            break
        }
        Start-Sleep -Seconds 2
    }
    docker logs --tail 20 $name 2>&1 | ForEach-Object { "  | $_" }
    $code = docker inspect -f "{{.State.ExitCode}}" $name
    $oom = docker inspect -f "{{.State.OOMKilled}}" $name
    Write-Host "exit code $code, OOM killed: $oom"

    & $python (Join-Path $root "scripts\check_output.py") (Join-Path $work "output") (Join-Path $root "data\practice.csv") (Join-Path $work "universe\universe.csv")
    exit $LASTEXITCODE
}
finally {
    Remove-Run
}
