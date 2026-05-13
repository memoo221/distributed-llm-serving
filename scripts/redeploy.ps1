# Redeploy Thunder GPU workers across both instances.
#
# What it does, in order:
#   1. Validates cloudflared tunnels and that both masters are reachable
#   2. tnr scp's thunder_worker.py + launch_workers.sh to each instance
#   3. Pipes launch_workers.sh through `tnr connect` with the env vars below
#      (kill old workers -> start gpu0 -> wait -> start gpu1 -> wait)
#   4. Verifies all 4 workers are visible to both masters
#
# Edit the $Config block below when:
#   - cloudflared trycloudflare URLs change (every cloudflared restart)
#   - you want a different BATCH_SIZE
#   - you want a different model
#   - your HF token rotates
#   - you provision new Thunder instances

# Pass -Force to redeploy even when workers are already healthy (e.g.
# after editing thunder_worker.py, you must force).
param([switch]$Force)

# -- EDIT ME -------------------------------------------------------------
$Config = @{
    HfToken    = "hf_***************"
    Master1Url = "https://chosen-launched-warner-venture.trycloudflare.com"
    Master2Url = "https://david-tree-settlement-irc.trycloudflare.com"
    BatchSize  = 64
    ModelName  = "meta-llama/Llama-3.1-8B-Instruct"
    # Each instance runs ONE worker (on cuda:0) and registers to the named
    # master. Symmetric: each master ends up with 1 GPU worker + 1 CPU worker.
    # We previously ran 2 workers per instance (cuda:0 + cuda:1) but Thunder's
    # virtualization stalls the second process — see launch_workers.sh notes.
    Instances = @(
        @{ Id = 0; Uuid = "gaxwh8fq"; WorkerPrefix = "thunder_a"; MasterTarget = "master1" },
        @{ Id = 1; Uuid = "xcidc8hb"; WorkerPrefix = "thunder_b"; MasterTarget = "master2" }
    )
}
# ------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
Set-Location $ProjectRoot

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Write-Ok([string]$msg) {
    Write-Host "    OK: $msg" -ForegroundColor Green
}

function Write-Fail([string]$msg) {
    Write-Host "    FAIL: $msg" -ForegroundColor Red
    throw $msg
}

function Invoke-RemoteBash([int]$instanceId, [hashtable]$envVars, [string]$scriptPath) {
    # Thunder runs SSH on non-standard ports per instance (e.g. :32004) and
    # provisions keys just-in-time via their API on every `tnr connect` call.
    # Raw ssh.exe to port 22 can't replicate that. So we pipe the bash command
    # via stdin into `tnr connect <id>`, which spawns an interactive shell;
    # the shell reads our piped commands, runs them, then exits.
    # Skip empty values — bash treats KEY='' as set-but-empty, which Python's
    # os.getenv(KEY, default) returns "" for (not the default). That just bit
    # us with MODEL_NAME='' producing an "invalid repo id" tokenizer crash.
    $envBlock = (
        $envVars.GetEnumerator() |
            Where-Object { -not [string]::IsNullOrEmpty([string]$_.Value) } |
            ForEach-Object { "$($_.Key)='$($_.Value)'" }
    ) -join " "
    $remoteCmd = "$envBlock bash $scriptPath"
    # `exit` on its own line so the SSH session terminates cleanly afterwards.
    $stdin = "$remoteCmd`nexit`n"
    Write-Host "    tnr connect $instanceId  <- piped launch_workers.sh ($($envVars.Count) env vars)"
    $stdin | & tnr connect $instanceId
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "tnr connect $instanceId returned exit $LASTEXITCODE"
    }
}

# 1. Sanity check tunnels and masters
Write-Step "Verifying cloudflared tunnels and masters"
foreach ($pair in @(@{Url=$Config.Master1Url; Name="master1"}, @{Url=$Config.Master2Url; Name="master2"})) {
    try {
        $r = Invoke-RestMethod -Uri "$($pair.Url)/scheduler/workers" -TimeoutSec 5
        if ($r.master_id -ne $pair.Name) {
            Write-Fail "$($pair.Url) responded but master_id is $($r.master_id), expected $($pair.Name)"
        }
        Write-Ok "$($pair.Name) reachable via $($pair.Url)"
    } catch {
        Write-Fail "$($pair.Name) ($($pair.Url)) not reachable: $($_.Exception.Message)"
    }
}

# 1a. Lock down Thunder SSH key permissions. After a snapshot restore (or a
# fresh `tnr` provision) the key files in ~/.thunder/keys land with broad
# Windows ACLs, OpenSSH refuses to use them ("UNPROTECTED PRIVATE KEY FILE"),
# and silently falls back to a password prompt — which Start-Job has no way
# to answer, so the parallel scp block hangs forever with no output.
Write-Step "Locking down Thunder SSH key permissions"
# After a snapshot restore (or fresh `tnr` provision) the key files in
# ~/.thunder/keys land with broad Windows ACLs, OpenSSH refuses to use
# them ("UNPROTECTED PRIVATE KEY FILE"), and silently falls back to a
# password prompt — which Start-Job has no way to answer, so the
# parallel scp block hangs forever with no output.
#
# Use the running user's SID (not $env:USERNAME) so we grant the actual
# token, not a name that might not resolve to it (e.g. profile imports
# leaving $env:USERNAME with a stray backslash that mangles the ACE and
# locks the file to a non-existent principal).
$mySid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
foreach ($inst in $Config.Instances) {
    $keyPath = Join-Path $env:USERPROFILE ".thunder\keys\$($inst.Uuid)"
    if (-not (Test-Path $keyPath)) {
        Write-Host "    skip: $keyPath not found (will be provisioned on next tnr call)"
        continue
    }
    # Try /reset first; if it fails (broken ACL → not owner), takeown
    # then retry. takeown emits a stderr error when you already own the
    # file, which PowerShell promotes to a terminating NativeCommandError
    # under $ErrorActionPreference=Stop — so we only call it on demand,
    # and even then redirect stderr to a file we discard rather than 2>&1.
    & icacls $keyPath /reset *> $null
    if ($LASTEXITCODE -ne 0) {
        cmd /c "takeown /F `"$keyPath`" >nul 2>&1"
        & icacls $keyPath /reset *> $null
    }
    & icacls $keyPath /inheritance:r *> $null
    & icacls $keyPath /grant:r "*${mySid}:(R)" *> $null
    Write-Ok "$($inst.Uuid) key permissions locked"
}

# 1a-bis. Ensure port 8000 is forwarded on each instance. Thunder does
# NOT auto-forward ports on new/restored instances, so without this the
# master's `/generate` callbacks land on Thunder's "Nothing running here"
# 404 page instead of the worker. `tnr ports forward` is idempotent —
# re-running it on an already-forwarded port is a no-op.
Write-Step "Ensuring port 8000 is forwarded on both instances"
# tnr writes "Fetching instances..." to stderr even on success, which
# PowerShell promotes to NativeCommandError under ErrorActionPreference=Stop.
# cmd /c suppresses stderr at the OS level, before PowerShell sees it.
foreach ($inst in $Config.Instances) {
    cmd /c "tnr ports forward $($inst.Id) --add 8000 -y >nul 2>&1"
    Write-Ok "$($inst.Uuid) port 8000 forwarded"
}

# 1b. Fast-path: if all thunder workers are already heartbeating with the
# right BATCH_SIZE, skip the entire redeploy — saves ~3 min when nothing
# actually changed. Pass -Force to override (e.g. after editing
# thunder_worker.py).
# Polls via cloudflared URLs (not localhost) so this script works from any
# machine, including a "Thunder admin" laptop separate from the docker host.
Write-Step "Checking if workers are already healthy"
$expectedWorkers = @{}
foreach ($inst in $Config.Instances) {
    $expectedWorkers["$($inst.WorkerPrefix)_gpu0"] = $true
}
$alreadyHealthy = $true
foreach ($url in @($Config.Master1Url, $Config.Master2Url)) {
    try {
        $r = Invoke-RestMethod -Uri "$url/scheduler/workers" -TimeoutSec 5
        $thunder = @($r.workers | Where-Object {
            $_.worker_id -like "thunder_*" -and
            $_.last_seen_sec_ago -lt 15 -and
            -not $_.in_cooldown -and
            $_.slots -eq $Config.BatchSize -and
            $expectedWorkers.ContainsKey($_.worker_id)
        })
        # Each master should have exactly 1 thunder worker after this redeploy.
        if ($thunder.Count -lt 1) {
            $alreadyHealthy = $false
            break
        }
    } catch {
        $alreadyHealthy = $false
        break
    }
}
if ($alreadyHealthy) {
    Write-Ok "both thunder workers live with slots=$($Config.BatchSize) - skipping redeploy"
    Write-Host ""
    Write-Host "Pass -Force to redeploy anyway (e.g. after editing thunder_worker.py)" -ForegroundColor Yellow
    if (-not $Force) {
        exit 0
    }
    Write-Host "(forcing redeploy)" -ForegroundColor Yellow
} else {
    Write-Host "    workers not healthy or wrong config - proceeding with redeploy"
}

# 2. Normalize line endings on the bash script so it doesn't choke on Thunder
#    if your local checkout has CRLF (default on Windows + git autocrlf=true).
$shPath = Join-Path $ProjectRoot "workers\launch_workers.sh"
$raw = [IO.File]::ReadAllText($shPath)
if ($raw -match "`r`n") {
    Write-Step "Normalizing launch_workers.sh to LF endings"
    $lf = $raw -replace "`r`n", "`n"
    [IO.File]::WriteAllText($shPath, $lf)
    Write-Ok "converted CRLF -> LF"
}

# 3. Push files + run launch script on each instance IN PARALLEL via Start-Job.
# Each instance is independent (scp + tnr connect both block on network/SSH),
# so running them as background jobs roughly halves wall-clock time.
Write-Step "Pushing + launching on both instances IN PARALLEL"
$jobs = @()
foreach ($inst in $Config.Instances) {
    $jobs += Start-Job -ArgumentList $inst, $Config, $ProjectRoot -ScriptBlock {
        param($inst, $cfg, $root)
        # Don't let tnr's stderr ("Fetching instances...") trip the parent's
        # ErrorActionPreference=Stop. Inherit-and-override.
        $ErrorActionPreference = "Continue"
        Set-Location $root
        function Tstamp { (Get-Date).ToString("HH:mm:ss.fff") }
        Write-Output "[$(Tstamp)][inst $($inst.Id)] JOB START"
        Write-Output "[$(Tstamp)][inst $($inst.Id)] tnr scp thunder_worker.py..."
        & tnr scp workers/thunder/thunder_worker.py "$($inst.Id):" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "scp thunder_worker.py failed on $($inst.Id)" }
        Write-Output "[$(Tstamp)][inst $($inst.Id)] tnr scp thunder_requirements.txt..."
        # Needed by launch_workers.sh's `pip install -r ...` step. On a fresh
        # instance (no snapshot) the worker won't start without this — uvicorn,
        # transformers, etc. aren't preinstalled.
        & tnr scp workers/thunder/thunder_requirements.txt "$($inst.Id):" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "scp thunder_requirements.txt failed on $($inst.Id)" }
        Write-Output "[$(Tstamp)][inst $($inst.Id)] tnr scp launch_workers.sh..."
        & tnr scp workers/launch_workers.sh "$($inst.Id):" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "scp launch_workers.sh failed on $($inst.Id)" }

        $envVars = @{
            HF_TOKEN      = $cfg.HfToken
            MASTER1_URL   = $cfg.Master1Url
            MASTER2_URL   = $cfg.Master2Url
            UUID_PREFIX   = $inst.Uuid
            WORKER_PREFIX = $inst.WorkerPrefix
            BATCH_SIZE    = $cfg.BatchSize
            MODEL_NAME    = $cfg.ModelName
            MASTER_TARGET = $inst.MasterTarget
        }
        $envBlock = (
            $envVars.GetEnumerator() |
                Where-Object { -not [string]::IsNullOrEmpty([string]$_.Value) } |
                ForEach-Object { "$($_.Key)='$($_.Value)'" }
        ) -join " "
        $stdin = "$envBlock bash ~/launch_workers.sh`nexit`n"
        Write-Output "[$(Tstamp)][inst $($inst.Id)] tnr connect..."
        $stdin | & tnr connect $inst.Id 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "tnr connect $($inst.Id) returned exit $LASTEXITCODE" }
        Write-Output "[$(Tstamp)][inst $($inst.Id)] JOB DONE"
    }
}
# Poll for live progress: drain each job's buffered Write-Output every
# second so we can see which step (scp / tnr connect) each instance is
# on. Without this, Wait-Job buffers all output until both jobs finish,
# making the parallel block look hung even when it's working fine.
while ($jobs | Where-Object { $_.State -eq "Running" }) {
    foreach ($j in $jobs) {
        $out = Receive-Job $j  # drains (no -Keep); each line printed once
        foreach ($line in $out) { Write-Host "    $line" }
    }
    Start-Sleep -Seconds 1
}
# Final drain after all jobs settled.
foreach ($j in $jobs) {
    $out = Receive-Job $j
    foreach ($line in $out) { Write-Host "    $line" }
    if ($j.State -eq "Failed") {
        $err = $j.ChildJobs[0].JobStateInfo.Reason.Message
        $jobs | Remove-Job -Force
        Write-Fail "instance launch failed: $err"
    }
}
$jobs | Remove-Job
Write-Ok "both instances launched"

# 4. Wait for workers to heartbeat. Polls master registries from the laptop
# (no extra Thunder-side overhead). Exits as soon as all 4 thunder workers
# show fresh heartbeats with the right slots, or fails after 6 min.
Write-Step "Waiting for both thunder workers to heartbeat (polling masters)"
# Map each expected worker to the cloudflared URL of the master it heartbeats
# to. Polling via the public URL (not http://localhost:7001) means this
# script can run from any machine, including a "Thunder admin" laptop
# separate from the docker host.
$expectedWorkers = @{}
foreach ($inst in $Config.Instances) {
    $url = if ($inst.MasterTarget -eq "master2") { $Config.Master2Url } else { $Config.Master1Url }
    $expectedWorkers["$($inst.WorkerPrefix)_gpu0"] = $url
}

$deadline = (Get-Date).AddSeconds(360)
$pollInterval = 5
$lastStatus = ""
while ((Get-Date) -lt $deadline) {
    $allHealthy = $true
    $statuses = @()
    foreach ($wid in $expectedWorkers.Keys | Sort-Object) {
        $url = $expectedWorkers[$wid]
        try {
            $r = Invoke-RestMethod -Uri "$url/scheduler/workers" -TimeoutSec 5
            $w = $r.workers | Where-Object { $_.worker_id -eq $wid }
            if ($w -and $w.last_seen_sec_ago -lt 10 -and -not $w.in_cooldown -and $w.slots -eq $Config.BatchSize) {
                $statuses += "[OK] $wid (last_seen=$([math]::Round($w.last_seen_sec_ago,1))s slots=$($w.slots))"
            } else {
                $allHealthy = $false
                if ($w) {
                    $statuses += "[..] $wid (last_seen=$([math]::Round($w.last_seen_sec_ago,1))s slots=$($w.slots))"
                } else {
                    $statuses += "[--] $wid (not yet in registry)"
                }
            }
        } catch {
            $allHealthy = $false
            $statuses += "[!!] $wid - $url unreachable"
        }
    }
    $statusLine = $statuses -join "  "
    if ($statusLine -ne $lastStatus) {
        Write-Host "    $statusLine"
        $lastStatus = $statusLine
    }
    if ($allHealthy) {
        Write-Ok "both thunder workers heartbeating with slots=$($Config.BatchSize)"
        break
    }
    Start-Sleep $pollInterval
}
if (-not $allHealthy) {
    Write-Host "    FAIL: timeout - some workers never heartbeated. Final state above." -ForegroundColor Red
    Write-Host "    Check worker logs:  `"tail -n 30 ~/gpu0.log; tail -n 30 ~/gpu1.log; exit`" | tnr connect <id>" -ForegroundColor Yellow
    Write-Fail "redeploy timed out waiting for workers"
}

Write-Step "Done"
Write-Host "Both Thunder GPU workers running with BATCH_SIZE=$($Config.BatchSize) (1 per master)" -ForegroundColor Green
Write-Host "Run a load test from http://localhost:8050"
