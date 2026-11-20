[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8100,
    [string]$ListenHost = "127.0.0.1",
    [switch]$Build,
    [switch]$NoBrowser,
    [switch]$Offline,
    [switch]$OfficialSources,
    [string]$Mirror = "",
    [switch]$PrepareOffline,
    [switch]$CheckOnly,
    [switch]$NonInteractive,
    [switch]$Help
)

# Keep this file ASCII-only. Windows PowerShell 5.1 reads a UTF-8 file without a
# BOM as the system code page, so non-ASCII source text is not portable.
Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$script:Root = [IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path))
$script:BackendDir = Join-Path $script:Root "backend"
$script:FrontendDir = Join-Path $script:Root "frontend"
$script:Requirements = Join-Path $script:BackendDir "requirements.txt"
$script:VenvDir = Join-Path $script:Root ".venv"
$script:VenvPython = Join-Path $script:VenvDir "Scripts\python.exe"
$script:RequirementStamp = Join-Path $script:VenvDir ".mancu-requirements.sha256"
$script:TranscriptStarted = $false

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Notice([string]$Message) { Write-Host "[--] $Message" -ForegroundColor Yellow }

function Wait-OnFailure {
    if ($NonInteractive -or $env:CI) { return }
    if ($Host.Name -eq "ConsoleHost") {
        try { [void](Read-Host "Press Enter to close") } catch { }
    }
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $script:Root,
        [switch]$AllowFailure
    )
    Push-Location -LiteralPath $WorkingDirectory
    try {
        # Send child stdout to the host instead of the function pipeline. If it
        # leaks into the pipeline, callers assigning `$code = Invoke-External`
        # receive an array of log lines plus the exit code and misread success.
        & $FilePath @Arguments | Out-Host
        $exitCode = $LASTEXITCODE
    }
    finally { Pop-Location }
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "Command failed with exit code $exitCode`: $FilePath $($Arguments -join ' ')"
    }
    return $exitCode
}

function Invoke-PythonCandidate {
    param($Python, [string[]]$Arguments, [switch]$Capture)
    $allArgs = @($Python.Prefix) + $Arguments
    if ($Capture) {
        # Windows PowerShell 5.1 turns native stderr into an ErrorRecord. With
        # ErrorActionPreference=Stop, probing an unavailable `py -3.x` would
        # abort the whole launcher instead of moving to the next candidate.
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & $Python.Command @allArgs 2>&1
            $exitCode = $LASTEXITCODE
        }
        catch {
            $output = @($_)
            $exitCode = 1
        }
        finally { $ErrorActionPreference = $oldPreference }
        return [pscustomobject]@{ ExitCode = $exitCode; Output = @($output) }
    }
    return Invoke-External -FilePath $Python.Command -Arguments $allArgs
}

function Find-BasePython {
    # Prefer the newest supported interpreter, independent of PATH order. The
    # Python launcher is standard on Windows and lets us request exact versions.
    $candidates = @()
    $pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($minor in @(14, 13, 12, 11)) {
            $candidates += [pscustomobject]@{
                Command = $pyLauncher.Source
                Prefix = @("-3.$minor")
                Rank = $minor
            }
        }
    }
    $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates += [pscustomobject]@{
            Command = $pythonCommand.Source
            Prefix = @()
            Rank = 0
        }
    }

    $usable = @()
    foreach ($candidate in $candidates) {
        $probe = Invoke-PythonCandidate -Python $candidate -Arguments @(
            "-c",
            "import struct,sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}|{struct.calcsize(chr(80))*8}')"
        ) -Capture
        if ($env:MANCU_LAUNCHER_DEBUG) {
            Write-Host "[debug] python candidate=$($candidate.Command) $($candidate.Prefix -join ' ') exit=$($probe.ExitCode) output=$($probe.Output -join ' | ')"
        }
        if ($probe.ExitCode -ne 0 -or $probe.Output.Count -eq 0) { continue }
        $facts = ([string]$probe.Output[-1]).Trim().Split("|")
        if ($facts.Count -ne 2) { continue }
        try {
            $version = [version]$facts[0]
            $bits = [int]$facts[1]
        }
        catch { continue }
        if ($version.Major -ne 3 -or $version.Minor -lt 11 -or $bits -ne 64) { continue }
        $usable += [pscustomobject]@{
            Command = $candidate.Command
            Prefix = $candidate.Prefix
            Version = $version
        }
    }
    if ($usable.Count -eq 0) {
        throw "No supported Python was found. Install 64-bit Python 3.11 or newer and enable 'Add python.exe to PATH'."
    }
    return $usable | Sort-Object -Property Version -Descending | Select-Object -First 1
}

function Test-VenvPython {
    if (-not (Test-Path -LiteralPath $script:VenvPython -PathType Leaf)) { return $false }
    & $script:VenvPython -c "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Assert-ChildPath([string]$Path, [string]$Parent) {
    $fullPath = [IO.Path]::GetFullPath($Path)
    $fullParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($fullParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the project: $fullPath"
    }
}

function Ensure-VirtualEnvironment {
    $basePython = Find-BasePython
    if (Test-VenvPython) {
        $version = [version]((& $script:VenvPython -c "import platform; print(platform.python_version())").Trim())
        if ($version -ge $basePython.Version) {
            Write-Ok "Project Python $version is ready in .venv"
            return
        }
        Write-Notice "A newer Python is available ($($basePython.Version) > $version); rebuilding .venv."
    }
    Write-Notice "Selected highest available Python $($basePython.Version): $($basePython.Command) $($basePython.Prefix -join ' ')"
    if (Test-Path -LiteralPath $script:VenvDir) {
        Assert-ChildPath -Path $script:VenvDir -Parent $script:Root
        Write-Notice "The existing .venv is incomplete; rebuilding it."
        Remove-Item -LiteralPath $script:VenvDir -Recurse -Force
    }
    Write-Step "Creating isolated Python environment"
    Invoke-PythonCandidate -Python $basePython -Arguments @("-m", "venv", $script:VenvDir) | Out-Null
    if (-not (Test-VenvPython)) {
        throw "Python created .venv, but .venv\Scripts\python.exe is not usable."
    }
    Write-Ok "Created .venv with Python $($basePython.Version)"
}

function Test-RequiredImports {
    $probe = @"
import fastapi, uvicorn, pydantic, httpx
import numpy, PIL, onnxruntime, tokenizers
from skimage.metrics import structural_similarity
"@
    & $script:VenvPython -c $probe 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Test-PipDependencyGraph {
    & $script:VenvPython -m pip check 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-RequirementHash {
    # Use .NET directly. Some remote/constrained Windows PowerShell hosts do not
    # auto-load Microsoft.PowerShell.Utility, so Get-FileHash may not exist even
    # though the engine reports version 5.1.
    $stream = [IO.File]::OpenRead($script:Requirements)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha.ComputeHash($stream)
        return ([BitConverter]::ToString($bytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}

function Test-RequirementStamp {
    if (-not (Test-Path -LiteralPath $script:RequirementStamp -PathType Leaf)) { return $false }
    return ((Get-Content -LiteralPath $script:RequirementStamp -Raw).Trim() -eq (Get-RequirementHash))
}

function Get-PipSources {
    if ($Mirror.Trim()) {
        return @([pscustomobject]@{ Name = "custom mirror"; Url = $Mirror.Trim() })
    }
    if ($OfficialSources) {
        return @([pscustomobject]@{ Name = "official PyPI"; Url = "" })
    }
    return @(
        [pscustomobject]@{ Name = "Tsinghua mirror"; Url = "https://pypi.tuna.tsinghua.edu.cn/simple" },
        [pscustomobject]@{ Name = "Aliyun mirror"; Url = "https://mirrors.aliyun.com/pypi/simple/" },
        [pscustomobject]@{ Name = "Tencent mirror"; Url = "https://mirrors.cloud.tencent.com/pypi/simple" },
        [pscustomobject]@{ Name = "official PyPI"; Url = "" }
    )
}

function Invoke-PipWithFallback([ValidateSet("install", "download")][string]$Action) {
    $common = @("-m", "pip", $Action, "--disable-pip-version-check", "--no-input", "--prefer-binary", "-r", $script:Requirements)
    if ($Action -eq "download") {
        $wheelDir = Join-Path $script:BackendDir "wheels"
        New-Item -ItemType Directory -Path $wheelDir -Force | Out-Null
        $common += @("-d", $wheelDir)
    }
    if ($Offline) {
        if ($Action -eq "download") { throw "-Offline and -PrepareOffline cannot be used together." }
        $wheelDir = Join-Path $script:BackendDir "wheels"
        if (-not (Test-Path -LiteralPath $wheelDir -PathType Container)) {
            throw "Offline installation requires backend\wheels. Run start.ps1 -PrepareOffline on a connected computer first."
        }
        Invoke-External -FilePath $script:VenvPython -Arguments ($common + @("--no-index", "--find-links", $wheelDir)) | Out-Null
        return
    }
    foreach ($source in (Get-PipSources)) {
        Write-Notice "Trying $($source.Name)..."
        $args = @($common)
        if ($source.Url) { $args += @("--index-url", $source.Url) }
        $code = Invoke-External -FilePath $script:VenvPython -Arguments $args -AllowFailure
        if ($code -eq 0) {
            Write-Ok "Python packages completed via $($source.Name)"
            return
        }
    }
    throw "Every Python package source failed. Check the network/proxy, or use -Mirror/-OfficialSources/-Offline."
}

function Ensure-PythonDependencies {
    if ($PrepareOffline) {
        Write-Step "Downloading Python wheels for offline installation"
        Invoke-PipWithFallback -Action "download"
        Write-Ok "Offline wheels are ready in backend\wheels"
        return
    }
    $complete = (Test-RequirementStamp) -and (Test-RequiredImports) -and (Test-PipDependencyGraph)
    if ($complete) {
        Write-Ok "All required Python packages and transitive dependencies are complete"
        return
    }
    Write-Step "Installing or repairing every Python dependency in requirements.txt"
    Invoke-PipWithFallback -Action "install"
    if (-not (Test-RequiredImports)) {
        throw "pip finished, but at least one required module still cannot be imported in .venv."
    }
    if (-not (Test-PipDependencyGraph)) {
        throw "pip finished, but 'pip check' found missing or incompatible transitive dependencies."
    }
    Set-Content -LiteralPath $script:RequirementStamp -Value (Get-RequirementHash) -Encoding ASCII
    Write-Ok "Dependency verification passed: direct imports and pip dependency graph are complete"
}

function Find-Tool([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    return ""
}

function Assert-NodeVersion([string]$NodeExe) {
    $raw = (& $NodeExe --version 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $raw) { throw "Node.js was found but could not run." }
    $major = [int](([string]$raw).Trim().TrimStart('v').Split('.')[0])
    if ($major -lt 20) { throw "Node.js 20 or newer is required to build the frontend; found $raw." }
}

function Install-FrontendDependencies([string]$NpmExe) {
    $cacheDir = Join-Path $script:FrontendDir ".npm-cache"
    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
    $registries = @(
        [pscustomobject]@{ Name = "npmmirror"; Url = "https://registry.npmmirror.com" },
        [pscustomobject]@{ Name = "Tencent mirror"; Url = "https://mirrors.cloud.tencent.com/npm/" },
        [pscustomobject]@{ Name = "official npm"; Url = "https://registry.npmjs.org" }
    )
    if ($OfficialSources) {
        $registries = @([pscustomobject]@{ Name = "official npm"; Url = "https://registry.npmjs.org" })
    }
    foreach ($registry in $registries) {
        Write-Notice "Trying $($registry.Name)..."
        $args = @("install", "--registry", $registry.Url, "--cache", $cacheDir, "--no-audit", "--no-fund", "--fetch-retries", "2", "--fetch-timeout", "60000")
        $code = Invoke-External -FilePath $NpmExe -Arguments $args -WorkingDirectory $script:FrontendDir -AllowFailure
        $esbuild = Join-Path $script:FrontendDir "node_modules\esbuild\package.json"
        if ($code -eq 0 -and (Test-Path -LiteralPath $esbuild -PathType Leaf)) {
            Write-Ok "Frontend packages completed via $($registry.Name)"
            return
        }
    }
    throw "Every npm registry failed. Configure npm proxy settings or check the network."
}

function Ensure-Frontend {
    $bundle = Join-Path $script:FrontendDir "dist\bundle.js"
    $esbuild = Join-Path $script:FrontendDir "node_modules\esbuild\package.json"
    $needsBuild = $Build -or -not (Test-Path -LiteralPath $bundle -PathType Leaf)
    if (-not $needsBuild) {
        Write-Ok "Prebuilt frontend is ready (use -Build after editing frontend source)"
        return
    }
    if ($Offline -and -not (Test-Path -LiteralPath $esbuild -PathType Leaf)) {
        throw "Offline frontend build requires frontend\node_modules, or a prebuilt frontend\dist\bundle.js."
    }
    $node = Find-Tool "node.exe"
    if (-not $node) { $node = Find-Tool "node" }
    if (-not $node) { throw "Node.js 20 or newer is required because the frontend must be built." }
    Assert-NodeVersion -NodeExe $node
    if (-not (Test-Path -LiteralPath $esbuild -PathType Leaf)) {
        $npm = Find-Tool "npm.cmd"
        if (-not $npm) { $npm = Find-Tool "npm" }
        if (-not $npm) { throw "npm was not found next to Node.js." }
        Write-Step "Installing frontend dependencies"
        Install-FrontendDependencies -NpmExe $npm
    }
    Write-Step "Building frontend"
    Invoke-External -FilePath $node -Arguments @("build.mjs") -WorkingDirectory $script:FrontendDir | Out-Null
    if (-not (Test-Path -LiteralPath $bundle -PathType Leaf)) {
        throw "Frontend build returned without creating frontend\dist\bundle.js."
    }
    Write-Ok "Frontend build completed"
}

function Get-ListeningProcessIds([int]$TargetPort) {
    $ids = @()
    if (Get-Command "Get-NetTCPConnection" -ErrorAction SilentlyContinue) {
        try { $ids = @(Get-NetTCPConnection -State Listen -LocalPort $TargetPort -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique) }
        catch { $ids = @() }
    }
    if ($ids.Count -eq 0) {
        foreach ($line in (& netstat.exe -ano -p tcp 2>$null)) {
            if ($line -match "^\s*TCP\s+\S+:$TargetPort\s+\S+\s+LISTENING\s+(\d+)\s*$") { $ids += [int]$Matches[1] }
        }
    }
    return @($ids | Select-Object -Unique)
}

function Clear-StaleMancuProcess([int]$TargetPort) {
    foreach ($processId in (Get-ListeningProcessIds -TargetPort $TargetPort)) {
        $commandLine = ""
        try { $commandLine = [string](Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction Stop).CommandLine }
        catch { }
        if ($commandLine -match "app\.serve" -and $commandLine -match "python") {
            Write-Notice "Stopping stale Mancu process PID $processId on port $TargetPort"
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Start-Sleep -Milliseconds 500
            continue
        }
        $name = "unknown"
        try { $name = (Get-Process -Id $processId -ErrorAction Stop).ProcessName } catch { }
        throw "Port $TargetPort is already used by PID $processId ($name). Use -Port with another value or stop that program."
    }
}

function Show-Usage {
    Write-Host @"
Mancu one-file setup and launcher

  powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1

Options:
  -Port 8200             Use another HTTP port
  -ListenHost 0.0.0.0   Listen on the network (secure it before exposing)
  -Build                 Rebuild the frontend
  -NoBrowser             Do not open a browser
  -Offline               Install Python packages from backend\wheels
  -PrepareOffline        Download backend\wheels, then exit
  -CheckOnly             Verify/setup everything, but do not start the server
  -OfficialSources       Skip mainland mirrors
  -Mirror <URL>          Use one custom pip source
  -NonInteractive        Never wait for Enter after an error
"@
}

if ($Help) { Show-Usage; exit 0 }

try {
    if ($PSVersionTable.PSVersion.Major -lt 5) { throw "PowerShell 5.1 or newer is required." }
    if (-not [Environment]::Is64BitOperatingSystem) { throw "Mancu requires 64-bit Windows." }
    if (-not (Test-Path -LiteralPath $script:Requirements -PathType Leaf)) {
        throw "backend\requirements.txt was not found. Keep start.ps1 in the project root."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $script:BackendDir "app\main.py") -PathType Leaf)) {
        throw "backend\app\main.py was not found. The project package is incomplete."
    }

    $logDir = Join-Path $script:Root "logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    try {
        Start-Transcript -Path (Join-Path $logDir "start-powershell.log") -Append | Out-Null
        $script:TranscriptStarted = $true
    }
    catch { }

    Write-Host "`n============================================================"
    Write-Host "  Mancu setup and launcher"
    Write-Host "  Project: $script:Root"
    Write-Host "============================================================"

    Ensure-VirtualEnvironment
    Ensure-PythonDependencies
    if ($PrepareOffline) { Write-Host "`nOffline package preparation finished."; exit 0 }
    Ensure-Frontend
    if ($CheckOnly) {
        Write-Ok "Environment check completed; server was not started"
        exit 0
    }
    Clear-StaleMancuProcess -TargetPort $Port

    if (-not $env:STUDIO_LLM_BASE_URL) { $env:STUDIO_LLM_BASE_URL = "https://api.deepseek.com/v1" }
    if (-not $env:STUDIO_LLM_MODEL) { $env:STUDIO_LLM_MODEL = "deepseek-chat" }
    if (-not $env:STUDIO_IMAGE_PROVIDER) { $env:STUDIO_IMAGE_PROVIDER = "atlas-cloud" }
    if (-not $env:STUDIO_MEDIA_MODE) { $env:STUDIO_MEDIA_MODE = "full" }
    if (-not $env:STUDIO_GATEWAY_SIZE) { $env:STUDIO_GATEWAY_SIZE = "small" }
    if (-not $env:STUDIO_GATEWAY_STEPS) { $env:STUDIO_GATEWAY_STEPS = "8" }
    if (-not $env:MANCU_MEDIA_BASE_URL) { $env:MANCU_MEDIA_BASE_URL = "http://127.0.0.1:3300" }
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONUNBUFFERED = "1"
    $env:STUDIO_DATA_DIR = Join-Path $script:BackendDir "userdata"
    $env:STUDIO_CONFIG_DIR = Join-Path $script:BackendDir "config"
    $env:STUDIO_PORT = [string]$Port
    $env:STUDIO_HOST = $ListenHost
    if (-not $NoBrowser -and $ListenHost -in @("127.0.0.1", "localhost")) { $env:MANCU_OPEN_BROWSER = "1" }
    else { Remove-Item Env:MANCU_OPEN_BROWSER -ErrorAction SilentlyContinue }

    if (-not $env:STUDIO_LLM_API_KEY) {
        Write-Notice "STUDIO_LLM_API_KEY is not set. The UI will start, but agents need a key configured in Settings."
    }
    if ($ListenHost -notin @("127.0.0.1", "localhost")) {
        Write-Notice "Listening on $ListenHost exposes the service to the network. Add firewall/authentication controls."
    }
    Write-Step "Starting Mancu"
    Write-Host "Open: http://127.0.0.1:$Port"
    Write-Host "Press Ctrl+C to stop.`n"
    Invoke-External -FilePath $script:VenvPython -Arguments @("-m", "app.serve", "--host", $ListenHost, "--port", [string]$Port) -WorkingDirectory $script:BackendDir | Out-Null
}
catch {
    Write-Host "`n[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Startup log: $script:Root\logs\start-powershell.log" -ForegroundColor DarkGray
    Wait-OnFailure
    exit 1
}
finally {
    if ($script:TranscriptStarted) { try { Stop-Transcript | Out-Null } catch { } }
}

