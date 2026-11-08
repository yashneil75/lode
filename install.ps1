# Install a lode standalone binary on Windows.
#
#   powershell -c "irm https://raw.githubusercontent.com/yashneil75/lode/main/install.ps1 | iex"
#
# Override with environment variables for custom hosting or a pinned version:
#   $env:LODE_VERSION, $env:LODE_REPO, $env:LODE_BASE_URL, $env:LODE_INSTALL_DIR

$ErrorActionPreference = "Stop"

$Repo = if ($env:LODE_REPO) { $env:LODE_REPO } else { "yashneil75/lode" }
$InstallRoot = if ($env:LODE_INSTALL_DIR) {
    $env:LODE_INSTALL_DIR
} else {
    Join-Path $env:LOCALAPPDATA "lode"
}

function Say($message) { Write-Host "  $message" }
function Die($message) {
    Write-Host "lode install: $message" -ForegroundColor Red
    exit 1
}

# --- platform -------------------------------------------------------------
$arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    "AMD64" { "x86_64" }
    "ARM64" { "arm64" }
    default { Die "no standalone build for $env:PROCESSOR_ARCHITECTURE; install with: pip install lode-code-search" }
}
Say "target windows-$arch"

# --- resolve the version --------------------------------------------------
$Version = $env:LODE_VERSION
if (-not $Version) {
    try {
        $latest = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
        $Version = $latest.tag_name
    } catch {
        Die "could not determine the latest release of $Repo; set `$env:LODE_VERSION"
    }
}
Say "version $Version"

$Base = if ($env:LODE_BASE_URL) {
    $env:LODE_BASE_URL
} else {
    "https://github.com/$Repo/releases/download/$Version"
}
$Asset = "lode-{0}-windows-{1}.zip" -f $Version.TrimStart("v"), $arch
$Url = "$Base/$Asset"

# --- download and verify --------------------------------------------------
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("lode-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Tmp | Out-Null
try {
    Say "downloading $Asset"
    Invoke-WebRequest -Uri $Url -OutFile (Join-Path $Tmp $Asset) -UseBasicParsing

    $SumsPath = Join-Path $Tmp "sha256sums.txt"
    try {
        Invoke-WebRequest -Uri "$Base/sha256sums.txt" -OutFile $SumsPath -UseBasicParsing
    } catch {
        Say "no sha256sums.txt at $Base; skipping verification"
    }

    if (Test-Path $SumsPath) {
        $line = Get-Content $SumsPath | Where-Object { $_ -match [regex]::Escape($Asset) } | Select-Object -First 1
        if (-not $line) { Die "$Asset is not listed in sha256sums.txt" }
        $expected = ($line -split '\s+')[0]
        $actual = (Get-FileHash (Join-Path $Tmp $Asset) -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $expected) { Die "checksum mismatch: $actual != $expected" }
        Say "checksum ok"
    }

    # --- install ----------------------------------------------------------
    # Unpack beside the old install first, then swap, so a failed download never
    # removes a working binary.
    $Stage = Join-Path $Tmp "stage"
    $Target = Join-Path $InstallRoot "lode"
    New-Item -ItemType Directory -Path $Stage | Out-Null
    Expand-Archive -Path (Join-Path $Tmp $Asset) -DestinationPath $Stage -Force
    if (-not (Test-Path (Join-Path $Stage "lode\lode.exe"))) { Die "archive did not contain lode\lode.exe" }

    New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
    $Backup = Join-Path $Tmp "previous"
    if (Test-Path $Target) {
        Move-Item $Target $Backup
        try {
            Move-Item (Join-Path $Stage "lode") $Target
        } catch {
            Move-Item $Backup $Target
            Die "could not move the new build into place"
        }
    } else {
        Move-Item (Join-Path $Stage "lode") $Target
    }
    Say "installed to $Target"

    # --- PATH -------------------------------------------------------------
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (($userPath -split ';') -notcontains $Target) {
        Say "adding $Target to your user PATH"
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$Target", "User")
        Say "reopen your terminal for it to take effect"
    }
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}

Write-Host ""
& (Join-Path $Target "lode.exe") version
if ($LASTEXITCODE -ne 0) { Die "the installed binary did not run" }
Say "next: lode index C:\path\to\your-project -n your-project"
