#Requires -Version 5.1
<#
Installs the latest gollama release for Windows and (if run interactively)
launches it. Usage:
  irm https://raw.githubusercontent.com/blaketylerfullerton/gollama/main/install.ps1 | iex
#>

$ErrorActionPreference = "Stop"

$Repo = "blaketylerfullerton/gollama"
$BinName = "gollama"

switch ($env:PROCESSOR_ARCHITECTURE) {
    "AMD64" { $GoArch = "amd64" }
    "ARM64" { $GoArch = "arm64" }
    default { throw "unsupported architecture: $($env:PROCESSOR_ARCHITECTURE)" }
}

Write-Host "Looking up the latest release of $Repo..."
$release = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repo/releases/latest"
$tag = $release.tag_name
if (-not $tag) { throw "could not determine the latest release" }
$version = $tag -replace '^v', ''

$asset = "${BinName}_${version}_windows_${GoArch}.zip"
$url = "https://github.com/$Repo/releases/download/$tag/$asset"

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $zipPath = Join-Path $tmp $asset
    Write-Host "Downloading $asset ($tag)..."
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zipPath

    Expand-Archive -Path $zipPath -DestinationPath $tmp -Force
    $exePath = Join-Path $tmp "$BinName.exe"
    if (-not (Test-Path $exePath)) { throw "archive didn't contain a '$BinName.exe' binary" }

    $installDir = Join-Path $env:LOCALAPPDATA "Programs\gollama"
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    $dest = Join-Path $installDir "$BinName.exe"
    Copy-Item -Path $exePath -Destination $dest -Force

    Write-Host "Installed $BinName $version to $dest"

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) { $userPath = "" }
    if (($userPath -split ';') -notcontains $installDir) {
        $newPath = if ($userPath) { "$userPath;$installDir" } else { $installDir }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        $env:Path += ";$installDir"
        Write-Host "Added $installDir to your user PATH (open a new terminal for it to take effect elsewhere)."
    }

    if ([Environment]::UserInteractive) {
        & $dest
    } else {
        Write-Host "Run '$BinName' to get started."
    }
} finally {
    Remove-Item -Path $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
