<#
.SYNOPSIS
    Turn dist\PDFtoWordOCR into a self-contained app (Tesseract + handwriting model
    inside the folder), self-test it with this PC's own installs hidden, and zip it.

.DESCRIPTION
    Run after .\build_exe.ps1. The resulting ZIP needs no installation on the target
    PC: extract it and double-click PDFtoWordOCR.exe. No internet is needed.

    - Tesseract is copied from its installed folder into dist\PDFtoWordOCR\tesseract
      (the app looks there first). Training tools and the uninstaller are left out.
    - The handwriting model is copied into dist\PDFtoWordOCR\models\trocr-base-handwritten
      (created with "python -m ocr_app.fetch_model --output ..." if missing).

.PARAMETER AppDir
    Built app folder to make portable. Default: dist\PDFtoWordOCR
.PARAMETER TesseractDir
    Tesseract folder to bundle. Default: the folder of the installed tesseract.exe.
.PARAMETER ModelDir
    Saved model folder. Default: models\trocr-base-handwritten in the project.
.PARAMETER ZipPath
    Output ZIP. Default: dist\PDFtoWordOCR-portable.zip
.PARAMETER NoZip
    Bundle and self-test only.
#>
[CmdletBinding()]
param(
    [string]$AppDir = "",
    [string]$TesseractDir = "",
    [string]$ModelDir = "",
    [string]$Python = "",
    [string]$ZipPath = "",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$dist = if ($AppDir) { (Resolve-Path -LiteralPath $AppDir).Path.TrimEnd("\") } else { Join-Path $PSScriptRoot "dist\PDFtoWordOCR" }
$exe = Join-Path $dist "PDFtoWordOCR.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "$exe not found. Run .\build_exe.ps1 first." }
$running = Get-Process PDFtoWordOCR -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($dist, [StringComparison]::OrdinalIgnoreCase) }
if ($running) { throw "PDF to Word OCR is running from $dist (PID $($running.Id -join ', ')). Close it first." }
if (-not $Python) {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
}
if (-not $ModelDir) { $ModelDir = Join-Path $PSScriptRoot "models\trocr-base-handwritten" }
if (-not $ZipPath) { $ZipPath = Join-Path $PSScriptRoot "dist\PDFtoWordOCR-portable.zip" }

# ---------------------------------------------------------------- Tesseract
if (-not $TesseractDir) {
    $candidates = @(
        (Join-Path $env:ProgramFiles "Tesseract-OCR"),
        (Join-Path $env:LOCALAPPDATA "Programs\Tesseract-OCR")
    )
    $onPath = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += (Split-Path $onPath.Source) }
    $TesseractDir = $candidates | Where-Object { Test-Path -LiteralPath (Join-Path $_ "tesseract.exe") } | Select-Object -First 1
}
if (-not $TesseractDir -or -not (Test-Path -LiteralPath (Join-Path $TesseractDir "tesseract.exe"))) {
    throw "Tesseract not found. Install it (winget install --id UB-Mannheim.TesseractOCR) or pass -TesseractDir."
}
if (-not (Test-Path -LiteralPath (Join-Path $TesseractDir "tessdata\eng.traineddata"))) {
    throw "$TesseractDir has no tessdata\eng.traineddata; this Tesseract cannot be bundled as-is."
}
Write-Host "== Bundling Tesseract from $TesseractDir"
$tessTarget = Join-Path $dist "tesseract"
if (Test-Path -LiteralPath $tessTarget) { Remove-Item -LiteralPath $tessTarget -Recurse -Force }
$sourceRoot = (Resolve-Path -LiteralPath $TesseractDir).Path.TrimEnd("\")
Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object {
    # Keep tesseract.exe, its DLLs, language data and licence docs; skip training tools,
    # man pages and the uninstaller.
    -not (($_.Extension -eq ".exe" -and $_.Name -ne "tesseract.exe") -or $_.Extension -eq ".html")
} | ForEach-Object {
    $relative = $_.FullName.Substring($sourceRoot.Length + 1)
    $destination = Join-Path $tessTarget $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $destination
}
$saveTessdata = $env:TESSDATA_PREFIX
Remove-Item Env:TESSDATA_PREFIX -ErrorAction SilentlyContinue
$langOutput = & (Join-Path $tessTarget "tesseract.exe") --list-langs 2>&1 | ForEach-Object { "$_" }
if ($saveTessdata) { $env:TESSDATA_PREFIX = $saveTessdata }
Write-Host "   $($langOutput[0])"
Write-Host "   bundled languages: $(($langOutput | Select-Object -Skip 1) -join ', ')"

# ---------------------------------------------------------------- handwriting model
if (-not (Test-Path -LiteralPath (Join-Path $ModelDir "model.safetensors"))) {
    Write-Host "== Saving the handwriting model to $ModelDir"
    & $Python -m ocr_app.fetch_model --output $ModelDir
    if ($LASTEXITCODE -ne 0) { throw "Could not save the handwriting model." }
}
Write-Host "== Bundling the handwriting model from $ModelDir"
$modelTarget = Join-Path $dist "models\trocr-base-handwritten"
if (Test-Path -LiteralPath $modelTarget) { Remove-Item -LiteralPath $modelTarget -Recurse -Force }
New-Item -ItemType Directory -Force -Path $modelTarget | Out-Null
Copy-Item -Path (Join-Path $ModelDir "*") -Destination $modelTarget -Recurse

# ---------------------------------------------------------------- user manual
# docs\USER_MANUAL.txt / .md are written for the person receiving the app.
$manualDir = Join-Path $PSScriptRoot "docs"
foreach ($ext in "txt", "md") {
    $manual = Join-Path $manualDir "USER_MANUAL.$ext"
    if (-not (Test-Path -LiteralPath $manual)) { throw "$manual is missing." }
    Copy-Item -LiteralPath $manual -Destination (Join-Path $dist "User Manual.$ext")
}
Remove-Item -LiteralPath (Join-Path $dist "HOW TO USE.txt") -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- self-test with this PC's installs hidden
Write-Host "== Self-test (local Tesseract install, model cache and network hidden)"
$report = Join-Path $PSScriptRoot "build\portable_selftest.json"
if (Test-Path -LiteralPath $report) { Remove-Item -LiteralPath $report }
$emptyHfHome = Join-Path ([IO.Path]::GetTempPath()) ("pdfocr_empty_hf_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $emptyHfHome | Out-Null
$saved = @{}
foreach ($name in "HF_HOME", "HF_HUB_OFFLINE", "TESSERACT_CMD", "TESSDATA_PREFIX", "PATH") {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
    $env:HF_HOME = $emptyHfHome
    $env:HF_HUB_OFFLINE = "1"
    Remove-Item Env:TESSERACT_CMD, Env:TESSDATA_PREFIX -ErrorAction SilentlyContinue
    $env:PATH = ($env:PATH -split ";" | Where-Object { $_ -and $_ -notmatch "Tesseract" }) -join ";"
    $proc = Start-Process -FilePath $exe -ArgumentList "--self-test `"$report`"" -Wait -PassThru
} finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process") }
    Remove-Item -LiteralPath $emptyHfHome -Recurse -Force -ErrorAction SilentlyContinue
}
if (-not (Test-Path -LiteralPath $report)) { throw "The executable did not write a self-test report." }
$result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
$problems = @()
foreach ($check in $result.checks) {
    Write-Host ("  {0,-18} {1,-8} {2}" -f $check.check, $check.status.ToUpper(), ($check.detail -split "`n")[0])
    if ($check.status -ne "pass") { $problems += $check.check }
}
$tessDetail = ($result.checks | Where-Object check -eq "tesseract_ocr").detail
$modelDetail = ($result.checks | Where-Object check -eq "handwriting_model").detail
if ($tessDetail -notlike "*\PDFtoWordOCR\tesseract\tesseract.exe*") { $problems += "tesseract not from bundle" }
if ($modelDetail -notlike "*\PDFtoWordOCR\models\trocr-base-handwritten*") { $problems += "model not from bundle" }
if ($proc.ExitCode -ne 0 -or $problems) { throw "Portable self-test failed: $($problems -join ', ')" }

# ---------------------------------------------------------------- zip
$sizeGB = (Get-ChildItem -LiteralPath $dist -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("`nPortable folder: {0} ({1:N2} GB)" -f $dist, $sizeGB)
if (-not $NoZip) {
    Write-Host "== Creating $ZipPath"
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath }
    & (Join-Path $env:SystemRoot "System32\tar.exe") -a -c -f $ZipPath -C (Split-Path $dist) (Split-Path $dist -Leaf)
    if ($LASTEXITCODE -ne 0) { throw "Creating the ZIP failed." }
    Write-Host ("ZIP ready: {0} ({1:N2} GB)" -f $ZipPath, ((Get-Item -LiteralPath $ZipPath).Length / 1GB))
    # Copies beside the ZIP, so the recipient can read them before downloading.
    foreach ($ext in "txt", "md") {
        $copy = Join-Path (Split-Path $ZipPath) "PDFtoWordOCR - User Manual.$ext"
        Copy-Item -LiteralPath (Join-Path $manualDir "USER_MANUAL.$ext") -Destination $copy
        Write-Host "Manual beside the ZIP: $copy"
    }
}
