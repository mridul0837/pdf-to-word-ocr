<#
.SYNOPSIS
    Build dist\PDFtoWordOCR\PDFtoWordOCR.exe (PyInstaller one-folder) and self-test it.

.DESCRIPTION
    1. Checks that the build dependencies are importable.
    2. Runs the pytest suite (skip with -SkipTests).
    3. Runs PyInstaller with PDFtoWordOCR.spec.
    4. Runs "PDFtoWordOCR.exe --self-test" and prints the report.

    Tesseract and the TrOCR model weights are NOT bundled; see README.md.

.PARAMETER Python
    Python interpreter to use. Default: .venv\Scripts\python.exe if it exists, else "python".
.PARAMETER SkipTests
    Do not run the test suite before building.
.PARAMETER Console
    Build with a console window, to see tracebacks when diagnosing start-up problems.

.EXAMPLE
    .\build_exe.ps1
.EXAMPLE
    .\build_exe.ps1 -Console -SkipTests
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$SkipTests,
    [switch]$Console
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not $Python) {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
}
Write-Host "Using Python: $Python"

# PyInstaller replaces dist\PDFtoWordOCR; a running copy locks its files and the
# build would fail half-way through deleting the old folder.
$distDir = Join-Path $PSScriptRoot "dist\PDFtoWordOCR"
$running = Get-Process PDFtoWordOCR -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($distDir, [StringComparison]::OrdinalIgnoreCase) }
if ($running) {
    throw "PDF to Word OCR is running from $distDir (PID $($running.Id -join ', ')). Close it and run the build again."
}

& $Python -c "import PyInstaller, PySide6, pymupdf, docx, pytesseract, cv2, torch, transformers"
if ($LASTEXITCODE -ne 0) {
    throw "Build dependencies are missing. Run: & '$Python' -m pip install -r requirements-dev.txt"
}

if (-not $SkipTests) {
    Write-Host "`n== Running tests"
    & $Python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed; not building. Use -SkipTests to override." }
}

Write-Host "`n== Running PyInstaller"
$env:PDFOCR_CONSOLE = if ($Console) { "1" } else { "0" }
& $Python -m PyInstaller --noconfirm --clean PDFtoWordOCR.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }

$exe = Join-Path $distDir "PDFtoWordOCR.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "PyInstaller finished but $exe was not created." }

Write-Host "`n== Self-test of the built executable"
$report = Join-Path $PSScriptRoot "build\selftest_report.json"
if (Test-Path -LiteralPath $report) { Remove-Item -LiteralPath $report }
$proc = Start-Process -FilePath $exe -ArgumentList "--self-test `"$report`"" -Wait -PassThru
if (Test-Path -LiteralPath $report) {
    $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    foreach ($check in $result.checks) {
        $line = $check.detail -split "`n" | Select-Object -First 1
        Write-Host ("  {0,-18} {1,-8} {2}" -f $check.check, $check.status.ToUpper(), $line)
    }
} else {
    Write-Warning "The executable did not write a self-test report."
}

$sizeGB = (Get-ChildItem -LiteralPath $distDir -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("`nBuild output: {0}  ({1:N2} GB folder)" -f $exe, $sizeGB)
Write-Host "Ship the whole dist\PDFtoWordOCR folder, not just the .exe."
Write-Host "Tesseract and the TrOCR model weights must be available on the target PC (see README.md),"
Write-Host "or run .\make_portable.ps1 to bundle both and create a ZIP that needs no installation."
if ($proc.ExitCode -ne 0) { throw "Self-test of the built executable failed (exit code $($proc.ExitCode))." }
