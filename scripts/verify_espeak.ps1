# Verify eSpeak-NG Installation
Write-Host "Checking for eSpeak-NG installation..." -ForegroundColor Cyan

# Check if espeak-ng is in PATH
$espeakPath = Get-Command espeak-ng -ErrorAction SilentlyContinue
if ($espeakPath) {
    Write-Host "✓ eSpeak-NG found in PATH: $($espeakPath.Source)" -ForegroundColor Green
    & espeak-ng --version
    exit 0
}

# Check common installation locations
$commonPaths = @(
    "C:\Program Files\eSpeak NG\espeak-ng.exe",
    "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    "C:\Program Files\eSpeak\espeak-ng.exe",
    "C:\Program Files (x86)\eSpeak\espeak-ng.exe"
)

foreach ($path in $commonPaths) {
    if (Test-Path $path) {
        Write-Host "✓ Found eSpeak-NG at: $path" -ForegroundColor Green
        $dir = Split-Path $path -Parent
        Write-Host "`nTo add to PATH, run this command as Administrator:" -ForegroundColor Yellow
        $pathCmd = "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';$dir', 'Machine')"
        Write-Host "  $pathCmd" -ForegroundColor White
        Write-Host "`nOr manually add to PATH: $dir" -ForegroundColor Yellow
        & $path --version
        exit 0
    }
}

Write-Host "✗ eSpeak-NG not found." -ForegroundColor Red
Write-Host "`nPlease:" -ForegroundColor Yellow
Write-Host "1. Download from: https://github.com/espeak-ng/espeak-ng/releases" -ForegroundColor White
Write-Host "2. Install the .msi file" -ForegroundColor White
Write-Host "3. Add 'C:\Program Files\eSpeak NG\' to your PATH" -ForegroundColor White
Write-Host "4. Restart your terminal" -ForegroundColor White

