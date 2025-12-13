# PowerShell script to enable Windows Long Path Support
# This script must be run as Administrator

Write-Host "Windows Long Path Support Enabler" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Check current status
Write-Host "`nChecking current Long Path status..." -ForegroundColor Yellow
try {
    $currentValue = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction SilentlyContinue
    if ($currentValue.LongPathsEnabled -eq 1) {
        Write-Host "Long Path Support is already ENABLED!" -ForegroundColor Green
        Write-Host "No changes needed." -ForegroundColor Green
        exit 0
    } else {
        Write-Host "Long Path Support is currently DISABLED" -ForegroundColor Red
    }
} catch {
    Write-Host "Long Path Support is currently DISABLED (registry key not found)" -ForegroundColor Red
}

# Enable Long Path Support
Write-Host "`nEnabling Long Path Support..." -ForegroundColor Yellow
try {
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force | Out-Null
    Write-Host "SUCCESS: Long Path Support has been enabled!" -ForegroundColor Green
    Write-Host "`nIMPORTANT: You must restart your computer for this change to take effect." -ForegroundColor Yellow
    Write-Host "After restarting, you can install CoquiTTS with: pip install -r requirements.txt" -ForegroundColor Cyan
    
    $restart = Read-Host "`nWould you like to restart now? (y/n)"
    if ($restart -eq 'y' -or $restart -eq 'Y') {
        Write-Host "Restarting computer in 10 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        Restart-Computer
    } else {
        Write-Host "Please restart your computer manually when ready." -ForegroundColor Yellow
    }
} catch {
    Write-Host "ERROR: Failed to enable Long Path Support" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
