<#
.SYNOPSIS
Builds the FSOC Tracker executable on Windows using PyInstaller.

.DESCRIPTION
This script sets up a Python virtual environment, installs the necessary dependencies, 
and packages the FSOC Tracker application into a standalone Windows executable (.exe).

.EXAMPLE
.\scripts\build_windows.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "Starting FSOC Tracker Windows Build..." -ForegroundColor Cyan

# Check for Python
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+."
    exit 1
}

$VenvDir = ".venv-win"

# 1. Create Virtual Environment
Write-Host "`n[1/4] Creating virtual environment in $VenvDir..."
python -m venv $VenvDir

# 2. Activate Virtual Environment
Write-Host "[2/4] Activating virtual environment..."
$ActivateScript = ".\$VenvDir\Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    . $ActivateScript
} else {
    Write-Error "Failed to find activation script."
    exit 1
}

# Upgrade pip
python -m pip install --upgrade pip

# 3. Install Dependencies
Write-Host "`n[3/4] Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller pytest

Write-Host "Installing PySide6 for GUI support..."
pip install PySide6

# 4. Build Executable with PyInstaller
Write-Host "`n[4/4] Building standalone executable with PyInstaller..."

# Clean previous builds
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist\fsoc-tracker-gui") { Remove-Item -Recurse -Force "dist\fsoc-tracker-gui" }

# Set environment variable for GUI build
$env:BUILD_GUI = "1"

# Run PyInstaller
pyinstaller --noconfirm fsoc-tracker.spec

Write-Host "`nBuild Complete!" -ForegroundColor Green
Write-Host "You can find your Windows executable in the 'dist\fsoc-tracker-gui' directory." -ForegroundColor Yellow
Write-Host "Run it by double-clicking 'fsoc-tracker.exe' in that folder."

# Deactivate venv
deactivate
