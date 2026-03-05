# Start Server with FFmpeg Support
# This script automatically sets up FFmpeg in PATH before starting the API server

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Dementia Backend Server Startup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Step 1: Add FFmpeg to PATH
Write-Host "`n[1] Setting up FFmpeg in PATH..." -ForegroundColor Yellow
$ffmpegPath = "C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"
$env:PATH = "$ffmpegPath;$env:PATH"

# Verify FFmpeg is available
$ffmpegCheck = & {
    try {
        $result = ffmpeg -version 2>&1 | Select-Object -First 1
        Write-Host "✅ FFmpeg ready: $result" -ForegroundColor Green
        $true
    } catch {
        Write-Host "❌ FFmpeg not found: $_" -ForegroundColor Red
        $false
    }
}

if (-not $ffmpegCheck) {
    Write-Host "`n⚠️  FFmpeg not found. Please ensure FFmpeg is installed at $ffmpegPath" -ForegroundColor Red
    Write-Host "To install: Download from https://ffmpeg.org/download.html" -ForegroundColor Yellow
    Read-Host "Press Enter to continue anyway (audio will not work)"
}

# Step 2: Activate Virtual Environment
Write-Host "`n[2] Activating Python virtual environment..." -ForegroundColor Yellow
. ".\venv\Scripts\Activate.ps1"
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Step 3: Verify pydub is installed
Write-Host "`n[3] Checking pydub installation..." -ForegroundColor Yellow
python -c "import pydub; print('✅ pydub is ready')" 2>&1 | ForEach-Object {
    if ($_ -match "ModuleNotFoundError") {
        Write-Host "❌ pydub not found, installing..." -ForegroundColor Red
        pip install pydub
    } else {
        Write-Host $_
    }
}

# Step 4: Start the server
Write-Host "`n[4] Starting API server..." -ForegroundColor Yellow
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Server is running. Press Ctrl+C to stop." -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

python run_api.py
