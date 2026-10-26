# Builds the portable demo folder (dist\HealthCoachDemo) on a Windows PC.
#
# Run once, on your own Windows machine, from the project folder:
#     powershell -ExecutionPolicy Bypass -File portable\build_portable.ps1
#
# Needs: Python 3.11 or 3.12, internet, and Ollama installed with gemma3:4b pulled
# (the script copies Gemma from your Ollama instead of downloading it again).
# Result: about 5 GB in dist\HealthCoachDemo. Copy that folder to the USB drive.

param(
    [string]$Out = "dist\HealthCoachDemo",
    [string]$Model = "gemma3:4b"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is far faster without the progress bar
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

# --- 1. Python environment --------------------------------------------------------
Step "Python environment"
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { & py -3.11 -m venv .venv }
    } else {
        & python -m venv .venv
    }
    Check "Creating the virtual environment"
}
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r requirements.txt pyinstaller --quiet
Check "Installing packages"

# --- 2. Speech models and Gemma check --------------------------------------------
Step "Speech models"
& $py setup_models.py --download-only
Check "Downloading models (is Ollama running, with $Model pulled?)"

# --- 3. Bundle the app with PyInstaller ------------------------------------------
Step "Bundling the app (a few minutes)"
& $py -m PyInstaller --noconfirm --clean --onedir --console --name HealthCoachDemo `
    --distpath build\pyi --workpath build\pyi-work --specpath build `
    --paths "$Root" `
    --add-data "$Root\static;static" `
    --add-data "$Root\data;data" `
    --collect-all moonshine_voice `
    --collect-all piper `
    --collect-all onnxruntime `
    --collect-submodules uvicorn `
    --collect-submodules websockets `
    --collect-submodules app `
    portable\launcher.py
Check "PyInstaller"
Write-Host "  Checking the bundled program loads everything:"
& "build\pyi\HealthCoachDemo\HealthCoachDemo.exe" --check-imports
Check "The bundled program's self-check"

# --- 4. Assemble the folder ------------------------------------------------------
Step "Assembling $Out"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
New-Item -ItemType Directory -Path $Out | Out-Null
Copy-Item "build\pyi\HealthCoachDemo" -Destination "$Out\app" -Recurse
Copy-Item "models" -Destination "$Out\models" -Recurse
Copy-Item "portable\Start Demo.bat", "portable\Stop Demo.bat" -Destination $Out
Copy-Item "portable\README-USB.txt" -Destination "$Out\README.txt"

# --- 5. Portable Ollama ----------------------------------------------------------
Step "Portable Ollama"
$zip = "build\ollama-windows-amd64.zip"
if (-not (Test-Path $zip)) {
    Write-Host "  Downloading ollama-windows-amd64.zip from GitHub..."
    Invoke-WebRequest "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip" -OutFile $zip
}
Expand-Archive $zip -DestinationPath "$Out\ollama" -Force
if (-not (Test-Path "$Out\ollama\ollama.exe")) { throw "ollama.exe not found after unpacking $zip" }

# --- 6. Gemma: copy from your Ollama (or download into the package) --------------
Step "Gemma ($Model)"
$dest = Join-Path (Resolve-Path $Out) "ollama\models"
New-Item -ItemType Directory -Path "$dest\blobs" -Force | Out-Null
$src = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE ".ollama\models" }
$name, $tag = $Model.Split(":")
$manifestRel = "manifests\registry.ollama.ai\library\$name\$tag"
$manifest = Join-Path $src $manifestRel

if (Test-Path $manifest) {
    Write-Host "  Copying from $src"
    New-Item -ItemType Directory -Path (Split-Path (Join-Path $dest $manifestRel)) -Force | Out-Null
    Copy-Item $manifest (Join-Path $dest $manifestRel)
    $m = Get-Content $manifest -Raw | ConvertFrom-Json
    $digests = @($m.config.digest) + @($m.layers | ForEach-Object { $_.digest })
    foreach ($d in $digests) {
        $blob = $d.Replace(":", "-")
        Copy-Item (Join-Path $src "blobs\$blob") "$dest\blobs\$blob"
    }
} else {
    Write-Host "  $Model not found in $src. Downloading it into the package instead..."
    $env:OLLAMA_MODELS = $dest
    $env:OLLAMA_HOST = "127.0.0.1:11436"
    $server = Start-Process "$Out\ollama\ollama.exe" -ArgumentList "serve" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 5
    & "$Out\ollama\ollama.exe" pull $Model
    $pullCode = $LASTEXITCODE
    Stop-Process -Id $server.Id -Force
    Remove-Item Env:\OLLAMA_MODELS, Env:\OLLAMA_HOST
    if ($pullCode -ne 0) { throw "Pulling $Model failed" }
}

# --- 7. Done ---------------------------------------------------------------------
$sizeGB = [math]::Round(((Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 1)
Step "Done"
Write-Host "  $Out  ($sizeGB GB)"
Write-Host ""
Write-Host "  1. Test it here first: double-click '$Out\Start Demo.bat'."
Write-Host "  2. Copy the whole HealthCoachDemo folder to the USB drive."
Write-Host "  3. On the other laptop, copy it to the Desktop and run 'Start Demo.bat'."
