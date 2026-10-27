# Cria o ambiente virtual e instala as dependencias do piano2sheet.
# Uso:  powershell -ExecutionPolicy Bypass -File instalar.ps1        (GPU NVIDIA)
#       powershell -ExecutionPolicy Bypass -File instalar.ps1 -Cpu   (sem GPU)
param([switch]$Cpu)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12 -m venv .venv }
    else { & python -m venv .venv }
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
if ($Cpu) { & $py -m pip install torch torchaudio }
else { & $py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
& $py -m pip install --no-deps transkun==2.0.1
& $py -m pip install -r requirements.txt
& $py -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.is_available())"
Write-Host "Pronto. Exemplo:  .\piano2sheet.bat `"https://www.youtube.com/watch?v=...`""
