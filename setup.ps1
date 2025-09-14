# Creates venv, installs deps
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
Write-Host 'Done. To run: .\.venv\Scripts\Activate.ps1 then python jarvis.py'

