# Jarvis Prime v6 (local‑first assistant)

**Default STT quality: HIGH** (`medium.en`). CPU‑friendly, reliable pickup, optional WebRTC VAD.

## Features
- Always‑listening voice with energy VAD (+ optional WebRTC VAD)
- One‑word intents: say “**google**”, “**gmail**”, “**youtube**”, “**calendar**”, “**drive**”, “**github**”
- Smart routing: factual → web (SerpAPI), math → solver (SymPy), chitchat → local LLM (Ollama)
- PDF/DOCX helpers, notes, reminders, clipboard, screenshots
- British voice auto‑pick on Windows (SAPI) + fallback to `pyttsx4`

## Requirements
- **Python 3.10 – 3.12** recommended (3.13 works for most deps, but wheels may lag)
- (Windows) Optional DOCX→PDF automation requires **Microsoft Word** (for `comtypes` export)
- (Optional) **Ollama** running a local model (default: `gemma3:4b`)
- (Optional) **SerpAPI** key for web answers

## Quick start (Windows PowerShell)
```powershell
# 1) clone + venv
git clone https://github.com/<your-username>/jarvis-prime-v6.git
cd jarvis-prime-v6
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2) deps (webrtcvad is optional; if it fails, Jarvis still works)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) (optional) set API keys / model
$env:SERPAPI_KEY="YOUR_KEY"     # omit if you don't want web answers
# Ensure Ollama is running a local model:
#   ollama run gemma3:4b

# 4) run
python jarvis.py
```

## Quick start (Linux / macOS)
```bash
git clone https://github.com/<your-username>/jarvis-prime-v6.git
cd jarvis-prime-v6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export SERPAPI_KEY="YOUR_KEY"   # optional
python jarvis.py
```

## Runtime tips
- **Pick your mic**: `audio list` → `audio in 5` → `audio autotune`
- **Change STT quality**:
  - `stt quality fast` → `base.en`
  - `stt quality balanced` → `small.en`
  - `stt quality high` (default) → `medium.en`
  - `stt quality best` → `large-v3`
- **Voice rate**: `rate 2` (default). Volume: `volume 100`
- **Open sites**: say “**google**”, “**gmail**”, “**youtube**” (no full URL spoken)
- **Math**: `solve x^2 - 5*x + 6`, `calc 5 km to m`, `factor 3233`, `rsa 3233 :: 17`
- **Reminders**: `remind 2025-11-07 09:00 :: standup`

## Project layout
```
jarvis-prime-v6/
├─ jarvis.py
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ docs/
└─ scripts/
```

## Common issues
- **`webrtcvad` fails to build on Windows**: it's optional; proceed without it.
- **No voice out**: Windows SAPI sometimes locks—relaunch the program or switch to `pyttsx4` (automatic fallback).
- **STT slow**: try `stt quality balanced` (`small.en`) or `fast` (`base.en`). Also run `audio autotune`.

## License
MIT © 2025 Sai Charan Kommi
