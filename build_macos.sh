#!/bin/bash
# Build macOS .app + .dmg for BatchTranscribe.
# MUST be run on a Mac - Windows cannot cross-compile macOS apps.
#
# Output:
#   dist/BatchTranscribe.app
#   dist/BatchTranscribe-2.0.0.dmg
set -euo pipefail

cd "$(dirname "$0")"

echo "=== BatchTranscribe macOS build ==="

echo "[1/5] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/5] Installing build dependencies..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

echo "[3/5] Generating icons (this produces app.iconset for iconutil)..."
python build_icon.py

echo "[4/5] Building .app (this can take 5-10 minutes the first time)..."
flet build macos \
    --product "BatchTranscribe" \
    --build-version "2.0.0" \
    --verbose

APP_PATH="build/macos/BatchTranscribe.app"
if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: $APP_PATH not found. Build may have failed."
    exit 1
fi

echo "[5/5] Wrapping .app in .dmg..."
if ! command -v create-dmg >/dev/null 2>&1; then
    echo "Installing create-dmg via Homebrew..."
    brew install create-dmg
fi

DMG_PATH="dist/BatchTranscribe-2.0.0.dmg"
mkdir -p dist
rm -f "$DMG_PATH"

create-dmg \
    --volname "BatchTranscribe" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "BatchTranscribe.app" 175 190 \
    --app-drop-link 425 190 \
    --no-internet-enable \
    "$DMG_PATH" \
    "$APP_PATH"

echo
echo "=== Build complete ==="
echo "App bundle:  $APP_PATH"
echo "Disk image:  $DMG_PATH"
echo
echo "To install: open the .dmg and drag BatchTranscribe.app to /Applications"
echo

