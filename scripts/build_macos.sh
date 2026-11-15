#!/usr/bin/env bash
# Package Hupesi TTS for macOS -> dist/Hupesi-TTS-v<version>-macos-<arch>.dmg
# Users open the .dmg and drag "Hupesi TTS" into Applications.
# (Linux build: build_linux.sh; Windows build: build_desktop_exe.bat - three independent scripts;
#  changing this script does not affect the other two).
#
# PyInstaller CANNOT cross-build: this must run on macOS. If you do not have a Mac, use CI:
# .github/workflows/build-macos.yml. The build matches the architecture of the build machine (arm64 / x86_64).
#
# Differences from the Windows/Linux builds:
#  - The .app bundle has a dedicated window (pywebview + macOS's built-in WebKit, bundled with PyObjC)
#    and can be closed with Cmd+Q like a normal app.
#  - Models are bundled INSIDE the .app and copied to ~/Library/Application Support on first
#    run; all writable data lives outside the bundle (see apps/runtime_paths_macos.py).
#  - CPU-only. Not signed with an Apple Developer ID (ad-hoc signature only): on first launch you
#    must click "Open Anyway" in System Settings -> Privacy & Security.
#
# Usage: ./build_macos.sh                  # requires uv + Node 20+
#        ./build_macos.sh --no-models      # do not bundle models; the app downloads them on first run
#        ./build_macos.sh --skip-frontend  # use a prebuilt UI (Node not required)
# Note: macOS uses bash 3.2 - avoid bash 4+ syntax (associative arrays, ${var,,}, etc.).
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root (script lives in scripts/)

[[ "$(uname -s)" == Darwin ]] || { echo "[ERROR] This script only runs on macOS. Linux: build_linux.sh"; exit 1; }

WITH_MODELS=1
BUILD_FRONTEND=1
for arg in "$@"; do
    case "$arg" in
        --no-models) WITH_MODELS=0 ;;
        --skip-frontend) BUILD_FRONTEND=0 ;;
        *) echo "[ERROR] Invalid argument: $arg"; exit 1 ;;
    esac
done

ARCH="$(uname -m)"
APP_NAME="Hupesi TTS"
APP="dist/${APP_NAME}.app"
# The version is read from installer.iss so every platform uses the same version number.
VERSION="$(sed -n 's/^#define AppVersion *"\(.*\)".*/\1/p' scripts/installer.iss)"
[[ -n "$VERSION" ]] || { echo "[ERROR] Could not read AppVersion from scripts/installer.iss"; exit 1; }
DMG="dist/Hupesi-TTS-v${VERSION}-macos-${ARCH}.dmg"
STAGE=".mac-stage"

if [[ $BUILD_FRONTEND == 1 ]]; then
    echo "[+] 1/5 Building the React UI..."
    (cd app-desktop/hupesi-tts-app && npm ci --no-audit --no-fund && npm run build)
elif [[ -f app-desktop/hupesi-tts-app/dist/index.html ]]; then
    echo "[+] 1/5 Using the prebuilt UI."
else
    echo "[ERROR] app-desktop/hupesi-tts-app/dist is missing - remove --skip-frontend or build it first."
    exit 1
fi

echo "[+] 2/5 Installing the Python environment (torch-free) + pywebview/PyObjC + PyInstaller..."
if [[ "$ARCH" == x86_64 ]]; then
    # onnxruntime 1.24 dropped Intel Macs (only macosx arm64 wheels remain); 1.23.2 is the
    # last release with an x86_64 wheel. Downgrade here only instead of changing uv.lock so
    # Windows/Linux keep their versions.
    uv sync --frozen --group desktop --no-install-package onnxruntime
    uv pip install "onnxruntime==1.23.2"
else
    uv sync --frozen --group desktop
fi
uv pip install pyinstaller
# From here, `uv run` must not sync again (it would reinstall onnxruntime per uv.lock).
export UV_NO_SYNC=1

rm -rf build "dist/${APP_NAME}" "$APP" "$STAGE" "$DMG"
MODEL_ARGS=()
if [[ $WITH_MODELS == 1 ]]; then
    echo "[+] 3/5 Pre-downloading the v3 Turbo int8 model (~312 MB)..."
    uv run python scripts/fetch_models.py
    # As with the Windows build: copy BY LIST (Turbo int8 + codec, ~312 MB), not the entire
    # cache - the build machine cache may contain other models (Kokoro ~650 MB) and cloned
    # voices from real people (custom_voices_*.json).
    uv run python scripts/copy_release_models.py "$STAGE/models"
    MODEL_ARGS=(--add-data "$STAGE/models/hf_cache:models/hf_cache")
else
    echo "[+] 3/5 Skipping models: the app will download them on first run."
fi

echo "[+] 4/5 Packaging with PyInstaller (.app)..."
uv run pyinstaller --clean --noconfirm --onedir --windowed \
    --name "$APP_NAME" \
    --icon "assets/icons/icon.icns" \
    --osx-bundle-identifier "com.festusxmutukugithub.hupesitts" \
    --paths src \
    --exclude-module torch --exclude-module torchaudio --exclude-module transformers \
    --add-data "docs/hupesi-tts.md:." \
    --add-data "DIEU-KHOAN.md:." \
    --add-data "app-desktop/hupesi-tts-app/dist:app-desktop/hupesi-tts-app/dist" \
    --collect-all onnxruntime --collect-all sea_g2p --collect-all soundfile \
    --collect-all tokenizers --collect-all vieneu --collect-all vieneu_utils \
    --collect-all pypdf --collect-all kaldi_native_fbank --collect-all soxr \
    --collect-all librosa --collect-all huggingface_hub \
    --collect-all webview \
    --hidden-import webview.platforms.cocoa \
    --hidden-import unittest.mock \
    --hidden-import uvicorn.logging --hidden-import uvicorn.loops --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
    apps/desktop_app.py

# Voice Cloning tab records via the microphone: WebKit on macOS terminates the app immediately
# when microphone access is requested if Info.plist does not declare a reason.
PLIST="$APP/Contents/Info.plist"
plutil -replace NSMicrophoneUsageDescription \
    -string "Hupesi TTS uses the microphone to record a voice sample for voice cloning." "$PLIST"
plutil -replace NSHighResolutionCapable -bool true "$PLIST"
# Minimum macOS version per onnxruntime wheel: arm64 1.24.4 = macosx_14_0, x86_64 1.23.2 = macosx_13_0.
if [[ "$ARCH" == x86_64 ]]; then MIN_MACOS="13.0"; else MIN_MACOS="14.0"; fi
plutil -replace LSMinimumSystemVersion -string "$MIN_MACOS" "$PLIST"
plutil -lint "$PLIST"

# Editing Info.plist invalidates the ad-hoc signature PyInstaller just applied -> sign again.
# Apple Silicon requires all executed code to be signed (even ad-hoc); without it macOS reports
# the app as "damaged".
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

echo "[+] 5/5 Creating the .dmg installer..."
DMGSRC="$STAGE/dmg"
rm -rf "$DMGSRC"; mkdir -p "$DMGSRC"
cp -R "$APP" "$DMGSRC/"
ln -s /Applications "$DMGSRC/Applications"
cp LICENSE NOTICE THIRD-PARTY-NOTICES.md PRIVACY.md DIEU-KHOAN.md packaging/macos/HUONG-DAN.txt "$DMGSRC/"
# hdiutil on CI machines sometimes reports "Resource busy" -> retry a few times.
ok=0
for i in 1 2 3 4 5; do
    if hdiutil create -volname "$APP_NAME" -srcfolder "$DMGSRC" -ov -format UDZO "$DMG"; then
        ok=1; break
    fi
    sleep 5
done
[[ $ok == 1 ]] || { echo "[ERROR] Could not create the .dmg file"; exit 1; }
rm -rf "$STAGE"
echo "[SUCCESS] $DMG  (app: $APP)"
