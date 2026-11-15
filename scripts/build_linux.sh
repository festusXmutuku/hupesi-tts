#!/usr/bin/env bash
# Package Hupesi TTS for Linux -> dist/Hupesi-TTS-v<version>-linux-<arch>.tar.gz
#                                   and dist/Hupesi-TTS-v<version>-<amd64|arm64>.deb
# (macOS build: build_macos.sh; Windows build: build_desktop_exe.bat - three independent scripts;
#  changing this script does not affect the other two).
#
# PyInstaller CANNOT cross-build: this must run on Linux. CI: .github/workflows/build-linux.yml
#
# Differences from the Windows build:
#  - No dedicated window: the app runs in the terminal and opens the UI in a browser
#    (desktop_app.py selects this automatically on Linux), so no `desktop` group and no --windowed.
#  - Bundles uv so the app can install NVIDIA GPU support on demand (see apps/gpu_installer_linux.py).
#  - The .deb installs to /opt/hupesi-tts (read-only) + the `hupesi-tts` command + an
#    application menu entry. Writable data is automatically moved to ~/HupesiTTS, and bundled
#    models are copied there on first run (apps/runtime_paths_linux.py).
#  - Build against an older glibc when possible (CI uses Ubuntu 22.04): a build made on a newer
#    glibc will not run on older distros.
#
# Usage: ./build_linux.sh                  # requires uv + Node 20+
#        ./build_linux.sh --no-models      # do not bundle models; the app downloads them on first run
#        ./build_linux.sh --skip-frontend  # use a prebuilt UI (Node not required)
#        ./build_linux.sh --no-deb         # create only .tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root (script lives in scripts/)

[[ "$(uname -s)" == Linux ]] || { echo "[ERROR] This script only runs on Linux. macOS: build_macos.sh"; exit 1; }

WITH_MODELS=1
BUILD_FRONTEND=1
BUILD_DEB=1
for arg in "$@"; do
    case "$arg" in
        --no-models) WITH_MODELS=0 ;;
        --skip-frontend) BUILD_FRONTEND=0 ;;
        --no-deb) BUILD_DEB=0 ;;
        *) echo "[ERROR] Invalid argument: $arg"; exit 1 ;;
    esac
done

ARCH="$(uname -m)"; [[ "$ARCH" == "aarch64" ]] && ARCH=arm64
NAME="Hupesi-TTS"
# The version is read from installer.iss so every platform uses the same version number.
VERSION="$(sed -n 's/^#define AppVersion *"\(.*\)".*/\1/p' scripts/installer.iss)"
[[ -n "$VERSION" ]] || { echo "[ERROR] Could not read AppVersion from scripts/installer.iss"; exit 1; }
OUT="dist/${NAME}-v${VERSION}-linux-${ARCH}"

if [[ $BUILD_FRONTEND == 1 ]]; then
    echo "[+] 1/4 Building the React UI..."
    (cd app-desktop/hupesi-tts-app && npm ci --no-audit --no-fund && npm run build)
elif [[ -f app-desktop/hupesi-tts-app/dist/index.html ]]; then
    echo "[+] 1/4 Using the prebuilt UI."
else
    echo "[ERROR] app-desktop/hupesi-tts-app/dist is missing - remove --skip-frontend or build it first."
    exit 1
fi

echo "[+] 2/4 Installing the Python environment (torch-free) + PyInstaller..."
uv sync --frozen
uv pip install pyinstaller

if [[ $WITH_MODELS == 1 ]]; then
    echo "[+] 3/4 Pre-downloading the v3 Turbo int8 model (~312 MB)..."
    uv run python scripts/fetch_models.py
else
    echo "[+] 3/4 Skipping models: the app will download them on first run."
fi

echo "[+] 4/4 Packaging with PyInstaller..."
rm -rf build "dist/${NAME}" "$OUT" "${OUT}.tar.gz"
UV_BIN="$(command -v uv)" || { echo "[ERROR] Could not find uv to bundle the GPU installer."; exit 1; }
# The --collect-all list matches the Windows build, except for webview/pythonnet/clr_loader.
uv run pyinstaller --clean --noconfirm --onedir \
    --name "$NAME" \
    --paths src \
    --exclude-module torch --exclude-module torchaudio --exclude-module transformers \
    --exclude-module webview \
    --add-data "docs/hupesi-tts.md:." \
    --add-data "DIEU-KHOAN.md:." \
    --add-data "app-desktop/hupesi-tts-app/dist:app-desktop/hupesi-tts-app/dist" \
    --add-binary "${UV_BIN}:tools" \
    --collect-all onnxruntime --collect-all sea_g2p --collect-all soundfile \
    --collect-all tokenizers --collect-all vieneu --collect-all vieneu_utils \
    --collect-all pypdf --collect-all kaldi_native_fbank --collect-all soxr \
    --collect-all librosa --collect-all huggingface_hub \
    --hidden-import unittest.mock \
    --hidden-import numpy.typing --hidden-import numpy.lib.stride_tricks \
    --collect-submodules joblib --collect-submodules apps._kokoro_torch \
    --hidden-import uvicorn.logging --hidden-import uvicorn.loops --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    apps/desktop_app.py

mv "dist/${NAME}" "$OUT"
cp LICENSE NOTICE THIRD-PARTY-NOTICES.md PRIVACY.md DIEU-KHOAN.md packaging/linux/HUONG-DAN.txt "$OUT/"

if [[ $WITH_MODELS == 1 && -d models/hf_cache ]]; then
    # As with the Windows build: copy BY LIST (Turbo int8 + codec, ~312 MB), not the entire
    # cache - the build machine cache may contain other models (Kokoro ~650 MB) and cloned
    # voices from real people (custom_voices_*.json).
    uv run python scripts/copy_release_models.py "$OUT/models"
fi

echo "[+] Compressing..."
tar -C dist -czf "${OUT}.tar.gz" "$(basename "$OUT")"
echo "[SUCCESS] ${OUT}.tar.gz"

[[ $BUILD_DEB == 1 ]] || exit 0

# ---- .deb package ----------------------------------------------------------
case "$ARCH" in x86_64) DEB_ARCH=amd64 ;; *) DEB_ARCH="$ARCH" ;; esac
# File name matches other releases (Hupesi-TTS-v<version>-...); the package name inside
# remains hupesi-tts so apt treats newer versions as upgrades.
DEB="dist/Hupesi-TTS-v${VERSION}-${DEB_ARCH}.deb"
PKG=".deb-stage"
echo "[+] Creating .deb package ${VERSION}..."
rm -rf "$PKG" "$DEB"
mkdir -p "$PKG/DEBIAN" "$PKG/opt" "$PKG/usr/bin" "$PKG/usr/share/applications" \
    "$PKG/usr/share/metainfo"
# Hard-link instead of copying ~600 MB (falls back to copy across filesystems).
cp -al "$OUT" "$PKG/opt/hupesi-tts" 2>/dev/null || cp -a "$OUT" "$PKG/opt/hupesi-tts"
printf '#!/bin/sh\nexec /opt/hupesi-tts/Hupesi-TTS "$@"\n' > "$PKG/usr/bin/hupesi-tts"
cp packaging/linux/hupesi-tts.desktop "$PKG/usr/share/applications/"
# Icons at multiple sizes: previously only 128px was included, so launchers/App Center could
# not show the logo when 256/512 were needed.
ICONS=assets/icons
for pair in "32x32:32x32.png" "64x64:64x64.png" "128x128:128x128.png" \
            "256x256:128x128@2x.png" "512x512:icon.png"; do
    size="${pair%%:*}"; src="${pair#*:}"
    mkdir -p "$PKG/usr/share/icons/hicolor/${size}/apps"
    cp "$ICONS/$src" "$PKG/usr/share/icons/hicolor/${size}/apps/hupesi-tts.png"
done
# AppStream metainfo: App Center / GNOME Software reads the name, description, license, and icon
# from here (without this file it shows "License unknown" and no logo).
sed -e "s/@VERSION@/${VERSION}/" -e "s/@DATE@/$(date +%Y-%m-%d)/" \
    packaging/linux/hupesi-tts.metainfo.xml > "$PKG/usr/share/metainfo/hupesi-tts.metainfo.xml"
# control is English-only ASCII: Ubuntu App Center rendered accented characters as
# "Chuy??n v?n b??n" (user report, 2026-09).
{
    echo "Package: hupesi-tts"
    echo "Version: ${VERSION}"
    echo "Architecture: ${DEB_ARCH}"
    echo "Maintainer: Festus Mutuku <noreply@users.noreply.github.com>"
    echo "Installed-Size: $(du -sk "$PKG" | cut -f1)"
    echo "Recommends: xdg-utils"
    echo "Section: sound"
    echo "Priority: optional"
    echo "Homepage: https://github.com/festusxmutukugithub/hupesi-tts"
    echo "Description: Vietnamese text-to-speech and voice cloning (VieNeu-TTS)"
    echo " Hupesi TTS turns Vietnamese text into natural speech and clones a"
    echo " voice from a short sample. Runs offline on the CPU, models included;"
    echo " optional NVIDIA GPU support can be installed from the app."
} > "$PKG/DEBIAN/control"
# Standard permissions for a system package; root ownership via --root-owner-group.
chmod -R u+rwX,go+rX,go-w "$PKG"
chmod 755 "$PKG/usr/bin/hupesi-tts"
find "$PKG/usr/share" -type f -exec chmod 644 {} +
# gzip instead of default xz/zstd: much faster for ~600 MB, and readable by any dpkg.
dpkg-deb --root-owner-group -Zgzip --build "$PKG" "$DEB"
rm -rf "$PKG"
echo "[SUCCESS] $DEB"
