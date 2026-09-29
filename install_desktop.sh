#!/usr/bin/env bash
# Register the TCAR Parser GUI as a desktop app (app menu, and a desktop icon with --desktop).
#
#   ./install_desktop.sh [--desktop] [--appimage <file>] [--uninstall]
#
# The launcher runs this checkout's gui.py with the checkout's .venv Python when there
# is one, else the python3 on PATH (set PYTHON=... to choose another). With --appimage
# it runs that AppImage (packaging/build_appimage.sh) instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP=0 UNINSTALL=0 APPIMAGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --desktop) DESKTOP=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --appimage) APPIMAGE="$(readlink -f "$2")"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
if [[ -z "${PYTHON:-}" && -x "$HERE/.venv/bin/python" ]]; then
    PY="$HERE/.venv/bin/python"
else
    PY="${PYTHON:-$(command -v python3)}"
fi
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
ENTRY="$APPS/tcar-parser.desktop"

if [[ $UNINSTALL == 1 ]]; then
    rm -f "$ENTRY" "$ICONS/tcar-parser.svg" "$DESKTOP_DIR/tcar-parser.desktop"
    update-desktop-database "$APPS" 2>/dev/null || true
    echo "removed"
    exit 0
fi

if [[ -n "$APPIMAGE" ]]; then
    [[ -f "$APPIMAGE" ]] || { echo "no such AppImage: $APPIMAGE" >&2; exit 1; }
    chmod +x "$APPIMAGE"
    EXEC="\"$APPIMAGE\""
else
    "$PY" -c "import PyQt5" 2>/dev/null || "$PY" -c "import PySide6" 2>/dev/null || {
        echo "$PY cannot import PyQt5 or PySide6: pip install -e '$HERE[gui]'" >&2
        exit 1
    }
    EXEC="\"$PY\" \"$HERE/gui.py\""
fi

mkdir -p "$APPS" "$ICONS"
cp "$HERE/assets/tcar-parser.svg" "$ICONS/tcar-parser.svg"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=TCAR Parser
Name[ko]=티카 파서
Comment=T-Car ROS 1 bags to nuScenes
Comment[ko]=티카 ROS 1 bag을 nuScenes 데이터셋으로 변환
Exec=$EXEC
Path=$HERE
Icon=tcar-parser
Terminal=false
Categories=Development;
StartupWMClass=tcar-parser
StartupNotify=true
EOF
chmod +x "$ENTRY"
update-desktop-database "$APPS" 2>/dev/null || true
gtk-update-icon-cache -q "${ICONS%/scalable/apps}" 2>/dev/null || true
echo "app menu: $ENTRY"

if [[ $DESKTOP == 1 ]]; then
    mkdir -p "$DESKTOP_DIR"
    cp "$ENTRY" "$DESKTOP_DIR/tcar-parser.desktop"
    chmod +x "$DESKTOP_DIR/tcar-parser.desktop"
    gio set "$DESKTOP_DIR/tcar-parser.desktop" metadata::trusted true 2>/dev/null || true
    echo "desktop:  $DESKTOP_DIR/tcar-parser.desktop"
fi
