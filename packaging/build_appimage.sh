#!/usr/bin/env bash
# Build dist/TCAR_Parser-x86_64.AppImage: the GUI, the converters and their Python in one file.
#
#   packaging/build_appimage.sh
#
# Needs uv (for a relocatable CPython and the wheels), network access, and FUSE to run
# appimagetool (downloaded to build/appimage/ on first use). The AppImage runs the GUI;
# `TCAR_Parser-x86_64.AppImage bag2nuscenes <args>` (or bag2nuscenes_full, yield_check,
# remove_log) runs that CLI instead.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYVER="${PYVER:-3.12}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
BUILD="$REPO/build/appimage"
APPDIR="$BUILD/AppDir"
OUT="$REPO/dist/TCAR_Parser-x86_64.AppImage"
TOOL="$BUILD/appimagetool-x86_64.AppImage"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/app" "$REPO/dist"

echo "== Python $PYVER (python-build-standalone via uv: relocatable)"
"$UV" python install "$PYVER"
PYBIN="$("$UV" python find --system --managed-python "$PYVER")"   # not a venv
PYROOT="$(dirname "$(dirname "$(readlink -f "$PYBIN")")")"
cp -a "$PYROOT" "$APPDIR/usr/python"
rm -f "$APPDIR"/usr/python/lib/python3*/EXTERNALLY-MANAGED
PY="$APPDIR/usr/python/bin/python3"

echo "== dependencies (pyproject.toml + gui extra)"
# nuscenes-devkit brings opencv-python-headless; drop opencv-python, whose bundled Qt
# would sit next to PyQt5's and which installs the same cv2 package anyway.
printf 'opencv-python; python_version < "0"\n' > "$BUILD/overrides.txt"
# Same versions as the checkout's tested .venv when there is one; devkit 1.1.x pins
# a shapely that does not build on Python 3.12.
echo "nuscenes-devkit>=1.2" > "$BUILD/constraints.txt"
if [[ -x "$REPO/.venv/bin/python" ]]; then
    "$UV" pip freeze --python "$REPO/.venv/bin/python" --exclude-editable \
        | grep -viE '^opencv-python==' >> "$BUILD/constraints.txt"
fi
"$UV" pip install --python "$PY" --override "$BUILD/overrides.txt" \
    --constraint "$BUILD/constraints.txt" -r "$REPO/pyproject.toml" --extra gui

echo "== app"
cp "$REPO"/*.py "$APPDIR/usr/app/"
cp -r "$REPO/assets" "$REPO/curation" "$REPO/msg" "$REPO/packet_decoder" "$REPO/scripts" "$APPDIR/usr/app/"
find "$APPDIR" -name __pycache__ -type d -prune -exec rm -rf {} +
# bytecode now: the image is read-only at run time
"$PY" -m compileall -q -j 0 "$APPDIR/usr/app" "$APPDIR"/usr/python/lib/python3*/site-packages >/dev/null || true

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
# TCAR Parser: the GUI, or one of the CLIs by name (bag2nuscenes, bag2nuscenes_full,
# yield_check, remove_log).
HERE="$(dirname "$(readlink -f "$0")")"
PY="$HERE/usr/python/bin/python3"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
case "${1:-}" in
    bag2nuscenes|bag2nuscenes_full|yield_check|remove_log)
        tool="$1"; shift
        exec "$PY" "$HERE/usr/app/$tool.py" "$@" ;;
esac
exec "$PY" "$HERE/usr/app/gui.py" "$@"
EOF
chmod +x "$APPDIR/AppRun"
cp "$REPO/assets/tcar-parser.svg" "$APPDIR/tcar-parser.svg"
ln -sf tcar-parser.svg "$APPDIR/.DirIcon"
cat > "$APPDIR/tcar-parser.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=TCAR Parser
Name[ko]=티카 파서
Comment=T-Car ROS 1 bags to nuScenes
Comment[ko]=티카 ROS 1 bag을 nuScenes 데이터셋으로 변환
Exec=AppRun
Icon=tcar-parser
Terminal=false
Categories=Development;
StartupWMClass=tcar-parser
EOF

echo "== appimagetool"
if [[ ! -x "$TOOL" ]]; then
    curl -fL -o "$TOOL" \
        https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$TOOL"
fi
rm -f "$OUT"
ARCH=x86_64 "$TOOL" --no-appstream "$APPDIR" "$OUT"
ls -lh "$OUT"
