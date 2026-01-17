#!/usr/bin/env bash

set -e

# Paths
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
UFBX_DIR="$SRC_DIR/../ufbx"
OUT_BIN="$SRC_DIR/fbx_to_json"

# Compile (always)
echo "Compiling fbx_to_json..."
clang++ "$SRC_DIR/fbx_to_json.cpp" "$UFBX_DIR/ufbx.c" \
  -std=c++17 -O2 -I"$UFBX_DIR" -o "$OUT_BIN"

echo "Build finished."

# Ask before execution
read -p "Execute fbx_to_json now? [y/n] " answer

case "$answer" in
  y|Y)
    read -p "Enter FBX file name (without extension): " name

    if [ -z "$name" ]; then
      echo "No name given. Aborting."
      exit 1
    fi

    IN_FBX="$SRC_DIR/../../engine/assets/models/${name}.fbx"

    if [ ! -f "$IN_FBX" ]; then
      echo "FBX file not found: $IN_FBX"
      exit 1
    fi

    echo "Running fbx_to_json..."
    # Exporter writes multiple JSON files automatically (mesh / skin / skeleton / animation)
    "$OUT_BIN" "$IN_FBX"

    echo "Expected outputs:"
    echo "  ${name}_mesh.json"
    echo "  ${name}_skin.json"
    echo "  ${name}_skeleton.json"
    echo "  ${name}_animation.json"
    echo "in engine/assets/models/"
    ;;
  *)
    echo "Execution skipped."
    ;;
esac
