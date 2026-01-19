#!/usr/bin/env bash

set -e

# Paths
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
UFBX_DIR="$SRC_DIR/../ufbx"
OUT_BIN="$SRC_DIR/fbx_loader"

# Compile (always)
echo "Compiling fbx_loader..."
clang++ "$SRC_DIR/fbx_loader.cpp" "$UFBX_DIR/ufbx.c" \
  -std=c++17 -O2 -I"$UFBX_DIR" -o "$OUT_BIN"

echo "Build finished."

# Ask before execution
read -p "Execute fbx_loader now? [y/n] " answer

case "$answer" in
  y|Y)
    read -p "Enter FBX file name (without extension): " name

    if [ -z "$name" ]; then
      echo "No name given. Aborting."
      exit 1
    fi

    IN_FBX="$SRC_DIR/../../assets/models/${name}.fbx"
    OUT_JSON="$SRC_DIR/../../assets/models/${name}.json"

    if [ ! -f "$IN_FBX" ]; then
      echo "FBX file not found: $IN_FBX"
      exit 1
    fi

    echo "Running fbx_loader..."
    "$OUT_BIN" "$IN_FBX" > "$OUT_JSON"

    echo "Written: $OUT_JSON"
    ;;
  *)
    echo "Execution skipped."
    ;;
esac
