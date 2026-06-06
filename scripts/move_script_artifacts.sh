#!/usr/bin/env bash
set -euo pipefail

# Move generated image and CSV artifacts from scripts/ to ../outputs/scripts/
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT_DIR/scripts"
DST_DIR="$ROOT_DIR/outputs/scripts"

mkdir -p "$DST_DIR"
shopt -s nullglob
moved=false
for ext in png csv; do
  files=("$SRC_DIR"/*.${ext})
  if [ ${#files[@]} -gt 0 ]; then
    mv "$SRC_DIR"/*.${ext} "$DST_DIR"/ || true
    moved=true
  fi
done

if [ "$moved" = true ]; then
  echo "Moved generated artifacts to $DST_DIR"
else
  echo "No png/csv artifacts found in $SRC_DIR"
fi
