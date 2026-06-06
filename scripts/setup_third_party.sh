#!/usr/bin/env bash
# Clone the external dependencies that live under third_party/ but are NOT
# committed to this repository (they have their own upstream repos).
#
# Vendored copies of `ldm` and `taming` ARE committed and need no action here.
# This script only fetches RoMA V2 and CO3D, pinned to the commits this project
# was developed against.
#
# Usage:
#   bash scripts/setup_third_party.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

clone_pinned () {
  local url="$1" dir="$2" commit="$3"
  local dest="$TP/$dir"
  if [ -d "$dest/.git" ]; then
    echo "[skip] $dir already present at $dest"
    return
  fi
  echo "[clone] $url -> $dest"
  git clone "$url" "$dest"
  git -C "$dest" checkout "$commit"
}

# RoMA V2 — dense feature matching (used by src/data/warp_dataset.py via third_party/RoMA2/src)
clone_pinned "git@github.com:Parskatt/RoMaV2.git" "RoMA2" "7151f3846ad0c89c213afb6803966484a6dd76e0"

# CO3D — Common Objects in 3D dataset tools (Meta)
clone_pinned "https://github.com/facebookresearch/co3d.git" "co3d" "eb51d7583c56ff23dc918d9deafee50f4d8178c3"

echo
echo "Done. Make sure third_party/ and third_party/RoMA2/src are on PYTHONPATH"
echo "(this repo installs them via the conda env's site-packages .pth file)."
