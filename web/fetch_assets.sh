#!/usr/bin/env bash
# Optional: copy the model next to the page, for a demo that does not load it
# from Hugging Face (offline use, or your own server).
#
#   bash fetch_assets.sh
#
# Then set BASE = "assets/" at the top of public/worker.js.
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://huggingface.co/henryhale/tiny-voca-lg/resolve/main"
mkdir -p public/assets
curl -fsSL -o public/assets/model.onnx "$BASE/model.onnx"
curl -fsSL -o public/assets/model.json "$BASE/model.json"
ls -l public/assets
