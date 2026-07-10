#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$ROOT/eval}"
RUN_WEB="${RUN_WEB:-1}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

mkdir -p "$OUT_DIR"

echo "== Collect eval tables =="
"$PYTHON_BIN" "$ROOT/utils/collect_eval.py" --out "$OUT_DIR"

if [[ "$RUN_WEB" == "1" ]]; then
  echo ""
  echo "== Build pipeline web =="
  ACTIVEFLY_STATIC_BUILD_ONLY=1 "$PYTHON_BIN" "$ROOT/pipeline_web.py" --static
fi

echo ""
echo "Eval outputs written to: $OUT_DIR"
