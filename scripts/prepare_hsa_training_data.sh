#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DATA_DIR="${DATA_DIR:-data/train/vidprom_filtered_extended}"
export SOURCE_URL="${SOURCE_URL:-https://huggingface.co/gdhe17/Self-Forcing/resolve/main/vidprom_filtered_extended.txt}"
export SOURCE_SHA256="${SOURCE_SHA256:-7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5}"
export OUTPUT_SHA256="${OUTPUT_SHA256:-c5ca345c5cb83db295dee0dda0f06530032e5ea2fe0e83c6fe686a4111b02623}"
export OUTPUT_PROMPTS="${OUTPUT_PROMPTS:-248217}"
export PYTHON="${PYTHON:-python3}"

source_file="${DATA_DIR}/source_prompts.txt"
output_file="${DATA_DIR}/prompts_train.txt"
manifest_file="${DATA_DIR}/manifest.json"
mkdir -p "${DATA_DIR}"

if [[ ! -f "${source_file}" ]]; then
  echo "[download] ${SOURCE_URL}"
  curl --fail --location --retry 5 --continue-at - \
    "${SOURCE_URL}" --output "${source_file}"
fi

actual_sha="$(${PYTHON} - "${source_file}" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
if [[ "${actual_sha}" != "${SOURCE_SHA256}" ]]; then
  echo "[error] source SHA256 mismatch: expected ${SOURCE_SHA256}, got ${actual_sha}" >&2
  exit 2
fi

"${PYTHON}" scripts/prepare_hsa_training_prompts.py \
  --source "${source_file}" \
  --output "${output_file}" \
  --manifest "${manifest_file}" \
  --exclude data/benchmarks/vbench_standard/full/prompts.txt \
  --exclude data/benchmarks/vbench_augmented/full/prompts.txt

"${PYTHON}" - "${manifest_file}" "${OUTPUT_SHA256}" "${OUTPUT_PROMPTS}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected_sha, expected_count = sys.argv[2], int(sys.argv[3])
if manifest["output_sha256"] != expected_sha:
    raise SystemExit(
        f"output SHA256 mismatch: expected {expected_sha}, got {manifest['output_sha256']}"
    )
if manifest["output_prompts"] != expected_count:
    raise SystemExit(
        f"output prompt count mismatch: expected {expected_count}, "
        f"got {manifest['output_prompts']}"
    )
PY

echo "[done] prompts=${output_file}"
echo "[done] manifest=${manifest_file}"
