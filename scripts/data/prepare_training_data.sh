#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export DATA_DIR="${DATA_DIR:-data/train/vidprom_filtered_extended}"
export SOURCE_URL="${SOURCE_URL:-https://huggingface.co/gdhe17/Self-Forcing/resolve/main/vidprom_filtered_extended.txt}"
export SOURCE_SHA256="${SOURCE_SHA256:-7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5}"
export SOURCE_PROMPTS="${SOURCE_PROMPTS:-248221}"
export OUTPUT_SHA256="${OUTPUT_SHA256:-c5ca345c5cb83db295dee0dda0f06530032e5ea2fe0e83c6fe686a4111b02623}"
export OUTPUT_PROMPTS="${OUTPUT_PROMPTS:-248217}"
export PYTHON="${PYTHON:-python3}"
export ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

source_file="${SOURCE_FILE:-${DATA_DIR}/source_prompts.txt}"
output_file="${DATA_DIR}/prompts_train.txt"
manifest_file="${DATA_DIR}/manifest.json"
mkdir -p "${DATA_DIR}"

validate_prompt_file() {
  local path="$1"
  local expected_sha="$2"
  local expected_count="$3"
  "${PYTHON}" - "${path}" "${expected_sha}" "${expected_count}" <<'PY'
import hashlib
import sys

path, expected_sha, expected_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
digest = hashlib.sha256()
count = 0
with open(path, "rb") as handle:
    for line in handle:
        digest.update(line)
        if line.strip():
            count += 1
actual_sha = digest.hexdigest()
if actual_sha != expected_sha or count != expected_count:
    raise SystemExit(1)
PY
}

if [[ -f "${output_file}" ]] && validate_prompt_file "${output_file}" "${OUTPUT_SHA256}" "${OUTPUT_PROMPTS}"; then
  echo "[reuse] validated prepared prompts; no network access required"
  echo "[done] prompts=${output_file}"
  if [[ -f "${manifest_file}" ]]; then
    echo "[done] manifest=${manifest_file}"
  fi
  exit 0
fi

if [[ ! -f "${source_file}" ]]; then
  for candidate in \
    "${DATA_DIR}/vidprom_filtered_extended.txt" \
    "data/train/vidprom_filtered_extended.txt" \
    "vidprom_filtered_extended.txt"; do
    if [[ -f "${candidate}" ]]; then
      source_file="${candidate}"
      echo "[reuse] local source=${source_file}"
      break
    fi
  done
fi

if [[ ! -f "${source_file}" && "${ALLOW_DOWNLOAD}" != "1" ]]; then
  echo "[error] local training prompts were not found and network download is disabled" >&2
  echo "[hint] place the prepared file at ${output_file}" >&2
  echo "[hint] or place vidprom_filtered_extended.txt under ${DATA_DIR}" >&2
  echo "[hint] set SOURCE_FILE=/path/to/vidprom_filtered_extended.txt for another local path" >&2
  echo "[hint] set ALLOW_DOWNLOAD=1 only on a host that can access Hugging Face" >&2
  exit 2
fi

if [[ ! -f "${source_file}" ]]; then
  echo "[download] ${SOURCE_URL}"
  curl --fail --location --retry 5 --continue-at - \
    "${SOURCE_URL}" --output "${source_file}"
fi

if ! validate_prompt_file "${source_file}" "${SOURCE_SHA256}" "${SOURCE_PROMPTS}"; then
  echo "[error] source prompt file failed SHA256/count validation: ${source_file}" >&2
  exit 2
fi

"${PYTHON}" scripts/data/prepare_prompts.py \
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
