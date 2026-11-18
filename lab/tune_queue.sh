#!/usr/bin/env bash
# Runs tune_vllm.py once per line of a queue file (args per line; '#' comments). Skips labels already done OK.
# Usage: bash lab/tune_queue.sh <queue_file>
set -u
H="$HOME/dory-lab"; PY="$H/.venv/bin/python"; Q="$1"
export CODER_DRIVE_ROOT="$H/drive" DORY_INT8_DIR="$H/drive/checkpoints/dory_int8_gptq_rtn_g128" VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="$H/hf_cache" HF_HUB_CACHE="$H/hf_cache/hub" HF_DATASETS_CACHE="$H/hf_cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$H/tune"
cd "$H/repo"
while IFS= read -r line; do
  line="${line%%#*}"; [ -z "${line// }" ] && continue
  label="$(sed -n 's/.*--label \([^ ]*\).*/\1/p' <<<"$line")"
  if [ -n "$label" ] && grep -q '"status": "OK"' "$H/tune/$label.json" 2>/dev/null; then echo "=== [pulado] $label (já OK) ==="; continue; fi
  # wait for the previous engine to release the GPU (EngineCore can outlive its parent for a few s)
  for _ in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)"
    [ "${used:-9999}" -lt 1200 ] && break
    pkill -f "VLLM::EngineCore" 2>/dev/null; sleep 2
  done
  export LAB_VRAM_GIB="${FIXED_LAB_VRAM_GIB:-$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | awk '{printf "%.2f", $1/1024-0.35}')}"
  echo "=== [tune] $label === $(date +%H:%M:%S) (VRAM em uso antes: ${used} MiB)"
  # shellcheck disable=SC2086
  timeout 45m "$PY" lab/tune_vllm.py $line 2>&1
  sleep 5
done < "$Q"
echo "=== [tune] fila concluída $(date +%H:%M:%S) ==="
