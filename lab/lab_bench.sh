#!/usr/bin/env bash
# Benchmark sequence for RTX 2060 SUPER lab PCs with a hard session deadline.
# Usage: bash lab/lab_bench.sh [--deadline HH:MM] [--max-problems N] [--skip-quality]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DORY_LAB_HOME="${DORY_LAB_HOME:-$HOME/dory-lab}"
DEADLINE="17:30"
MAX_PROBLEMS=16
SKIP_QUALITY=0

usage() {
  cat <<'EOF'
Uso:
  bash lab/lab_bench.sh [--deadline HH:MM] [--max-problems N] [--skip-quality]

Opções:
  --deadline HH:MM   Horário-limite local para não iniciar novas rodadas (padrão: 17:30)
  --max-problems N   Máximo de problemas no quality (padrão: 16)
  --skip-quality     Roda apenas env e smokes (pula a geração longa F1a)
  -h, --help         Mostra esta ajuda
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --deadline)
      DEADLINE="${2:-}"
      [ -n "$DEADLINE" ] || { echo "ERRO: --deadline exige HH:MM" >&2; exit 1; }
      shift 2
      ;;
    --max-problems)
      MAX_PROBLEMS="${2:-}"
      [ -n "$MAX_PROBLEMS" ] || { echo "ERRO: --max-problems exige um número" >&2; exit 1; }
      shift 2
      ;;
    --skip-quality)
      SKIP_QUALITY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERRO: argumento desconhecido: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SETUP_OK_FILE="$DORY_LAB_HOME/setup_ok.json"
if [ ! -f "$SETUP_OK_FILE" ]; then
  echo "ERRO: $SETUP_OK_FILE não encontrado." >&2
  echo "O que fazer: rode primeiro 'bash lab/lab_setup.sh <origem>' antes do benchmark." >&2
  exit 1
fi

mkdir -p "$DORY_LAB_HOME/logs" "$DORY_LAB_HOME/tmp"
TS="$(date '+%Y-%m-%d_%H%M%S')"
LOG_FILE="$DORY_LAB_HOME/logs/bench_${TS}.log"

RAW_HOST="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "desconhecido")"
HOST="$(printf '%s' "$RAW_HOST" | tr -cd '[:alnum:]._-')"
[ -n "$HOST" ] || HOST="desconhecido"

DRIVE_DIR="$DORY_LAB_HOME/drive"
REPO_DIR="$DORY_LAB_HOME/repo"
VENV_PY="$DORY_LAB_HOME/.venv/bin/python"
RCLONE_BIN="$DORY_LAB_HOME/bin/rclone"
SOURCE_FILE="$DORY_LAB_HOME/source.txt"
T4_OUT="$DRIVE_DIR/results/t4_validation"
OUT_DIR="$DRIVE_DIR/results/lab_validation/$HOST"

if [ ! -x "$VENV_PY" ] || [ ! -f "$REPO_DIR/scripts/t4_validate.py" ]; then
  echo "ERRO: ambiente incompleto em $DORY_LAB_HOME (.venv ou repo/scripts/t4_validate.py ausente)." >&2
  echo "O que fazer: rode novamente 'bash lab/lab_setup.sh'." >&2
  exit 1
fi

export CODER_DRIVE_ROOT="$DRIVE_DIR"
export DORY_INT8_DIR="$DRIVE_DIR/checkpoints/dory_int8_gptq_rtn_g128"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_HOME="$DORY_LAB_HOME/hf_cache"
export HF_HUB_CACHE="$DORY_LAB_HOME/hf_cache/hub"
export HF_DATASETS_CACHE="$DORY_LAB_HOME/hf_cache/datasets"

if [ -z "${LAB_VRAM_GIB:-}" ]; then
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || echo "")"
  if [ -z "$FREE_MIB" ]; then
    echo "ERRO: não foi possível consultar memory.free via nvidia-smi." >&2
    exit 1
  fi
  LAB_VRAM_GIB="$("$VENV_PY" - "$FREE_MIB" <<'PY'
import sys
free_mib = float(sys.argv[1])
vram_gib = max(1.0, round(free_mib / 1024.0 - 0.35, 2))
print(f"{vram_gib:.2f}")
PY
)"
  VRAM_SOURCE="calculado automaticamente (VRAM livre ${FREE_MIB} MiB - 0.35 GiB)"
else
  VRAM_SOURCE="definido pelo usuário no ambiente"
fi
export LAB_VRAM_GIB

is_local_source() {
  local src="$1"
  [ -d "$src" ] && return 0
  case "$src" in
    /*|./*|../*) return 0 ;;
    *:*) return 1 ;;
    *) return 0 ;;
  esac
}

sync_quality_to_out() {
  if [ -d "$T4_OUT/quality" ]; then
    mkdir -p "$OUT_DIR/quality"
    cp -a "$T4_OUT/quality/." "$OUT_DIR/quality/"
  fi
}

run_smoke_mode() {
  local mode_label="$1"
  local flag="$2"
  local dest_json="$3"
  local dest_err="$4"
  local tmp_log="$DORY_LAB_HOME/tmp/smoke_${mode_label}.log"

  rm -f "$T4_OUT/smoke_int8.json" "$dest_err"
  set +e
  "$VENV_PY" scripts/t4_validate.py smoke --variant int8 "$flag" 2>&1 | tee "$tmp_log"
  local rc=${PIPESTATUS[0]}
  set -e

  if [ "$rc" -eq 0 ] && [ -f "$T4_OUT/smoke_int8.json" ]; then
    mv "$T4_OUT/smoke_int8.json" "$dest_json"
    return 0
  fi

  {
    echo "exit_code=$rc"
    echo "--- últimas 40 linhas do log ---"
    tail -n 40 "$tmp_log" 2>/dev/null || true
  } > "$dest_err"
  return 1
}

main_bench() {
  local interrupted=0
  local drv_ver=""
  local git_sha=""
  drv_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || echo '')"
  git_sha="$(cat "$REPO_DIR/.git_sha" 2>/dev/null | tr -d '[:space:]' || echo '')"

  handle_interrupt() {
    if [ "$interrupted" -eq 1 ]; then
      return
    fi
    interrupted=1
    echo ""
    echo "[INTERROMPIDO] Ctrl-C recebido. Preservando todo o progresso concluído até aqui..."
    sync_quality_to_out || true
    if [ -x "$VENV_PY" ] && [ -f "$SCRIPT_DIR/bench_summary.py" ]; then
      "$VENV_PY" "$SCRIPT_DIR/bench_summary.py" \
        --out-dir "$OUT_DIR" \
        --drive-root "$DRIVE_DIR" \
        --repo-root "$REPO_DIR" \
        --hostname "$HOST" \
        --driver-version "$drv_ver" \
        --git-sha "$git_sha" \
        --lab-vram-gib "$LAB_VRAM_GIB" || true
    fi
    echo "Progresso preservado em:"
    echo "  - $T4_OUT/quality/"
    echo "  - $OUT_DIR/"
    exit 130
  }
  trap handle_interrupt INT TERM

  mkdir -p "$OUT_DIR" "$T4_OUT"
  cd "$REPO_DIR"

  local deadline_epoch now_epoch
  deadline_epoch="$("$VENV_PY" - "$DEADLINE" <<'PY'
import sys
from datetime import datetime
hh, mm = map(int, sys.argv[1].split(":"))
now = datetime.now()
dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
print(int(dt.timestamp()))
PY
)"
  now_epoch="$(date +%s)"

  echo "========================================================================"
  echo "DORY LAB BENCH"
  echo "  Host          : $HOST"
  echo "  Saída         : $OUT_DIR"
  echo "  LAB_VRAM_GIB  : $LAB_VRAM_GIB ($VRAM_SOURCE)"
  echo "  Deadline      : $DEADLINE ($(($((deadline_epoch - now_epoch)) / 60)) min restantes)"
  echo "  Max problemas : $MAX_PROBLEMS"
  echo "  Skip quality  : $SKIP_QUALITY"
  echo "  Log           : $LOG_FILE"
  echo "========================================================================"

  if "$VENV_PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) < 6.5 else 1)" "$LAB_VRAM_GIB"; then
    echo "ATENÇÃO: LAB_VRAM_GIB=$LAB_VRAM_GIB está baixo (algo está ocupando VRAM, ex. desktop/Xorg)."
    echo "Se der OOM, saia da sessão gráfica e rode em um TTY (Ctrl+Alt+F3)."
  fi

  # --- Step 1: env ---
  echo ""
  echo "=== [1/7] Coletar ambiente (t4_validate.py env) ==="
  if ! "$VENV_PY" scripts/t4_validate.py env || [ ! -f "$T4_OUT/env.json" ]; then
    echo "ERRO: t4_validate.py env falhou. O ambiente não está pronto; rode de novo 'bash lab/lab_setup.sh'."
    return 1
  fi
  cp "$T4_OUT/env.json" "$OUT_DIR/env.json"

  # --- Step 2: smoke eager ---
  echo ""
  echo "=== [2/7] Smoke INT8 com --enforce-eager ==="
  local eager_ok=0
  if run_smoke_mode "eager" "--enforce-eager" "$OUT_DIR/smoke_int8_eager.json" "$OUT_DIR/smoke_int8_eager.error.txt"; then
    eager_ok=1
    echo "[OK] Smoke eager salvo em $OUT_DIR/smoke_int8_eager.json"
  else
    echo "[FALHOU] Smoke eager falhou. Detalhes salvos em $OUT_DIR/smoke_int8_eager.error.txt"
  fi

  # --- Step 3: smoke CUDA graphs ---
  echo ""
  echo "=== [3/7] Smoke INT8 com CUDA graphs (--no-enforce-eager) ==="
  local graphs_ok=0
  if run_smoke_mode "graphs" "--no-enforce-eager" "$OUT_DIR/smoke_int8_graphs.json" "$OUT_DIR/smoke_int8_graphs.error.txt"; then
    graphs_ok=1
    echo "[OK] Smoke CUDA graphs salvo em $OUT_DIR/smoke_int8_graphs.json"
  else
    echo "[FALHOU] Smoke CUDA graphs falhou (seguindo normalmente). Detalhes em $OUT_DIR/smoke_int8_graphs.error.txt"
  fi

  # --- Step 4: compare smokes and choose mode ---
  echo ""
  echo "=== [4/7] Comparação Eager vs CUDA Graphs ==="
  local chosen_flag=""
  chosen_flag="$("$VENV_PY" - "$OUT_DIR/smoke_int8_eager.json" "$OUT_DIR/smoke_int8_graphs.json" <<'PY'
import json
import sys
from pathlib import Path

def load(p):
    path = Path(p)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

e = load(sys.argv[1])
g = load(sys.argv[2])

print(f"{'modo':<14} {'status':<8} {'tok_s':>8} {'kv_tokens':>11} {'max_conc':>9} {'load_s':>8}")
print("-" * 64)
for name, d in [("eager", e), ("cuda_graphs", g)]:
    if d is None:
        print(f"{name:<14} {'FALHOU':<8} {'—':>8} {'—':>11} {'—':>9} {'—':>8}")
    else:
        print(
            f"{name:<14} {'OK':<8} "
            f"{str(d.get('tok_s', '—')):>8} "
            f"{str(d.get('kv_cache_size_tokens', '—')):>11} "
            f"{str(d.get('max_concurrency', '—')):>9} "
            f"{str(d.get('load_s', '—')):>8}"
        )

if e is not None and g is not None:
    print("--no-enforce-eager" if (g.get("tok_s") or 0) > (e.get("tok_s") or 0) else "--enforce-eager")
elif g is not None:
    print("--no-enforce-eager")
elif e is not None:
    print("--enforce-eager")
else:
    print("NONE")
PY
)"
  local selected_flag
  selected_flag="$(printf '%s\n' "$chosen_flag" | tail -n 1)"
  printf '%s\n' "$chosen_flag" | sed '$d'

  if [ "$selected_flag" = "NONE" ]; then
    echo ""
    echo "ERRO: ambos os smokes (eager e CUDA graphs) falharam."
    echo "O que fazer: veja smoke_int8_eager.error.txt e smoke_int8_graphs.error.txt em $OUT_DIR."
    echo "Se for OOM por causa do desktop/Xorg usando VRAM, saia da sessão gráfica (Ctrl+Alt+F3) e rode novamente."
    "$VENV_PY" "$SCRIPT_DIR/bench_summary.py" \
      --out-dir "$OUT_DIR" \
      --drive-root "$DRIVE_DIR" \
      --repo-root "$REPO_DIR" \
      --hostname "$HOST" \
      --driver-version "$drv_ver" \
      --git-sha "$git_sha" \
      --lab-vram-gib "$LAB_VRAM_GIB" || true
    return 1
  fi
  echo "Modo escolhido para o quality: $selected_flag"

  # --- Step 5: quality loop with hard deadline ---
  echo ""
  echo "=== [5/7] Qualidade & Throughput longo F1a ($selected_flag) ==="
  if [ "$SKIP_QUALITY" -eq 1 ]; then
    echo "[PULADO] --skip-quality informado."
  else
    local samples_file="$T4_OUT/quality/samples_efficient_int8.jsonl"
    local done_count=0
    if [ -f "$samples_file" ]; then
      done_count="$("$VENV_PY" - "$samples_file" <<'PY'
import sys
from pathlib import Path
from lib.io import ResumableProgress
print(len(ResumableProgress(Path(sys.argv[1]))))
PY
)"
    fi

    local next_n=$((done_count + 1))
    local last_run_s=0
    if [ "$done_count" -ge "$MAX_PROBLEMS" ]; then
      echo "[PULADO] Já há $done_count problemas concluídos (>= --max-problems $MAX_PROBLEMS)."
    else
      while [ "$next_n" -le "$MAX_PROBLEMS" ]; do
        now_epoch="$(date +%s)"
        if [ "$last_run_s" -eq 0 ]; then
          if [ "$now_epoch" -ge "$deadline_epoch" ]; then
            echo "[DEADLINE] Horário atual já atingiu ou passou o limite $DEADLINE. Não iniciando problema $next_n."
            break
          fi
        else
          local can_start
          can_start="$("$VENV_PY" - "$now_epoch" "$last_run_s" "$deadline_epoch" <<'PY'
import sys
now_s, last_s, dead_s = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print("1" if (now_s + 1.3 * last_s) < dead_s else "0")
PY
)"
          if [ "$can_start" != "1" ]; then
            echo "[DEADLINE] Próxima rodada (estimativa 1.3 × ${last_run_s}s) cruzaria o limite $DEADLINE. Parando com segurança."
            break
          fi
        fi

        echo ""
        echo "--- Rodando quality --n-problems $next_n --batch 1 ($selected_flag) ---"
        local t_start t_end
        t_start="$(date +%s)"
        local q_log="$DORY_LAB_HOME/tmp/quality_${next_n}.log"
        "$VENV_PY" scripts/t4_validate.py quality --variant int8 --n-problems "$next_n" --batch 1 "$selected_flag" 2>&1 | tee "$q_log"
        local q_rc=${PIPESTATUS[0]}
        t_end="$(date +%s)"
        if [ "$q_rc" -ne 0 ]; then
          # main_bench roda com errexit desligado: sem este break, uma falha se repetiria até --max-problems
          {
            echo "exit_code=$q_rc (quality --n-problems $next_n)"
            echo "--- últimas 40 linhas do log ---"
            tail -n 40 "$q_log" 2>/dev/null || true
          } > "$OUT_DIR/quality.error.txt"
          echo "[FALHOU] quality --n-problems $next_n saiu com código $q_rc. Detalhes em $OUT_DIR/quality.error.txt. Parando o loop."
          sync_quality_to_out
          break
        fi
        last_run_s=$((t_end - t_start))
        [ "$last_run_s" -ge 1 ] || last_run_s=1
        echo "Rodada --n-problems $next_n concluída em ${last_run_s}s."
        sync_quality_to_out
        next_n=$((next_n + 1))
      done
    fi
  fi

  # --- Step 6: summary ---
  echo ""
  echo "=== [6/7] Copiar resultados e gerar bench_summary ==="
  sync_quality_to_out
  "$VENV_PY" "$SCRIPT_DIR/bench_summary.py" \
    --out-dir "$OUT_DIR" \
    --drive-root "$DRIVE_DIR" \
    --repo-root "$REPO_DIR" \
    --hostname "$HOST" \
    --driver-version "$drv_ver" \
    --git-sha "$git_sha" \
    --lab-vram-gib "$LAB_VRAM_GIB"

  echo ""
  cat "$OUT_DIR/bench_summary.md"

  # --- Step 7: upload or local notice ---
  echo ""
  echo "=== [7/7] Entrega dos resultados ==="
  local origem=""
  if [ -s "$SOURCE_FILE" ]; then
    origem="$(head -n 1 "$SOURCE_FILE" | tr -d '\r\n')"
  fi

  if [ -n "$origem" ] && ! is_local_source "$origem" && [ -x "$RCLONE_BIN" ]; then
    local remote_dest="${origem%/}/results/lab_validation/$HOST"
    echo "Enviando $OUT_DIR -> $remote_dest ..."
    if "$RCLONE_BIN" copy --progress "$OUT_DIR" "$remote_dest"; then
      echo "[OK] Resultados enviados para $remote_dest"
    else
      echo "AVISO: falha no upload via rclone. Os arquivos locais continuam salvos em: $OUT_DIR"
    fi
  else
    echo "Origem local ou rclone indisponível; copie manualmente os arquivos de:"
    echo "  $OUT_DIR"
  fi
  echo "========================================================================"
}

set +e
main_bench 2>&1 | tee -i "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"
