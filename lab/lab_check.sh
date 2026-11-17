#!/usr/bin/env bash
# Read-only hardware, OS, GPU, network, and persistence diagnostics for the lab PCs.
# Always exits 0 and writes a timestamped report under $DORY_LAB_HOME.
set -euo pipefail

DORY_LAB_HOME="${DORY_LAB_HOME:-$HOME/dory-lab}"
mkdir -p "$DORY_LAB_HOME"

RAW_HOST="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "desconhecido")"
HOST="$(printf '%s' "$RAW_HOST" | tr -cd '[:alnum:]._-')"
[ -n "$HOST" ] || HOST="desconhecido"

TS="$(date '+%Y-%m-%d_%H%M')"
OUT_FILE="$DORY_LAB_HOME/lab_check_${HOST}_${TS}.txt"

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

version_lt() {
  # Returns 0 (true) if $1 < $2 for dotted numeric versions (major.minor).
  local a_maj a_min b_maj b_min
  a_maj="${1%%.*}"
  a_min="${1#*.}"
  a_min="${a_min%%[^0-9]*}"
  b_maj="${2%%.*}"
  b_min="${2#*.}"
  b_min="${b_min%%[^0-9]*}"
  a_maj="${a_maj:-0}"; a_min="${a_min:-0}"
  b_maj="${b_maj:-0}"; b_min="${b_min:-0}"
  if [ "$a_maj" -lt "$b_maj" ]; then
    return 0
  elif [ "$a_maj" -eq "$b_maj" ] && [ "$a_min" -lt "$b_min" ]; then
    return 0
  fi
  return 1
}

run_diagnostics() {
  local verdict_blocks=()
  local verdict_warns=()
  local verdict_oks=()

  echo "========================================================================"
  echo "DORY LAB CHECK - Host: $HOST | Data: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "Arquivo de saída: $OUT_FILE"
  echo "========================================================================"

  echo
  echo "--- 1. Sistema Operacional, Kernel e glibc ---"
  echo "Hostname: $(hostname 2>/dev/null || echo "$HOST")"
  if [ -r /etc/os-release ]; then
    cat /etc/os-release || true
  else
    echo "/etc/os-release não encontrado."
  fi
  if has_cmd lsb_release; then
    echo ""
    lsb_release -a 2>/dev/null || true
  fi
  echo "Kernel: $(uname -a 2>/dev/null || echo "desconhecido")"

  local glibc_line="" glibc_ver=""
  if has_cmd ldd; then
    glibc_line="$(ldd --version 2>&1 | head -n 1 || true)"
    echo "glibc: $glibc_line"
    glibc_ver="$(printf '%s' "$glibc_line" | grep -oE '[0-9]+\.[0-9]+' | tail -n 1 || true)"
  else
    echo "glibc: ldd não encontrado"
  fi
  if [ -n "$glibc_ver" ]; then
    if version_lt "$glibc_ver" "2.28"; then
      verdict_blocks+=("glibc $glibc_ver < 2.28 (wheels manylinux_2_28 não vão rodar neste sistema).")
    else
      verdict_oks+=("glibc $glibc_ver >= 2.28.")
    fi
  else
    verdict_warns+=("Não foi possível detectar a versão da glibc via ldd.")
  fi

  echo
  echo "--- 2. CPU, Memória RAM e Swap ---"
  local cpu_model="desconhecido"
  if [ -r /proc/cpuinfo ]; then
    cpu_model="$(grep -m1 -E '^(model name|Hardware|Processor)' /proc/cpuinfo | sed 's/^[^:]*:[[:space:]]*//' || true)"
  fi
  echo "CPU: ${cpu_model:-desconhecido} ($(nproc 2>/dev/null || echo '?') vCPUs)"
  if has_cmd free; then
    free -m || true
  fi
  local ram_mib=0
  if [ -r /proc/meminfo ]; then
    local mem_kib
    mem_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    ram_mib=$(( ${mem_kib:-0} / 1024 ))
  elif has_cmd free; then
    ram_mib="$(free -m | awk '/^Mem:/ {print $2}' 2>/dev/null || echo 0)"
  fi
  if [ "${ram_mib:-0}" -gt 0 ]; then
    local ram_gib_str
    ram_gib_str="$(awk -v m="$ram_mib" 'BEGIN {printf "%.2f", m/1024.0}')"
    if [ "$ram_mib" -lt 7168 ]; then
      verdict_warns+=("RAM total ${ram_gib_str} GiB (< 7 GiB): risco de OOM durante carga do modelo.")
    else
      verdict_oks+=("RAM total ${ram_gib_str} GiB (>= 7 GiB).")
    fi
  else
    verdict_warns+=("Não foi possível ler o total de RAM.")
  fi

  echo
  echo "--- 3. Disco e Tipo de Montagem ---"
  df -h "$HOME" /tmp 2>/dev/null || true
  if has_cmd findmnt; then
    echo ""
    echo "Ponto de montagem de \$HOME:"
    findmnt -T "$HOME" 2>/dev/null || true
  else
    echo ""
    echo "Sistema de arquivos de \$HOME (df -T):"
    df -T "$HOME" 2>/dev/null || true
  fi
  local home_avail_kib=0 home_avail_gib="0"
  home_avail_kib="$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)"
  home_avail_gib="$(awk -v k="${home_avail_kib:-0}" 'BEGIN {printf "%.1f", k/1048576.0}')"
  if [ "${home_avail_kib:-0}" -lt 31457280 ]; then
    verdict_blocks+=("Espaço livre em \$HOME é ${home_avail_gib} GiB (< 30 GiB necessários para wheels, checkpoint, repo e venv).")
  else
    verdict_oks+=("Espaço livre em \$HOME: ${home_avail_gib} GiB (>= 30 GiB).")
  fi

  echo
  echo "--- 4. GPU e Driver NVIDIA ---"
  if ! has_cmd nvidia-smi; then
    echo "nvidia-smi NÃO encontrado no PATH."
    verdict_blocks+=("nvidia-smi ausente: sem driver NVIDIA/GPU acessível.")
  else
    local smi_out=""
    if smi_out="$(nvidia-smi 2>&1)"; then
      printf '%s\n' "$smi_out"
      echo ""
      echo "Consulta estruturada (driver_version, name, memory.total, memory.used, memory.free, compute_cap):"
      local q_out=""
      q_out="$(nvidia-smi --query-gpu=driver_version,name,memory.total,memory.used,memory.free,compute_cap --format=csv,noheader 2>/dev/null || true)"
      printf '%s\n' "$q_out"

      echo ""
      echo "Processos de compute na GPU:"
      local procs_out=""
      procs_out="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
      if [ -n "$procs_out" ]; then
        printf '%s\n' "$procs_out"
      else
        echo "(nenhum processo compute listado; veja a tabela Processes acima para Xorg/desktop)"
      fi

      local first_line drv_ver gpu_name cap drv_major
      first_line="$(printf '%s\n' "$q_out" | head -n 1)"
      drv_ver="$(printf '%s' "$first_line" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$1); print $1}')"
      gpu_name="$(printf '%s' "$first_line" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2}')"
      cap="$(printf '%s' "$first_line" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$6); print $6}')"
      verdict_oks+=("GPU detectada: ${gpu_name:-desconhecida} (driver ${drv_ver:-?}, compute_cap ${cap:-?}).")

      if [ -n "$cap" ] && [ "$cap" != "7.5" ]; then
        verdict_warns+=("compute_cap=$cap (diferente de 7.5 esperado para RTX 2060 SUPER).")
      fi
      drv_major="${drv_ver%%.*}"
      if [ -n "$drv_major" ] && [ "$drv_major" -eq "$drv_major" ] 2>/dev/null; then
        if [ "$drv_major" -lt 580 ]; then
          verdict_warns+=("Driver NVIDIA $drv_ver (< 580): provavelmente antigo para torch cu130 (CUDA 13); o teste definitivo é o passo GPU do lab_setup.")
        fi
      fi
    else
      echo "nvidia-smi falhou ao executar:"
      printf '%s\n' "$smi_out"
      verdict_blocks+=("nvidia-smi falhou ao comunicar com o driver NVIDIA.")
    fi
  fi

  echo
  echo "--- 5. Ferramentas Disponíveis e Sudo ---"
  local tools=(python3 python3.13 uv git curl wget rclone tar unzip)
  local has_py313=0
  for t in "${tools[@]}"; do
    local found_path=""
    if has_cmd "$t"; then
      found_path="$(command -v "$t")"
    elif [ -x "$DORY_LAB_HOME/bin/$t" ]; then
      found_path="$DORY_LAB_HOME/bin/$t (local dory-lab)"
    fi

    if [ -n "$found_path" ]; then
      local ver=""
      case "$t" in
        python3)
          ver="($("$t" --version 2>&1 | head -n 1))"
          if "$t" -c 'import sys; exit(0 if sys.version_info[:2] == (3, 13) else 1)' 2>/dev/null; then
            has_py313=1
          fi
          ;;
        python3.13)
          ver="($("$t" --version 2>&1 | head -n 1))"
          has_py313=1
          ;;
        uv|git|curl|wget|rclone|tar|unzip)
          ver="($(${found_path%% *} --version 2>&1 | head -n 1 || true))"
          ;;
      esac
      echo "  [OK] $t -> $found_path $ver"
    else
      echo "  [--] $t -> ausente"
    fi
  done

  if has_cmd sudo; then
    if sudo -n true 2>/dev/null; then
      echo "sudo sem senha: sim"
    else
      echo "sudo sem senha: não"
    fi
  else
    echo "sudo sem senha: não (comando sudo ausente)"
  fi

  echo
  echo "--- 6. Conectividade Internet ---"
  local urls=(
    "https://pypi.org"
    "https://github.com"
    "https://codeload.github.com"
    "https://api.github.com"
    "https://astral.sh"
    "https://downloads.rclone.org"
    "https://huggingface.co"
    "https://www.googleapis.com"
  )
  local astral_ok=0
  if has_cmd curl; then
    for u in "${urls[@]}"; do
      local code=""
      if code="$(curl -sI -o /dev/null -w '%{http_code}' --max-time 8 "$u" 2>/dev/null)"; then
        echo "  [OK]     $u (HTTP $code)"
        if [ "$u" = "https://astral.sh" ]; then
          astral_ok=1
        fi
      else
        echo "  [FALHOU] $u"
      fi
    done
  else
    echo "  [FALHOU] curl ausente; não foi possível testar URLs HTTP."
  fi

  if [ "$astral_ok" -eq 0 ] && [ "$has_py313" -eq 0 ]; then
    verdict_blocks+=("Sem acesso a https://astral.sh E sem python3.13 local: não será possível baixar uv/Python 3.13.")
  elif [ "$astral_ok" -eq 1 ]; then
    verdict_oks+=("Acesso a astral.sh OK (ou python3.13 local disponível).")
  else
    verdict_oks+=("python3.13 disponível localmente.")
  fi

  echo
  echo "--- 7. Rede Local (LAN) ---"
  if has_cmd ip; then
    ip -4 -o addr show 2>/dev/null || true
  elif has_cmd hostname; then
    echo "IPs: $(hostname -I 2>/dev/null || echo 'desconhecido')"
  fi
  echo "Velocidade das interfaces (/sys/class/net/*/speed):"
  local any_iface=0
  for iface_path in /sys/class/net/*; do
    [ -e "$iface_path" ] || continue
    any_iface=1
    local ifname speed state
    ifname="$(basename "$iface_path")"
    state="$(cat "$iface_path/operstate" 2>/dev/null || echo '?')"
    speed="$(cat "$iface_path/speed" 2>/dev/null || echo 'N/A')"
    if [ "$speed" = "-1" ]; then
      speed="sem link"
    elif [ "$speed" != "N/A" ]; then
      speed="${speed} Mb/s"
    fi
    echo "  $ifname: estado=$state velocidade=$speed"
  done
  if [ "$any_iface" -eq 0 ]; then
    echo "  (nenhuma interface encontrada em /sys/class/net)"
  fi

  echo
  echo "--- 8. Persistência do Disco ---"
  local marker="$HOME/.dory_lab_marker"
  if [ -f "$marker" ]; then
    local created_ts
    created_ts="$(cat "$marker" 2>/dev/null || echo 'timestamp ilegível')"
    echo "Marcador encontrado ($marker): o disco persistiu desde $created_ts"
  else
    local now_ts
    now_ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf '%s\n' "$now_ts" > "$marker"
    echo "Marcador criado em $marker ($now_ts)."
    echo "IMPORTANTE: rode de novo depois de um reboot para confirmar se o Ubuntu guarda arquivos."
  fi

  echo
  echo "========================================================================"
  echo "VEREDITO"
  echo "========================================================================"
  local item
  if [ ${#verdict_oks[@]} -gt 0 ]; then
    for item in "${verdict_oks[@]}"; do
      echo "[OK]       $item"
    done
  fi
  if [ ${#verdict_warns[@]} -gt 0 ]; then
    for item in "${verdict_warns[@]}"; do
      echo "[ATENÇÃO]  $item"
    done
  fi
  if [ ${#verdict_blocks[@]} -gt 0 ]; then
    for item in "${verdict_blocks[@]}"; do
      echo "[BLOQUEIO] $item"
    done
    echo ""
    echo "Resumo: há ${#verdict_blocks[@]} BLOQUEIO(s) neste PC. Veja as linhas [BLOQUEIO] acima."
  else
    echo ""
    echo "Resumo: nenhum BLOQUEIO detectado. Este PC está apto a tentar o lab_setup.sh."
  fi
  echo "========================================================================"
  echo "Relatório salvo em: $OUT_FILE"
}

run_diagnostics 2>&1 | tee "$OUT_FILE" || true
exit 0
