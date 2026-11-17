#!/usr/bin/env bash
# Idempotent, no-sudo setup for the Dory 4B INT8 lab environment under $DORY_LAB_HOME.
# Usage: bash lab/lab_setup.sh [<origem>] [--allow-index]
set -euo pipefail

DORY_LAB_HOME="${DORY_LAB_HOME:-$HOME/dory-lab}"
ALLOW_INDEX=0
ORIGEM_ARG=""

usage() {
  cat <<'EOF'
Uso:
  bash lab/lab_setup.sh <origem> [--allow-index]
  bash lab/lab_setup.sh [--allow-index]   (reutiliza $DORY_LAB_HOME/source.txt)

Exemplos de <origem>:
  gdrive:coder-comprimido      (caminho rclone do Google Drive)
  /media/usb/coder-comprimido  (diretório local / pendrive com layout drive/)

Opções:
  --allow-index  Permite consultar o PyPI apenas para dependências faltantes
                 se a instalação offline das wheels falhar.
  -h, --help     Mostra esta ajuda.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --allow-index)
      ALLOW_INDEX=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "ERRO: opção desconhecida: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [ -n "$ORIGEM_ARG" ]; then
        echo "ERRO: mais de uma <origem> informada ('$ORIGEM_ARG' e '$1')." >&2
        usage >&2
        exit 1
      fi
      ORIGEM_ARG="$1"
      shift
      ;;
  esac
done

mkdir -p "$DORY_LAB_HOME" "$DORY_LAB_HOME/bin" "$DORY_LAB_HOME/drive" "$DORY_LAB_HOME/logs" "$DORY_LAB_HOME/tmp"

TS="$(date '+%Y-%m-%d_%H%M%S')"
LOG_FILE="$DORY_LAB_HOME/logs/setup_${TS}.log"
SOURCE_FILE="$DORY_LAB_HOME/source.txt"

if [ -n "$ORIGEM_ARG" ]; then
  printf '%s\n' "$ORIGEM_ARG" > "$SOURCE_FILE"
  ORIGEM="$ORIGEM_ARG"
elif [ -s "$SOURCE_FILE" ]; then
  ORIGEM="$(head -n 1 "$SOURCE_FILE" | tr -d '\r\n')"
else
  echo "ERRO: <origem> não informada e $SOURCE_FILE não existe." >&2
  usage >&2
  exit 1
fi

is_local_source() {
  local src="$1"
  if [ -d "$src" ]; then
    return 0
  fi
  case "$src" in
    /*|./*|../*)
      return 0
      ;;
    *:*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

DRIVE_DIR="$DORY_LAB_HOME/drive"
WHEELS_DIR="$DRIVE_DIR/wheels/py313-vllm-pr60149"
CKPT_DIR="$DRIVE_DIR/checkpoints/dory_int8_gptq_rtn_g128"
DATA_F1A_DIR="$DRIVE_DIR/data/f1a"
PROBLEMS_FILE="$DATA_F1A_DIR/problems_v2.jsonl"
REF_SAMPLES_FILE="$DATA_F1A_DIR/samples_efficient_L2_n8_t1_p0.95_m8192.jsonl"
REPO_DIR="$DORY_LAB_HOME/repo"
VENV_DIR="$DORY_LAB_HOME/.venv"
VENV_PY="$VENV_DIR/bin/python"
UV_BIN="$DORY_LAB_HOME/bin/uv"
RCLONE_BIN="$DORY_LAB_HOME/bin/rclone"
SETUP_OK_FILE="$DORY_LAB_HOME/setup_ok.json"

export UV_CACHE_DIR="$DORY_LAB_HOME/uv_cache"
export UV_PYTHON_INSTALL_DIR="$DORY_LAB_HOME/uv_python"
export UV_TOOL_DIR="$DORY_LAB_HOME/uv_tools"
export HF_HOME="$DORY_LAB_HOME/hf_cache"
export HF_HUB_CACHE="$DORY_LAB_HOME/hf_cache/hub"
export HF_DATASETS_CACHE="$DORY_LAB_HOME/hf_cache/datasets"

check_drive_complete() {
  [ -f "$WHEELS_DIR/OK" ] || return 1
  local whl_count
  whl_count="$(find "$WHEELS_DIR" -maxdepth 1 -type f -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')"
  [ "${whl_count:-0}" -ge 200 ] || return 1
  [ -s "$CKPT_DIR/config.json" ] || return 1
  [ -s "$CKPT_DIR/model-00001.safetensors" ] || return 1
  [ -s "$CKPT_DIR/model-00002.safetensors" ] || return 1
  [ -s "$PROBLEMS_FILE" ] || return 1
  [ -s "$REF_SAMPLES_FILE" ] || return 1
  return 0
}

main_setup() {
  local started_iso
  started_iso="$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')"
  local any_step_changed=0

  echo "========================================================================"
  echo "DORY LAB SETUP"
  echo "  DORY_LAB_HOME : $DORY_LAB_HOME"
  echo "  Origem        : $ORIGEM"
  echo "  Allow index   : $ALLOW_INDEX"
  echo "  Log           : $LOG_FILE"
  echo "========================================================================"

  # --- Step 1: Pre-checks ---
  local t0=$SECONDS
  echo ""
  echo "=== [Passo 1/9] Pré-checagens (GPU e espaço em disco) ==="
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERRO: comando 'nvidia-smi' não encontrado."
    echo "O que fazer: este PC não tem driver NVIDIA acessível. Escolha outro PC do lab ou peça ao suporte para instalar o driver."
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERRO: 'nvidia-smi' falhou ao consultar a GPU."
    echo "O que fazer: o driver NVIDIA pode estar quebrado ou a GPU indisponível. Reinicie o PC ou troque de máquina."
    return 1
  fi

  local free_kib existing_kib total_budget_kib free_gib
  free_kib="$(df -Pk "$DORY_LAB_HOME" | awk 'NR==2 {print $4}')"
  existing_kib="$(du -sk "$DRIVE_DIR" "$VENV_DIR" 2>/dev/null | awk '{s+=$1} END {print s+0}')"
  total_budget_kib=$(( free_kib + existing_kib ))
  free_gib="$(awk -v k="$free_kib" 'BEGIN {printf "%.1f", k/1048576.0}')"
  # Require >= 30 GiB total headroom (free + already populated drive/venv) and >= 2 GiB currently free.
  if [ "$total_budget_kib" -lt 31457280 ] || [ "$free_kib" -lt 2097152 ]; then
    echo "ERRO: espaço livre insuficiente no filesystem de $DORY_LAB_HOME (${free_gib} GiB livres)."
    echo "O que fazer: são necessários pelo menos 30 GiB livres para wheels, checkpoint INT8, repo e venv. Libere espaço em \$HOME ou defina DORY_LAB_HOME em outro disco."
    return 1
  fi
  echo "[OK] GPU visível e espaço suficiente (${free_gib} GiB livres agora). ($((SECONDS - t0))s)"

  # --- Step 2: uv ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 2/9] Instalar uv em $DORY_LAB_HOME/bin ==="
  if [ -x "$UV_BIN" ]; then
    echo "[PULADO] uv já existe ($("$UV_BIN" --version 2>&1)). ($((SECONDS - t0))s)"
  else
    if ! command -v curl >/dev/null 2>&1; then
      echo "ERRO: 'curl' não está instalado; não é possível baixar o uv."
      echo "O que fazer: copie o binário do uv manualmente para $UV_BIN."
      return 1
    fi
    if ! curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$DORY_LAB_HOME/bin" INSTALLER_NO_MODIFY_PATH=1 sh; then
      echo "ERRO: falha ao baixar/instalar o uv de https://astral.sh/uv/install.sh."
      echo "O que fazer: confira a internet para astral.sh/github.com ou copie o binário 'uv' manualmente para $UV_BIN."
      return 1
    fi
    any_step_changed=1
    echo "[OK] uv instalado: $("$UV_BIN" --version 2>&1). ($((SECONDS - t0))s)"
  fi

  # --- Step 3: Source check & rclone ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 3/9] Verificar origem ($ORIGEM) ==="
  local source_mode="local"
  if is_local_source "$ORIGEM"; then
    source_mode="local"
    if [ ! -d "$ORIGEM" ]; then
      echo "ERRO: diretório local '$ORIGEM' não existe."
      echo "O que fazer: confira o caminho montado do pendrive/disco ou passe um caminho rclone (ex.: gdrive:coder-comprimido)."
      return 1
    fi
    if [ ! -d "$ORIGEM/wheels/py313-vllm-pr60149" ] || [ ! -d "$ORIGEM/checkpoints/dory_int8_gptq_rtn_g128" ] || [ ! -d "$ORIGEM/data/f1a" ]; then
      echo "ERRO: o diretório '$ORIGEM' não tem a estrutura esperada (wheels/py313-vllm-pr60149, checkpoints/dory_int8_gptq_rtn_g128, data/f1a)."
      echo "O que fazer: confira se o pendrive contém a pasta raiz espelhada do Drive."
      return 1
    fi
    echo "[OK] Origem local válida: $ORIGEM ($((SECONDS - t0))s)"
  else
    source_mode="rclone"
    if [ ! -x "$RCLONE_BIN" ]; then
      echo "Baixando rclone para $RCLONE_BIN..."
      local rc_tmp="$DORY_LAB_HOME/tmp/rclone_dl"
      rm -rf "$rc_tmp" && mkdir -p "$rc_tmp"
      if ! curl -LsSf -o "$rc_tmp/rclone.zip" "https://downloads.rclone.org/rclone-current-linux-amd64.zip"; then
        echo "ERRO: falha ao baixar rclone de downloads.rclone.org."
        echo "O que fazer: confira a internet ou use um pendrive local como <origem>."
        return 1
      fi
      if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$rc_tmp/rclone.zip" -d "$rc_tmp/unzipped"
      elif command -v python3 >/dev/null 2>&1; then
        python3 -m zipfile -e "$rc_tmp/rclone.zip" "$rc_tmp/unzipped"
      else
        echo "ERRO: nem 'unzip' nem 'python3' disponíveis para extrair o rclone.zip."
        return 1
      fi
      local extracted_bin
      extracted_bin="$(find "$rc_tmp/unzipped" -type f -name rclone | head -n 1)"
      if [ -z "$extracted_bin" ] || [ ! -f "$extracted_bin" ]; then
        echo "ERRO: binário 'rclone' não encontrado dentro do arquivo baixado."
        return 1
      fi
      cp "$extracted_bin" "$RCLONE_BIN"
      chmod +x "$RCLONE_BIN"
      rm -rf "$rc_tmp"
      any_step_changed=1
    fi

    local remote_name="${ORIGEM%%:*}"
    if ! "$RCLONE_BIN" listremotes 2>/dev/null | grep -qx "${remote_name}:"; then
      cat <<EOF
ERRO: o remote rclone '${remote_name}:' ainda não está configurado.

Configure o remote agora seguindo estes passos:
  1. Rode:
       "$RCLONE_BIN" config
  2. Escolha 'n' (New remote) e dê o nome: ${remote_name}
  3. Em Storage, escolha 'drive' (Google Drive)
  4. Em client_id e client_secret, deixe em branco (aperte Enter)
  5. Em scope, escolha '1' (drive - acesso completo, necessário para ler e enviar resultados)
  6. Em service_account_file, deixe em branco (Enter)
  7. Em Edit advanced config, escolha 'n'
  8. Em Use auto config, escolha 'y' -> o navegador vai abrir para login na conta que enxerga a pasta do projeto
  9. Confirme com 'y' e saia com 'q'
 10. Depois rode novamente:
       bash "$0" "$ORIGEM"
EOF
      return 2
    fi

    local dirs_out=""
    if ! dirs_out="$("$RCLONE_BIN" lsf --dirs-only "$ORIGEM" 2>&1)"; then
      echo "ERRO: falha ao listar '$ORIGEM' via rclone:"
      printf '%s\n' "$dirs_out"
      echo ""
      echo "O que fazer: se o token expirou, rode '\"$RCLONE_BIN\" config reconnect ${remote_name}:'. Se o caminho não existir, confira o nome da pasta ou use um remote com root_folder_id."
      return 1
    fi

    local missing_dirs=()
    for d in wheels checkpoints data; do
      if ! printf '%s\n' "$dirs_out" | grep -Eq "^${d}/?$"; then
        missing_dirs+=("$d/")
      fi
    done
    if [ ${#missing_dirs[@]} -gt 0 ]; then
      cat <<EOF
ERRO: as pastas esperadas (${missing_dirs[*]}) não apareceram em '$ORIGEM'.
Pastas encontradas:
$dirs_out

Possível causa: '$ORIGEM' aponta para um atalho (shortcut) no Google Drive que o rclone não seguiu.
Alternativa recomendada:
  1. Rode "$RCLONE_BIN" config (edite o remote '${remote_name}' -> advanced config -> root_folder_id)
  2. Informe o ID da pasta do projeto (você sabe qual é) em root_folder_id
  3. Rode novamente usando apenas o remote na raiz:
       bash "$0" ${remote_name}:
EOF
      return 1
    fi
    echo "[OK] Remote rclone '$ORIGEM' verificado (wheels/, checkpoints/, data/ presentes). ($((SECONDS - t0))s)"
  fi

  # --- Step 4: Sync drive/ ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 4/9] Sincronizar wheels, checkpoint INT8 e dados F1a para $DRIVE_DIR ==="
  if check_drive_complete; then
    echo "[PULADO] Arquivos já presentes e validados em $DRIVE_DIR ($(du -sh "$DRIVE_DIR" | cut -f1)). ($((SECONDS - t0))s)"
  else
    mkdir -p "$WHEELS_DIR" "$CKPT_DIR" "$DATA_F1A_DIR"
    if [ "$source_mode" = "rclone" ]; then
      echo "Copiando wheels/py313-vllm-pr60149..."
      if ! "$RCLONE_BIN" copy --transfers 8 --checkers 16 --progress \
        "${ORIGEM%/}/wheels/py313-vllm-pr60149" "$WHEELS_DIR"; then
        echo "ERRO: falha ao baixar wheels via rclone. Verifique cota do Drive ou conexão e rode novamente (o rclone retoma de onde parou)."
        return 1
      fi
      echo "Copiando checkpoints/dory_int8_gptq_rtn_g128..."
      if ! "$RCLONE_BIN" copy --transfers 8 --checkers 16 --progress \
        "${ORIGEM%/}/checkpoints/dory_int8_gptq_rtn_g128" "$CKPT_DIR"; then
        echo "ERRO: falha ao baixar checkpoint INT8 via rclone. Rode novamente para retomar."
        return 1
      fi
      echo "Copiando data/f1a (apenas problems_v2.jsonl e samples_efficient_L2_n8_t1_p0.95_m8192.jsonl)..."
      if ! "$RCLONE_BIN" copy --transfers 8 --checkers 16 --progress \
        "${ORIGEM%/}/data/f1a" "$DATA_F1A_DIR" \
        --include "problems_v2.jsonl" \
        --include "samples_efficient_L2_n8_t1_p0.95_m8192.jsonl"; then
        echo "ERRO: falha ao baixar arquivos de data/f1a via rclone."
        return 1
      fi
    else
      echo "Copiando de diretório local '$ORIGEM' (cp -rn)..."
      cp -rn "${ORIGEM%/}/wheels/py313-vllm-pr60149/." "$WHEELS_DIR/"
      cp -rn "${ORIGEM%/}/checkpoints/dory_int8_gptq_rtn_g128/." "$CKPT_DIR/"
      cp -rn "${ORIGEM%/}/data/f1a/problems_v2.jsonl" "$PROBLEMS_FILE"
      cp -rn "${ORIGEM%/}/data/f1a/samples_efficient_L2_n8_t1_p0.95_m8192.jsonl" "$REF_SAMPLES_FILE"
    fi

    # Verify downloaded/copied artifacts
    local whl_count
    whl_count="$(find "$WHEELS_DIR" -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')"
    if [ ! -f "$WHEELS_DIR/OK" ] || [ "${whl_count:-0}" -lt 200 ]; then
      echo "ERRO: verificação de wheels falhou (OK existe? $([ -f "$WHEELS_DIR/OK" ] && echo sim || echo não), .whl encontrados: $whl_count, esperado >= 200)."
      echo "O que fazer: confira se o download do Drive completou sem atingir cota de download e rode novamente."
      return 1
    fi
    if [ ! -s "$CKPT_DIR/config.json" ] || [ ! -s "$CKPT_DIR/model-00001.safetensors" ] || [ ! -s "$CKPT_DIR/model-00002.safetensors" ]; then
      echo "ERRO: checkpoint INT8 incompleto em $CKPT_DIR (config.json ou model-00001/00002.safetensors ausente/vazio)."
      echo "O que fazer: apague arquivos parciais se necessário e rode o lab_setup.sh novamente."
      return 1
    fi
    if [ ! -s "$PROBLEMS_FILE" ] || [ ! -s "$REF_SAMPLES_FILE" ]; then
      echo "ERRO: arquivos F1a ausentes ou vazios em $DATA_F1A_DIR."
      return 1
    fi
    any_step_changed=1
    echo "[OK] Sincronização validada: $whl_count wheels + checkpoint INT8 + 2 arquivos F1a. Tamanho total em drive/: $(du -sh "$DRIVE_DIR" | cut -f1). ($((SECONDS - t0))s)"
  fi

  # --- Step 5: Code into repo/ ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 5/9] Código do repositório em $REPO_DIR ==="
  local repo_url="https://github.com/Arthur-galvao/experimentos-llm-local.git"
  local tarball_url="https://codeload.github.com/Arthur-galvao/experimentos-llm-local/tar.gz/refs/heads/main"
  local api_url="https://api.github.com/repos/Arthur-galvao/experimentos-llm-local/commits/main"

  if command -v git >/dev/null 2>&1; then
    if [ -d "$REPO_DIR/.git" ]; then
      if git -C "$REPO_DIR" pull --ff-only >/dev/null 2>&1; then
        git -C "$REPO_DIR" rev-parse HEAD > "$REPO_DIR/.git_sha"
        echo "[PULADO/ATUALIZADO] Repositório git atualizado ($(cat "$REPO_DIR/.git_sha")). ($((SECONDS - t0))s)"
      else
        git -C "$REPO_DIR" rev-parse HEAD > "$REPO_DIR/.git_sha"
        echo "[PULADO] git pull falhou (possivelmente sem rede), mantendo clone existente ($(cat "$REPO_DIR/.git_sha")). ($((SECONDS - t0))s)"
      fi
    elif [ -f "$REPO_DIR/scripts/t4_validate.py" ]; then
      if [ ! -s "$REPO_DIR/.git_sha" ]; then
        { curl -LsSf --max-time 8 "$api_url" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null || echo "desconhecido"; } > "$REPO_DIR/.git_sha"
      fi
      echo "[PULADO] Código já presente em $REPO_DIR (sha $(cat "$REPO_DIR/.git_sha")). ($((SECONDS - t0))s)"
    else
      rm -rf "$REPO_DIR"
      if ! git clone --branch main --depth 1 "$repo_url" "$REPO_DIR"; then
        echo "ERRO: falha ao clonar $repo_url."
        echo "O que fazer: verifique acesso a github.com."
        return 1
      fi
      git -C "$REPO_DIR" rev-parse HEAD > "$REPO_DIR/.git_sha"
      any_step_changed=1
      echo "[OK] Repositório clonado ($(cat "$REPO_DIR/.git_sha")). ($((SECONDS - t0))s)"
    fi
  else
    if [ -f "$REPO_DIR/scripts/t4_validate.py" ]; then
      # repo/ já veio do bootstrap do README (tarball): só registra o sha, sem trocar a pasta de onde este script roda
      if [ ! -s "$REPO_DIR/.git_sha" ]; then
        { curl -LsSf --max-time 8 "$api_url" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null || echo "desconhecido"; } > "$REPO_DIR/.git_sha"
      fi
      echo "[PULADO] Código já presente em $REPO_DIR (sha $(cat "$REPO_DIR/.git_sha")). ($((SECONDS - t0))s)"
    else
      local repo_tmp="$DORY_LAB_HOME/tmp/repo_dl"
      rm -rf "$repo_tmp" && mkdir -p "$repo_tmp"
      if ! curl -LsSf -o "$repo_tmp/main.tar.gz" "$tarball_url"; then
        echo "ERRO: falha ao baixar tarball de $tarball_url."
        echo "O que fazer: verifique acesso a codeload.github.com."
        return 1
      fi
      tar -xzf "$repo_tmp/main.tar.gz" -C "$repo_tmp"
      local extracted_dir
      extracted_dir="$(find "$repo_tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
      if [ -z "$extracted_dir" ] || [ ! -f "$extracted_dir/scripts/t4_validate.py" ]; then
        echo "ERRO: tarball extraído não contém scripts/t4_validate.py."
        return 1
      fi
      local api_sha="desconhecido"
      if command -v python3 >/dev/null 2>&1; then
        api_sha="$(curl -LsSf --max-time 8 "$api_url" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null || echo "desconhecido")"
      fi
      printf '%s\n' "$api_sha" > "$extracted_dir/.git_sha"
      rm -rf "${REPO_DIR}.old"
      if [ -e "$REPO_DIR" ]; then
        mv "$REPO_DIR" "${REPO_DIR}.old"
      fi
      mv "$extracted_dir" "$REPO_DIR"
      rm -rf "${REPO_DIR}.old" "$repo_tmp"
      any_step_changed=1
      echo "[OK] Código baixado via tarball e instalado em $REPO_DIR (sha $api_sha). ($((SECONDS - t0))s)"
    fi
  fi

  # --- Step 6: venv + offline wheels ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 6/9] Criar .venv (Python 3.13) e instalar wheels offline ==="
  if [ ! -x "$VENV_PY" ]; then
    if ! "$UV_BIN" venv --python 3.13 "$VENV_DIR"; then
      echo "ERRO: falha ao criar .venv com Python 3.13."
      echo "O que fazer: o uv precisa baixar CPython 3.13 se não houver python3.13 local. Verifique conexão com github.com/astral.sh."
      return 1
    fi
    any_step_changed=1
  fi

  local pkgs=(vllm accelerate datasets evalplus matplotlib pytest trl peft)
  if "$VENV_PY" -c "import vllm, torch, transformers, accelerate, datasets, evalplus, matplotlib, pytest, trl, peft" >/dev/null 2>&1; then
    echo "[PULADO] .venv e pacotes já instalados. ($((SECONDS - t0))s)"
  else
    local pip_log="$DORY_LAB_HOME/tmp/uv_pip_err.log"
    set +e
    "$UV_BIN" pip install --python "$VENV_PY" --no-index --find-links "$WHEELS_DIR" "${pkgs[@]}" >"$pip_log" 2>&1
    local pip_rc=$?
    set -e
    cat "$pip_log"

    if [ "$pip_rc" -ne 0 ]; then
      if [ "$ALLOW_INDEX" -eq 1 ]; then
        echo ""
        echo "Instalação --no-index falhou; --allow-index ativo -> tentando novamente permitindo PyPI apenas para o que falta (mantendo vllm/torch locais fixados)..."
        local pinned_pkgs=("${pkgs[@]}")
        local vllm_whl torch_whl
        vllm_whl="$(find "$WHEELS_DIR" -maxdepth 1 -type f -name 'vllm-*.whl' | head -n 1 || true)"
        torch_whl="$(find "$WHEELS_DIR" -maxdepth 1 -type f -name 'torch-*.whl' | head -n 1 || true)"
        if [ -n "$vllm_whl" ]; then
          pinned_pkgs[0]="$vllm_whl"
        fi
        if [ -n "$torch_whl" ]; then
          pinned_pkgs+=("$torch_whl")
        fi
        if ! "$UV_BIN" pip install --python "$VENV_PY" --find-links "$WHEELS_DIR" "${pinned_pkgs[@]}"; then
          echo "ERRO: instalação falhou mesmo com --allow-index."
          echo "O que fazer: verifique o log acima e a conectividade com pypi.org."
          return 1
        fi
      else
        echo ""
        echo "ERRO: falha na resolução offline das wheels (--no-index)."
        echo "Distribuições possivelmente ausentes na pasta de wheels:"
        grep -iE 'not found|no solution|because|unsatisfiable|missing' "$pip_log" | head -n 20 || true
        echo ""
        echo "O que fazer: rode novamente permitindo PyPI apenas para o que faltar:"
        echo "  bash \"$0\" \"$ORIGEM\" --allow-index"
        return 1
      fi
    fi
    any_step_changed=1
    echo "[OK] Pacotes instalados na .venv. ($((SECONDS - t0))s)"
  fi

  # --- Step 7: patch_vllm_dory.py ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 7/9] Aplicar e conferir patch do Dory no vLLM ==="
  if "$VENV_PY" "$REPO_DIR/scripts/patch_vllm_dory.py" --check >/dev/null 2>&1; then
    echo "[PULADO] Patch do Dory já aplicado no vLLM. ($((SECONDS - t0))s)"
  else
    if ! "$VENV_PY" "$REPO_DIR/scripts/patch_vllm_dory.py"; then
      echo "ERRO: falha ao aplicar patch_vllm_dory.py."
      echo "O que fazer: confira se a wheel instalada do vLLM é a da PR #60149."
      return 1
    fi
    if ! "$VENV_PY" "$REPO_DIR/scripts/patch_vllm_dory.py" --check; then
      echo "ERRO: patch_vllm_dory.py --check falhou após aplicar o patch."
      return 1
    fi
    any_step_changed=1
    echo "[OK] Patch aplicado e verificado. ($((SECONDS - t0))s)"
  fi

  # --- Step 8: GPU gate ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 8/9] Portão de GPU (PyTorch + CUDA) ==="
  if ! "$VENV_PY" - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
avail = torch.cuda.is_available()
print("cuda_available:", avail)
if not avail:
    sys.exit(1)
print("device_name:", torch.cuda.get_device_name(0))
print("device_capability:", torch.cuda.get_device_capability(0))
PY
  then
    local drv_ver
    drv_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo 'desconhecida')"
    echo ""
    echo "[BLOQUEIO] torch.cuda.is_available() retornou False."
    echo "O driver NVIDIA instalado ($drv_ver) provavelmente é antigo demais para as wheels cu130 (CUDA 13)."
    echo "O que fazer: atualizar o driver NVIDIA exige TI/sudo e está fora do escopo desta sessão; tente outro PC com driver mais novo ou encerre aqui."
    return 1
  fi
  echo "[OK] Portão de GPU aprovado. ($((SECONDS - t0))s)"

  # --- Step 9: setup_ok.json ---
  t0=$SECONDS
  echo ""
  echo "=== [Passo 9/9] Gravar setup_ok.json ==="
  if [ -f "$SETUP_OK_FILE" ] && [ "$any_step_changed" -eq 0 ]; then
    echo "[PULADO] $SETUP_OK_FILE já existe e nenhum passo anterior mudou. ($((SECONDS - t0))s)"
  else
    local drv_ver
    drv_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || echo '')"
    "$VENV_PY" - "$SETUP_OK_FILE" "$ORIGEM" "$REPO_DIR/.git_sha" "$DRIVE_DIR" "$WHEELS_DIR" "$CKPT_DIR" "$started_iso" "$drv_ver" "$ALLOW_INDEX" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import vllm

out_path = Path(sys.argv[1])
origem = sys.argv[2]
sha_file = Path(sys.argv[3])
drive_dir = Path(sys.argv[4])
wheels_dir = Path(sys.argv[5])
ckpt_dir = Path(sys.argv[6])
started_at = sys.argv[7]
driver_ver = sys.argv[8]
allow_index = bool(int(sys.argv[9]))

def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

git_sha = sha_file.read_text(encoding="utf-8").strip() if sha_file.exists() else "desconhecido"
props = torch.cuda.get_device_properties(0)
payload = {
    "started_at": started_at,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "source": origem,
    "git_sha": git_sha,
    "allow_index": allow_index,
    "versions": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": vllm.__version__,
        "transformers": transformers.__version__,
        "driver": driver_ver,
        "gpu": props.name,
        "capability": f"{props.major}.{props.minor}",
        "vram_gib": round(props.total_memory / 2**30, 2),
    },
    "sizes": {
        "drive_bytes": dir_bytes(drive_dir),
        "wheels_count": len(list(wheels_dir.glob("*.whl"))),
        "checkpoint_bytes": dir_bytes(ckpt_dir),
    },
}
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
    echo "[OK] Gravado em $SETUP_OK_FILE. ($((SECONDS - t0))s)"
  fi

  echo ""
  echo "========================================================================"
  echo "SETUP CONCLUÍDO COM SUCESSO!"
  echo "Próximo passo: bash lab/lab_bench.sh"
  echo "========================================================================"
}

set +e
main_setup 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"
