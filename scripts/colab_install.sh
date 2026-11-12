#!/usr/bin/env bash
# Instala o vLLM da PR #60149 e as dependências no Colab.
# 1a vez: monta as wheels de tudo e guarda no Drive (demora como antes, só uma vez).
# Depois: instala só dessas wheels (sem git clone, sem compilar, sem baixar da internet).
# A versão do vLLM fica congelada na que foi validada. Para atualizar a PR, apague a pasta $WHEELS no Drive.
set -euo pipefail

VLLM_SPEC="vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head"
EXTRAS="accelerate datasets evalplus matplotlib pytest trl peft"
PY=$(python -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')
WHEELS="${WHEELS_DIR:-/content/drive/MyDrive/coder-comprimido/wheels}/${PY}-vllm-pr60149"
LOCAL=/content/wheels_tmp

nvidia-smi --query-gpu=name,memory.total --format=csv

if [ ! -f "$WHEELS/OK" ]; then
  echo ">>> Primeira vez: gerando as wheels (só acontece uma vez; depois fica rápido)."
  rm -rf "$LOCAL" && mkdir -p "$LOCAL"
  VLLM_USE_PRECOMPILED=1 pip wheel -q -w "$LOCAL" "$VLLM_SPEC" $EXTRAS
  mkdir -p "$WHEELS"
  cp "$LOCAL"/*.whl "$WHEELS"/
  touch "$WHEELS/OK"  # só marca como pronto depois da cópia inteira; sessão que cair no meio refaz
  echo ">>> Wheels salvas em $WHEELS ($(du -sh "$WHEELS" | cut -f1))."
fi

echo ">>> Instalando das wheels em $WHEELS"
pip install -q --no-index --find-links "$WHEELS" vllm $EXTRAS
python -c "import transformers, vllm; print('transformers', transformers.__version__, '| vllm', vllm.__version__)"
