# experimentos-llm-local — modelo coder com raciocínio comprimido

Projeto de pesquisa: a partir do NVIDIA Nemotron-Labs Dory 1.0 4B (transformer em loop), construir um modelo coder que resolva tarefas de programação gastando o mínimo de tokens de raciocínio, medindo acerto × tokens contra o baseline reproduzido. Visão completa em [VISAO_GERAL.md](VISAO_GERAL.md).

## Onde fica cada coisa

| Lugar | Responsabilidade | Não guarda |
|---|---|---|
| **Este repositório** | Fonte da verdade do código (`lib/`, `scripts/`, `tests/`), notebooks sem saída e docs que podem ser públicos | Dados, pesos, resultados, conversas ou qualquer coisa privada (o repo é público) |
| **Drive central** (`coder-comprimido`, conta do projeto, acesso restrito) | Único lugar dos artefatos: `wheels/`, `checkpoints/`, `data/`, `eval_cache/`, `results/`, `docs/` privado e `_arquivo/`. Tem o próprio `README.md` | Código editável |
| **Colab** (A100 / T4) | Só compute. Clona este repo na Célula 0 e grava tudo no Drive | Estado |
| **Lab da faculdade** (15 × RTX 2060 SUPER + PC chefe) | Geração em paralelo (F1a, rollouts) com o Dory INT8; o PC chefe roda fila, dashboard e sincroniza com o Drive | Gates (só A100 em BF16) |

Cada resultado registra o `git_sha` do commit que o gerou, que é o elo entre o Drive e este repo.

## Estrutura do repo

```
lib/            código compartilhado (modelo HF do Dory, geração vLLM, avaliação, recompensas, registro)
scripts/        pontos de entrada por fase (f0_*, f1a_*, t4_validate.py, patch_vllm_dory.py, colab_install.sh)
notebooks/      notebooks finos do Colab: chamam lib/ e scripts/, sem lógica própria
tests/          testes (tests/fixtures/ guarda o dory.py da PR #60149 do vLLM)
docs/arquivo/   documentos obsoletos mantidos por histórico (ESPEC_ANTIGRAVITY.md)
VISAO_GERAL.md  objetivo, métricas, eixos de compressão, roadmap
RESULTADOS_F0.md  baselines de referência (F0)
BENCH_CUSTOM.md   benchmark custom usado entre fases
```

## Como rodar no Colab

1. Abra o notebook da fase (`notebooks/*.ipynb`) no Colab, com a conta que enxerga o Drive central em `MyDrive/coder-comprimido` (dono da pasta, ou atalho com esse nome).
2. A Célula 0 monta o Drive, clona/atualiza este repo e define `CODER_DRIVE_ROOT`.
3. A Célula 1 instala o vLLM da PR #60149 a partir de `wheels/` no Drive (`scripts/colab_install.sh`).

Se o Drive estiver em outro caminho, defina antes de tudo:

```python
os.environ["CODER_DRIVE_ROOT"] = "/content/drive/MyDrive/<pasta ou atalho>"
```

`lib/io.py`, `scripts/colab_install.sh`, `scripts/t4_validate.py` e todos os notebooks leem essa variável.

## Registro de resultados

- Cada rodada grava em `results/runs/<run_id>.jsonl` (`lib/registry.py`), para Colab e lab poderem gravar ao mesmo tempo no Drive, que não tem lock.
- `results/results.jsonl` é o registro legado do F0: continua sendo lido, não recebe mais linhas.
- `load_results` junta os dois em ordem cronológica.

## Testes

```bash
python -m pytest tests/            # test_critical.py precisa de torch
python -m pytest tests/test_registry_runs.py   # só biblioteca padrão
```

## Regras

- Números em FP16/INT8 (T4, lab) não entram nos gates. Gates rodam na A100 em BF16, com a configuração fixa de `VISAO_GERAL.md`.
- Nada de segredos (tokens, senhas, links de compartilhamento) neste repo.
