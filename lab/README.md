# Runbook do Lab — Validação do Dory 4B (INT8 GPTQ) em RTX 2060 SUPER

Kit de scripts defensivos e resumíveis para instalar e medir o **Dory 4B (INT8 GPTQ)** no vLLM (PR #60149) em **uma única sessão de 5 horas (13:00–18:00)** nos PCs do laboratório, **sem `sudo`**, **sem `git` obrigatório** e sem tocar em arquivos fora de `$DORY_LAB_HOME` (padrão: `~/dory-lab`, além do marcador `~/.dory_lab_marker`).

---

## 1. O que é este kit

- **Hardware-alvo:** RTX 2060 SUPER (8 GB VRAM, Turing `sm_75`), 8 GB RAM, Ubuntu x86_64.
- **Objetivo da sessão:** descobrir rapidamente se o PC do lab roda o stack `cu130` + `vllm` PR #60149, comparar `--enforce-eager` vs `--no-enforce-eager` (CUDA graphs), medir `tok/s` real em sequências longas (`F1a efficient`, até 8192 tokens) e gerar um resumo comparativo com a referência A100 (BF16).
- **Scripts:**
  - `lab/lab_check.sh` — diagnóstico somente leitura (~1 min, sempre retorna `0`).
  - `lab/lab_setup.sh` — instalação idempotente em `$DORY_LAB_HOME`.
  - `lab/lab_bench.sh` — sequência de benchmark com deadline rígido e retomada automática.
  - `lab/bench_summary.py` — gera `bench_summary.json` e `bench_summary.md`.

---

## 2. Checklist antes de ir para o lab

- [ ] **Quem faz login no Google Drive (`rclone`):** confirmar quem da equipe estará presente com acesso à conta que enxerga a pasta do projeto no Drive (e com acesso ao autenticador/2FA, se houver).
- [ ] **Plano B (pendrive USB):** preparar um pendrive formatado (exFAT ou ext4) com pelo menos **10 GB** livres contendo exatamente a subárvore do Drive:
  ```text
  coder-comprimido/
    wheels/
      py313-vllm-pr60149/          (~241 arquivos .whl + arquivo OK)
    checkpoints/
      dory_int8_gptq_rtn_g128/     (config.json, tokenizer*, model-00001.safetensors, model-00002.safetensors, index...)
    data/
      f1a/
        problems_v2.jsonl
        samples_efficient_L2_n8_t1_p0.95_m8192.jsonl
  ```
- [ ] **Tamanho esperado de download por PC:** **~8,5 GiB** (4,08 GiB de wheels em 241 arquivos + 4,42 GiB do checkpoint INT8 + 2 arquivos F1a) + ~200 MB de Python 3.13/`uv`/`rclone`/código. Depois de instalado, o PC precisa de ~30 GiB livres.
- [ ] **Hugging Face acessível:** o primeiro `smoke` baixa o tokenizer do Dory de `huggingface.co` (poucos MB). O `lab_check.sh` testa essa rota.
- [ ] **Ter anotado fora do repo** (em papel/celular) o ID da pasta raiz no Google Drive, caso o `rclone` não siga atalho e seja preciso configurar `root_folder_id`.

---

## 3. Cronograma da sessão (quinta-feira, 13:00–18:00)

**Primeiro de tudo, em cada PC (baixa o kit, ~1 MB, sem git):**

```bash
mkdir -p ~/dory-lab && cd ~/dory-lab
curl -L https://codeload.github.com/Arthur-galvao/experimentos-llm-local/tar.gz/refs/heads/main | tar xz
mv experimentos-llm-local-main repo
cd ~/dory-lab/repo   # todos os comandos abaixo rodam daqui
```

Sem internet: copie a pasta `repo/` do pendrive para `~/dory-lab/repo`.


| Horário | O que fazer | Comando |
|---|---|---|
| **13:00** | Ligar 2–3 PCs no Ubuntu, baixar os scripts do repo (ou copiar do pendrive) e rodar o diagnóstico em cada um | `bash lab/lab_check.sh` |
| **~13:15** | Escolher o melhor PC sem `BLOQUEIO` e iniciar a instalação (via rclone ou pendrive) | `bash lab/lab_setup.sh gdrive:coder-comprimido` *(ou caminho do pendrive)* |
| **~14:00** | Com `setup_ok.json` aprovado, iniciar o benchmark com deadline padrão (`17:30`) | `bash lab/lab_bench.sh` |
| **17:30** | **Deadline rígido:** o `lab_bench.sh` para automaticamente antes de iniciar qualquer problema que cruzaria `17:30` | *(automático)* |
| **17:45** | Conferir se o upload via `rclone` terminou (ou copiar `~/dory-lab/drive/results/lab_validation/<hostname>/` para o pendrive) e liberar o laboratório | `ls -lh ~/dory-lab/drive/results/lab_validation/` |

> **Quanto demora o `quality`:** cada problema gera 8 amostras de até 8.192 tokens (até ~65 mil tokens). Na 2060S isso pode levar de dezenas de minutos a mais de 1 hora por problema, sem saída nova na tela enquanto gera. É normal sair da sessão com só 2–4 problemas: o objetivo principal é o tok/s real, não a qualidade.
> Por isso, antes do `quality`, o bench roda um **smoke longo** (4 sequências × até 4.096 tokens, ~15–40 min) que garante um tok/s em sequência longa mesmo que nenhum problema do `quality` termine. Os resultados (e os `lab_check_*.txt` e `logs/`) **sobem para o Drive depois de cada etapa** e também no Ctrl-C, então pode interromper às 17:40 sem perder nada.

> Dica: se quiser rodar só os smokes primeiro (sem `quality`), use `bash lab/lab_bench.sh --skip-quality`. Para testar em outro horário, ajuste `--deadline HH:MM`.

---

## 4. Como ler os vereditos do `lab_check.sh`

No final do relatório (e em `~/dory-lab/lab_check_<hostname>_<data>.txt`), aparece a seção **VEREDITO**:

- **`[OK]`**: requisito atendido (ex.: `glibc >= 2.28`, disco `>= 30 GiB`, GPU detectada).
- **`[ATENÇÃO]`**: não impede tentar o setup, mas exige cuidado:
  - *Driver `< 580`*: pode ser antigo para `torch cu130` (CUDA 13), mas não é bloqueio antecipado — o teste definitivo é o Passo 8 do `lab_setup.sh`.
  - *RAM `< 7 GiB`*: risco de OOM na carga; feche navegador e apps pesados.
  - *`compute_cap != 7.5`*: GPU diferente da RTX 2060 SUPER esperada.
- **`[BLOQUEIO]`**: impede continuar neste PC:
  - `nvidia-smi` ausente ou falhando;
  - `glibc < 2.28` (incompatível com wheels `manylinux_2_28`);
  - espaço livre em `$HOME < 30 GiB`;
  - sem acesso a `https://astral.sh` **e** sem `python3.13` local.

---

## 5. Troubleshooting

### Driver NVIDIA antigo para `cu130`
- **Sintoma:** Passo 8 do `lab_setup.sh` imprime `[BLOQUEIO] torch.cuda.is_available() retornou False`.
- **Causa:** o driver NVIDIA instalado é antigo demais para CUDA 13 (`cu130`).
- **Ação:** atualizar driver exige `sudo`/TI e está fora do escopo desta sessão. Teste outro PC do lab (se todos tiverem o mesmo driver antigo, encerrar e documentar a versão do driver).

### `glibc < 2.28`
- **Sintoma:** `[BLOQUEIO]` no `lab_check.sh`.
- **Ação:** Ubuntu antigo demais para as wheels compiladas (`manylinux_2_28`). Troque de PC.

### Disco insuficiente (`< 30 GiB`)
- **Sintoma:** `[BLOQUEIO]` no `lab_check.sh` ou Passo 1 do `lab_setup.sh`.
- **Ação:** verifique `df -h` se há outra partição gravável (ex.: `/scratch`, `/dados`, SSD secundário) e rode exportando `export DORY_LAB_HOME=/caminho/com/espaco/dory-lab`.

### VRAM insuficiente (ambiente gráfico ocupando GPU)
- **Sintoma:** `LAB_VRAM_GIB` cai abaixo de ~6.8 GiB ou os smokes falham com CUDA OOM.
- **Causa:** Xorg / Wayland / navegador usando VRAM da 2060 SUPER (8 GB no total).
- **Ação:**
  1. Feche navegadores e apps gráficos.
  2. Se ainda faltar VRAM, **saia da sessão gráfica** e mude para um TTY texto com **`Ctrl+Alt+F3`**, faça login no terminal e rode `bash lab/lab_bench.sh` de lá (isso costuma liberar 300–700 MiB de VRAM).
  3. Para voltar ao modo gráfico depois: `Ctrl+Alt+F1` ou `Ctrl+Alt+F2`.

### `rclone` não segue atalho do Google Drive
- **Sintoma:** Passo 3 do `lab_setup.sh` acusa que `wheels/`, `checkpoints/`, `data/` não apareceram em `<origem>`.
- **Ação:** configure o remote com o `root_folder_id` direto da pasta do projeto:
  1. Rode `~/dory-lab/bin/rclone config` → `e` (Edit existing remote) → selecione o remote → `y` (Edit advanced config) → informe o ID da pasta do projeto em `root_folder_id` → confirme até sair.
  2. Rode novamente usando a raiz do remote: `bash lab/lab_setup.sh gdrive:`.

### Cota de download do Google Drive excedida
- **Sintoma:** `rclone copy` falha com erro `downloadQuotaExceeded` / `403`.
- **Ação:** use o **Plano B (pendrive)** passando o diretório montado do pendrive como `<origem>`:
  `bash lab/lab_setup.sh /media/$USER/PENDRIVE/coder-comprimido`.

### Wheels faltando na instalação offline (`--no-index`)
- **Sintoma:** Passo 6 do `lab_setup.sh` falha listando distribuições ausentes.
- **Ação:** reexecute permitindo PyPI apenas para o que faltar (mantendo `vllm` e `torch` locais fixados):
  `bash lab/lab_setup.sh <origem> --allow-index`.

---

## 6. O que trazer de volta

Antes de ir embora, garanta que os seguintes arquivos foram enviados ao Drive (ou copiados para o pendrive):

1. `~/dory-lab/drive/results/lab_validation/<hostname>/bench_summary.md`
2. `~/dory-lab/drive/results/lab_validation/<hostname>/bench_summary.json`
3. `~/dory-lab/drive/results/lab_validation/<hostname>/quality/` (inclusive `samples_efficient_int8.jsonl` e `throughput_int8.jsonl`)
4. Todos os arquivos `~/dory-lab/lab_check_<hostname>_*.txt` de cada PC testado.
5. Os logs em `~/dory-lab/logs/` (se houver qualquer falha para depurar depois).

**Na próxima sessão no laboratório:** rode `bash lab/lab_check.sh` novamente no mesmo PC após um reboot. Se `~/.dory_lab_marker` ainda existir, o script confirmará que o disco persiste entre reinicializações (ou avisará se o sistema restaura o disco no boot).

---

## 7. Tabela de decisão (pós-sessão)

| Pergunta | O que olhar em `bench_summary.md` | Critério de decisão |
|---|---|---|
| **CUDA graphs vs Eager** | Seção 2 (`tok/s`, `max_concurrency`, erro OOM) | Usar `--no-enforce-eager` apenas se completou sem OOM e deu ganho claro de `tok/s` sem esmagar `kv_cache_size_tokens`; caso contrário, ficar em `--enforce-eager`. |
| **O farm de 15 PCs vale a pena?** | Seção 3 (`tok/s real por PC` × 15 vs `800 tok/s` da A100) | Se `15 × tok/s_pc` ficar competitivo com `800 tok/s` (ou próximo disso, considerando custo zero de CU no lab), vale montar a distribuição de geração em paralelo. Se ficar muito abaixo ou exigir manutenção alta por reboot/driver, concentrar na A100/Colab. |
| **Qualidade INT8 vs A100 BF16** | Seção 4 (`p_hat`, `solved`, `tokens`, `truncated`) | **Apenas indicativa** (amostra pequena, `<= 16` problemas). Serve só para detectar regressão grotesca/colapso do INT8. **Estes números nunca entram nos gates do projeto** — os gates oficiais rodam exclusivamente na A100 em BF16. |
