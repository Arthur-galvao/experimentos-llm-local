# Especificação para o Agente: Notebooks Colab por Fase

> Leia `VISAO_GERAL.md` antes. Este documento diz **o que** implementar, em que ordem e com quais critérios de aceite. Você (agente) escreve o código.

---

## 0. Regras para o agente

1. **Não invente APIs.** O Dory usa código próprio (`trust_remote_code=True`). Antes de usar qualquer parâmetro de loop, cache ou forward, **abra e leia** o `modeling_*.py` e o `config.json` do repositório no Hub. Campos conhecidos no config: `n_recurrent_loops`, `n_input_layers`, `n_recurrent_layers`, `n_output_layers`, `recurrent_kv_cache_mode`, `vocab_size=131072`, `hidden_size=2560`, `tie_word_embeddings=false`. Confirme como mudar o número de loops na inferência (argumento do forward/generate ou config) e documente no `lib/dory.py`.
2. **Versões:** instale primeiro o vLLM da PR (`#60149`), sem fixar `transformers`, `safetensors`, `tokenizers` ou `huggingface_hub` (ele exige `transformers>=5.16.1,<5.21` e instala as próprias versões compatíveis). Por cima, instale só pacotes compatíveis: `accelerate`, `datasets`, `evalplus`, `matplotlib`, `pytest`, e `trl`/`peft` em versões que suportem `transformers` 5.x. Imprima as versões de `transformers` e `vllm` logo após a instalação.
3. **Todo notebook é retomável.** O Colab desconecta. Checkpoints e resultados vão para o Google Drive. Ao reiniciar, o notebook detecta o último checkpoint e continua de onde parou.
4. **Smoke test antes da corrida real.** Toda tarefa de treino ou avaliação tem uma célula `SMOKE=True` que roda com ~5 exemplos e poucos passos, para validar o pipeline antes de gastar compute.
5. **Célula de estimativa de custo.** Antes da corrida real, estime o tempo (exemplos × tempo medido no smoke) e as compute units (A100 ≈ 11–13 CU/h, confirmar no painel).
6. **Seeds fixas** e config de cada execução serializado no registro de resultados.
7. **Notebooks finos e nada de código duplicado:** lógica compartilhada fica em `lib/` e a lógica executável em `scripts/`. Os notebooks (como `F0_setup_baseline.ipynb`) têm apenas a célula 0 (Drive + `git clone`/`git pull`), instalação e chamadas `!python scripts/...`. Todo script em `scripts/` deve ter `main()` + `if __name__ == "__main__":` e nenhum código solto no nível do módulo (pois o vLLM usa `spawn` e reimporta o arquivo).

---

## 1. Ambiente

### 1.1 Verificação de hardware (primeira célula de todo notebook)
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
- **Atenção:** no Colab, "RAM alta" se refere à **RAM do sistema** (~80GB), não necessariamente à VRAM. A A100 do Colab pode ter **40GB ou 80GB**.
- `lib/hw.py` lê a VRAM e define `PROFILE = "a100_80" | "a100_40"`. As configs de treino dependem disso:

| Tarefa | a100_80 | a100_40 |
|---|---|---|
| SFT 4B | full FT bf16 + AdamW 8-bit + gradient checkpointing | LoRA r=64 (+ embed/lm_head em `modules_to_save` quando a fase exigir) |
| GRPO 4B | LoRA r=64; referência = adapter desligado | LoRA r=32, batch menor |
| Embeddings novos (F3) | full FT | LoRA + `modules_to_save=["embed_tokens","lm_head"]` |
| Coconut (F4) | LoRA + gradient checkpointing, poucos pensamentos | provavelmente inviável; avisar |

### 1.2 Persistência e Cache Local
- Montar o Drive em `/content/drive`. Raiz de artefatos persistentes (`checkpoints`, `eval_cache`, `results`, `data`): `/content/drive/MyDrive/coder-comprimido/`.
- **Cache do Hugging Face sempre no disco local (`/root/.cache/huggingface`):** `setup_drive_and_cache` em `lib/io.py` **nunca** deve apontar `HF_HOME`, `HF_HUB_CACHE` ou `HF_DATASETS_CACHE` para `/content/drive` (o Drive corrompia `tokenizer.json`). `hf_cache` é removido dos caminhos.
- **Validação de integridade (`ensure_model_files(repo)` em `lib/io.py`):** valida `config.json`, `tokenizer_config.json` e `tokenizer.json` com `json.load` e baixa novamente com `force_download=True` caso algum esteja corrompido.
- Checkpoints a cada N passos (padrão: 15 min de treino), mantendo os 2 últimos.

### 1.3 Backend de geração, isolamento de processos e notebooks finos
- `lib/gen.py` com interface única: `generate(model, prompts, max_new_tokens, temperature, n_loops, n_samples) -> list[Output]`, onde `Output` tem `text`, `cot_text`, `answer_text`, `n_cot_tokens`, `n_total_tokens`, `latency_s`, `backend`.
- **Backend principal de avaliação: vLLM da PR oficial** (`VLLM_USE_PRECOMPILED=1 pip install "git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head"`), com loops definidos via `hf_overrides={"n_recurrent_loops": k, "recurrent_kv_cache_mode": mode}`.
- **Lotes grandes no vLLM:** com `backend="vllm"`, enviar os problemas pendentes em lotes de 128 (`VLLM_EVAL_BATCH_SIZE = 128`), gravando checkpoint em `eval_cache` a cada lote, em vez de usar o `batch_size` do HF.
- **Backend HF** (`lib/dory.py`): usado para treino, logit lens e testes curtos. É PyTorch puro, lento demais para gerar 16 mil tokens em centenas de problemas.
- **Sem troca silenciosa:** cada saída registra o backend **efetivamente usado** (`vllm` ou `hf`), e o `results.jsonl` grava esse valor. Com `smoke=False`, uma falha do vLLM **interrompe** a execução (`allow_hf_fallback=False` por padrão). O fallback para HF só é permitido em smoke test.
- **Um processo por motor vLLM:** antes de criar um motor vLLM, tire qualquer modelo HF da GPU. Cada baseline do F0 (Dory L1, L2, L4 e Qwen) roda num subprocesso separado chamando um script de `scripts/`, garantindo liberação de 100% da VRAM quando o processo termina. Todo script em `scripts/` tem `main()` + `if __name__ == "__main__":` e nenhum código solto no nível do módulo (o vLLM usa `spawn` e reimporta o arquivo).
- **Notebooks finos:** toda a lógica fica em `lib/` e `scripts/`. O `F0_setup_baseline.ipynb` tem apenas a célula 0 (Drive + `git pull`), instalação e chamadas `!python scripts/...`.
- **Alinhamento de Tokenizer (`fix_mistral_regex`):** comparar a tokenização de 20 textos (código, inglês, espaços e quebras de linha) entre o tokenizer HF e o tokenizer do vLLM; se divergirem, usar `fix_mistral_regex=True` onde necessário para que produzam exatamente os mesmos IDs.
- **Referência oficial do modelo:** `hf://models/tiny-random/dory-1.0/vllm_dory.py` (cópia do código da PR do vLLM). Qualquer mudança em `lib/dory.py` precisa continuar batendo com ele.
- Separar CoT e resposta pelos delimitadores de raciocínio do chat template de cada modelo (verificar no `tokenizer_config.json`), contando tokens de CoT exatamente pelos `token_ids` até `</think>`.

---

## 2. Estrutura do repositório

```
coder-comprimido/
├── lib/
│   ├── hw.py           # perfil de hardware
│   ├── io.py           # Drive, checkpoints, retomada
│   ├── dory.py         # carregar Dory, setar loops, logit lens por loop
│   ├── gen.py          # backend de geração (HF / vLLM)
│   ├── evals.py        # HumanEval+, MBPP+, LCB v6, LCB Pro, SciCode
│   ├── sandbox.py      # execução de código com timeout/limite de memória
│   ├── metrics.py      # acerto por token, Pareto, gráficos
│   ├── registry.py     # results.jsonl
│   ├── rewards.py      # recompensas do GRPO
│   └── dsl/            # tokenizer analysis, gramática, parser (F3)
├── notebooks/
│   ├── F0_setup_baseline.ipynb
│   ├── F1a_self_distill.ipynb
│   ├── F1b_sft_short_cot.ipynb
│   ├── F1c_grpo_length.ipynb
│   ├── F2_adaptive_loops.ipynb
│   ├── F3a_dsl_design.ipynb
│   ├── F3b_dsl_corpus.ipynb
│   ├── F3c_dsl_tokens_sft.ipynb
│   ├── F3d_dsl_grpo.ipynb
│   ├── F4_coconut.ipynb
│   ├── F5_vocab_prune.ipynb
│   └── EVAL_phase_gate.ipynb
├── data/               # splits fixos, gerados uma vez
└── results/
    ├── results.jsonl
    └── plots/
```
`lib/` fica no Drive e é importado com `sys.path.append`, ou é instalado como pacote editável a partir de um repositório Git (preferível, se houver).

---

## 3. Protocolo de avaliação

### 3.1 Dois níveis
- **`eval_quick`** (a cada experimento): HumanEval+ completo + **subconjunto fixo de 200 problemas do LCB v6** (os mais recentes por data de publicação; o split é gerado uma vez em `data/lcb_quick.json` e nunca muda).
- **`eval_gate`** (fim de fase, notebook `EVAL_phase_gate.ipynb`): LCB v6 completo + LCB Pro + SciCode.

### 3.2 Implementação
- **HumanEval+/MBPP+:** biblioteca `evalplus`.
- **LCB v6:** dataset oficial do LiveCodeBench (`livecodebench/code_generation_lite`, versão v6; confirmar o nome e a versão no Hub). Execução local dos testes pelo `lib/sandbox.py`, com timeout por teste.
- **LCB Pro:** verificar se há dataset e juiz públicos executáveis localmente. Se não houver, registrar como indisponível e seguir sem ele.
- **SciCode:** repositório oficial (`scicode-bench`); seguir o protocolo de avaliação deles.
- **Parâmetros fixos de geração** para todas as comparações: temperatura, top_p, `max_new_tokens` (definir no F0 e registrar). Mudar algum deles invalida a comparação com o baseline.

### 3.3 Registro (`results.jsonl`, uma linha por avaliação)
```json
{"run_id": "...", "phase": "F1c", "model": "dory", "checkpoint": "...", "n_loops": 2,
 "eval": "quick|gate", "bench": "lcb_v6_quick", "pass1": 0.0,
 "tokens_cot_mean": 0, "tokens_total_mean": 0, "latency_mean_s": 0.0,
 "acerto_por_1k_tokens": 0.0, "config": {...}, "git_sha": "...", "timestamp": "..."}
```

### 3.4 Gráficos (`lib/metrics.py`)
- Dispersão **pass@1 × tokens_cot_mean**, um ponto por run, com a fronteira de Pareto destacada e o baseline marcado.
- Gerado ao fim de cada notebook e salvo em `results/plots/`.

### 3.5 Critério de gate (implementar como função em `metrics.py`)
Uma fase **passa** se:
1. o melhor run da fase domina pelo menos um ponto do Pareto anterior (menos tokens com acerto ≥, ou mais acerto com tokens ≤) no LCB v6 completo; **e**
2. o SciCode não cai mais que 2 pontos em relação ao baseline.

O notebook de gate imprime `PASSOU` ou `REPROVOU` com os números. Se reprovar, a fase seguinte parte do melhor checkpoint anterior.

---

## 4. Fases e tarefas

### F0: Setup e baseline (`F0_setup_baseline.ipynb`)
**Tarefas**
1. Implementar `lib/` completo (exceto `dsl/`).
2. Carregar o Dory e validar a troca de loops: o mesmo prompt com 1, 2 e 4 loops deve gerar saídas diferentes e tempos crescentes.
3. Implementar o **logit lens por loop** em `lib/dory.py`: dado um prompt, retornar os top-k tokens ao fim de cada loop para as últimas posições.
4. Gerar os splits fixos (`data/lcb_quick.json`).
5. **Teste de equivalência** (`F0b_equivalence.ipynb`, sessão limpa): 5 prompts × {1, 2, 4} loops × 32 tokens gulosos. Rodar no vLLM oficial e, na nossa reimplementação, forçar a mesma sequência de tokens para comparar os logprobs posição por posição. Se o top-1 divergir em mais de 2% das posições, a execução para. **Nada abaixo roda sem esse teste aprovado.**
6. Rodar os baselines com a configuração fixa (`max_new_tokens=16000`, `temperature=1.0`, `top_p=0.95`, `seed=42`, vLLM):
   - **Dory** com 1/2/4 loops: `eval_quick` + `eval_gate`.
   - **Qwen3.5-4B** (controle sem loop, só texto): apenas `eval_quick`, só pelo vLLM.
   - O **Nanbeige não entra no F0** (ver seção 4, Comparação final).
7. A estimativa de compute units inclui o `eval_quick` e o `eval_gate` de todos os baselines. O `RUN_FULL_EVAL` só é ligado manualmente depois de revisar a estimativa.

**Aceite:** equivalência aprovada (`results/equivalence_vllm_hf.json`); 4 baselines em `results.jsonl` (Dory L1/L2/L4 + Qwen), todos com `backend=vllm`; gráfico Pareto inicial; documentação de como setar loops.

---

### F1: CoT curta

#### F1a: Auto-destilação (`F1a_self_distill.ipynb`)
1. Escolher o **dataset de treino com testes**, separado dos benchmarks (candidatos: KodCode, TACO, CodeContests). Remover problemas que tenham sobreposição textual com o LCB/HumanEval (dedup por n-grama).
2. Para cada problema, amostrar N=8 soluções do Dory (2 loops, temperatura ~0.7).
3. Executar os testes. Para cada problema, guardar **o trace correto com menos tokens de CoT**.
4. Incluir uma fração (~20%) de exemplos com **CoT vazia** nos problemas em que alguma amostra acertou com raciocínio mínimo (pensamento adaptativo).

**Saída:** `data/short_cot_sft.jsonl` + estatísticas (redução média de tokens em relação à média das amostras).

**Como foi implementado** (`scripts/f1a_self_distill.py`, notebook `F1a_self_distill.ipynb`):
- **Dataset:** `agentica-org/DeepCoder-Preview-Dataset`, config `taco` (TACO-verified filtrado, MIT, enunciados humanos, >= 5 testes). Ficam de fora o KodCode (gerado pelo GPT-4o), o `primeintellect` (sintético) e o `lcbv5` (são problemas do LCB v6). Só problemas stdin/stdout: os funcionais do TACO não trazem a assinatura no enunciado.
- **Dedup:** 13-gramas de palavras contra LCB v6, HumanEval+, MBPP+ e o custom; sai o problema que divide >= 30% dos 13-gramas com algum problema de benchmark.
- **Amostragem:** a oficial do Dory (temperatura 1.0, top_p 0.95), no lugar dos ~0.7, igual à da avaliação e à dos rollouts da F1c. Orçamento de 16384 tokens; amostra truncada conta como erro. Prompt e sandbox iguais aos do LCB (6 s por teste).
- **Guarda todas as amostras**, inclusive as erradas, com tipo do erro e teste que falhou: são os ramos mortos e os casos de borda do SFT arbóreo (`CONCEITO_RACIOCINIO_ARBOREO.md`, seção 8.1). O p̂ de cada problema (acertos / 8) separa direto de armadilha para o currículo e para o bônus (1 − p̂)² da F1c.
- **CoT vazia:** vai para os problemas com p̂ = 1 de menor CoT, até 20% dos exemplos.

#### F1b: SFT (`F1b_sft_short_cot.ipynb`)
- SFT do Dory em `short_cot_sft.jsonl`, com a config do perfil de hardware.
- Treinar com o número de loops padrão (2). Avaliar com 1/2/4.
- **Aceite:** `eval_quick` registrado para cada loop.

#### F1c: GRPO com penalidade de comprimento (`F1c_grpo_length.ipynb`)
- `trl.GRPOTrainer` (verificar compatibilidade com o modelo `trust_remote_code`; se não houver, implementar um loop GRPO mínimo).
- Recompensa em `lib/rewards.py`: `r = pass_tests(answer) - λ * n_cot_tokens / L_ref`, com bônus pequeno para o formato correto.
- **Sweep:** λ ∈ {0, 0.1, 0.3, 1.0} × loops de treino ∈ {1, 2, 4}. Começar por λ ∈ {0.1, 0.3} com 2 loops e expandir se o compute permitir.
- Rollouts com `max_new_tokens` limitado (ex.: 4096) para viabilizar o HF generate.
- **Aceite:** gráfico Pareto com todas as combinações, mostrando quantos tokens de CoT cada loop extra substitui.

#### F1d (opcional, recomendado para o artigo): mesma F1 no Qwen3.5-4B
- Repetir F1a → F1b → F1c no `Qwen/Qwen3.5-4B`, com o mesmo dataset, a mesma recompensa e os mesmos valores de λ.
- **Aceite:** curvas acerto × tokens do Dory comprimido e do Qwen comprimido no mesmo gráfico. É esse gráfico que mostra se os loops substituem tokens de raciocínio.
- Só executar com a confirmação dos autores (dobra o custo da F1).

**Gate F1:** `EVAL_phase_gate.ipynb` no melhor checkpoint.

---

### F2: Loops adaptativos (`F2_adaptive_loops.ipynb`)
1. **Diagnóstico:** medir por token a convergência entre loops (distância entre estados ocultos consecutivos e KL entre as distribuições do logit lens). Plotar a distribuição.
2. **Saída antecipada sem treino (heurística):** parar o loop do token quando a KL entre loops consecutivos ficar abaixo de um limiar. Sweep do limiar.
3. **Cabeça de parada treinada:** MLP pequeno sobre o estado oculto que prevê se mais um loop muda o token. Os rótulos vêm do próprio modelo rodando com 4 loops. Base congelada.
4. Verificar a compatibilidade com o KV cache (`recurrent_kv_cache_mode`). Se o loop variável por token quebrar o cache, documentar e propor uma solução.

**Aceite:** loops médios por token × pass@1 comparados com os loops fixos. **Gate F2.**

---

### F3: Linguagem otimizada (DSL)

#### F3a: Análise e design (`F3a_dsl_design.ipynb`)
1. Extrair dos traces da F1 os ~200 conceitos e operadores mais frequentes (n-gramas e termos técnicos).
2. Para cada um, medir o custo no tokenizer do Dory em quatro formas: inglês, glifo APL equivalente, caractere/termo chinês equivalente, e o candidato a token novo. Gerar a tabela `data/dsl_cost_table.csv`.
3. Propor a DSL. **Padrão: tokens customizados**, um por conceito, inicializados a partir do embedding do termo em inglês. Glifos APL/Mandarim só se já forem 1 token e mostrarem vantagem na tabela. Incluir as regras de composição. Escrever `lib/dsl/SPEC.md` e a gramática `lib/dsl/dsl.gbnf`.
4. Implementar `lib/dsl/parser.py`: DSL → AST → inglês (tradutor reverso determinístico).

**Ponto de parada humano:** apresentar a tabela e a spec para os autores aprovarem antes da F3b.

#### F3b: Corpus (`F3b_dsl_corpus.ipynb`)
- **Professor: DeepSeek V4.1 Flash** (pesos abertos, licença MIT). Modelos proprietários (Claude, Gemini) **não** são usados para gerar ou validar dados de treino, por restrição dos termos de uso, e porque um professor aberto deixa o artigo reprodutível.
- **Onde roda:** esta etapa **não precisa de GPU**. Roda como script Python local (`scripts/translate_dsl.py`), no PC dos autores ou num runtime de CPU do Colab, e grava `data/dsl_sft.jsonl` no Drive. Não gastar compute units de A100 aqui.
- **Backend principal: `OpencodeCLI`** (em `lib/teacher.py`), usando a **cota gratuita do DeepSeek V4.1 Flash no opencode**.
  - Chamar `opencode run` em modo não interativo. **Verificar** na documentação do opencode: o identificador do modelo gratuito, a flag de modelo, o formato de saída (preferir JSON) e como **desligar todas as ferramentas** do agente (a tarefa é só texto → texto; o agente não pode ler nem editar arquivos). Rodar num diretório de trabalho vazio, por segurança.
  - **Cota gratuita:** detectar erro de limite ou cota esgotada, pausar com backoff e retomar do checkpoint. Registrar quantos traces cabem por janela de cota para estimar o tempo total.
  - O agente envia um system prompt grande a cada chamada, e isso consome a cota. Por isso os lotes de vários traces por chamada são obrigatórios.
- **Backend reserva: `OpenAICompatHTTP`** (chave paga do DeepSeek ou de outro provedor). Só usar se a cota gratuita não der conta do volume, e só com a confirmação dos autores.
- **Lotes:** enviar **vários traces por chamada** (ex.: 10–20, em JSON) para diluir o overhead fixo do prompt. A spec da DSL vai num prefixo fixo, para aproveitar o cache do provedor.
- **Piloto:** 200 traces, no **modo sem raciocínio** e no modo com raciocínio (Max). Registrar taxa de aceitação, compressão e custo por trace aceito em cada combinação de backend × modo.
- **Produção (cascata):** a combinação mais barata por trace aceito traduz tudo. Os rejeitados vão uma vez para o modo com raciocínio. Os que falharem de novo são descartados.
- **Robustez:** checkpoint por lote (o script retoma de onde parou), limite de taxa com backoff e chaves de API em variável de ambiente ou Secrets do Colab, nunca no código.
- Validar cada item: (a) passa na gramática; (b) a volta pelo `parser.py` preserva o raciocínio, conferido por um LLM-juiz ou por heurística; (c) a resposta final continua passando nos testes.
- **Saída:** `data/dsl_sft.jsonl` + taxa de rejeição.

#### F3c: Tokens novos + SFT (`F3c_dsl_tokens_sft.ipynb`)
1. Adicionar os tokens novos ao tokenizer. `resize_token_embeddings`. Inicializar `embed_tokens` **e** `lm_head` (não são compartilhados) pela média dos embeddings dos subtokens originais.
2. **Warmup:** treinar só as linhas novas de embedding e lm_head (resto congelado) por poucos passos.
3. SFT completo em `dsl_sft.jsonl`.
4. Avaliar com e sem decodificação restrita pela gramática (no HF: `transformers` com `LogitsProcessor` próprio, ou biblioteca compatível com GBNF; verificar).

#### F3d: GRPO na DSL (`F3d_dsl_grpo.ipynb`)
- Mesmo setup da F1c, partindo da F3c. Recompensa extra para a gramática válida.
- **Aceite:** curva Pareto da F3 sobreposta à da F1. **Gate F3:** a curva da DSL tem que ficar à esquerda da F1.

---

### F4: Coconut sobre o Dory (`F4_coconut.ipynb`): ADIADA
**Não implementar agora.** Fica registrada como bônus experimental. Só será retomada por decisão explícita dos autores, depois do gate F3.
1. Estudar a implementação de referência (`facebookresearch/coconut`) e adaptar ao Dory: o estado oculto final (após Output) volta como embedding de entrada da próxima posição. Verificar se a normalização do nGPT dispensa a projeção. Se não dispensar, adicionar uma camada linear treinável.
2. **Currículo:** no estágio k, os primeiros k passos da DSL viram k pensamentos contínuos (delimitados por tokens especiais `<bot>`/`<eot>`).
3. Treino sequencial (sem paralelismo de teacher forcing nos pensamentos), com gradient checkpointing. Começar com k ≤ 3.
4. **Probe de interpretabilidade:** decodificador pequeno, estado latente → passo da DSL, treinado com os pares do currículo.

**Aceite e gate F4:** como nas outras fases.

---

### F5: Poda de vocabulário (`F5_vocab_prune.ipynb`)
1. Tokenizar o corpus de treino + saídas do modelo final + benchmarks. Contar a frequência de cada token.
2. Manter: os 256 tokens de byte, os tokens especiais, os tokens da DSL e os tokens com frequência acima do limiar. Remover o resto.
3. **Integridade do BPE:** não remover tokens intermediários de cadeias de merge usadas por tokens mantidos. Reconstruir os merges, ou remover apenas tokens "folha". Validar com round-trip encode/decode em todo o corpus.
4. Remapear `embed_tokens` e `lm_head` (separados) para os novos ids.
5. Medir VRAM, tamanho em disco e tokens/s antes e depois.

**Aceite:** nenhuma perda de pass@1 no `eval_gate`, mais os ganhos de VRAM e velocidade registrados.

---

## 5. Ordem de execução e checklist

- [x] F0: lib + teste de equivalência + baselines (Dory 1/2/4, Qwen3.5-4B). Resultado em `RESULTADOS_F0.md`; base da F1 = Dory L2
- [ ] F1a → F1b → F1c → **gate F1**
- [ ] F2 → **gate F2**
- [ ] F3a → **aprovação humana** → F3b → F3c → F3d → **gate F3**
- [ ] F4 (condicional) → **gate F4**
- [ ] F5 → **gate final**
- [ ] Comparação final (`FINAL_external.ipynb`): Nanbeige4.2-3B apenas no `eval_quick`, com a configuração fixa. Se não rodar no vLLM, usar o HF com `max_new_tokens=8192` e registrar o orçamento diferente; se o custo for inviável, usar o número divulgado pelo fabricante, marcado como não medido.
- [ ] Relatório final: Pareto de todas as fases, comparação com o Qwen3.5-4B (e com o Nanbeige, se medido), exemplos de traces (original → curto → DSL → latente com logit lens)

## 6. Fluxo de implementação no Antigravity

- **Uma fase de cada vez.** Implementar a F0, rodar e validar o baseline, e só então pedir a próxima fase. Fases posteriores dependem de dados e decisões das anteriores.
- **Divisão de modelos:**
  - **Gemini 4 Argon** implementa ou revisa os **módulos críticos**, onde erro silencioso invalida resultados: `lib/dory.py` (loops, cache, logit lens), `lib/gen.py` (separação CoT/resposta e contagem de tokens), `lib/rewards.py`, `lib/sandbox.py`, `lib/evals.py`, F1c (GRPO), F2 (loops adaptativos), F3c (tokens novos com embedding/lm_head separados), F4 e F5 (integridade do BPE).
  - **Gemini 3.8 Flash** implementa o resto: `io.py`, `registry.py`, `metrics.py`, gráficos, estrutura dos notebooks, scripts de dados e `teacher.py`.
  - **Argon revisa o diff** de tudo que o Flash fizer nos módulos críticos.
- **Testes obrigatórios para os módulos críticos** (`tests/`, rodam numa célula antes de qualquer corrida):
  - Contagem de tokens de CoT num exemplo com resultado conhecido.
  - Recompensa em casos sintéticos (correto/curto, correto/longo, errado).
  - Sandbox respeitando timeout e limite de memória.
  - Mudança de loops alterando a saída e o tempo.
  - Tokens novos com embeddings inicializados corretamente (embedding e lm_head) e round-trip encode/decode.

## 7. Decisões que o agente **não** toma sozinho
- Escolha final do professor após o piloto (F3b): apresentar a tabela do piloto.
- Aprovação da spec da DSL (F3a).
- Retomar a F4.
- Qualquer corrida estimada em mais de 30 compute units: pedir confirmação antes.
