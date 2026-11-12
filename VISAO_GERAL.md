# Projeto: Modelo Coder com Raciocínio Comprimido

> Documento de visão geral. A especificação de implementação (notebooks do Colab) está em `ESPEC_ANTIGRAVITY.md`.

## 1. Objetivo

Construir, a partir de um modelo open-weight pequeno, um **modelo coder (inglês + código)** que resolva tarefas de programação gastando **o mínimo possível de tokens de raciocínio**, deslocando o "pensar" de tokens visíveis para computação latente, e o que sobrar de raciocínio visível para uma **linguagem otimizada** e densa.

Projeto experimental: o objetivo é descobrir até onde dá para ir, medindo cada passo.

## 2. Métrica central

**Acerto por token**, sempre medido contra o baseline reproduzido do modelo original.

- `pass@1` em código (testes executados).
- `tokens_cot`: tokens médios de raciocínio por problema.
- `tokens_total`: tokens gerados (raciocínio + resposta).
- Curva **acerto × tokens** (fronteira de Pareto). O resultado do projeto é empurrar essa curva para a esquerda (menos tokens) sem descer (mesmo acerto).

Encurtar a resposta e perder acerto não conta como ganho.

## 3. Modelo base

### Principal: NVIDIA Nemotron-Labs Dory 1.0 4B (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`)
- Normalized-GPT **em loop (latent-recurrent)**: grupos de camadas Input → Recurrent → Output. O bloco recorrente roda N vezes na mesma posição, com a entrada reinjetada a cada loop.
- O **número de loops é ajustável na inferência** sem retreinar (1 = mínimo, 2 = padrão, 4 = mais acurácia).
- Pré-treinado em **inglês + 43 linguagens de programação**: já é o recorte "só inglês + código" que queríamos, sem poda estrutural de idiomas.
- 256K de contexto, licença OpenMDW 1.1 (permite uso comercial).
- Vocabulário de 131.072 tokens, `tie_word_embeddings: false`.
- SciCode (self-reported, por loops): 27.3 / 34.94 / 35.11. SWE-bench Verified: 50.3 / 59.5 / 61.8. **LiveCodeBench não é reportado**, então o baseline é nosso.

### Controle: Qwen3.5-4B (`Qwen/Qwen3.5-4B`)
- Transformer **convencional, sem loop**, do mesmo porte do Dory (4B). Suporte maduro no vLLM, por isso barato de avaliar.
- É o modelo de comparação que a própria NVIDIA usa no card do Dory. Self-reported nesse card: SciCode 25.82 e SWE-bench Verified 54.7, contra 27.3 / 50.3 do Dory com 1 loop e 35.11 / 61.8 com 4 loops.
- Serve para responder: *quanto do ganho vem dos loops e quanto vem do nosso método?*
- É multimodal; usamos só entrada de texto.
- **Experimento-chave opcional para o artigo:** aplicar a mesma F1 (CoT curta + GRPO com penalidade de comprimento) ao Qwen3.5-4B. Se a curva acerto × tokens do Dory comprimido for melhor que a do Qwen comprimido pelo mesmo método, fica demonstrado que os loops substituem tokens de raciocínio.

### Comparação externa (só no final): Nanbeige4.2-3B (`Nanbeige/Nanbeige4.2-3B`)
- Outro modelo em loop (~4B), Apache-2.0. Self-reported: LCB v6 72.5, SWE-bench Verified 63.6, SciCode 35.6. CoT muito longa (até 131K tokens recomendados).
- **Fora do F0**: provavelmente não roda no vLLM, e no HF a avaliação completa levaria dias de A100.
- Avaliado **uma vez, no fim do projeto**, só no `eval_quick`, para a tabela do artigo. Se o custo for inviável, cita-se o número divulgado pelo fabricante, deixando explícito que não foi medido por nós.

### Descartados
- **LFM3-1.7B:** forte em código (LCB v6 73.46), mas sem loop. Raciocínio latente exigiria Coconut desde o início, a fase mais cara.
- **coder-lfm3-1.7b, Qwen3.5-4B-3D-Coder:** fine-tunes pequenos ou de nicho, sem benchmark. Não servem como base.

## 4. Os eixos de compressão

O raciocínio pode ser comprimido em eixos diferentes e complementares:

| Eixo | Técnica | O que troca | Fase |
|---|---|---|---|
| Tokens | CoT curta (SFT + RL com penalidade de comprimento) | Menos tokens visíveis | F1 |
| Profundidade | Loops do Dory (fixos → adaptativos) | Mais cálculo por token, menos tokens | F1–F2 |
| Linguagem | DSL otimizada com tokens customizados | Cada token carrega mais informação | F3 |
| Tempo latente | Coconut (pensamentos contínuos) | Tokens de CoT viram posições latentes | F4 (condicional) |
| Modelo | Poda de vocabulário | VRAM e velocidade (ganho modesto) | F5 |

### 4.1 Pensamento vetorial: profundidade × tempo

Três coisas diferentes costumam ser chamadas de "pensar em vetores":

1. **Activation steering (CAA):** somar um vetor fixo às ativações. Controla estilo, não pensa. **Descartado como pilar.** O RL faz o mesmo trabalho melhor.
2. **Coconut (eixo do tempo):** o último estado oculto vira a entrada da *próxima posição* no lugar de um token. A cadeia de raciocínio continua existindo (ocupa posições e KV cache), mas é latente: sem gargalo de vocabulário, e um vetor pode carregar várias hipóteses.
3. **Loops do Dory (eixo da profundidade):** o bloco recorrente roda várias vezes *na mesma posição*. Mais cálculo por token, sem posições novas, e o KV cache não cresce com os loops.

Loops conseguem substituir parte da CoT: há resultado teórico e empírico de que k loops simulam passos de raciocínio explícito (Saunshi et al., ICLR 2025, *Reasoning with Latent Thoughts: On the Power of Looped Transformers*). Mas problemas com muitos passos sequenciais ainda precisam de "espaço" na sequência. Por isso os eixos se combinam:

| | Tokens de CoT | Profundidade por token |
|---|---|---|
| CoT normal | muitos, visíveis | fixa |
| Coconut | posições latentes | fixa |
| Dory | menos, visíveis | variável (loops) |
| Dory + Coconut | posições latentes | variável |

**Ordem:** primeiro esgotar o eixo nativo e barato (loops + CoT curta). Coconut só entra se a curva saturar, porque o backprop atravessa loops × pensamentos sequenciais e o custo se multiplica.

### 4.2 Loops adaptativos

Uma cabeça de parada decide, token a token, quantos loops rodar: boilerplate e sintaxe usam 1, pontos de decisão usam 4. A referência é o Huginn (Geiping et al. 2025), que fazia saída antecipada quando o estado convergia entre loops. É o equivalente, no eixo da profundidade, do "pensamento adaptativo" (pular a CoT em passos triviais).

## 5. Linguagem otimizada (DSL)

A ideia original era uma CoT híbrida: **Mandarim** para a semântica das entidades e **APL** para a lógica operacional. Com o modelo restrito a inglês + código, a abordagem fica assim:

- **O que importa é o custo em tokens, não em caracteres.** Glifos de APL costumam custar 1–3 tokens. Termos técnicos em inglês ("payload", "database") já costumam ser 1 token.
- **APL e Mandarim viram um *pool de candidatos*, não uma regra.** Para cada um dos ~200 conceitos e operadores mais frequentes nos traces, escolhe-se a forma mais barata no tokenizer: inglês, glifo APL, caractere chinês ou **token novo criado sob medida**.
- **Tokens customizados:** o que ainda custar mais de 1 token vira um token novo no vocabulário. O embedding é inicializado pela média dos subtokens originais e treinado primeiro isolado, depois no SFT completo.
- **Gramática formal (GBNF):** a DSL é definida como gramática, e o bloco de raciocínio usa decodificação restrita. Isso garante a sintaxe, torna o parser reverso determinístico e permite medir a densidade exata.
- **Teste decisivo:** a curva acerto × tokens da DSL tem que ficar à esquerda da curva da CoT curta em inglês. Se não ficar, a DSL é descartada.

## 6. Interpretabilidade (tradutor reverso)

O modelo vai operar comprimido. As ferramentas para entender o que ele fez:

1. **Parser da DSL:** a gramática formal permite traduzir a DSL de volta para inglês de forma determinística.
2. **Logit lens por loop:** projetar o estado oculto no `lm_head` ao fim de cada loop mostra o "rascunho" convergindo. É nativo do Dory e barato.
3. **Probe para pensamentos Coconut** (se a F4 acontecer): um decodificador pequeno treinado com os pares (estado latente → passo da DSL) que o próprio currículo gera.

Um LLM traduzindo um log comprimido gera uma explicação plausível, mas não fiel. Por isso foi descartado como mecanismo principal.

## 7. Avaliação

**Durante o desenvolvimento, toda comparação entre fases usa só o benchmark custom** ([BENCH_CUSTOM.md](BENCH_CUSTOM.md)): problemas de código gerados por semente, com modo rápido de 126 e completo de 504, medindo acerto, tokens, truncamento, tempo e motivo das falhas. Os benchmarks públicos abaixo ficam para a demonstração final.

### Benchmarks (demonstração final)
- **HumanEval+ / MBPP+ (EvalPlus):** filtro rápido, roda em minutos.
- **LiveCodeBench v6:** métrica principal. Usar os problemas mais recentes para evitar contaminação. Subconjunto fixo de ~200 para iterar, completo no fim de cada fase.
- **LiveCodeBench Pro:** difícil e com ruído alto. Só no fim de fase.
- **SciCode:** alarme de perda de conhecimento (exige conhecimento científico, não só código).

### Fluxo
- **A cada experimento:** `custom_quick`.
- **No fim de cada fase:** `custom_full`, contra o baseline reproduzido nos mesmos problemas.
- **Critério de corte:** a fase só fica se melhorar a curva acerto × tokens no `custom_full` sem perder acerto em nenhum nível (easy/medium/hard). Ganho no hard só conta como ganho de raciocínio se cair o `wrong_answer` do `hard_fail_breakdown`, e não só o truncamento. Se não passar, a fase é descartada e o projeto segue da melhor versão anterior.
- **Demonstração final:** os benchmarks públicos acima, contra os números de [RESULTADOS_F0.md](RESULTADOS_F0.md).

### Baseline
Os números dos model cards são vendor-reported. Reproduzimos o Dory (1, 2 e 4 loops) e o Qwen3.5-4B na nossa configuração, e **esses** passam a ser os números de referência.

**Configuração fixa de geração** (vale para todas as comparações): `max_new_tokens=16000`, `temperature=1.0`, `top_p=0.95` (o recomendado pela NVIDIA), `seed=42`, backend vLLM. Mudar algum desses invalida a comparação.

**Validação da reimplementação:** o Transformers não tem implementação do Dory. Usamos a nossa (`lib/dory.py`), conferida linha a linha contra o código oficial da PR do vLLM e validada por um teste numérico de equivalência (`F0b_equivalence.ipynb`).

## 8. Roadmap

| Fase | Conteúdo | Gate |
|---|---|---|
| F0 | Ambiente, teste de equivalência, harness de avaliação, baselines (Dory 1/2/4 loops + Qwen3.5-4B) | Equivalência aprovada; baselines registrados |
| F1 | CoT curta: auto-destilação (menor trace correto) → SFT → GRPO com acerto − λ·tokens × loops; pensamento adaptativo. Opcional: mesma F1 no Qwen3.5-4B | Curva acerto × tokens melhor que o baseline |
| F2 | Loops adaptativos por token (cabeça de parada) | Menos loops médios no mesmo acerto |
| F3 | DSL: análise do tokenizer → spec + gramática → corpus → tokens novos → SFT → GRPO | Curva à esquerda da F1 |
| F4 | Coconut sobre o Dory (só se F1–F3 saturarem) | Curva à esquerda da F3 |
| F5 | Poda de vocabulário guiada pelo uso real | Sem perda de acerto; medir VRAM e velocidade |

## 9. Descartado ou adiado (e por quê)

- **Poda estrutural de idiomas:** desnecessária, porque o Dory já é inglês + código no pré-treino.
- **Activation steering como pilar:** controla estilo, não raciocina. No máximo, um experimento comparativo.
- **Pré-treino continuado em código:** opcional, já que o modelo base é forte em código.
- **Otimizações de input (soft prompt, gist tokens):** fora do escopo atual, que é otimizar o modelo em si.

## 10. Decisões

### Tomadas
1. **F4 (Coconut) adiada.** É bônus experimental, fora do plano principal.
2. **DSL = tokens customizados** como base. Cada token novo é inicializado a partir do embedding do conceito em inglês, então nasce com a semântica do conceito. Glifos APL/Mandarim só entram se já forem 1 token e mostrarem vantagem na tabela de custo. O design da linguagem será definido em discussão própria.
3. **Professor da tradução para a DSL: DeepSeek V4.1 Flash** (pesos abertos, MIT), via **opencode CLI usando a cota gratuita**. HTTP pago fica só como reserva. Cascata: o modo sem raciocínio traduz tudo, e os rejeitados vão para o modo com raciocínio. Roda em CPU (local), sem gastar A100.
   - **Claude e Gemini estão fora**: os Termos de Consumidor da Anthropic (seção 3.2) proíbem usar o serviço para desenvolver ou treinar modelos de IA, sem exceção para pesquisa ou uso sem fins lucrativos. Os termos do Google precisam ser checados, mas tendem a ter cláusula parecida.
   - **Vantagem para o artigo:** com um professor de pesos abertos, o pipeline de dados é reprodutível por terceiros.
   - Modelo aberto rodando no Colab (ex.: DavidAU Qwen3.8-27B) foi descartado: custaria horas de A100 e o comportamento é instável para uma tradução de formato estrita.

### Em aberto
1. **Dataset de treino com testes**, separado dos benchmarks (candidatos: KodCode, TACO, CodeContests). Verificar sobreposição com o LCB.
2. **Orçamento de compute units** por fase.
3. **Termos de uso da API do DeepSeek** para gerar dados de treino: confirmar antes da produção.
