# Benchmark custom (medição durante o desenvolvimento)

Enquanto o projeto está em desenvolvimento, as fases são comparadas **só** neste benchmark. HumanEval+, LCB, SciCode e os outros benchmarks públicos ficam para a demonstração final.

Código em `lib/custom_bench.py`. Conferência em `tests/test_custom_bench.py`, que roda na Célula 2.

## Como rodar

Use a Célula 4 do `notebooks/F0_setup_baseline.ipynb`, ou o script:

```bash
python scripts/f0_run_baseline.py --target L2 --scope custom_quick
```

```bash
python scripts/f0_run_baseline.py --target plot
```

| Modo | Problemas | Uso |
|---|---|---|
| `custom_quick` | 126 (7 por família) | Iterar. Cabe num único lote do vLLM, então o tempo é o da geração mais longa. |
| `custom_full` | 504 (28 por família, contém o quick) | Confirmar quando a diferença entre duas versões é pequena. |

**Geração (custom_v3):**
- **Orçamento de 81.920 tokens** (`CUSTOM_MAX_NEW_TOKENS`) para todos os modelos. É o que a Qwen recomenda para programação de competição, e o Dory aceita até 256K. Assim o truncamento não decide o acerto.
- **Amostragem oficial de cada modelo**, copiada da página dele no HF e guardada em `OFFICIAL_SAMPLING` (`lib/dory.py`):
  - Dory: `temperature=1.0`, `top_p=0.95`, sem `top_k` nem penalidade.
  - Qwen3.5-4B, modo de pensamento para tarefas gerais: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`. A penalidade é o que a Qwen indica contra repetição infinita.
- `seed=42`, vLLM.
- O que já está em `results.jsonl` (mesmo `run_id`, bench e protocolo) não roda de novo.

**GPU:** 81.920 tokens por resposta pedem memória. O ideal é G4 (96 GB) ou A100 de 80 GB. Na A100 de 40 GB e na L4, o Dory L4 (cache separado por loop) pode ficar lento, com poucas respostas ao mesmo tempo, ou não caber.

A amostragem da Qwen entra no protocolo e no cache (`custom_v3_k20_pp1.5`). Assim, uma rodada com amostragem diferente nunca reaproveita outra. A tabela do `plot` filtra pelo `custom_protocol`.

## O que é

São 18 famílias de problemas de código, 6 por nível:

| Nível | Para que serve | Famílias |
|---|---|---|
| easy | O Dory acerta tudo. Mede se ele pensa demais em problema simples (F1). | soma de dígitos filtrada, transformação do k-ésimo caractere, soma com filtro de resto, blocos repetidos, pares com soma divisível, ordenação por chave |
| medium | O Dory acerta a maioria, e os loops se separam (na v1: L1 64%, L2 88%). Avisa cedo se uma fase quebra o raciocínio médio. | espiral a partir de um canto, contagem de escadas mod M, maior trecho balanceado de parênteses, grade com passes grátis (Dijkstra com estado), intervalos com peso e folga, jogo de pilhas (Grundy) |
| hard | O Dory acerta pouco (alvo de 10% a 25%). É aqui que se vê se uma fase ganhou raciocínio ou só economizou tokens. | DP de dígitos com 3 condições até 10^18, strings sem padrões proibidos até 10^18 (autômato + potência de matriz), conjunto de nós de uma árvore com distância mínima (DP em árvore), menor número de trocas para agrupar valores (DP em bitmask sobre inversões), jogo de pilhas com divisão (Sprague-Grundy), subsequências distintas de tamanho k com restrição entre vizinhos |

**Como ler o hard:** com o orçamento grande, o `hard_pass1` mede se o modelo sabe resolver, não se a resposta coube.
- **Ganho de raciocínio:** o `hard_pass1` (81.920 tokens) da fase sobe acima do L2 base.
- **Ganho de tokens:** o `hard_pass1` fica igual e os tokens caem (`hard_tokens_cot_median`, `tokens_cot_mean_correct`). Ou então o acerto com orçamento menor sobe: `hard_pass1_at_budget["16000"]` passa a chegar perto do acerto completo.
- O `hard_fail_breakdown` separa três tipos de falha:
  - `wrong_answer`: errou o raciocínio;
  - `truncated`: ainda estava raciocinando quando acabou o orçamento;
  - `loop`: estourou o orçamento repetindo o mesmo trecho.

No quick, o hard tem 42 problemas, e um a mais muda ~2,4 pontos. Para afirmar ganho no hard, use o `custom_full` (168 problemas no hard).

- **Cada instância muda o problema, não só os nomes.** A semente sorteia parâmetros como o conjunto de passos, o tipo de movimento, o canto e o sentido da espiral, o limite de itens ou o módulo, e a resposta certa muda com eles.
- **O enunciado traz restrições, dois exemplos e uma assinatura fixa.** A correção usa cerca de 14 testes escondidos por problema, com casos de borda. Nos níveis medium e hard há também entradas grandes, que reprovam força bruta exponencial. O limite é de 10 s por problema, folgado para não reprovar uma solução polinomial correta.
- **É determinístico.** A mesma versão gera sempre os mesmos problemas. Qualquer mudança numa família exige subir `CUSTOM_PROTOCOL_ID`, o que invalida os resultados antigos em vez de misturá-los: o cache por problema é separado por versão, e a tabela e o gráfico só mostram a versão atual.

## Por que confiar na medição

O `tests/test_custom_bench.py` verifica cinco coisas:
1. Em cada família medium/hard, a solução de referência bate com uma **força bruta escrita de outro jeito** em pelo menos 100 entradas pequenas. A espiral, por exemplo, é conferida por rotação da matriz em vez de simulação; o agrupamento por trocas, por BFS sobre as trocas; a árvore, testando todos os subconjuntos. Também verifica que a referência resolve cada problema em menos de 3 s, com folga para o limite de 10 s do sandbox.
2. Os testes de cada problema **aceitam a referência e rejeitam uma resposta qualquer**, e nenhuma instância tem todas as saídas iguais.
3. Determinismo, e `quick` é prefixo de `full`.
4. O caminho completo funciona: gerador falso → `eval_custom` → sandbox → `results.jsonl`. Resposta truncada conta como erro, e uma segunda execução não duplica a linha.
5. As métricas agregadas.

## O que mede, e para qual parte do projeto

Tudo fica no registro de `results.jsonl`: os campos padrão mais o `config`.

| Métrica | Campo | Serve para |
|---|---|---|
| Acerto | `pass1`, `{easy,medium,hard}_pass1`, `family_pass1` | todas as fases (não pode cair) |
| Tokens de raciocínio | `tokens_cot_mean`, `{nível}_tokens_cot_mean` | F1 (CoT curta), F3 (DSL) |
| Tokens só nos acertos | `tokens_cot_mean_correct` | F1/F3: separa "comprimiu" de "desistiu cedo" |
| Pensar demais no fácil | `easy_tokens_cot_mean` | F1 (pensamento adaptativo) |
| Estouro do orçamento | `{nível}_truncated_frac` | F1/F3 |
| Tempo | `latency_mean_s` (por problema, amortizado no lote), `throughput_tok_s` | F2 (loops adaptativos), F5 (vocabulário) |
| Loops | `n_loops` do registro (L1/L2/L4) | F2 |
| Acerto com limite menor | `pass1_at_budget` e `{nível}_pass1_at_budget`: pass@1 se o limite fosse 4k, 8k, 16k ou 32k (passou e coube). Sai da mesma rodada. No Dory, o ponto de 16k é comparável à v2; na Qwen não é, porque a amostragem mudou. | F1 (curva acerto × orçamento) |
| Tokens sem cauda | `{nível}_tokens_cot_median` | F1/F3 (algumas respostas em loop até o orçamento puxam a média) |
| Por que errou | `fail_breakdown` e `{nível}_fail_breakdown`: `truncated`, `loop`, `no_code`, `wrong_answer`, `timeout`, `runtime_error` | todas (diagnóstico); no hard, separa raciocínio de tokens |
| Repetição | `loop_score` por problema: fração de 8-gramas repetidos nas últimas 2.000 palavras da CoT. Vira `loop` acima de 0,5. As truncadas guardam `tail`, os últimos 1.500 caracteres da CoT, no cache do problema. | diagnóstico; o limiar de 0,5 ainda precisa ser calibrado lendo os `tail`. Limites conhecidos: com o `presence_penalty`, o loop da Qwen tende a virar paráfrase e o score subestima; um bloco repetido com mais de ~1.000 palavras também fica abaixo de 0,5; e só a CoT é medida, não a resposta. |
| Eficiência | `acerto_por_1k_tokens` e o gráfico `F0_pareto_custom_quick.png` | métrica central |

Ainda não mede: VRAM (F5) e média de loops por token (F2 adaptativo), porque dependem do código dessas fases.

## Versões

- **custom_v1** (commit 574a5c3): o Dory saturou. Easy 100%, hard ~90% no L2, e 12 das 18 famílias deram 100% em todos os loops. Pass@1 / CoT médio: L1 0.881 / 6486, L2 0.960 / 5974, L4 0.944 / 5848, Qwen 0.468 / 12292. A ordem foi a mesma dos benchmarks públicos.
- **custom_v2**: o easy ficou igual. O medium passou a ser formado pelas 6 famílias da v1 que separavam os loops. As 6 que davam sempre 100% saíram, e entraram 6 hard novas. A semente é a mesma, então o easy e as famílias que mudaram de nível geram as mesmas instâncias da v1.
  - Rodada quick (commit 2840852), pass@1 easy / medium / hard: L1 1.00 / 0.74 / 0.19, L2 1.00 / 0.81 / 0.29, L4 1.00 / 0.83 / 0.24, Qwen 0.88 / 0.19 / 0.00.
  - No hard, 29 das 30 falhas do L2 foram truncamento em 16k. `tree_spread` deu 0% em todos os modelos e `distinct_subseq` só passou de 0 no L4 (14%). Elas não foram aliviadas: o orçamento da v3 mostra se são difíceis de verdade ou só longas.
  - Erro de protocolo: a Qwen rodou com a amostragem do Dory, sem `top_k=20` nem `presence_penalty=1.5`, e com 16k tokens, quando a Qwen recomenda de 32k a 81k. Os números dela na v1 e na v2 não valem como controle.
- **custom_v3** (atual): mesmas instâncias da v2, com orçamento de 81.920 tokens e amostragem oficial de cada modelo. Também passa a medir pass@1 por orçamento, mediana de tokens e `loop`. Os baselines precisam ser refeitos.

Depois de cada rodada de baselines, confira o `family_pass1`. No hard, uma família que fique em 0% para todos os loops não mede nada; nesse caso, alivie os parâmetros dela e suba a versão.
