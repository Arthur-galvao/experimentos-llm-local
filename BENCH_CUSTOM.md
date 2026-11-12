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

A configuração fixa de geração é a mesma do F0: `max_new_tokens=16000`, `temperature=1.0`, `top_p=0.95`, `seed=42`, vLLM. O que já está em `results.jsonl` (mesmo `run_id`, bench e protocolo) não roda de novo.

## O que é

São 18 famílias de problemas de código, 6 por nível:

| Nível | Para que serve | Famílias |
|---|---|---|
| easy | O Dory acerta tudo. Mede se ele pensa demais em problema simples (F1). | soma de dígitos filtrada, transformação do k-ésimo caractere, soma com filtro de resto, blocos repetidos, pares com soma divisível, ordenação por chave |
| medium | O Dory acerta a maioria, e os loops se separam (na v1: L1 64%, L2 88%). Avisa cedo se uma fase quebra o raciocínio médio. | espiral a partir de um canto, contagem de escadas mod M, maior trecho balanceado de parênteses, grade com passes grátis (Dijkstra com estado), intervalos com peso e folga, jogo de pilhas (Grundy) |
| hard | O Dory acerta pouco (alvo de 10% a 25%). É aqui que se vê se uma fase ganhou raciocínio ou só economizou tokens. | DP de dígitos com 3 condições até 10^18, strings sem padrões proibidos até 10^18 (autômato + potência de matriz), conjunto de nós de uma árvore com distância mínima (DP em árvore), menor número de trocas para agrupar valores (DP em bitmask sobre inversões), jogo de pilhas com divisão (Sprague-Grundy), subsequências distintas de tamanho k com restrição entre vizinhos |

**Como ler o hard:** na v2, quase toda falha do Dory no hard é `truncated`: ele não erra a resposta, ele não termina dentro de 16k tokens. Por isso o `wrong_answer` sozinho não separa raciocínio de tokens. A comparação é com o **teto**: o mesmo hard rodado no L2 base com 64k tokens (`--scope custom_quick_teto`, registro `F0_dory_L2_teto64k` no bench `custom_quick_hard`).
- A fase, em 16k, acerta **mais que o teto**: ganhou raciocínio. Resolve o que o modelo base não resolve nem com 4x mais tokens.
- Acerta mais que o L2 em 16k, mas **até o teto**: ganhou tokens. Faz o mesmo raciocínio em menos espaço.

No quick, o hard tem 42 problemas, e um a mais muda ~2,4 pontos. Para afirmar ganho no hard, use o `custom_full` e o teto do full (`custom_full_teto`), com 168 problemas no hard.

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
| Por que errou | `fail_breakdown` e `{nível}_fail_breakdown`: `truncated`, `no_code`, `wrong_answer`, `timeout`, `runtime_error` | todas (diagnóstico); no hard, separa raciocínio de tokens |
| Eficiência | `acerto_por_1k_tokens` e o gráfico `F0_pareto_custom_quick.png` | métrica central |

Ainda não mede: VRAM (F5) e média de loops por token (F2 adaptativo), porque dependem do código dessas fases.

## Versões

- **custom_v1** (commit 574a5c3): o Dory saturou. Easy 100%, hard ~90% no L2, e 12 das 18 famílias deram 100% em todos os loops. Pass@1 / CoT médio: L1 0.881 / 6486, L2 0.960 / 5974, L4 0.944 / 5848, Qwen 0.468 / 12292. A ordem foi a mesma dos benchmarks públicos.
- **custom_v2** (atual): o easy ficou igual. O medium passou a ser formado pelas 6 famílias da v1 que separavam os loops. As 6 que davam sempre 100% saíram, e entraram 6 hard novas. A semente é a mesma, então o easy e as famílias que mudaram de nível geram as mesmas instâncias da v1.
  - Rodada quick (commit 2840852), pass@1 easy / medium / hard: L1 1.00 / 0.74 / 0.19, L2 1.00 / 0.81 / 0.29, L4 1.00 / 0.83 / 0.24, Qwen 0.88 / 0.19 / 0.00.
  - No hard, 29 das 30 falhas do L2 foram truncamento em 16k. `tree_spread` deu 0% em todos os modelos e `distinct_subseq` só passou de 0 no L4 (14%). Elas não foram aliviadas: o teto em 64k mostra se são difíceis de verdade ou só longas.

Depois de cada rodada de baselines, confira o `family_pass1`. No hard, uma família que fique em 0% para todos os loops não mede nada; nesse caso, alivie os parâmetros dela e suba a versão.
