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

| Nível | Famílias |
|---|---|
| easy | soma de dígitos filtrada, transformação do k-ésimo caractere, soma com filtro de resto, blocos repetidos, pares com soma divisível, ordenação por chave |
| medium | janela com k letras distintas, união de intervalos com folga, espiral a partir de um canto, contagem de escadas mod M, menor caminho em grade (4/8 vizinhos/cavalo), maior trecho balanceado de parênteses |
| hard | maior subsequência com diferença limitada, mochila com limite de itens, grade com passes grátis (Dijkstra com estado), intervalos com peso e folga, contagem de strings sem bloco longo, jogo de pilhas (Grundy) |

- **Cada instância muda o problema, não só os nomes.** A semente sorteia parâmetros como o conjunto de passos, o tipo de movimento, o canto e o sentido da espiral, o limite de itens ou o módulo, e a resposta certa muda com eles.
- **O enunciado traz restrições, dois exemplos e uma assinatura fixa.** A correção usa cerca de 14 testes escondidos por problema, com casos de borda. Nos níveis medium e hard há também entradas grandes, que reprovam força bruta exponencial. O limite é de 10 s por problema, folgado para não reprovar uma solução polinomial correta.
- **É determinístico.** A mesma versão gera sempre os mesmos problemas. Qualquer mudança numa família exige subir `CUSTOM_PROTOCOL_ID`, o que invalida os resultados antigos em vez de misturá-los.

## Por que confiar na medição

O `tests/test_custom_bench.py` verifica cinco coisas:
1. Em cada família medium/hard, a solução de referência bate com uma **força bruta escrita de outro jeito** em pelo menos 100 entradas pequenas. A espiral, por exemplo, é conferida por rotação da matriz em vez de simulação.
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
| Por que errou | `fail_breakdown`: `truncated`, `no_code`, `wrong_answer`, `timeout`, `runtime_error` | todas (diagnóstico) |
| Eficiência | `acerto_por_1k_tokens` e o gráfico `F0_pareto_custom_quick.png` | métrica central |

Ainda não mede: VRAM (F5) e média de loops por token (F2 adaptativo), porque dependem do código dessas fases.

Na primeira rodada, se uma família ficar em 0% ou 100% para todos os modelos, ela não separa nada. Nesse caso, ajuste os parâmetros da família e suba a versão.
