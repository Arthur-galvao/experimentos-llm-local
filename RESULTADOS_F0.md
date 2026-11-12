# F0: Setup e baselines (resultado)

Concluído em 2026-11-12. Os números de referência das fases seguintes são os desta página. A fonte de verdade é `MyDrive/coder-comprimido/results/results.jsonl`. As saídas por item ficam em `eval_cache/` e os gráficos em `results/plots/`, todos no Drive.

**Configuração fixa** (igual em todas as linhas): vLLM (PR #60149), `max_new_tokens=16000`, `temperature=1.0`, `top_p=0.95`, `seed=42`. **Truncado** = atingiu 16k tokens sem fechar `</think>`; truncado conta como **erro** (ver "Correções").

> **Aviso sobre a Qwen (descoberto depois):** as linhas da Qwen3.5-4B rodaram com a amostragem do Dory, sem o `top_k=20` e o `presence_penalty=1.5` que a página da Qwen recomenda, e com 16k tokens, quando ela recomenda de 32k a 81k. Os números dela abaixo subestimam o modelo e não valem como controle. Os do Dory estão na configuração oficial. A amostragem oficial agora sai de `OFFICIAL_SAMPLING`. Ela entra no protocolo e no cache da Qwen (sufixo `_k20_pp1.5`), então a próxima rodada pública da Qwen refaz tudo sozinha, sem reaproveitar as gerações antigas. Ainda falta decidir o orçamento público da Qwen para a demonstração final: 16k como o do Dory, ou os 32k a 81k que ela recomenda.

## 1. Equivalência vLLM × reimplementação HF (`lib/dory.py`)

5 prompts × 32 tokens gulosos por loop; limite de 2% de divergência no top-1.

| Loops | Posições | Divergências | Taxa | |Δ logprob| médio |
|---|---|---|---|---|
| 1 | 160 | 1 | 0,6% | 0,004 |
| 2 | 160 | 0 | 0% | — |
| 4 | 160 | 0 | 0% | — |

Aprovado. A única divergência do L1 é `"I"` × `" I"` no último passo, um empate quase exato (Δ 0,09). Os arquivos estão em `results/equivalence_vllm_hf_L{1,2,4}.json`.

## 2. eval_quick (HumanEval+ completo + 200 problemas mais recentes do LCB v6)

| Modelo | HE+ pass@1 | LCB quick pass@1 | LCB truncado | CoT média LCB | Latência LCB |
|---|---|---|---|---|---|
| Dory L1 | 0,945 | 0,455 | 46% | 11,4k | 6,6 s |
| **Dory L2** | 0,915 | **0,535** | 39,5% | 10,9k | 9,7 s |
| Dory L4 | 0,915 | 0,510 | 42% | 11,0k | 17,9 s |
| Qwen3.5-4B | 0,854 | 0,230 | 73,5% | 13,9k | 4,7 s |

Teste pareado de McNemar no LCB quick:
- **L2 > L1** é significativo (p = 0,002).
- **L4 ≈ L2** (p = 0,68), com cerca de 2× a latência.
- **No HumanEval+** as diferenças entre loops são ruído (benchmark saturado, ~1–5 problemas).

**Decisão: a base da F1 é o Dory com 2 loops (L2).**

## 3. eval_gate do Dory L2 (referência do critério de corte)

| Benchmark | n | pass@1 | Truncado | CoT média | Latência |
|---|---|---|---|---|---|
| HumanEval+ | 164 | 0,902 | 0% | 5,3k | 5,1 s |
| MBPP+ | 378 | 0,786 | 1,3% | 4,2k | 3,7 s |
| LCB v6 completo | 1055 | 0,685 | 23,6% | 9,2k | 8,1 s |
| SciCode (teste, sem background) | 288 passos / 65 problemas | **0,144** subproblema (oficial) · 1/65 = 1,5% problema principal | 51% | 13,8k | 20,4 s |
| LCB Pro | — | não rodou | — | — | — |

- **SciCode:** o critério de corte das fases ("SciCode não cai mais que 2 pontos") usa a **taxa oficial de subproblema, 0,144**. A taxa por passo do harness dá 0,146, e a diferença está só no denominador do protocolo oficial. O `test_data.h5` é baixado sozinho: primeiro tenta o Drive oficial; se falhar, usa a cópia do HF, que só é aceita se as soluções gold do split de validação passarem em pelo menos 90% dos passos.
- **LCB Pro:** `eval_lcb_pro` não encontrou dataset com casos de teste no Hub. A espec permite pular; fica fora do gate até existir uma fonte com testes.
- **HE+ quick 0,915 × gate 0,902:** a diferença é de 2 problemas, ruído de amostragem entre execuções (temperatura 1,0).
- **Desvio da espec:** o `eval_gate` foi rodado só no L2, não no L1/L4. Com o L2 escolhido como base pelo `eval_quick`, o gate do L1/L4 não muda nenhuma decisão e custaria compute units. Se o artigo precisar, basta rodar `f0_run_baseline.py --target L1 --scope gate` (idem L4).

## 4. Leituras que orientam a F1

- **O orçamento de 16k é o gargalo.** De 24% a 51% das respostas difíceis truncam: LCB completo 24%, LCB quick 40%, SciCode 51%. Boa parte do pass@1 mede "quem termina dentro do orçamento". Comprimir a CoT deve converter truncado em acerto, e é esse o efeito que a F1 precisa mostrar na curva acerto × tokens.
- **O Qwen3.5-4B trunca 73,5%** no LCB quick. Por isso fica tão abaixo do Dory, apesar de ser mais rápido por token.
- **LCB quick (0,535) < LCB completo (0,685):** os 200 problemas mais recentes são os mais difíceis.
- **Contaminação:** em 2026-11-11 não havia no Hub nenhum LCB oficial posterior ao v6 (jun/2025) nem LCB Pro posterior a out/2025. Os dois modelos são de 2026, então não existe split do LCB garantidamente limpo. Um conjunto novo (ex.: AtCoder recente) fica para a F5 ou para o artigo.

## 5. Correções feitas durante o F0

- **Resposta truncada contava como acerto** (commit `91b55a4`). Sem `</think>`, `lib/evals.py` e `lib/rewards.py` corrigiam o código rascunho da CoT. Agora só a resposta pós-`</think>` é avaliada, e truncado vale erro/0 na recompensa, para o GRPO não premiar truncamento. Os runs já feitos foram recorrigidos por `scripts/f0_rescore_truncated.py`, idempotente e sem regerar; os registros recorrigidos levam `config.rescored_truncated_as_fail=true`. O LCB quick caiu 2–3 pontos no Dory e 6,5 no Qwen; os números acima já estão corrigidos.
- **SciCode `test_data.h5` sem passo manual** (commit `2e79ac7`). O gdown do Drive oficial batia na cota; o fallback para a cópia do HF foi descrito acima.
- **Harness:** EvalPlus oficial (base + plus), prompt oficial do LCB, sandbox sem tracing, um subprocesso por motor vLLM, retomada por `eval_cache` (`has_completed_run` pula runs completos sem duplicar linhas).

## 6. Como setar loops

- **vLLM:** `build_dory_hf_overrides(n_loops=k)` em `lib/gen.py` monta o `hf_overrides={"n_recurrent_loops": k, "recurrent_kv_cache_mode": "per_loop"}`. Um motor por valor de `k`.
- **HF (`lib/dory.py`):** `model(..., n_loops=k)` no `forward`; sem o argumento, vale `config.n_recurrent_loops`.
- **Logit lens por loop:** `logit_lens_per_loop` em `lib/dory.py`.
- **Notebook:** `notebooks/F0_setup_baseline.ipynb`, célula 4 (`SCOPE="quick"|"gate"`, `GATE_TARGETS`). Por script: `python scripts/f0_run_baseline.py --target {L1,L2,L4,qwen,plot} --scope {quick,gate}`.
