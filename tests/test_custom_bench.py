"""Confere o benchmark custom: referência de cada família contra força bruta independente, testes que
discriminam, determinismo e o caminho completo pelo sandbox."""

from __future__ import annotations

import itertools
import time
import unittest
from collections import Counter
from functools import lru_cache

from lib.custom_bench import (
    CUSTOM_PROTOCOL_ID,
    FAMILIES,
    INSTANCES_PER_FAMILY,
    build_custom_suite,
    custom_extra_metrics,
    make_problem,
)
from lib.sandbox import check_solution


# ---------------------------------------------------------------- forças brutas (só entradas pequenas)


def _dd(p, lo, hi):
    def ok(x):
        t = str(x)
        return (sum(map(int, t)) % p["m"] == 0 and x % p["q"] == 0
                and all(abs(int(a) - int(b)) >= p["D"] for a, b in zip(t, t[1:])))

    return sum(ok(x) for x in range(lo, hi + 1))


def _fp(p, n):
    alpha = "abcdefghijklmnopqrstuvwxyz"[: p["a"]]
    return sum(not any(x in "".join(t) for x in p["patterns"]) for t in itertools.product(alpha, repeat=n))


def _ts(p, n, edges, weights):
    adj = {v: [] for v in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    dist = {}
    for s0 in range(n):  # BFS de cada nó
        d, frontier = {s0: 0}, [s0]
        for v in frontier:
            for u in adj[v]:
                if u not in d:
                    d[u] = d[v] + 1
                    frontier.append(u)
        dist[s0] = d
    assert all(len(dist[v]) == n for v in range(n))  # é conexa (árvore)
    best = 0
    for mask in range(1 << n):
        chosen = [v for v in range(n) if mask >> v & 1]
        if all(dist[a][b] > p["d"] for a, b in itertools.combinations(chosen, 2)):
            best = max(best, sum(weights[v] for v in chosen))
    return best


def _gc(p, nums):
    def done(t):
        firsts = [t.index(v) for v in set(t)]
        grouped = all(t[i:i + t.count(v)].count(v) == t.count(v) for v, i in zip(set(t), firsts))
        return grouped and (p["first"] is None or p["first"] not in t or t[: t.count(p["first"])] == (p["first"],) * t.count(p["first"]))

    start = tuple(nums)
    dist, frontier = {start: 0}, [start]
    for t in frontier:  # BFS por trocas adjacentes
        if done(t):
            return dist[t]
        for i in range(len(t) - 1):
            u = t[:i] + (t[i + 1], t[i]) + t[i + 2 :]
            if u not in dist:
                dist[u] = dist[t] + 1
                frontier.append(u)
    raise AssertionError("sem estado final")


def _sg(p, piles):
    @lru_cache(None)
    def win(state):
        for i, x in enumerate(state):
            rest = state[:i] + state[i + 1 :]
            nxt = [rest + (x - s,) for s in p["moves"] if s <= x]
            nxt += [rest + (a, x - a) for a in range(1, x) if p["split"] == "any" or 2 * a != x]
            if any(not win(tuple(sorted(y for y in t if y))) for t in nxt):
                return True
        return False

    return win(tuple(sorted(y for y in piles if y)))


def _ds(p, s, k):
    bad = (lambda x, y: x == y) if p["rule"] == "equal" else (lambda x, y: abs(ord(x) - ord(y)) == 1)
    subs = {"".join(c) for c in itertools.combinations(s, k)}
    return sum(all(not bad(a, b) for a, b in zip(t, t[1:])) for t in subs) % (10**9 + 7)


def _seg(p, cost0, cost1):
    R, C = len(cost0), len(cost0[0])
    cells = [(r, c) for r in range(R) for c in range(C)]
    near = [(a, b) for a, b in itertools.combinations(cells, 2)
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 and (p["nbr"] == 8 or a[0] == b[0] or a[1] == b[1])]
    best = None
    for lab in itertools.product((0, 1), repeat=len(cells)):
        L = dict(zip(cells, lab))
        if p["border0"] and any(L[(r, c)] for r, c in cells if r in (0, R - 1) or c in (0, C - 1)):
            continue
        cost = sum((cost1 if L[x] else cost0)[x[0]][x[1]] for x in cells) + p["P"] * sum(L[a] != L[b] for a, b in near)
        best = cost if best is None else min(best, cost)
    return best


def _wm(p, a, w):
    # existe ótimo com b[i] -/+ i*D num dos valores a[j] -/+ j*D; testa toda sequência monótona deles
    n, D = len(a), p["D"]
    sg = 1 if p["dir"] == "inc" else -1
    cands = sorted({a[j] - sg * j * D for j in range(n)})
    best = 0 if n == 0 else None
    for combo in itertools.combinations_with_replacement(cands, n):
        seq = combo if sg == 1 else combo[::-1]
        b = [x + sg * i * D for i, x in enumerate(seq)]
        assert all((b[i + 1] - b[i]) * sg >= D for i in range(n - 1))
        cost = sum(wi * abs(bi - ai) for wi, bi, ai in zip(w, b, a))
        best = cost if best is None else min(best, cost)
    return best


def _cg(p, n, edges, targets, runner, chaser):
    T = set(targets)
    nxt = {u: ([u] if p["stay"] else []) + [v for x, v in edges if x == u] for u in range(n)}
    val = {}  # ponto fixo: repete até nenhum estado mudar
    states = [(r, c, t) for r in range(n) for c in range(n) for t in (0, 1)]
    changed = True
    while changed:
        changed = False
        for r, c, t in states:
            if (r, c, t) in val:
                continue
            if r == c:
                v = "chaser"
            elif r in T:
                v = "runner"
            else:
                kids = [(x, c, 1) for x in nxt[r]] if t == 0 else [(r, x, 0) for x in nxt[c] if x not in T]
                me, other = ("runner", "chaser") if t == 0 else ("chaser", "runner")
                if any(val.get(k) == me for k in kids):
                    v = me
                elif all(val.get(k) == other for k in kids):
                    v = other
                else:
                    continue
            val[(r, c, t)] = v
            changed = True
    return val.get((runner, chaser, 0 if p["first"] == "runner" else 1), "draw")


def _rg(p, n, c):
    def forms(col):
        rots = [col[i:] + col[:i] for i in range(n)]
        return rots + [r[::-1] for r in rots] if p["flip"] else rots

    seen = set()
    for col in itertools.product(range(c), repeat=n):
        if all(col[i] != col[(i + 1) % n] for i in range(n)):
            seen.add(min(forms(col)))
    return len(seen) % (10**9 + 7)


def _cs(p, perm):
    n = len(perm)
    return sum(sorted(perm[l:r + 1]) == list(range(min(perm[l:r + 1]), max(perm[l:r + 1]) + 1))
               for l in range(n) for r in range(l + p["L"] - 1, n))


def _ko(p, intervals):
    best = 0
    for k in range(len(intervals) + 1):
        for combo in itertools.combinations(intervals, k):
            pts = Counter(x for l, r, _ in combo for x in range(l, r + 1 if p["closed"] else r))
            if all(v <= p["k"] for v in pts.values()):
                best = max(best, sum(w for _, _, w in combo))
    return best


BRUTE = {
    "digit_count": (_dd, lambda lo, hi: hi <= 6000),
    "avoid_patterns": (_fp, lambda n: n <= 8),
    "tree_spread": (_ts, lambda n, e, w: n <= 12),
    "group_values": (_gc, lambda nums: len(nums) <= 7),
    "split_game": (_sg, lambda piles: sum(piles) <= 9),
    "distinct_subseq": (_ds, lambda s, k: len(s) <= 12),
    "min_cut_grid": (_seg, lambda c0, c1: len(c0) * len(c0[0]) <= 12),
    "monotone_fit": (_wm, lambda a, w: len(a) <= 7),
    "chase_game": (_cg, lambda n, e, t, r, c: n <= 8),
    "rings": (_rg, lambda n, c: n <= 7 and c <= 4 and c**n <= 5000),
    "consecutive_runs": (_cs, lambda perm: len(perm) <= 60),
    "k_overlap": (_ko, lambda iv: len(iv) <= 12),
}


class TestCustomBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = build_custom_suite("full")

    def test_tamanho_ids_e_niveis(self):
        self.assertEqual(len(self.full), 18 * INSTANCES_PER_FAMILY["full"])
        self.assertEqual(len({p["task_id"] for p in self.full}), len(self.full))
        self.assertEqual({f.tier for f in FAMILIES}, {"easy", "medium", "hard"})
        for tier in ("easy", "medium", "hard"):
            self.assertEqual(sum(f.tier == tier for f in FAMILIES), 6)

    def test_determinismo_e_quick_prefixo_do_full(self):
        quick = build_custom_suite("quick")
        self.assertEqual([p["prompt"] for p in quick], [p["prompt"] for p in self.full[: len(quick)]])
        self.assertEqual(make_problem(FAMILIES[7], 3)["test_code"], make_problem(FAMILIES[7], 3)["test_code"])

    def test_toda_familia_medium_hard_tem_forca_bruta(self):
        self.assertEqual({f.name for f in FAMILIES if f.tier != "easy"}, set(BRUTE))

    def test_referencia_bate_com_forca_bruta(self):
        n_checked: dict[str, int] = {}
        for p in self.full:
            if p["family"] not in BRUTE:
                continue
            brute, small = BRUTE[p["family"]]
            for args, expected in p["cases"]:
                if small(*args):
                    self.assertEqual(brute(p["params"], *args), expected, (p["task_id"], args))
                    n_checked[p["family"]] = n_checked.get(p["family"], 0) + 1
        for fam in BRUTE:
            self.assertGreaterEqual(n_checked.get(fam, 0), 100, fam)

    def test_testes_discriminam(self):
        # Nenhuma instância tem todas as saídas iguais (uma constante passaria em tudo).
        for p in self.full:
            outs = {repr(e) for _, e in p["cases"]}
            self.assertGreater(len(outs), 1, p["task_id"])

    def test_test_code_aceita_referencia_e_rejeita_errada(self):
        for p in self.full:
            fam = next(f for f in FAMILIES if f.name == p["family"])
            ns = {fam.fn: lambda *a, _f=fam, _p=p["params"]: _f.solve(_p, *a)}
            t0 = time.perf_counter()
            exec(p["test_code"], ns)  # não deve levantar
            # a referência roda com folga no limite do sandbox (10 s), então uma solução polinomial correta cabe
            self.assertLess(time.perf_counter() - t0, 3.0, p["task_id"])
            ns_bad = {fam.fn: lambda *a: None}
            with self.assertRaises(AssertionError):
                exec(p["test_code"], ns_bad)

    def test_sandbox_ponta_a_ponta(self):
        p = next(x for x in self.full if x["family"] == "pair_count")
        k = p["params"]["k"]
        good = (
            "```python\ndef count_pairs(nums):\n    n = len(nums)\n"
            f"    return sum((nums[i] + nums[j]) % {k} == 0 for i in range(n) for j in range(i + 1, n))\n```"
        )
        bad = "```python\ndef count_pairs(nums):\n    return 0\n```"
        self.assertTrue(check_solution(good, test_code=p["test_code"], entry_point="count_pairs", **p["sandbox_kwargs"]).passed)
        res = check_solution(bad, test_code=p["test_code"], entry_point="count_pairs", **p["sandbox_kwargs"])
        self.assertFalse(res.passed)
        self.assertEqual(res.error_type, "AssertionError")

    def test_eval_custom_ponta_a_ponta_com_gerador_falso(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from lib.evals import eval_custom
        from lib.gen import Output

        one_per_family = self.full[:18]
        k = next(p for p in one_per_family if p["family"] == "pair_count")["params"]["k"]
        good = (
            "```python\ndef count_pairs(nums):\n    n = len(nums)\n"
            f"    return sum((nums[i] + nums[j]) % {k} == 0 for i in range(n) for j in range(i + 1, n))\n```"
        )

        def fake_generate(model, prompts, **kwargs):  # noqa: ANN001, ANN003
            outs = []
            for pr in prompts:
                if "def count_pairs" in pr:
                    outs.append(Output(text=f"x</think>{good}", cot_text="x", answer_text=good, n_cot_tokens=50,
                                       n_total_tokens=80, latency_s=1.0, backend="hf"))
                else:  # CoT truncada: sem </think>, resposta vazia
                    outs.append(Output(text="pensando...", cot_text="pensando...", answer_text="", n_cot_tokens=64,
                                       n_total_tokens=64, latency_s=1.0, backend="hf"))
            return outs

        with tempfile.TemporaryDirectory() as tmp:
            kw = dict(run_id="r", subset="quick", max_new_tokens=64, backend="hf", cache_dir=Path(tmp) / "c",
                      results_path=Path(tmp) / "results.jsonl")
            with mock.patch("lib.evals.generate", fake_generate), \
                    mock.patch("lib.custom_bench.build_custom_suite", lambda subset: one_per_family):
                res = eval_custom(None, None, **kw)
                again = eval_custom(None, None, **kw)  # já registrado: pula sem duplicar
            n_lines = len((Path(tmp) / "results.jsonl").read_text(encoding="utf-8").splitlines())
        self.assertEqual(res["bench"], "custom_quick")
        self.assertAlmostEqual(res["pass1"], 1 / 18, places=5)
        cfg = res["config"]
        self.assertEqual(cfg["eval_protocol"], CUSTOM_PROTOCOL_ID)
        self.assertAlmostEqual(cfg["easy_pass1"], round(1 / 6, 4))
        self.assertEqual(cfg["hard_pass1"], 0.0)
        self.assertEqual(cfg["fail_breakdown"], {"truncated": 17})
        self.assertEqual(cfg["family_pass1"]["pair_count"], 1.0)
        self.assertEqual(again["timestamp"], res["timestamp"])
        self.assertEqual(n_lines, 1)

    def test_metricas_extras(self):
        rows = {
            "easy/digit_sum/0": {"passed": True, "n_cot_tokens": 100, "n_total_tokens": 150, "latency_s": 1.0},
            "hard/pile_game/0": {"passed": False, "n_cot_tokens": 16000, "n_total_tokens": 16000, "latency_s": 3.0, "truncated": True, "error_type": "EmptyCode"},
            "hard/run_count/0": {"passed": False, "n_cot_tokens": 900, "n_total_tokens": 1000, "latency_s": 1.0, "error_type": "AssertionError"},
            "medium/spiral/0": None,
        }
        m = custom_extra_metrics(rows)
        self.assertEqual(m["easy_pass1"], 1.0)
        self.assertEqual(m["hard_pass1"], 0.0)
        self.assertIsNone(m["medium_pass1"])
        self.assertEqual(m["hard_truncated_frac"], 0.5)
        self.assertEqual(m["tokens_cot_mean_correct"], 100.0)
        self.assertEqual(m["throughput_tok_s"], round(17150 / 5.0, 1))
        self.assertEqual(m["fail_breakdown"], {"truncated": 1, "wrong_answer": 1})
        self.assertEqual(m["hard_fail_breakdown"], {"truncated": 1, "wrong_answer": 1})
        self.assertEqual(m["easy_fail_breakdown"], {})
        # corte por orçamento: passou e coube (150 <= 4000; exatamente no limite também conta)
        self.assertEqual(m["easy_pass1_at_budget"], {"4000": 1.0, "8000": 1.0, "16000": 1.0, "32000": 1.0})
        self.assertEqual(custom_extra_metrics(rows, max_new_tokens=16000)["pass1_at_budget"], {"4000": round(1 / 3, 4), "8000": round(1 / 3, 4)})
        edge = {"easy/a/0": {"passed": True, "n_cot_tokens": 10, "n_total_tokens": 4000}}
        self.assertEqual(custom_extra_metrics(edge)["pass1_at_budget"]["4000"], 1.0)
        self.assertEqual(m["hard_tokens_cot_median"], 8450.0)
        # truncada repetindo vira "loop"; truncada ainda raciocinando continua "truncated"
        from lib.metrics import repetition_score
        looping = " ".join(["vou conferir de novo o caso base da recursão agora"] * 60)
        self.assertGreater(repetition_score(looping), 0.9)
        self.assertEqual(repetition_score("curto demais"), 0.0)
        rows["hard/pile_game/0"]["loop_score"] = repetition_score(looping)
        self.assertEqual(custom_extra_metrics(rows)["hard_fail_breakdown"], {"loop": 1, "wrong_answer": 1})


if __name__ == "__main__":
    unittest.main()
