"""Benchmark custom de código, gerado por semente (sem dataset externo).

18 famílias de problemas (6 por nível: easy, medium, hard). Cada instância sorteia parâmetros que mudam
a semântica do problema (não só nomes), e os testes vêm da solução de referência da família. Toda
família de medium/hard tem uma força bruta independente em `tests/test_custom_bench.py` que confere a
referência. Determinístico: mesma semente => mesmos problemas; `quick` é um prefixo de `full`.

Níveis (desde a custom_v4): easy = Dory acerta tudo (mede pensar demais no simples); medium = técnica
conhecida mas longa, o acerto depende do orçamento (mede economia de tokens); hard = é preciso achar o
modelo certo do problema e o Dory acerta pouco, para separar ganho de raciocínio de ganho só de tokens.

Mede, por problema: acerto (testes escondidos), tokens de CoT/total, truncamento e latência; o
`custom_extra_metrics` agrega por nível, por família e por tipo de falha.
"""

from __future__ import annotations

import bisect
import copy
import functools
import heapq
import itertools
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from .metrics import LOOP_SCORE_THRESHOLD

# Mude ao alterar qualquer família ou o protocolo de geração: invalida resultados antigos.
# v3 = mesmas instâncias da v2, com orçamento de CUSTOM_MAX_NEW_TOKENS e amostragem oficial de cada modelo.
# v4 = hard da v3 vira medium, 6 hard novas, o medium da v3 sai.
CUSTOM_PROTOCOL_ID = "custom_v4"
# O easy e o medium da v4 são as mesmas instâncias da v3 (a semente não depende do nível), geradas com o
# mesmo orçamento e amostragem: o cache da v3 vale para eles, e só o hard novo precisa gerar.
REUSE_CACHE_FROM = "custom_v3"


def reused_task_id(task_id: str) -> str | None:
    """task_id da v3 com a mesma geração que este task_id da v4 (None = problema novo)."""
    tier, rest = task_id.split("/", 1)
    return {"easy": task_id, "medium": f"hard/{rest}"}.get(tier)
CUSTOM_SEED = "coder-comprimido-custom-v1"
# Orçamento grande para o truncamento não decidir o acerto: 81.920 é o que a Qwen recomenda para
# programação de competição (o Dory aceita até 256K). Os cortes menores saem da mesma rodada (BUDGETS).
CUSTOM_MAX_NEW_TOKENS = 81920
# pass@1 "se o limite fosse B": a resposta passou e coube em B tokens gerados. O mesmo começo de resposta
# sairia com limite B (o limite não muda a amostragem), a menos de ruído numérico do lote.
BUDGETS = (4000, 8000, 16000, 32000)
INSTANCES_PER_FAMILY = {"quick": 7, "full": 28}  # 18 famílias => 126 / 504 problemas
SANDBOX_KWARGS = {"timeout_s": 10.0, "max_memory_mb": 1024}
TIERS = ("easy", "medium", "hard")

_LOWER = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Family:
    name: str
    tier: str
    signature: str
    make_params: Callable[[random.Random], dict[str, Any]]
    describe: Callable[[dict[str, Any]], str]
    make_inputs: Callable[[random.Random, dict[str, Any]], list[tuple]]  # os 2 primeiros viram exemplos
    solve: Callable[..., Any]  # solve(params, *args)

    @property
    def fn(self) -> str:
        return self.signature.split("def ", 1)[1].split("(", 1)[0]


def _word(rng: random.Random, n: int, alphabet: str = _LOWER) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


# ----------------------------------------------------------------------------------------- easy


def _digit_params(rng):
    if rng.random() < 0.5:
        return {"mode": "gt", "d": rng.randint(2, 7)}
    return {"mode": "div", "k": rng.randint(2, 5)}


def _digit_desc(p):
    cond = f"strictly greater than {p['d']}" if p["mode"] == "gt" else f"divisible by {p['k']}"
    return (
        f"Return the sum of the decimal digits of the non-negative integer `n` that are {cond}.\n\n"
        "Constraints: 0 <= n <= 10**12."
    )


def _digit_inputs(rng, p):
    out = [(rng.randint(100, 99999),), (rng.randint(10**5, 10**9),), (0,), (10**12,), (9_999_999_999,)]
    return out + [(rng.randint(0, 10 ** rng.randint(1, 12)),) for _ in range(9)]


def _digit_solve(p, n):
    ds = [int(c) for c in str(n)]
    return sum(x for x in ds if (x > p["d"] if p["mode"] == "gt" else x % p["k"] == 0))


def _kth_params(rng):
    return {"k": rng.randint(2, 5), "op": rng.choice(["upper", "remove", "double"])}


def _kth_desc(p):
    act = {"upper": "is converted to uppercase", "remove": "is removed", "double": "is written twice"}[p["op"]]
    return (
        "`s` consists of lowercase English letters. Return a new string in which every character whose "
        f"1-based position in `s` is a multiple of {p['k']} {act}; all other characters stay unchanged "
        "and in order.\n\nConstraints: 0 <= len(s) <= 40."
    )


def _kth_inputs(rng, p):
    out = [(_word(rng, rng.randint(5, 10)),), (_word(rng, rng.randint(10, 15)),), ("",)]
    out += [(_word(rng, p["k"] - 1),), (_word(rng, p["k"]),)]
    return out + [(_word(rng, rng.randint(0, 40)),) for _ in range(9)]


def _kth_solve(p, s):
    parts = []
    for i, ch in enumerate(s, 1):
        if i % p["k"]:
            parts.append(ch)
        elif p["op"] == "upper":
            parts.append(ch.upper())
        elif p["op"] == "double":
            parts.append(ch * 2)
    return "".join(parts)


def _modsum_params(rng):
    a = rng.randint(2, 7)
    return {"a": a, "b": rng.randint(0, a - 1), "c": rng.randint(2, 5)}


def _modsum_desc(p):
    return (
        f"Return the sum of {p['c']} * x over all elements x of `nums` with x % {p['a']} == {p['b']} "
        f"(Python's % operator, so the remainder is always between 0 and {p['a'] - 1}, also for negative x). "
        "Return 0 if no element qualifies.\n\nConstraints: 0 <= len(nums) <= 30, -100 <= nums[i] <= 100."
    )


def _modsum_inputs(rng, p):
    rnd = lambda n: [rng.randint(-100, 100) for _ in range(n)]  # noqa: E731
    out = [(rnd(rng.randint(5, 8)),), (rnd(rng.randint(8, 12)),), ([],), ([p["b"]],), ([-p["a"] + p["b"]],)]
    return out + [(rnd(rng.randint(0, 30)),) for _ in range(9)]


def _modsum_solve(p, nums):
    return sum(p["c"] * x for x in nums if x % p["a"] == p["b"])


def _runs_params(rng):
    return {"m": rng.randint(2, 4), "alphabet": rng.choice(["ab", "abc", "xyz"])}


def _runs_desc(p):
    return (
        f"`s` consists of the characters {', '.join(repr(c) for c in p['alphabet'])}. A run is a maximal "
        "block of equal consecutive characters. Return a list with one two-element list [character, length] "
        f"for each run of length at least {p['m']}, in order of appearance (an empty list if there is none)."
        "\n\nConstraints: 0 <= len(s) <= 40."
    )


def _runs_str(rng, p, n_runs):
    return "".join(rng.choice(p["alphabet"]) * rng.randint(1, 5) for _ in range(n_runs))[:40]


def _runs_inputs(rng, p):
    out = [(_runs_str(rng, p, 5),), (_runs_str(rng, p, 8),), ("",), (p["alphabet"][0] * p["m"],)]
    out += [(p["alphabet"][0] * (p["m"] - 1),)]
    return out + [(_runs_str(rng, p, rng.randint(1, 12)),) for _ in range(9)]


def _runs_solve(p, s):
    return [[ch, n] for ch, g in itertools.groupby(s) if (n := len(list(g))) >= p["m"]]


def _pairs_params(rng):
    return {"k": rng.randint(2, 9)}


def _pairs_desc(p):
    return (
        "Return the number of index pairs (i, j) with i < j such that "
        f"(nums[i] + nums[j]) % {p['k']} == 0.\n\nConstraints: 0 <= len(nums) <= 60, 0 <= nums[i] <= 1000."
    )


def _pairs_inputs(rng, p):
    rnd = lambda n: [rng.randint(0, 1000) for _ in range(n)]  # noqa: E731
    out = [(rnd(rng.randint(4, 6)),), (rnd(rng.randint(6, 9)),), ([],), ([p["k"]],), ([p["k"]] * 4,)]
    return out + [(rnd(rng.randint(0, 60)),) for _ in range(9)]


def _pairs_solve(p, nums):
    n = len(nums)
    return sum((nums[i] + nums[j]) % p["k"] == 0 for i in range(n) for j in range(i + 1, n))


_SORT_KEYS = {
    "vowels": ("the number of vowels (a, e, i, o, u) in the word", lambda w: sum(c in "aeiou" for c in w)),
    "distinct": ("the number of distinct letters in the word", lambda w: len(set(w))),
    "length": ("the length of the word", len),
}


def _sort_params(rng):
    return {"key": rng.choice(sorted(_SORT_KEYS)), "desc": rng.random() < 0.5}


def _sort_desc(p):
    order = "descending" if p["desc"] else "ascending"
    return (
        f"Return a new list with the words of `words` sorted by {_SORT_KEYS[p['key']][0]}, in {order} "
        "order. Words with the same value are ordered alphabetically (ascending). Duplicate words are kept."
        "\n\nConstraints: 0 <= len(words) <= 25; each word has 1 to 8 lowercase English letters."
    )


def _sort_inputs(rng, p):
    def words(n):
        pool = [_word(rng, rng.randint(1, 8), "aeioubcdlmnrst") for _ in range(max(1, n))]
        return [rng.choice(pool) for _ in range(n)]

    out = [(words(rng.randint(4, 6)),), (words(rng.randint(6, 8)),), ([],), (["b", "a"],)]
    return out + [(words(rng.randint(0, 25)),) for _ in range(10)]


def _sort_solve(p, words):
    key = _SORT_KEYS[p["key"]][1]
    return sorted(words, key=lambda w: (-key(w) if p["desc"] else key(w), w))


# --------------------------------------------------------------------------------------- medium


_MOD = 10**9 + 7

# Nível medium (era o hard da v2/v3): técnica conhecida mas longa (DP de dígitos, autômato + potência de
# matriz, DP em árvore com distância, DP em bitmask sobre custos de inversão, Sprague-Grundy com divisão,
# autômato de subsequências) + entradas grandes que reprovam força bruta. Na v3 o L2 acertou 83%, com
# acerto que sobe muito de 16k para 32k tokens: mede economia de tokens.


def _dd_params(rng):
    return {"m": rng.randint(2, 9), "q": rng.randint(2, 13), "D": rng.randint(1, 3)}


def _dd_desc(p):
    return (
        "Count the integers x with lo <= x <= hi such that all three conditions hold: the sum of the "
        f"decimal digits of x is divisible by {p['m']}; x is divisible by {p['q']}; and every two adjacent "
        f"digits of x differ by at least {p['D']} (x written in decimal without leading zeros; a one-digit "
        "number has no adjacent digits, and 0 is the one-digit number 0).\n\n"
        "Constraints: 0 <= lo <= hi <= 10**18."
    )


def _dd_inputs(rng, p):
    def rng_pair(lo_max, width):
        lo = rng.randint(0, lo_max)
        return (lo, lo + rng.randint(0, width))

    out = [rng_pair(100, 400), rng_pair(1000, 2000), (0, 0), (7, 7), (0, 3000)]
    out += [rng_pair(3000, 3000) for _ in range(7)]
    big = rng.randint(10**17, 10**18 // 2)
    return out + [(0, 10**18), (big, rng.randint(big, 10**18))]


def _dd_solve(p, lo, hi):
    m, q, D = p["m"], p["q"], p["D"]

    @functools.lru_cache(None)
    def free(r, prev, sm, md):  # r dígitos ainda livres depois do dígito `prev`
        if r == 0:
            return int(sm == 0 and md == 0)
        return sum(free(r - 1, d, (sm + d) % m, (md * 10 + d) % q) for d in range(10) if abs(d - prev) >= D)

    def upto(n):  # quantos x em [0, n]
        if n < 0:
            return 0
        if n == 0:
            return 1
        s = str(n)
        total = 1 + sum(free(length - 1, d, d % m, d % q) for length in range(1, len(s)) for d in range(1, 10))
        prev = sm = md = 0
        for i, ch in enumerate(s):
            top = int(ch)
            for d in range(1 if i == 0 else 0, top):
                if i == 0 or abs(d - prev) >= D:
                    total += free(len(s) - i - 1, d, (sm + d) % m, (md * 10 + d) % q)
            if i and abs(top - prev) < D:
                return total
            prev, sm, md = top, (sm + top) % m, (md * 10 + top) % q
        return total + int(sm == 0 and md == 0)

    return upto(hi) - upto(lo - 1)


def _fp_params(rng):
    a = rng.randint(2, 3)
    pats: set[str] = set()
    target = rng.randint(2, 3)
    while len(pats) < target:
        pats.add(_word(rng, rng.randint(2, 4), _LOWER[:a]))
    return {"a": a, "patterns": sorted(pats)}


def _fp_desc(p):
    letters = ", ".join(repr(c) for c in _LOWER[: p["a"]])
    return (
        f"Count the strings of length `n` over the alphabet {{{letters}}} that do not contain any of the "
        f"strings {p['patterns']} as a contiguous substring. Return the count modulo 1000000007. For n = 0 "
        "the answer is 1 (the empty string).\n\nConstraints: 0 <= n <= 10**18."
    )


def _fp_inputs(rng, p):
    out = [(rng.randint(3, 4),), (rng.randint(5, 6),), (0,), (1,), (2,)]
    out += [(rng.randint(0, 8),) for _ in range(6)]
    return out + [(rng.randint(1000, 5000),), (rng.randint(10**17, 10**18),), (10**18,)]


def _fp_solve(p, n):
    alpha, pats = _LOWER[: p["a"]], p["patterns"]
    prefixes = sorted({x[:i] for x in pats for i in range(len(x))})  # estados vivos (nenhum padrão completo)
    idx = {s: i for i, s in enumerate(prefixes)}
    size = len(prefixes)
    mat = [[0] * size for _ in range(size)]
    for s in prefixes:
        for c in alpha:
            w = s + c
            if any(w.endswith(x) for x in pats):
                continue
            nxt = next(w[k:] for k in range(len(w) + 1) if w[k:] in idx)
            mat[idx[s]][idx[nxt]] += 1

    def mul(x, y):
        return [[sum(x[i][k] * y[k][j] for k in range(size)) % _MOD for j in range(size)] for i in range(len(x))]

    vec = [[int(i == idx[""]) for i in range(size)]]
    while n:
        if n & 1:
            vec = mul(vec, mat)
        mat = mul(mat, mat)
        n >>= 1
    return sum(vec[0]) % _MOD


def _ts_params(rng):
    return {"d": rng.randint(2, 4)}


def _ts_desc(p):
    return (
        "A tree has n nodes numbered 0 to n-1 and the n-1 undirected edges in `edges`; weights[i] is the "
        "weight of node i. Choose a set of nodes such that the distance (number of edges on the path) "
        f"between any two chosen nodes is greater than {p['d']}. Return the maximum possible total weight "
        "of the chosen nodes.\n\nConstraints: 1 <= n <= 2000, 1 <= weights[i] <= 1000."
    )


def _ts_tree(rng, n, span):
    """Árvore aleatória com rótulos e arestas embaralhados (span pequeno => árvore funda)."""
    perm = list(range(n))
    rng.shuffle(perm)
    edges = []
    for i in range(1, n):
        a, b = perm[i], perm[rng.randint(max(0, i - span), i - 1)]
        edges.append([a, b] if rng.random() < 0.5 else [b, a])
    rng.shuffle(edges)
    return (n, edges, [rng.randint(1, 1000 if n > 12 else 20) for _ in range(n)])


def _ts_inputs(rng, p):
    out = [_ts_tree(rng, rng.randint(6, 8), 3), _ts_tree(rng, rng.randint(8, 10), 10), (1, [], [5])]
    out += [(6, [[i, i + 1] for i in range(5)], [rng.randint(1, 20) for _ in range(6)])]
    out += [(7, [[0, i] for i in range(1, 7)], [rng.randint(1, 20) for _ in range(7)])]
    out += [_ts_tree(rng, rng.randint(2, 12), rng.choice([1, 2, 12])) for _ in range(7)]
    return out + [_ts_tree(rng, 2000, 2000), _ts_tree(rng, 1500, 3)]


def _ts_solve(p, n, edges, weights):
    K = p["d"] + 1  # distância mínima permitida entre escolhidos
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    order, parent = [0], [-1] * n
    for v in order:
        for u in adj[v]:
            if u != parent[v]:
                parent[u] = v
                order.append(u)
    F: list[list[int]] = [[]] * n  # F[v][i]: melhor peso na subárvore com escolhidos a profundidade >= i
    for v in reversed(order):
        ch = [F[u] for u in adj[v] if u != parent[v]]
        f = [0] * (K + 1)
        f[K] = sum(c[K - 1] for c in ch)
        for i in range(K - 1, 0, -1):
            f[i] = f[i + 1]
            if ch:
                mi = max(i - 1, K - i - 1)
                f[i] = max(f[i], sum(c[mi] for c in ch) + max(c[i - 1] - c[mi] for c in ch))
        f[0] = max(weights[v] + f[K], f[1])
        F[v] = f
    return F[0][0]


def _gc_params(rng):
    c = rng.randint(3, 7)
    return {"c": c, "first": rng.choice([None, rng.randint(0, c - 1)])}


def _gc_desc(p):
    extra = (
        f", and the occurrences of the value {p['first']} (if any) come before every other element"
        if p["first"] is not None
        else ""
    )
    return (
        f"`nums` contains integers from 0 to {p['c'] - 1}. In one operation you may swap two adjacent "
        "elements. Return the minimum number of operations needed so that, for every value, all of its "
        f"occurrences form one contiguous block{extra}.\n\nConstraints: 0 <= len(nums) <= 10**5."
    )


def _gc_inputs(rng, p):
    rnd = lambda n: [rng.randrange(p["c"]) for _ in range(n)]  # noqa: E731
    out = [(rnd(6),), (rnd(7),), ([],), ([0],), ([1, 1, 1],)]
    out += [(rnd(rng.randint(1, 7)),) for _ in range(8)]
    blocks = [x for _ in range(400) for x in [rng.randrange(p["c"])] * rng.randint(1, 500)][: 10**5]
    return out + [(rnd(10**5),), (blocks,)]


def _gc_solve(p, nums):
    c = p["c"]
    cost = [[0] * c for _ in range(c)]  # cost[a][b]: pares (b antes de a) a desfazer se o bloco a vier antes de b
    seen = [0] * c
    for x in nums:
        row = cost[x]
        for b in range(c):
            row[b] += seen[b]
        seen[x] += 1
    full = (1 << c) - 1
    inf = float("inf")
    dp = [inf] * (1 << c)
    start = 0 if p["first"] is None else 1 << p["first"]
    dp[start] = 0 if p["first"] is None else sum(cost[p["first"]][b] for b in range(c) if b != p["first"])
    for mask in range(1 << c):
        if dp[mask] == inf:
            continue
        for a in range(c):
            if not mask >> a & 1:
                new = dp[mask] + sum(cost[a][b] for b in range(c) if b != a and not mask >> b & 1)
                if new < dp[mask | 1 << a]:
                    dp[mask | 1 << a] = new
    return dp[full]


def _sg_params(rng):
    return {"moves": sorted(rng.sample(range(1, 6), rng.randint(1, 3))), "split": rng.choice(["unequal", "any"])}


def _sg_desc(p):
    how = "two non-empty piles of different sizes" if p["split"] == "unequal" else "two non-empty piles (of any sizes)"
    return (
        "Two players alternate turns on the piles of stones in `piles`. On a turn the player chooses one pile "
        f"and either removes exactly s stones from it, for some s in {p['moves']} with s <= the size of that "
        f"pile, or splits it into {how}. The player who cannot make a move loses. Both play optimally. "
        "Return True if the player who moves first wins, otherwise False.\n\n"
        "Constraints: 1 <= len(piles) <= 5, 0 <= piles[i] <= 1000."
    )


def _sg_inputs(rng, p):
    def small():
        piles = [rng.randint(0, 6) for _ in range(rng.randint(1, 3))]
        while sum(piles) > 9:
            piles[piles.index(max(piles))] -= 1
        return (piles,)

    out = [small(), small(), ([0],), ([1],), ([2],), ([3],)]
    out += [small() for _ in range(7)]
    return out + [([rng.randint(0, 1000) for _ in range(5)],), ([rng.randint(500, 1000) for _ in range(4)],)]


def _sg_solve(p, piles):
    top = max(piles, default=0)
    gr = [0] * (top + 1)
    for x in range(1, top + 1):
        seen = {gr[x - s] for s in p["moves"] if s <= x}
        seen |= {gr[a] ^ gr[x - a] for a in range(1, x // 2 + 1) if p["split"] == "any" or 2 * a != x}
        g = 0
        while g in seen:
            g += 1
        gr[x] = g
    acc = 0
    for x in piles:
        acc ^= gr[x]
    return acc != 0


_RULES = {
    "equal": "are equal",
    "consecutive": "are consecutive letters of the alphabet (such as 'b' and 'c', in either order)",
}


def _ds_params(rng):
    rule = rng.choice(sorted(_RULES))  # alfabeto grande o bastante para a contagem não zerar
    return {"rule": rule, "a": rng.randint(3, 5) if rule == "equal" else rng.randint(4, 6)}


def _ds_desc(p):
    return (
        "Count the distinct strings t of length exactly k such that t is a subsequence of s (characters "
        "taken in order, not necessarily contiguous) and no two adjacent characters of t "
        f"{_RULES[p['rule']]}. Each different string counts once, however many ways it can be taken from s. "
        "Return the count modulo 1000000007. For k = 0 the answer is 1 (the empty string).\n\n"
        "Constraints: 1 <= len(s) <= 500, 0 <= k <= len(s), s consists of lowercase English letters."
    )


def _ds_inputs(rng, p):
    letters = _LOWER[: p["a"]]
    w = lambda n: _word(rng, n, letters)  # noqa: E731
    out = [(w(rng.randint(5, 7)), rng.randint(2, 3)), (w(rng.randint(7, 9)), rng.randint(3, 4))]
    out += [(w(4), 0), (letters[0] * 5, 1), (letters[0] * 5, 2), (w(6), 6)]
    out += [(lambda s: (s, rng.randint(0, len(s))))(w(rng.randint(1, 12))) for _ in range(7)]
    return out + [(w(500), rng.randint(60, 150)), (w(500), rng.randint(150, 250))]


def _ds_solve(p, s, k):
    if k == 0:
        return 1
    n = len(s)
    bad = (lambda x, y: x == y) if p["rule"] == "equal" else (lambda x, y: abs(ord(x) - ord(y)) == 1)
    nxt = [dict() for _ in range(n + 1)]  # nxt[i][c]: primeira posição j >= i com s[j] == c
    for i in range(n - 1, -1, -1):
        nxt[i] = dict(nxt[i + 1])
        nxt[i][s[i]] = i
    # f[i]: quantas continuações de comprimento r existem depois de escolher s[i] (casamento guloso)
    f = [1] * n
    for _ in range(k - 1):
        f = [
            sum(f[j] for c, j in nxt[i + 1].items() if not bad(s[i], c)) % _MOD
            for i in range(n)
        ]
    return sum(f[j] for j in nxt[0].values()) % _MOD



# ----------------------------------------------------------------------------------------- hard

# Nível hard (desde a custom_v4): o truque não é uma DP conhecida, é perceber o modelo certo do problema
# (corte mínimo, fluxo de custo mínimo, slope trick, análise retrógrada com empate, Burnside, árvore de
# segmentos sobre max - min). Entradas grandes reprovam força bruta. Alvo: Dory acerta de 10% a 25%.


def _maxflow(graph, s, t):
    """Dinic. graph[u] = lista de [v, cap, índice da aresta reversa em graph[v]]."""
    total = 0
    while True:
        level = [-1] * len(graph)
        level[s] = 0
        queue = [s]
        for u in queue:
            for v, cap, _ in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        if level[t] < 0:
            return total
        it = [0] * len(graph)
        while True:  # caminhos aumentantes no grafo de níveis, sem recursão
            path, u = [], s
            while u != t:
                while it[u] < len(graph[u]):
                    v, cap, _ = graph[u][it[u]]
                    if cap > 0 and level[v] == level[u] + 1:
                        break
                    it[u] += 1
                if it[u] < len(graph[u]):
                    path.append(u)
                    u = graph[u][it[u]][0]
                elif u == s:
                    break
                else:
                    level[u] = -1  # beco sem saída
                    u = path.pop()
                    it[u] += 1
            if u != t:
                break
            f = min(graph[x][it[x]][1] for x in path)
            for x in path:
                e = graph[x][it[x]]
                e[1] -= f
                graph[e[0]][e[2]][1] += f
            total += f


def _seg_params(rng):
    return {"P": rng.randint(2, 15), "nbr": rng.choice([4, 8]), "border0": rng.random() < 0.4}


def _seg_desc(p):
    nb = "share a side" if p["nbr"] == 4 else "share a side or a corner"
    border = (
        " Every cell in the first or last row or in the first or last column must get the label 0."
        if p["border0"]
        else ""
    )
    return (
        "`cost0` and `cost1` are R x C grids of non-negative integers. Give every cell (r, c) the label 0 "
        "or the label 1: label 0 costs cost0[r][c] and label 1 costs cost1[r][c]. In addition, every pair "
        f"of neighbouring cells that get different labels costs {p['P']}. Two cells are neighbours if they "
        f"{nb}.{border} Return the minimum possible total cost.\n\n"
        "Constraints: 1 <= R, C <= 25, 0 <= cost0[r][c], cost1[r][c] <= 100."
    )


def _seg_inputs(rng, p):
    def grid(r, c, hi):
        return ([[rng.randint(0, hi) for _ in range(c)] for _ in range(r)],
                [[rng.randint(0, hi) for _ in range(c)] for _ in range(r)])

    out = [grid(2, 3, 20), grid(3, 3, 20), grid(1, 1, 20), grid(1, 5, 20), grid(4, 1, 20)]
    out += [([[7, 7], [7, 7]], [[7, 7], [7, 7]])]
    out += [grid(*rng.choice([(3, 4), (2, 6), (4, 3), (2, 5), (3, 3)]), 25) for _ in range(6)]
    return out + [grid(25, 25, 100), grid(20, 25, 100)]


def _seg_solve(p, cost0, cost1):
    R, C = len(cost0), len(cost0[0])
    s, t, big = R * C, R * C + 1, 10**9
    graph: list[list[list[int]]] = [[] for _ in range(R * C + 2)]

    def add(u, v, cap_uv, cap_vu=0):
        graph[u].append([v, cap_uv, len(graph[v])])
        graph[v].append([u, cap_vu, len(graph[u]) - 1])

    steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if p["nbr"] == 8 else [])
    for r in range(R):
        for c in range(C):
            v = r * C + c
            forced = p["border0"] and (r in (0, R - 1) or c in (0, C - 1))
            add(s, v, big if forced else cost1[r][c])  # cortada se v fica do lado do sorvedouro (rótulo 1)
            add(v, t, cost0[r][c])  # cortada se v fica do lado da fonte (rótulo 0)
            for dr, dc in steps:
                if 0 <= r + dr < R and 0 <= c + dc < C:
                    add(v, (r + dr) * C + c + dc, p["P"], p["P"])
    return _maxflow(graph, s, t)


def _wm_params(rng):
    return {"D": rng.choice([0, 0, 1, 2, 5]), "dir": rng.choice(["inc", "dec"])}


def _wm_desc(p):
    if p["dir"] == "inc":
        rule = "b is non-decreasing" if p["D"] == 0 else f"b[i+1] >= b[i] + {p['D']} for every i"
    else:
        rule = "b is non-increasing" if p["D"] == 0 else f"b[i+1] <= b[i] - {p['D']} for every i"
    return (
        "Given `a` and `w` of the same length n, choose an integer array b of length n such that "
        f"{rule}. The cost of b is the sum of w[i] * abs(b[i] - a[i]). Return the minimum possible cost.\n\n"
        "Constraints: 0 <= n <= 3 * 10**4, -10**9 <= a[i] <= 10**9, 1 <= w[i] <= 1000."
    )


def _wm_inputs(rng, p):
    def arr(n, hi, wmax):
        return ([rng.randint(-hi, hi) for _ in range(n)], [rng.randint(1, wmax) for _ in range(n)])

    out = [arr(5, 20, 9), arr(6, 20, 9), ([], []), ([5], [3]), ([1, 2, 3, 4], [1, 1, 1, 1])]
    out += [([4, 3, 2, 1], [2, 1, 3, 1])]
    out += [arr(rng.randint(2, 7), rng.choice([5, 30]), rng.choice([1, 10])) for _ in range(6)]
    n = 3 * 10**4  # O(n^2) fica inviável sem estourar o tamanho do test_code (10**5 dava 3 MB, ~350 MB de pico)
    return out + [arr(n, 10**9, 1000), ([10**9 - 50 - 7 * i + rng.randint(0, 50) for i in range(n)],
                                        [rng.randint(1, 1000) for _ in range(n)])]


def _wm_solve(p, a, w):
    if p["dir"] == "dec":  # b[i+1] <= b[i] - D é a condição crescente lida de trás para frente
        a, w = a[::-1], w[::-1]
    heap: list[tuple[int, int]] = []  # slope trick: (-ponto de quebra, peso)
    ans = 0
    for i, (x, wi) in enumerate(zip(a, w)):
        c = x - i * p["D"]  # com b'[i] = b[i] - i*D a condição vira b' não decrescente
        heapq.heappush(heap, (-c, 2 * wi))
        need = wi
        while need:
            nv, cnt = heap[0]
            take = min(cnt, need)
            ans += (-nv - c) * take
            need -= take
            if take == cnt:
                heapq.heappop(heap)
            else:
                heapq.heapreplace(heap, (nv, cnt - take))
    return ans


def _cg_params(rng):
    return {"first": rng.choice(["runner", "chaser"]), "stay": rng.random() < 0.5}


def _cg_desc(p):
    move = (
        "either keep its token where it is or move it along one outgoing edge"
        if p["stay"]
        else "move its token along one outgoing edge"
    )
    return (
        "A directed graph has nodes 0 to n-1 and the directed edges [u, v] (from u to v) in `edges`. A runner "
        "token starts on node `runner` and a chaser token on node `chaser` (different nodes, neither in "
        f"`targets`). The two players alternate turns and the {p['first']} moves first. On its turn a player "
        f"must {move} of the node its token is on. The chaser may never move onto a node in `targets`. A "
        "player with no allowed move on its turn loses. The chaser wins as soon as both tokens are on the "
        "same node. The runner wins as soon as its token is on a node in `targets`. If the game can go on "
        "forever, it is a draw. Both play optimally (a player who cannot win prefers a draw to losing). "
        "Return 'runner', 'chaser' or 'draw'.\n\n"
        "Constraints: 2 <= n <= 60, 0 <= len(edges) <= 200, no self-loops or repeated edges."
    )


def _cg_graph(rng, n, maxdeg):
    edges = [[u, v] for u in range(n) for v in rng.sample([x for x in range(n) if x != u], rng.randint(0, min(maxdeg, n - 1)))]
    rng.shuffle(edges)
    nodes = list(range(n))
    rng.shuffle(nodes)
    targets = sorted(nodes[: rng.randint(1, min(2, n - 2))])
    return (n, edges, targets, nodes[-1], nodes[-2])


def _cg_inputs(rng, p):
    out = [_cg_graph(rng, 5, 2), _cg_graph(rng, 6, 2), (3, [], [2], 0, 1)]
    out += [(3, [[0, 2], [1, 0]], [2], 0, 1), (4, [[0, 1], [1, 0], [2, 3], [3, 2]], [3], 0, 2)]
    out += [_cg_graph(rng, rng.randint(3, 8), rng.choice([1, 2, 3])) for _ in range(7)]
    return out + [_cg_graph(rng, 60, 3), _cg_graph(rng, 60, 2)]


def _cg_solve(p, n, edges, targets, runner, chaser):
    T = set(targets)
    out_r = [[v] if p["stay"] else [] for v in range(n)]
    for u, v in edges:
        out_r[u].append(v)
    out_c = [[v for v in out_r[u] if v not in T] for u in range(n)]
    in_r = [[] for _ in range(n)]  # quem chega em v num lance do runner (idem chaser)
    in_c = [[] for _ in range(n)]
    for u in range(n):
        for v in out_r[u]:
            in_r[v].append(u)
        for v in out_c[u]:
            in_c[v].append(u)
    # estado (r, c, vez): vez 0 = runner joga, 1 = chaser joga; valor = quem vence (análise retrógrada)
    res: dict[tuple[int, int, int], str] = {}
    left: dict[tuple[int, int, int], int] = {}
    queue = []
    for r in range(n):
        for c in range(n):
            for turn in (0, 1):
                st = (r, c, turn)
                if r == c:
                    res[st] = "chaser"
                elif r in T:
                    res[st] = "runner"
                else:
                    left[st] = len(out_r[r]) if turn == 0 else len(out_c[c])
                    if left[st] == 0:
                        res[st] = "chaser" if turn == 0 else "runner"
                    else:
                        continue
                queue.append(st)
    for r, c, turn in queue:
        winner = res[(r, c, turn)]
        preds = [(x, c, 0) for x in in_r[r]] if turn == 1 else [(r, x, 1) for x in in_c[c]]
        for pr in preds:
            if pr in res:
                continue
            mover = "runner" if pr[2] == 0 else "chaser"
            if winner == mover:
                res[pr] = mover
                queue.append(pr)
            else:
                left[pr] -= 1
                if left[pr] == 0:
                    res[pr] = winner
                    queue.append(pr)
    return res.get((runner, chaser, 0 if p["first"] == "runner" else 1), "draw")


def _rg_params(rng):
    return {"flip": rng.random() < 0.5}


def _rg_desc(p):
    same = "rotating the ring or turning it over" if p["flip"] else "rotating the ring (turning it over is not allowed)"
    return (
        "A ring has n beads, bead i next to bead i+1 and bead n-1 next to bead 0. Each bead is painted with "
        "one of c colors so that no two neighbouring beads have the same color. Two paintings are the same "
        f"ring if one can be obtained from the other by {same}. Return the number of different rings "
        "modulo 1000000007.\n\nConstraints: 3 <= n <= 10**9, 1 <= c <= 10**9."
    )


def _rg_inputs(rng, p):
    small = [(n, c) for n in range(3, 8) for c in range(1, 5) if c**n <= 5000]
    out = [(4, 3), (5, 3), (3, 1), (4, 2), (6, 2)]
    out += rng.sample(small, 6)
    return out + [(10**9, 10**9), (720720, rng.randint(2, 10**9)), (999999937, rng.randint(2, 50)),
                  (2**29, 3)]


def _rg_solve(p, n, c):
    fac: dict[int, int] = {}
    m, d = n, 2
    while d * d <= m:
        while m % d == 0:
            fac[d] = fac.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        fac[m] = fac.get(m, 0) + 1
    divs = [(1, 1)]  # (e, phi(e)) para todo divisor e de n
    for q, k in fac.items():
        divs = [(e * q**j, ph * (q**j - q ** (j - 1) if j else 1)) for e, ph in divs for j in range(k + 1)]
    # as phi(e) rotações de ordem e fixam as pinturas boas de um anel de n/e contas
    total = sum(ph * (pow(c - 1, n // e, _MOD) + (-1) ** (n // e) * (c - 1)) for e, ph in divs) % _MOD
    group = n
    if p["flip"]:
        group = 2 * n  # n ímpar: todo eixo cruza uma aresta, que ficaria com a mesma cor dos dois lados
        if n % 2 == 0:  # eixos por duas contas fixam um caminho de n/2 + 1 contas; eixos por arestas, nada
            total += n // 2 * c * pow(c - 1, n // 2, _MOD)
    return total * pow(group, _MOD - 2, _MOD) % _MOD


def _cs_params(rng):
    return {"L": rng.choice([1, 2, 3, 5])}


def _cs_desc(p):
    return (
        "`p` is a permutation of the integers 0 to n-1. Count the contiguous subarrays p[l..r] of length at "
        f"least {p['L']} whose values, once sorted, are consecutive integers (each value exactly one more "
        "than the previous one).\n\nConstraints: 0 <= n <= 2 * 10**4."
    )


def _cs_inputs(rng, p):
    def perm(n):
        xs = list(range(n))
        rng.shuffle(xs)
        return (xs,)

    def blocky(n, size):  # blocos de valores consecutivos embaralhados: muitas respostas grandes
        xs = list(range(n))
        blocks = [xs[i : i + size] for i in range(0, n, size)]
        for b in blocks:
            if rng.random() < 0.5:
                b.reverse()
        rng.shuffle(blocks)
        return ([x for b in blocks for x in b],)

    out = [perm(6), perm(7), ([],), ([0],), ([0, 1, 2, 3, 4],), ([4, 3, 2, 1, 0],)]
    out += [rng.choice([perm, lambda n: blocky(n, rng.randint(2, 3))])(rng.randint(2, 9)) for _ in range(7)]
    # pares trocados: max-min fica sempre perto do comprimento, então O(n^2) com corte antecipado não passa
    return out + [perm(2 * 10**4), blocky(5000, 7), ([i ^ 1 for i in range(2 * 10**4)],)]


def _cs_solve(p, perm):
    n, L = len(perm), p["L"]
    if n < L:
        return 0
    size = 1 << max(1, (n - 1).bit_length())
    # folha l guarda g(l) = max(l..r) - min(l..r) + l; p[l..r] é bom se g(l) == r (sempre g(l) >= r)
    mn = [0] * (2 * size)
    cnt = [0] * (2 * size)
    lz = [0] * (2 * size)
    for i in range(size):
        mn[size + i], cnt[size + i] = (i, 1) if i < n else (10**9, 0)
    for i in range(size - 1, 0, -1):
        a, b = mn[2 * i], mn[2 * i + 1]
        mn[i] = min(a, b)
        cnt[i] = (cnt[2 * i] if a == mn[i] else 0) + (cnt[2 * i + 1] if b == mn[i] else 0)

    def add(lo, hi, v, node=1, nl=0, nr=size):  # soma v em [lo, hi)
        if hi <= nl or nr <= lo:
            return
        if lo <= nl and nr <= hi:
            mn[node] += v
            lz[node] += v
            return
        mid = (nl + nr) // 2
        add(lo, hi, v, 2 * node, nl, mid)
        add(lo, hi, v, 2 * node + 1, mid, nr)
        a, b = mn[2 * node], mn[2 * node + 1]
        m = min(a, b)
        mn[node] = m + lz[node]
        cnt[node] = (cnt[2 * node] if a == m else 0) + (cnt[2 * node + 1] if b == m else 0)

    def query(hi, node=1, nl=0, nr=size):  # (mínimo, quantas vezes) em [0, hi), sem empurrar o lazy
        if hi <= nl:
            return (10**18, 0)
        if nr <= hi:
            return (mn[node], cnt[node])
        mid = (nl + nr) // 2
        a, ca = query(hi, 2 * node, nl, mid)
        b, cb = query(hi, 2 * node + 1, mid, nr)
        m = min(a, b)
        return (m + lz[node], (ca if a == m else 0) + (cb if b == m else 0))

    hi_st: list[int] = []  # índices com máximos decrescentes (idem mínimos crescentes)
    lo_st: list[int] = []
    total = 0
    for r, x in enumerate(perm):
        while hi_st and perm[hi_st[-1]] < x:
            j = hi_st.pop()
            add(hi_st[-1] + 1 if hi_st else 0, j + 1, x - perm[j])
        hi_st.append(r)
        while lo_st and perm[lo_st[-1]] > x:
            j = lo_st.pop()
            add(lo_st[-1] + 1 if lo_st else 0, j + 1, perm[j] - x)
        lo_st.append(r)
        if r - L + 1 >= 0:
            m, k = query(r - L + 2)
            total += k if m == r else 0
    return total


def _ko_params(rng):
    return {"k": rng.randint(2, 4), "closed": rng.random() < 0.5}


def _ko_desc(p):
    if p["closed"]:
        cover = "[l, r, w] covers the integer points l, l+1, ..., r (both ends included)"
        lim = "0 <= l <= r <= 10**9"
    else:
        cover = "[l, r, w] covers the integer points l, l+1, ..., r-1 (r is not included)"
        lim = "0 <= l < r <= 10**9"
    return (
        f"Each interval {cover} and has weight w. Choose a subset of the intervals such that every integer "
        f"point is covered by at most {p['k']} of the chosen intervals. Return the maximum possible total "
        f"weight of the chosen intervals.\n\nConstraints: 0 <= len(intervals) <= 300, {lim}, 1 <= w <= 10**6."
    )


def _ko_inputs(rng, p):
    def ivs(n, hi, maxlen, wmax):
        out = []
        for _ in range(n):
            lo = rng.randint(0, hi - 1)
            r = min(hi, lo + rng.randint(0 if p["closed"] else 1, maxlen))
            out.append([lo, max(r, lo + (0 if p["closed"] else 1)), rng.randint(1, wmax)])
        return (out,)

    k = p["k"]
    out = [ivs(6, 12, 6, 20), ivs(8, 15, 8, 20), ([],), ([[3, 9, 5]],), ([[0, 5, 2]] * (k + 1),)]
    out += [([[i, i + 4, 3] for i in range(0, 8, 2)] + [[0, 12, 10]],)]
    out += [ivs(rng.randint(3, 12), rng.choice([10, 20, 40]), rng.choice([3, 10]), 20) for _ in range(6)]
    return out + [ivs(300, 10**9, 10**8, 10**6), ivs(300, 2000, 300, 1000)]


def _ko_solve(p, intervals):
    if not intervals:
        return 0
    spans = [(l, r + 1 if p["closed"] else r, w) for l, r, w in intervals]  # [l, fim) semiaberto
    xs = sorted({x for l, e, _ in spans for x in (l, e)})
    idx = {x: i for i, x in enumerate(xs)}
    V = len(xs)
    graph: list[list[list[int]]] = [[] for _ in range(V)]  # [v, cap, custo, índice reverso]

    def add(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for i in range(V - 1):
        add(i, i + 1, p["k"], 0)  # a linha do tempo carrega até k intervalos ao mesmo tempo
    for l, e, w in spans:
        add(idx[l], idx[e], 1, -w)
    best, flow = 0, 0
    while flow < p["k"]:  # caminhos mínimos sucessivos (Bellman-Ford: há custos negativos)
        dist = [float("inf")] * V
        prev: list[tuple[int, int] | None] = [None] * V
        dist[0] = 0
        inq = [False] * V
        queue = [0]
        while queue:
            nxt = []
            for u in queue:
                inq[u] = False
                for j, (v, cap, cost, _) in enumerate(graph[u]):
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prev[v] = (u, j)
                        if not inq[v]:
                            inq[v] = True
                            nxt.append(v)
            queue = nxt
        f, v = p["k"] - flow, V - 1
        while v:
            u, j = prev[v]
            f = min(f, graph[u][j][1])
            v = u
        v = V - 1
        while v:
            u, j = prev[v]
            graph[u][j][1] -= f
            graph[v][graph[u][j][3]][1] += f
            v = u
        best -= dist[V - 1] * f
        flow += f
    return best


FAMILIES: list[Family] = [
    Family("digit_sum", "easy", "def digit_sum(n: int) -> int:", _digit_params, _digit_desc, _digit_inputs, _digit_solve),
    Family("kth_char", "easy", "def transform(s: str) -> str:", _kth_params, _kth_desc, _kth_inputs, _kth_solve),
    Family("mod_sum", "easy", "def filtered_sum(nums: list[int]) -> int:", _modsum_params, _modsum_desc, _modsum_inputs, _modsum_solve),
    Family("long_runs", "easy", "def long_runs(s: str) -> list:", _runs_params, _runs_desc, _runs_inputs, _runs_solve),
    Family("pair_count", "easy", "def count_pairs(nums: list[int]) -> int:", _pairs_params, _pairs_desc, _pairs_inputs, _pairs_solve),
    Family("sort_words", "easy", "def sort_words(words: list[str]) -> list[str]:", _sort_params, _sort_desc, _sort_inputs, _sort_solve),
    Family("digit_count", "medium", "def count_numbers(lo: int, hi: int) -> int:", _dd_params, _dd_desc, _dd_inputs, _dd_solve),
    Family("avoid_patterns", "medium", "def count_strings(n: int) -> int:", _fp_params, _fp_desc, _fp_inputs, _fp_solve),
    Family("tree_spread", "medium", "def max_weight(n: int, edges: list[list[int]], weights: list[int]) -> int:", _ts_params, _ts_desc, _ts_inputs, _ts_solve),
    Family("group_values", "medium", "def min_swaps(nums: list[int]) -> int:", _gc_params, _gc_desc, _gc_inputs, _gc_solve),
    Family("split_game", "medium", "def first_player_wins(piles: list[int]) -> bool:", _sg_params, _sg_desc, _sg_inputs, _sg_solve),
    Family("distinct_subseq", "medium", "def count_subsequences(s: str, k: int) -> int:", _ds_params, _ds_desc, _ds_inputs, _ds_solve),
    Family("min_cut_grid", "hard", "def min_total_cost(cost0: list[list[int]], cost1: list[list[int]]) -> int:", _seg_params, _seg_desc, _seg_inputs, _seg_solve),
    Family("monotone_fit", "hard", "def min_cost(a: list[int], w: list[int]) -> int:", _wm_params, _wm_desc, _wm_inputs, _wm_solve),
    Family("chase_game", "hard", "def outcome(n: int, edges: list[list[int]], targets: list[int], runner: int, chaser: int) -> str:", _cg_params, _cg_desc, _cg_inputs, _cg_solve),
    Family("rings", "hard", "def count_rings(n: int, c: int) -> int:", _rg_params, _rg_desc, _rg_inputs, _rg_solve),
    Family("consecutive_runs", "hard", "def count_segments(p: list[int]) -> int:", _cs_params, _cs_desc, _cs_inputs, _cs_solve),
    Family("k_overlap", "hard", "def max_weight(intervals: list[list[int]]) -> int:", _ko_params, _ko_desc, _ko_inputs, _ko_solve),
]


def _call_repr(fn: str, args: tuple) -> str:
    return f"{fn}({', '.join(repr(a) for a in args)})"


def make_problem(family: Family, idx: int) -> dict[str, Any]:
    """Uma instância determinística (família, índice) no formato do `_run_problems_loop`."""
    rng = random.Random(f"{CUSTOM_SEED}:{family.name}:{idx}")
    params = family.make_params(rng)
    cases = [(args, family.solve(params, *copy.deepcopy(args))) for args in family.make_inputs(rng, params)]
    examples = "\n".join(f"{_call_repr(family.fn, a)} == {e!r}" for a, e in cases[:2])
    statement = f"{family.describe(params)}\n\nExamples:\n{examples}"
    prompt = (
        "Solve the following programming problem in Python 3. Implement the function with exactly the given "
        "signature and return the full implementation inside a ```python ... ``` code block.\n\n"
        f"{statement}\n\nStarter code:\n```python\n{family.signature}\n```"
    )
    test_code = (
        "import copy\n"
        f"_CASES = {cases!r}\n"
        "def check(candidate):\n"
        "    for args, expected in _CASES:\n"
        "        got = candidate(*copy.deepcopy(args))\n"
        "        assert got == expected, (repr(args)[:200], repr(got)[:200], repr(expected)[:200])\n"
        f"check({family.fn})\n"
    )
    return {
        "task_id": f"{family.tier}/{family.name}/{idx}",
        "prompt": prompt,
        "test_code": test_code,
        "entry_point": family.fn,
        "sandbox_kwargs": dict(SANDBOX_KWARGS),
        "tier": family.tier,
        "family": family.name,
        "params": params,
        "cases": cases,
    }


def build_custom_suite(subset: str = "quick") -> list[dict[str, Any]]:
    """`quick` (126) ou `full` (504). Ordem: índice, depois família, então qualquer prefixo cobre todas."""
    if subset not in INSTANCES_PER_FAMILY:
        raise ValueError(f"subset deve ser um de {sorted(INSTANCES_PER_FAMILY)}, recebido {subset!r}")
    return [make_problem(f, i) for i in range(INSTANCES_PER_FAMILY[subset]) for f in FAMILIES]


def _fail_bucket(row: dict[str, Any]) -> str:
    if row.get("truncated"):
        # "loop" = estourou o orçamento repetindo o mesmo bloco; "truncated" = ainda estava raciocinando
        return "loop" if float(row.get("loop_score", 0.0)) >= LOOP_SCORE_THRESHOLD else "truncated"
    return {"EmptyCode": "no_code", "AssertionError": "wrong_answer", "TimeoutError": "timeout"}.get(
        str(row.get("error_type")), "runtime_error"
    )


def _median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    mid = len(xs) // 2
    return round(xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2, 1)


def custom_extra_metrics(
    rows_by_id: dict[str, dict[str, Any] | None], max_new_tokens: int | None = None
) -> dict[str, Any]:
    """Métricas por nível/família/falha para o `config` do registro em `results.jsonl`.
    `max_new_tokens` (orçamento da rodada) limita os cortes de `BUDGETS` aos menores que ele."""
    rows = {tid: r for tid, r in rows_by_id.items() if r is not None}
    out: dict[str, Any] = {"custom_protocol": CUSTOM_PROTOCOL_ID}
    budgets = [b for b in BUDGETS if max_new_tokens is None or b < max_new_tokens]

    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else None

    def pass_at(rs, b):  # passou e coube em b tokens gerados (CoT + resposta)
        return mean(float(bool(r.get("passed")) and float(r.get("n_total_tokens", 0)) <= b) for r in rs)

    out["pass1_at_budget"] = {str(b): pass_at(rows.values(), b) for b in budgets}
    for tier in TIERS:
        rs = [r for tid, r in rows.items() if tid.split("/")[0] == tier]
        out[f"{tier}_pass1"] = mean(float(bool(r.get("passed"))) for r in rs)
        out[f"{tier}_pass1_at_budget"] = {str(b): pass_at(rs, b) for b in budgets}
        out[f"{tier}_tokens_cot_mean"] = mean(float(r.get("n_cot_tokens", 0)) for r in rs)
        # mediana: poucas respostas em loop até o orçamento puxam a média para cima
        out[f"{tier}_tokens_cot_median"] = _median([float(r.get("n_cot_tokens", 0)) for r in rs])
        out[f"{tier}_truncated_frac"] = mean(float(bool(r.get("truncated"))) for r in rs)
        # separa errou o raciocínio (wrong_answer/timeout), não coube (truncated) e travou repetindo (loop)
        out[f"{tier}_fail_breakdown"] = dict(Counter(_fail_bucket(r) for r in rs if not r.get("passed")))
    ok = [r for r in rows.values() if r.get("passed")]
    out["tokens_cot_mean_correct"] = mean(float(r.get("n_cot_tokens", 0)) for r in ok)
    lat = sum(float(r.get("latency_s", 0.0)) for r in rows.values())
    tok = sum(float(r.get("n_total_tokens", 0)) for r in rows.values())
    out["throughput_tok_s"] = round(tok / lat, 1) if lat > 0 else None
    out["fail_breakdown"] = dict(Counter(_fail_bucket(r) for r in rows.values() if not r.get("passed")))
    fam: dict[str, list[float]] = {}
    for tid, r in rows.items():
        fam.setdefault(tid.split("/")[1], []).append(float(bool(r.get("passed"))))
    out["family_pass1"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(fam.items())}
    return out
