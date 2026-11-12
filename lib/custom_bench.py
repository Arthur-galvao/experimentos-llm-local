"""Benchmark custom de código, gerado por semente (sem dataset externo).

18 famílias de problemas (6 por nível: easy, medium, hard). Cada instância sorteia parâmetros que mudam
a semântica do problema (não só nomes), e os testes vêm da solução de referência da família. Toda
família de medium/hard tem uma força bruta independente em `tests/test_custom_bench.py` que confere a
referência. Determinístico: mesma semente => mesmos problemas; `quick` é um prefixo de `full`.

Mede, por problema: acerto (testes escondidos), tokens de CoT/total, truncamento e latência; o
`custom_extra_metrics` agrega por nível, por família e por tipo de falha.
"""

from __future__ import annotations

import bisect
import copy
import heapq
import itertools
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable

CUSTOM_PROTOCOL_ID = "custom_v1"  # mude ao alterar qualquer família: invalida resultados antigos
CUSTOM_SEED = "coder-comprimido-custom-v1"
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


def _kd_params(rng):
    k = rng.randint(1, 3)
    return {"mode": rng.choice(["at most", "exactly"]), "k": k, "alpha": rng.randint(max(k + 1, 3), 6)}


def _kd_desc(p):
    letters = _LOWER[: p["alpha"]]
    return (
        "Return the length of the longest contiguous substring of `s` that contains "
        f"{p['mode']} {p['k']} distinct characters (0 if there is no such non-empty substring).\n\n"
        f"Constraints: 0 <= len(s) <= 2000; `s` uses only the letters '{letters[0]}' to '{letters[-1]}'."
    )


def _kd_inputs(rng, p):
    letters = _LOWER[: p["alpha"]]
    out = [(_word(rng, rng.randint(8, 12), letters),), (_word(rng, rng.randint(12, 16), letters),), ("",)]
    out += [(letters[0],), (letters[0] * 5,)]
    out += [(_word(rng, rng.randint(1, 60), letters),) for _ in range(8)]
    return out + [(_word(rng, 2000, letters[: p["k"] + 1]),), (_word(rng, 2000, letters),)]


def _kd_solve(p, s):
    cnt: dict[str, int] = {}
    left = best = 0
    for right, ch in enumerate(s):
        cnt[ch] = cnt.get(ch, 0) + 1
        while len(cnt) > p["k"]:
            c = s[left]
            cnt[c] -= 1
            if not cnt[c]:
                del cnt[c]
            left += 1
        if p["mode"] == "at most" or len(cnt) == p["k"]:
            best = max(best, right - left + 1)
    return best


def _mg_params(rng):
    return {"g": rng.randint(0, 5)}


def _mg_desc(p):
    return (
        "Each interval [start, end] of `intervals` has start <= end. Repeatedly merge any two intervals "
        f"whose gap is at most {p['g']}, where the gap between [a, b] and [c, d] is max(a, c) - min(b, d) "
        "(zero or negative when they touch or overlap); the two are replaced by [min(a, c), max(b, d)]. "
        "When no more merges are possible, return the sum of (end - start) over the remaining intervals. "
        "The input is not sorted. Return 0 for an empty list.\n\n"
        "Constraints: 0 <= len(intervals) <= 2000, 0 <= start <= end <= 10**6."
    )


def _mg_inputs(rng, p):
    def ivs(n, hi, maxlen):
        out = []
        for _ in range(n):
            s = rng.randint(0, hi)
            out.append([s, min(hi, s + rng.randint(0, maxlen))])
        return out

    out = [(ivs(4, 30, 6),), (ivs(6, 40, 6),), ([],), ([[5, 5]],), ([[0, 3], [3 + p["g"], 9]],)]
    out += [(ivs(rng.randint(1, 12), 60, 8),) for _ in range(8)]
    return out + [(ivs(2000, 10**6, 1000),), (ivs(2000, 10**6, 300),)]


def _mg_solve(p, intervals):
    total, cur = 0, None
    for s, e in sorted(map(tuple, intervals)):
        if cur is not None and s - cur[1] <= p["g"]:
            cur[1] = max(cur[1], e)
        else:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [s, e]
    return total + (cur[1] - cur[0] if cur is not None else 0)


_DIR_NAMES = ["right", "down", "left", "up"]
_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
_SPIRAL_START = {  # (canto, rotação) -> direção inicial, sempre ao longo da borda
    ("top-left", "clockwise"): "right",
    ("top-right", "clockwise"): "down",
    ("bottom-right", "clockwise"): "left",
    ("bottom-left", "clockwise"): "up",
    ("top-left", "counterclockwise"): "down",
    ("top-right", "counterclockwise"): "left",
    ("bottom-right", "counterclockwise"): "up",
    ("bottom-left", "counterclockwise"): "right",
}


def _sp_params(rng):
    corner, rot = rng.choice(sorted(_SPIRAL_START))
    return {"corner": corner, "rot": rot, "facing": _SPIRAL_START[(corner, rot)]}


def _sp_desc(p):
    return (
        "Traverse `matrix` (a non-empty rectangular grid of integers; row 0 is the top row, column 0 the "
        "left column) in a spiral and return the list of visited values in order. Start at the "
        f"{p['corner']} cell, facing {p['facing']}. Repeatedly: record the value of the current cell; if "
        "every cell has been recorded, stop; otherwise, if the next cell in the facing direction is inside "
        f"the grid and not yet recorded, move to it, else turn 90 degrees {p['rot']} and move one cell in "
        "the new direction.\n\nConstraints: 1 <= number of rows, number of columns <= 8."
    )


def _sp_inputs(rng, p):
    def mat(r, c):
        vals = rng.sample(range(1, 100), r * c)
        return [vals[i * c : (i + 1) * c] for i in range(r)]

    out = [(mat(3, 3),), (mat(3, 4),), ([[7]],), (mat(1, 5),), (mat(5, 1),), (mat(2, 2),)]
    return out + [(mat(rng.randint(1, 8), rng.randint(1, 8)),) for _ in range(8)]


def _sp_solve(p, matrix):
    rows, cols = len(matrix), len(matrix[0])
    r = 0 if p["corner"].startswith("top") else rows - 1
    c = 0 if p["corner"].endswith("left") else cols - 1
    d = _DIR_NAMES.index(p["facing"])
    turn = 1 if p["rot"] == "clockwise" else -1
    inside = lambda rr, cc: 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in seen  # noqa: E731
    seen: set[tuple[int, int]] = set()
    out = []
    while True:
        seen.add((r, c))
        out.append(matrix[r][c])
        if len(seen) == rows * cols:
            return out
        if not inside(r + _DIRS[d][0], c + _DIRS[d][1]):
            d = (d + turn) % 4
            if not inside(r + _DIRS[d][0], c + _DIRS[d][1]):
                raise AssertionError("spiral: a curva leva a uma célula inválida (enunciado ambíguo)")
        r, c = r + _DIRS[d][0], c + _DIRS[d][1]


_MODS = [10**9 + 7, 998244353, 10**9 + 9, 1000003]


def _steps_params(rng):
    return {"steps": sorted(rng.sample(range(1, 8), rng.randint(2, 4))), "mod": rng.choice(_MODS)}


def _steps_desc(p):
    return (
        f"You climb a staircase of `n` steps; each move climbs exactly s steps for some s in {p['steps']}. "
        "Return the number of different ordered sequences of moves that end exactly on step n, modulo "
        f"{p['mod']}. For n = 0 the answer is 1 (the empty sequence).\n\nConstraints: 0 <= n <= 100000."
    )


def _steps_inputs(rng, p):
    out = [(rng.randint(5, 9),), (rng.randint(10, 14),), (0,), (1,), (p["steps"][0] - 1,)]
    out += [(rng.randint(0, 60),) for _ in range(7)]
    return out + [(100000,), (99991,)]


def _steps_solve(p, n):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = sum(dp[i - s] for s in p["steps"] if s <= i) % p["mod"]
    return dp[n]


_MOVES = {
    "4": ("to one of the 4 side-adjacent cells", [(0, 1), (1, 0), (0, -1), (-1, 0)]),
    "8": (
        "to one of the 8 adjacent cells (diagonals included)",
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc],
    ),
    "knight": (
        "like a chess knight: 2 cells in one direction and 1 cell in a perpendicular direction, jumping "
        "over whatever is in between",
        [(a, b) for a in (-2, -1, 1, 2) for b in (-2, -1, 1, 2) if abs(a) != abs(b)],
    ),
}


def _gp_params(rng):
    return {"moves": rng.choice(sorted(_MOVES))}


def _gp_desc(p):
    return (
        "`grid` is a list of equal-length strings; grid[r][c] is '.' (free) or '#' (blocked). Starting at "
        "the top-left cell (0, 0), reach the bottom-right cell with the fewest moves, where one move goes "
        f"{_MOVES[p['moves']][0]}. You may only stand on free cells and never leave the grid. Return the "
        "minimum number of moves, or -1 if it is impossible (also when the start or the target cell is "
        "blocked). A 1x1 grid whose cell is free has answer 0.\n\n"
        "Constraints: 1 <= number of rows, number of columns <= 30."
    )


def _gp_grid(rng, rows, cols, density, free_ends=True):
    g = [["#" if rng.random() < density else "." for _ in range(cols)] for _ in range(rows)]
    if free_ends:
        g[0][0] = g[rows - 1][cols - 1] = "."
    return ["".join(r) for r in g]


def _gp_inputs(rng, p):
    out = [(_gp_grid(rng, 4, 5, 0.25),), (_gp_grid(rng, 5, 6, 0.3),), (["."],), (["#."],), (["..", ".#"],)]
    out += [(_gp_grid(rng, rng.randint(1, 8), rng.randint(1, 8), rng.choice([0.15, 0.3, 0.45])),) for _ in range(8)]
    return out + [(_gp_grid(rng, 30, 30, 0.3),), (_gp_grid(rng, 30, 30, 0.2),)]


def _gp_solve(p, grid):
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == "#" or grid[rows - 1][cols - 1] == "#":
        return -1
    dist = {(0, 0): 0}
    q = deque([(0, 0)])
    while q:
        r, c = q.popleft()
        if (r, c) == (rows - 1, cols - 1):
            return dist[(r, c)]
        for dr, dc in _MOVES[p["moves"]][1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "." and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return -1


_BRACKETS = ["()", "()[]", "()[]{}"]


def _br_params(rng):
    return {"pairs": rng.choice(_BRACKETS)}


def _br_desc(p):
    pairs = [p["pairs"][i : i + 2] for i in range(0, len(p["pairs"]), 2)]
    return (
        "A string is balanced if it can be reduced to the empty string by repeatedly deleting an adjacent "
        f"matched pair {', '.join(repr(x) for x in pairs)}. Return the length of the longest contiguous "
        "substring of `s` that is balanced (0 if none; the empty substring counts as balanced).\n\n"
        f"Constraints: 0 <= len(s) <= 3000; `s` uses only the characters {p['pairs']!r}."
    )


def _br_balanced(rng, pairs, n_pairs):
    opens, closes = pairs[0::2], pairs[1::2]
    s, stack, left = [], [], n_pairs
    while left or stack:
        if left and (not stack or rng.random() < 0.5):
            i = rng.randrange(len(opens))
            s.append(opens[i])
            stack.append(closes[i])
            left -= 1
        else:
            s.append(stack.pop())
    return "".join(s)


def _br_str(rng, pairs, length):
    parts, total = [], 0
    while total < length:
        chunk = _br_balanced(rng, pairs, rng.randint(1, 4)) if rng.random() < 0.6 else rng.choice(pairs)
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)[:length]


def _br_inputs(rng, p):
    pr = p["pairs"]
    out = [(_br_str(rng, pr, rng.randint(8, 12)),), (_br_str(rng, pr, rng.randint(12, 16)),), ("",)]
    out += [(pr[1] + pr[0],), (pr[:2] * 3,)]
    out += [(_br_str(rng, pr, rng.randint(1, 20)),) for _ in range(8)]
    return out + [(_br_str(rng, pr, 3000),), (_br_str(rng, pr, 3000),)]


def _br_solve(p, s):
    match = {p["pairs"][i + 1]: p["pairs"][i] for i in range(0, len(p["pairs"]), 2)}
    stack, best = [-1], 0  # índices: aberturas pendentes ou barreiras
    for i, ch in enumerate(s):
        if ch in match and len(stack) > 1 and s[stack[-1]] == match[ch]:
            stack.pop()
            best = max(best, i - stack[-1])
        else:
            stack.append(i)
    return best


# ----------------------------------------------------------------------------------------- hard


def _chain_params(rng):
    lo = rng.randint(-3, 2)
    return {"lo": lo, "hi": lo + rng.randint(0, 4)}


def _chain_desc(p):
    return (
        "Return the length of the longest subsequence of `nums` (elements kept in their original order, "
        "not necessarily contiguous) in which every element minus the element before it is between "
        f"{p['lo']} and {p['hi']} inclusive. A single element is a valid subsequence; return 0 for an empty "
        "list.\n\nConstraints: 0 <= len(nums) <= 1000, -50 <= nums[i] <= 50."
    )


def _chain_inputs(rng, p):
    rnd = lambda n, v: [rng.randint(-v, v) for _ in range(n)]  # noqa: E731
    out = [(rnd(rng.randint(6, 9), 10),), (rnd(rng.randint(8, 11), 10),), ([],), ([3],)]
    out += [(rnd(rng.randint(1, 12), 8),) for _ in range(8)]
    return out + [(rnd(1000, 50),), (rnd(1000, 15),)]


def _chain_solve(p, nums):
    best: dict[int, int] = {}
    ans = 0
    for x in nums:
        cur = 1 + max((best.get(x - d, 0) for d in range(p["lo"], p["hi"] + 1)), default=0)
        best[x] = max(best.get(x, 0), cur)
        ans = max(ans, cur)
    return ans


def _knap_params(rng):
    return {"c": rng.randint(2, 6)}


def _knap_desc(p):
    return (
        "Item i has weight weights[i] and value values[i]. Choose a subset of the items (each item at most "
        f"once) with total weight at most `capacity` and at most {p['c']} items. Return the maximum total "
        "value (0 if nothing can be chosen).\n\nConstraints: 0 <= len(weights) == len(values) <= 40, "
        "1 <= weights[i] <= 50, 1 <= values[i] <= 100, 0 <= capacity <= 500."
    )


def _knap_inputs(rng, p):
    def items(n, wmax, cap):
        return ([rng.randint(1, wmax) for _ in range(n)], [rng.randint(1, 100) for _ in range(n)], cap)

    out = [items(rng.randint(4, 6), 15, rng.randint(10, 40)), items(rng.randint(5, 7), 20, rng.randint(15, 50))]
    out += [([], [], 10), items(3, 10, 0), ([5] * 8, list(range(10, 90, 10)), 40)]
    out += [items(rng.randint(1, 10), 20, rng.randint(0, 60)) for _ in range(7)]
    return out + [items(40, 50, 500), items(40, 50, 120)]


def _knap_solve(p, weights, values, capacity):
    dp = [[-1] * (capacity + 1) for _ in range(p["c"] + 1)]  # dp[k][w]: melhor valor com k itens e peso w
    dp[0][0] = 0
    for wt, v in zip(weights, values):
        for k in range(p["c"], 0, -1):
            prev, row = dp[k - 1], dp[k]
            for w in range(capacity, wt - 1, -1):
                if prev[w - wt] >= 0 and prev[w - wt] + v > row[w]:
                    row[w] = prev[w - wt] + v
    return max(max(r) for r in dp)


def _pass_params(rng):
    return {"K": rng.randint(0, 3)}


def _pass_desc(p):
    return (
        "`grid` is a rectangular grid of integers from 1 to 9. Walk from the top-left cell to the "
        "bottom-right cell, moving each step to a side-adjacent cell (up, down, left or right). Every time "
        "you are on a cell you pay its value: you pay for the start cell, and again each time you enter a "
        f"cell. You hold {p['K']} free passes; each pass makes one of those payments 0. Return the minimum "
        "total payment.\n\nConstraints: 1 <= number of rows, number of columns <= 25."
    )


def _pass_inputs(rng, p):
    g = lambda r, c: [[rng.randint(1, 9) for _ in range(c)] for _ in range(r)]  # noqa: E731
    out = [(g(3, 3),), (g(3, 4),), ([[5]],), (g(1, 6),), (g(4, 1),)]
    out += [(g(rng.randint(1, 6), rng.randint(1, 6)),) for _ in range(7)]
    return out + [(g(25, 25),), (g(25, 20),)]


def _pass_solve(p, grid):
    rows, cols, K = len(grid), len(grid[0]), p["K"]
    dist = {(0, 0, 0): grid[0][0]}
    heap = [(grid[0][0], 0, 0, 0)]
    if K:
        dist[(0, 0, 1)] = 0
        heap.append((0, 0, 0, 1))
    heapq.heapify(heap)
    while heap:
        d, r, c, k = heapq.heappop(heap)
        if d > dist[(r, c, k)]:
            continue
        if (r, c) == (rows - 1, cols - 1):
            return d
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            for nk, nd in ((k, d + grid[nr][nc]), (k + 1, d)):
                if nk <= K and nd < dist.get((nr, nc, nk), float("inf")):
                    dist[(nr, nc, nk)] = nd
                    heapq.heappush(heap, (nd, nr, nc, nk))
    raise AssertionError("inalcançável")


def _wi_params(rng):
    return {"g": rng.randint(0, 3)}


def _wi_desc(p):
    return (
        "Each element of `intervals` is [start, end, weight] with start < end. Choose a set of intervals "
        f"such that for any two chosen intervals A and B, either B.start - A.end >= {p['g']} or "
        f"A.start - B.end >= {p['g']}. Return the maximum possible total weight of the chosen intervals "
        "(0 for an empty list).\n\nConstraints: 0 <= len(intervals) <= 1000, 0 <= start < end <= 10**6, "
        "1 <= weight <= 1000."
    )


def _wi_inputs(rng, p):
    def ivs(n, hi, maxlen, wmax):
        out = []
        for _ in range(n):
            s = rng.randint(0, hi - 1)
            out.append([s, min(hi, s + rng.randint(1, maxlen)), rng.randint(1, wmax)])
        return out

    out = [(ivs(4, 20, 6, 20),), (ivs(6, 25, 6, 20),), ([],), ([[0, 2, 5], [2, 4, 7]],)]
    out += [(ivs(rng.randint(1, 12), 30, 8, 20),) for _ in range(8)]
    return out + [(ivs(1000, 10**6, 5000, 1000),), (ivs(1000, 10**5, 2000, 1000),)]


def _wi_solve(p, intervals):
    iv = sorted((e, s, w) for s, e, w in intervals)
    ends = [e for e, _, _ in iv]
    dp = [0] * (len(iv) + 1)
    for i, (e, s, w) in enumerate(iv, 1):
        j = bisect.bisect_right(ends, s - p["g"], 0, i - 1)
        dp[i] = max(dp[i - 1], dp[j] + w)
    return dp[-1]


_MOD = 10**9 + 7


def _rc_params(rng):
    return {"a": rng.randint(2, 5), "r": rng.randint(1, 4)}


def _rc_desc(p):
    return (
        f"Count the strings of length `n` over an alphabet of {p['a']} letters in which every block of equal "
        f"consecutive letters has length at most {p['r']}. Return the count modulo 1000000007. "
        "For n = 0 the answer is 1.\n\n"
        "Constraints: 0 <= n <= 1000."
    )


def _rc_inputs(rng, p):
    out = [(rng.randint(3, 4),), (rng.randint(5, 6),), (0,), (1,), (p["r"],), (p["r"] + 1,)]
    out += [(rng.randint(0, 6),) for _ in range(4)]
    return out + [(rng.randint(20, 60),), (1000,), (999,)]


def _rc_solve(p, n):
    if n == 0:
        return 1
    ends = [0] * (p["r"] + 1)  # ends[j]: strings terminando num bloco de tamanho j
    ends[1] = p["a"]
    for _ in range(n - 1):
        new = [0] * (p["r"] + 1)
        new[1] = sum(ends) * (p["a"] - 1) % _MOD
        for j in range(2, p["r"] + 1):
            new[j] = ends[j - 1]
        ends = new
    return sum(ends) % _MOD


def _game_params(rng):
    return {"moves": sorted(rng.sample(range(1, 10), rng.randint(2, 4)))}


def _game_desc(p):
    return (
        "Two players alternate turns on the piles of stones in `piles`. On a turn the player chooses one pile "
        f"and removes exactly s stones from it, for some s in {p['moves']} with s <= the size of that pile. "
        "The player who cannot make a move loses. Both play optimally. Return True if the player who moves "
        "first wins, otherwise False.\n\nConstraints: 1 <= len(piles) <= 5, 0 <= piles[i] <= 10000."
    )


def _game_inputs(rng, p):
    rnd = lambda n, v: [rng.randint(0, v) for _ in range(n)]  # noqa: E731
    out = [(rnd(2, 15),), (rnd(3, 15),), ([0],), ([0, 0],), ([p["moves"][0]],)]
    out += [(rnd(rng.randint(1, 3), 12),) for _ in range(8)]
    return out + [(rnd(5, 10000),), (rnd(4, 10000),)]


def _game_solve(p, piles):
    top = max(piles, default=0)
    gr = [0] * (top + 1)
    for x in range(1, top + 1):
        seen = {gr[x - s] for s in p["moves"] if s <= x}
        g = 0
        while g in seen:
            g += 1
        gr[x] = g
    acc = 0
    for x in piles:
        acc ^= gr[x]
    return acc != 0


FAMILIES: list[Family] = [
    Family("digit_sum", "easy", "def digit_sum(n: int) -> int:", _digit_params, _digit_desc, _digit_inputs, _digit_solve),
    Family("kth_char", "easy", "def transform(s: str) -> str:", _kth_params, _kth_desc, _kth_inputs, _kth_solve),
    Family("mod_sum", "easy", "def filtered_sum(nums: list[int]) -> int:", _modsum_params, _modsum_desc, _modsum_inputs, _modsum_solve),
    Family("long_runs", "easy", "def long_runs(s: str) -> list:", _runs_params, _runs_desc, _runs_inputs, _runs_solve),
    Family("pair_count", "easy", "def count_pairs(nums: list[int]) -> int:", _pairs_params, _pairs_desc, _pairs_inputs, _pairs_solve),
    Family("sort_words", "easy", "def sort_words(words: list[str]) -> list[str]:", _sort_params, _sort_desc, _sort_inputs, _sort_solve),
    Family("k_distinct", "medium", "def longest_window(s: str) -> int:", _kd_params, _kd_desc, _kd_inputs, _kd_solve),
    Family("merge_gap", "medium", "def covered_length(intervals: list[list[int]]) -> int:", _mg_params, _mg_desc, _mg_inputs, _mg_solve),
    Family("spiral", "medium", "def spiral(matrix: list[list[int]]) -> list[int]:", _sp_params, _sp_desc, _sp_inputs, _sp_solve),
    Family("step_ways", "medium", "def count_ways(n: int) -> int:", _steps_params, _steps_desc, _steps_inputs, _steps_solve),
    Family("grid_path", "medium", "def shortest_path(grid: list[str]) -> int:", _gp_params, _gp_desc, _gp_inputs, _gp_solve),
    Family("brackets", "medium", "def longest_balanced(s: str) -> int:", _br_params, _br_desc, _br_inputs, _br_solve),
    Family("diff_chain", "hard", "def longest_chain(nums: list[int]) -> int:", _chain_params, _chain_desc, _chain_inputs, _chain_solve),
    Family("knapsack_k", "hard", "def best_value(weights: list[int], values: list[int], capacity: int) -> int:", _knap_params, _knap_desc, _knap_inputs, _knap_solve),
    Family("grid_passes", "hard", "def min_cost(grid: list[list[int]]) -> int:", _pass_params, _pass_desc, _pass_inputs, _pass_solve),
    Family("intervals_gap", "hard", "def max_weight(intervals: list[list[int]]) -> int:", _wi_params, _wi_desc, _wi_inputs, _wi_solve),
    Family("run_count", "hard", "def count_strings(n: int) -> int:", _rc_params, _rc_desc, _rc_inputs, _rc_solve),
    Family("pile_game", "hard", "def first_player_wins(piles: list[int]) -> bool:", _game_params, _game_desc, _game_inputs, _game_solve),
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
        return "truncated"
    return {"EmptyCode": "no_code", "AssertionError": "wrong_answer", "TimeoutError": "timeout"}.get(
        str(row.get("error_type")), "runtime_error"
    )


def custom_extra_metrics(rows_by_id: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Métricas por nível/família/falha para o `config` do registro em `results.jsonl`."""
    rows = {tid: r for tid, r in rows_by_id.items() if r is not None}
    out: dict[str, Any] = {"custom_protocol": CUSTOM_PROTOCOL_ID}

    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else None

    for tier in TIERS:
        rs = [r for tid, r in rows.items() if tid.split("/")[0] == tier]
        out[f"{tier}_pass1"] = mean(float(bool(r.get("passed"))) for r in rs)
        out[f"{tier}_tokens_cot_mean"] = mean(float(r.get("n_cot_tokens", 0)) for r in rs)
        out[f"{tier}_truncated_frac"] = mean(float(bool(r.get("truncated"))) for r in rs)
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
