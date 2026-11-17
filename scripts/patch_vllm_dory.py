"""Aplica (de forma idempotente) o patch mínimo para suporte GPTQ/SupportsQuant em `vllm.model_executor.models.dory`.

Fonte das âncoras: vLLM PR #60149 (commit `876edc971507f602b18d1ec523c174a068cc02d7`),
arquivo `vllm/model_executor/models/dory.py`.

Uso:
  python scripts/patch_vllm_dory.py          # localiza dory.py instalado e aplica o patch
  python scripts/patch_vllm_dory.py --check  # apenas verifica se o patch está aplicado (exit 0/1)
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import sys
from pathlib import Path

PATCH_MARKER = "# PATCH_VLLM_DORY_GPTQ_V1"

_IMPORT_ANCHOR = (
    "from vllm.sequence import IntermediateTensors\n"
    "\n"
    "from .utils import maybe_prefix\n"
)
_IMPORT_REPLACEMENT = (
    "from vllm.model_executor.models.interfaces import SupportsQuant\n"
    "from vllm.sequence import IntermediateTensors\n"
    "\n"
    "from .utils import maybe_prefix\n"
)

_CLASS_ANCHOR = (
    "class DoryForCausalLM(nn.Module):\n"
    "    packed_modules_mapping = {\n"
)
_CLASS_REPLACEMENT = (
    "class DoryForCausalLM(nn.Module, SupportsQuant):\n"
    "    packed_modules_mapping = {\n"
)

_LOAD_WEIGHTS_ANCHOR = (
    "        for name, w in weights:\n"
    "            mapped = False\n"
    "            for target, shard_name, shard_id in stacked:\n"
    '                if f".{shard_name}." not in name:\n'
    "                    continue\n"
    '                pname = name.replace(f".{shard_name}.", f".{target}.")\n'
    "                param = params[pname]\n"
    '                if target == "qkv_proj":\n'
    "                    heads = (\n"
    "                        self.config.num_attention_heads\n"
    '                        if shard_id == "q"\n'
    "                        else self.config.num_key_value_heads\n"
    "                    )\n"
    "                    rows = heads * self.config.head_dim\n"
    "                else:\n"
    "                    rows = self.config.intermediate_size\n"
    "                if isinstance(param, PerTensorScaleParameter):\n"
    "                    shape = () if w.ndim == 0 else (1,)\n"
    "                else:\n"
    "                    # Preserve packed input columns and per-block scale columns.\n"
    "                    shape = (rows, *param.shape[1:])\n"
    "                self._check_weight_shape(name, w, shape)\n"
    "                param.weight_loader(param, w, shard_id)\n"
    "                loaded.add(pname)  # report the MODEL param name, not the artifact name\n"
    "                mapped = True\n"
    "                break\n"
    "            if mapped:\n"
    "                continue\n"
    "\n"
    "            param = params[name]\n"
    "            shape = (\n"
    "                (self.config.vocab_size, self.config.hidden_size)\n"
    '                if name in ("backbone.embeddings.weight", "lm_head.weight")\n'
    "                else tuple(param.shape)\n"
    "            )\n"
    "            if isinstance(param, PerTensorScaleParameter) and w.ndim == 0:\n"
    "                shape = ()\n"
    "            self._check_weight_shape(name, w, shape)\n"
    '            getattr(param, "weight_loader", default_weight_loader)(param, w)\n'
    "            loaded.add(name)\n"
    "        return loaded\n"
)

_LOAD_WEIGHTS_REPLACEMENT = (
    f"        {PATCH_MARKER}\n"
    '        _quant_suffixes = (".qweight", ".qzeros", ".scales", ".g_idx")\n'
    "        for name, w in weights:\n"
    "            mapped = False\n"
    "            for target, shard_name, shard_id in stacked:\n"
    '                if f".{shard_name}." not in name:\n'
    "                    continue\n"
    '                pname = name.replace(f".{shard_name}.", f".{target}.")\n'
    '                if pname not in params and pname.endswith(".g_idx"):\n'
    "                    mapped = True\n"
    "                    break\n"
    "                param = params[pname]\n"
    "                is_quant = (\n"
    "                    name.endswith(_quant_suffixes)\n"
    '                    or getattr(param, "output_dim", 0) != 0\n'
    '                    or hasattr(param, "packed_dim")\n'
    "                )\n"
    "                if not is_quant:\n"
    '                    if target == "qkv_proj":\n'
    "                        heads = (\n"
    "                            self.config.num_attention_heads\n"
    '                            if shard_id == "q"\n'
    "                            else self.config.num_key_value_heads\n"
    "                        )\n"
    "                        rows = heads * self.config.head_dim\n"
    "                    else:\n"
    "                        rows = self.config.intermediate_size\n"
    "                    if isinstance(param, PerTensorScaleParameter):\n"
    "                        shape = () if w.ndim == 0 else (1,)\n"
    "                    else:\n"
    "                        # Preserve packed input columns and per-block scale columns.\n"
    "                        shape = (rows, *param.shape[1:])\n"
    "                    self._check_weight_shape(name, w, shape)\n"
    "                param.weight_loader(param, w, shard_id)\n"
    "                loaded.add(pname)  # report the MODEL param name, not the artifact name\n"
    "                mapped = True\n"
    "                break\n"
    "            if mapped:\n"
    "                continue\n"
    "\n"
    '            if name not in params and name.endswith(".g_idx"):\n'
    "                continue\n"
    "            param = params[name]\n"
    "            is_quant = (\n"
    "                name.endswith(_quant_suffixes)\n"
    '                or getattr(param, "output_dim", 0) != 0\n'
    '                or hasattr(param, "packed_dim")\n'
    "            )\n"
    "            if not is_quant:\n"
    "                shape = (\n"
    "                    (self.config.vocab_size, self.config.hidden_size)\n"
    '                    if name in ("backbone.embeddings.weight", "lm_head.weight")\n'
    "                    else tuple(param.shape)\n"
    "                )\n"
    "                if isinstance(param, PerTensorScaleParameter) and w.ndim == 0:\n"
    "                    shape = ()\n"
    "                self._check_weight_shape(name, w, shape)\n"
    '            getattr(param, "weight_loader", default_weight_loader)(param, w)\n'
    "            loaded.add(name)\n"
    "        return loaded\n"
)

_HUNKS: tuple[tuple[str, str, str], ...] = (
    ("import_SupportsQuant", _IMPORT_ANCHOR, _IMPORT_REPLACEMENT),
    ("class_SupportsQuant", _CLASS_ANCHOR, _CLASS_REPLACEMENT),
    ("load_weights_gptq", _LOAD_WEIGHTS_ANCHOR, _LOAD_WEIGHTS_REPLACEMENT),
)


def find_installed_dory_path() -> Path:
    """Localiza `vllm/model_executor/models/dory.py` no ambiente Python ativo."""
    spec = importlib.util.find_spec("vllm.model_executor.models.dory")
    if spec is None or not spec.origin:
        raise FileNotFoundError(
            "Módulo `vllm.model_executor.models.dory` não encontrado no ambiente atual."
        )
    return Path(spec.origin)


def is_patched(src: str) -> bool:
    """Retorna True se todos os trechos do patch estiverem presentes no código-fonte."""
    norm = src.replace("\r\n", "\n")
    return (
        PATCH_MARKER in norm
        and _IMPORT_REPLACEMENT in norm
        and _CLASS_REPLACEMENT in norm
        and _LOAD_WEIGHTS_REPLACEMENT in norm
    )


def patch_text(src: str, *, path_label: str = "dory.py") -> tuple[str, str, bool]:
    """Aplica o patch em memória sobre `src`.

    Retorna `(novo_texto, unified_diff, mudou)`. Falha antes de alterar qualquer coisa se
    alguma âncora faltar ou for ambígua.
    """
    norm = src.replace("\r\n", "\n")
    if is_patched(norm):
        return src, "", False

    missing_or_ambiguous: list[str] = []
    for hunk_name, anchor, _ in _HUNKS:
        count = norm.count(anchor)
        if count != 1:
            missing_or_ambiguous.append(f"{hunk_name} (ocorrências={count})")
    if missing_or_ambiguous:
        raise ValueError(
            f"Patch abortado em {path_label}: âncoras inválidas: {', '.join(missing_or_ambiguous)}. "
            "Nenhuma alteração foi gravada."
        )

    patched = norm
    for _, anchor, replacement in _HUNKS:
        patched = patched.replace(anchor, replacement, 1)

    ast.parse(patched, filename=path_label)
    diff = "".join(
        difflib.unified_diff(
            norm.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{path_label}",
            tofile=f"b/{path_label}",
        )
    )
    return patched, diff, True


def patch_file(path: Path, *, check: bool = False) -> tuple[bool, str]:
    """Verifica ou aplica o patch em `path` no disco.

    Quando `check=True`, retorna `(True, msg)` se já estiver aplicado e `(False, msg)` caso contrário.
    Quando `check=False`, aplica atomicamente (só escreve após validar todas as âncoras e o AST)
    e retorna `(True, diff)`.
    """
    src = path.read_text(encoding="utf-8")
    if check:
        ok = is_patched(src)
        return ok, ("OK: patch já aplicado." if ok else "PENDENTE: patch NÃO está aplicado.")

    patched, diff, changed = patch_text(src, path_label=str(path))
    if not changed:
        return True, ""
    path.write_text(patched, encoding="utf-8")
    return True, diff


def check_installed(target: Path | None = None) -> tuple[bool, Path, str]:
    """Verifica se `target` (ou o `dory.py` do vLLM instalado) está com o patch aplicado."""
    path = target if target is not None else find_installed_dory_path()
    ok, msg = patch_file(path, check=True)
    return ok, path, msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="apenas verifica se o patch já está aplicado")
    ap.add_argument("--target", type=Path, default=None, help="caminho alternativo de dory.py (para testes)")
    args = ap.parse_args()

    try:
        path = args.target if args.target is not None else find_installed_dory_path()
        ok, info = patch_file(path, check=args.check)
    except Exception as e:  # noqa: BLE001
        print(f"[patch_vllm_dory] ERRO: {e}", file=sys.stderr)
        sys.exit(1)

    if args.check:
        print(f"[patch_vllm_dory] {path}: {info}")
        sys.exit(0 if ok else 1)

    if info:
        print(f"[patch_vllm_dory] Patch aplicado em {path}:\n{info}")
    else:
        print(f"[patch_vllm_dory] {path}: já estava aplicado (idempotente).")


if __name__ == "__main__":
    main()
