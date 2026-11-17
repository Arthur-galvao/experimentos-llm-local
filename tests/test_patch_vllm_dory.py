"""Testes em CPU para scripts/patch_vllm_dory.py usando o dory.py da PR #60149 como fixture."""

from __future__ import annotations

import ast
import importlib.util
import py_compile
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

T4_ROOT = Path(__file__).resolve().parent.parent
PATCH_SCRIPT = T4_ROOT / "scripts" / "patch_vllm_dory.py"
FIXTURE_PATH = T4_ROOT / "tests" / "fixtures" / "vllm_dory_pr60149.py"


@pytest.fixture(scope="module")
def patch_mod():
    spec = importlib.util.spec_from_file_location("patch_vllm_dory_under_test", PATCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patch_apply_idempotent_and_compile(patch_mod, tmp_path: Path) -> None:
    target = tmp_path / "dory.py"
    shutil.copyfile(FIXTURE_PATH, target)

    # Antes do patch: --check deve retornar código != 0
    proc_before = subprocess.run(
        [sys.executable, str(PATCH_SCRIPT), "--check", "--target", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_before.returncode == 1
    assert not patch_mod.is_patched(target.read_text(encoding="utf-8"))

    # Primeira aplicação
    ok, diff1 = patch_mod.patch_file(target, check=False)
    assert ok is True
    assert diff1 != ""
    assert "+class DoryForCausalLM(nn.Module, SupportsQuant):" in diff1
    assert "from vllm.model_executor.models.interfaces import SupportsQuant" in diff1
    assert patch_mod.PATCH_MARKER in diff1

    patched_src = target.read_text(encoding="utf-8")
    assert patch_mod.is_patched(patched_src)
    assert "packed_modules_mapping" in patched_src

    # Validação sintática e py_compile
    ast.parse(patched_src, filename=str(target))
    py_compile.compile(str(target), doraise=True)

    # Segunda aplicação (idempotente)
    ok2, diff2 = patch_mod.patch_file(target, check=False)
    assert ok2 is True
    assert diff2 == ""
    assert target.read_text(encoding="utf-8") == patched_src

    # Depois do patch: --check via CLI deve retornar 0
    proc_after = subprocess.run(
        [sys.executable, str(PATCH_SCRIPT), "--check", "--target", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_after.returncode == 0, proc_after.stderr


def test_patch_aborts_atomically_on_missing_anchor(patch_mod, tmp_path: Path) -> None:
    target = tmp_path / "dory_modified.py"
    orig = FIXTURE_PATH.read_text(encoding="utf-8")
    broken = orig.replace("class DoryForCausalLM(nn.Module):", "class DoryForCausalLM(nn.Module, CustomBase):")
    target.write_text(broken, encoding="utf-8")

    with pytest.raises(ValueError, match="class_SupportsQuant"):
        patch_mod.patch_file(target, check=False)

    # Nunca aplica meio patch: import_SupportsQuant não pode ter sido gravado
    after = target.read_text(encoding="utf-8")
    assert after == broken
    assert "SupportsQuant" not in after


def test_patched_dory_load_weights_gptq_and_fp16_behavior(patch_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executa o load_weights do dory.py patchado com stubs mínimos do vLLM para checar GPTQ + FP16."""
    target = tmp_path / "dory.py"
    shutil.copyfile(FIXTURE_PATH, target)
    patch_mod.patch_file(target, check=False)

    # Stub mínimo da árvore vllm.* para importar o dory.py patchado sem vllm instalado
    class FakeSupportsQuant:
        packed_modules_mapping: dict[str, list[str]]

    class FakePerTensorScaleParameter(torch.nn.Parameter):
        pass

    stub_names = [
        "vllm",
        "vllm.compilation",
        "vllm.compilation.decorators",
        "vllm.config",
        "vllm.distributed",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.layers.logits_processor",
        "vllm.model_executor.layers.rotary_embedding",
        "vllm.model_executor.layers.vocab_parallel_embedding",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.weight_utils",
        "vllm.model_executor.models",
        "vllm.model_executor.models.interfaces",
        "vllm.model_executor.models.utils",
        "vllm.model_executor.parameter",
        "vllm.sequence",
    ]
    pkg = types.ModuleType("vllm.model_executor.models")
    pkg.__path__ = [str(tmp_path)]  # permite `from .utils import maybe_prefix`
    mods: dict[str, types.ModuleType] = {name: types.ModuleType(name) for name in stub_names}
    mods["vllm.model_executor.models"] = pkg
    mods["vllm.compilation.decorators"].support_torch_compile = lambda cls: cls
    mods["vllm.config"].VllmConfig = object
    mods["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 1
    mods["vllm.model_executor.layers.attention"].Attention = object
    for attr in ("ColumnParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear", "RowParallelLinear"):
        setattr(mods["vllm.model_executor.layers.linear"], attr, object)
    mods["vllm.model_executor.layers.logits_processor"].LogitsProcessor = object
    mods["vllm.model_executor.layers.rotary_embedding"].get_rope = lambda *a, **kw: None
    mods["vllm.model_executor.layers.vocab_parallel_embedding"].ParallelLMHead = object
    mods["vllm.model_executor.layers.vocab_parallel_embedding"].VocabParallelEmbedding = object
    mods["vllm.model_executor.model_loader.weight_utils"].default_weight_loader = (
        lambda param, w: param.data.copy_(w)
    )
    mods["vllm.model_executor.models.interfaces"].SupportsQuant = FakeSupportsQuant
    mods["vllm.model_executor.models.utils"].maybe_prefix = lambda p, n: f"{p}.{n}" if p else n
    mods["vllm.model_executor.parameter"].PerTensorScaleParameter = FakePerTensorScaleParameter
    mods["vllm.sequence"].IntermediateTensors = object

    for k, m in mods.items():
        monkeypatch.setitem(sys.modules, k, m)

    spec = importlib.util.spec_from_file_location("vllm.model_executor.models.dory", target)
    assert spec is not None and spec.loader is not None
    dory_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dory_mod)

    DoryForCausalLM = dory_mod.DoryForCausalLM
    assert issubclass(DoryForCausalLM, FakeSupportsQuant)
    assert DoryForCausalLM.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Instancia sem rodar __init__ pesado e testa load_weights
    calls: list[tuple] = []

    class QuantParam(torch.nn.Parameter):
        def __new__(cls, data, *, output_dim=None, packed_dim=None):
            p = torch.nn.Parameter.__new__(cls, data, requires_grad=False)
            p.output_dim = output_dim
            p.packed_dim = packed_dim
            p.weight_loader = lambda param, w, shard_id=None: calls.append((tuple(w.shape), shard_id))
            return p

    inst = DoryForCausalLM.__new__(DoryForCausalLM)
    torch.nn.Module.__init__(inst)
    inst.config = types.SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=64,
        vocab_size=32,
        hidden_size=32,
    )
    fake_params = {
        "backbone.layers.0.mixer.qkv_proj.qweight": QuantParam(torch.zeros(8, 128, dtype=torch.int32), output_dim=1, packed_dim=0),
        "backbone.layers.0.mixer.qkv_proj.scales": QuantParam(torch.zeros(1, 128, dtype=torch.float16), output_dim=1),
        "backbone.layers.0.mixer.qkv_proj.qzeros": QuantParam(torch.zeros(1, 32, dtype=torch.int32), output_dim=1, packed_dim=1),
        "backbone.layers.0.mixer.o_proj.qweight": QuantParam(torch.zeros(16, 32, dtype=torch.int32), output_dim=1, packed_dim=0),
        "backbone.embeddings.weight": torch.nn.Parameter(torch.zeros(32, 32), requires_grad=False),
    }
    monkeypatch.setattr(inst, "named_parameters", lambda: fake_params.items())

    weights = [
        ("backbone.layers.0.mixer.q_proj.qweight", torch.ones(8, 64, dtype=torch.int32)),
        ("backbone.layers.0.mixer.k_proj.scales", torch.ones(1, 32, dtype=torch.float16)),
        ("backbone.layers.0.mixer.v_proj.qzeros", torch.ones(1, 8, dtype=torch.int32)),
        ("backbone.layers.0.mixer.q_proj.g_idx", torch.zeros(32, dtype=torch.int32)),  # ignorado
        ("backbone.layers.0.mixer.o_proj.qweight", torch.ones(16, 32, dtype=torch.int32)),
        ("backbone.layers.0.mixer.o_proj.g_idx", torch.zeros(64, dtype=torch.int32)),  # ignorado
        ("backbone.embeddings.weight", torch.ones(32, 32)),
    ]
    loaded = inst.load_weights(weights)
    assert "backbone.layers.0.mixer.qkv_proj.qweight" in loaded
    assert "backbone.layers.0.mixer.qkv_proj.scales" in loaded
    assert "backbone.layers.0.mixer.qkv_proj.qzeros" in loaded
    assert "backbone.layers.0.mixer.o_proj.qweight" in loaded
    assert "backbone.embeddings.weight" in loaded
    assert calls == [((8, 64), "q"), ((1, 32), "k"), ((1, 8), "v"), ((16, 32), None)]
    assert torch.equal(fake_params["backbone.embeddings.weight"], torch.ones(32, 32))
