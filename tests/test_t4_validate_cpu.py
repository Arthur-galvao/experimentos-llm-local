"""Testes em CPU para scripts/t4_validate.py (sem GPU e sem vLLM)."""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import py_compile
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

T4_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = T4_ROOT / "scripts" / "t4_validate.py"
DEFAULT_SCRATCH_REPO = T4_ROOT  # lib/ vive no próprio repo
REPO_WITH_LIB = Path(os.environ.get("DORY_REPO_ROOT", str(DEFAULT_SCRATCH_REPO)))
if REPO_WITH_LIB.exists() and str(REPO_WITH_LIB) not in sys.path:
    sys.path.insert(0, str(REPO_WITH_LIB))


@pytest.fixture(scope="module")
def t4():
    """Importa scripts/t4_validate.py como módulo sem modificar o arquivo original."""
    spec = importlib.util.spec_from_file_location("t4_validate_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unpack_gptq_8bit(tensors: dict[str, torch.Tensor], zero_offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Desempacota qweight/qzeros (int32 tratando negativos como uint32) e reconstrói W [out, in]."""
    qw = tensors["qweight"].to(torch.int64) & 0xFFFFFFFF  # [in/4, out]
    qz = tensors["qzeros"].to(torch.int64) & 0xFFFFFFFF  # [in/128, out/4]
    scales = tensors["scales"].float()  # [in/128, out]

    in_q4, out_f = qw.shape
    in_f = in_q4 * 4
    n_groups = scales.shape[0]
    group_size = in_f // n_groups

    q_parts = [(qw >> (8 * k)) & 0xFF for k in range(4)]
    q = torch.stack(q_parts, dim=1).reshape(in_f, out_f)  # [in, out]

    z_parts = [(qz >> (8 * k)) & 0xFF for k in range(4)]
    zeros = torch.stack(z_parts, dim=2).reshape(n_groups, out_f)  # [G, out]

    scales_exp = scales.repeat_interleave(group_size, dim=0)  # [in, out]
    zeros_exp = (zeros + zero_offset).repeat_interleave(group_size, dim=0)  # [in, out]
    w_rec = (scales_exp * (q - zeros_exp).float()).t().contiguous()  # [out, in]
    return q, zeros, w_rec


def _autogptq_ref_pack(q_in_out: torch.Tensor, zeros_g_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Implementação independente de referência do empacotamento 8-bit do AutoGPTQ/GPTQModel."""
    in_f, out_f = q_in_out.shape
    n_groups = zeros_g_out.shape[0]
    pack = 4  # 32 // 8

    intweight = q_in_out.to(torch.int64)
    qweight_u = torch.zeros((in_f // pack, out_f), dtype=torch.int64)
    i = 0
    row = 0
    while row < qweight_u.shape[0]:
        for j in range(i, i + pack):
            qweight_u[row] |= intweight[j] << (8 * (j - i))
        i += pack
        row += 1

    z_int = zeros_g_out.to(torch.int64)
    qzeros_u = torch.zeros((n_groups, out_f // pack), dtype=torch.int64)
    i = 0
    col = 0
    while col < qzeros_u.shape[1]:
        for j in range(i, i + pack):
            qzeros_u[:, col] |= z_int[:, j] << (8 * (j - i))
        i += pack
        col += 1

    qweight = torch.where(qweight_u >= 2**31, qweight_u - 2**32, qweight_u).to(torch.int32)
    qzeros = torch.where(qzeros_u >= 2**31, qzeros_u - 2**32, qzeros_u).to(torch.int32)
    return qweight, qzeros


# --------------------------------------------------------------------------- 1. quantize_rtn_int8
def test_to_int32_no_overflow_high_bit(t4) -> None:
    vals = torch.tensor([0, 1, 2**31 - 1, 2**31, 0x80808080, 0xFFFFFFFF], dtype=torch.int64)
    packed = t4._to_int32(vals)
    assert packed.dtype == torch.int32
    roundtrip = packed.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(roundtrip, vals)


@pytest.mark.parametrize(
    "shape",
    [
        (4096, 2560),
        (1024, 2560),
        (2560, 4096),
        (7168, 2560),
        (2560, 7168),
    ],
)
def test_quantize_rtn_int8_model_shapes_and_unpack(t4, shape: tuple[int, int]) -> None:
    torch.manual_seed(123)
    out_f, in_f = shape
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02

    tensors, deq = t4.quantize_rtn_int8(w, group_size=128)

    assert set(tensors.keys()) == {"qweight", "qzeros", "scales"}
    assert "g_idx" not in tensors
    assert tensors["qweight"].shape == (in_f // 4, out_f)
    assert tensors["qweight"].dtype == torch.int32
    assert tensors["qzeros"].shape == (in_f // 128, out_f // 4)
    assert tensors["qzeros"].dtype == torch.int32
    assert tensors["scales"].shape == (in_f // 128, out_f)
    assert tensors["scales"].dtype == torch.float16

    q, zeros, w_rec = _unpack_gptq_8bit(tensors, zero_offset=t4.ZERO_OFFSET)
    assert int(q.min()) >= 1
    assert int(q.max()) <= 255
    assert torch.all(zeros == 127)

    assert torch.allclose(w_rec, deq, atol=1e-6, rtol=1e-6)
    rel_err = ((deq - w.float()).norm() / w.float().norm()).item()
    assert 0.002 <= rel_err <= 0.007, f"Erro relativo inesperado: {rel_err}"


def test_quantize_rtn_int8_outliers_and_autogptq_reference(t4) -> None:
    torch.manual_seed(7)
    out_f, in_f = 512, 1024
    w = torch.randn(out_f, in_f, dtype=torch.float32) * 0.02
    w[:, ::7] *= 6.0  # outliers por coluna
    w[13, 256:384] *= 15.0  # outlier localizado em um grupo

    tensors, deq = t4.quantize_rtn_int8(w, group_size=128)
    assert "g_idx" not in tensors
    q, zeros, w_rec = _unpack_gptq_8bit(tensors, zero_offset=1)

    assert int(q.min()) >= 1 and int(q.max()) <= 255
    assert torch.all(zeros == 127)
    assert torch.allclose(w_rec, deq, atol=1e-6, rtol=1e-6)
    rel_err = ((deq - w).norm() / w.norm()).item()
    assert 0.002 <= rel_err <= 0.02  # outliers ×15 num grupo inflam a escala: ~1,2% é o esperado do RTN

    ref_qw, ref_qz = _autogptq_ref_pack(q, zeros)
    assert torch.equal(tensors["qweight"], ref_qw)
    assert torch.equal(tensors["qzeros"], ref_qz)


def test_quantize_rtn_int8_zero_group_underflow_bug(t4) -> None:
    """Regressão: grupo todo zero não pode gerar scale 0 em FP16 nem NaN."""
    w = torch.zeros(16, 128, dtype=torch.float32)
    tensors, deq = t4.quantize_rtn_int8(w, group_size=128)
    q, _, _ = _unpack_gptq_8bit(tensors, zero_offset=1)
    assert not torch.isnan(deq).any()
    assert int(q.min()) >= 1 and int(q.max()) <= 255


# --------------------------------------------------------------------------- 2. cmd_quantize E2E
def _make_fake_dory_checkpoint(src: Path, *, single_file: bool = False) -> dict[str, torch.Tensor]:
    src.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(99)
    shard1 = {
        "model.embeddings.weight": (torch.randn(64, 256) * 0.02).bfloat16(),
        "model.layers.0.mixer.q_proj.weight": (torch.randn(256, 256) * 0.02).bfloat16(),
        "model.layers.0.mixer.k_proj.weight": (torch.randn(128, 256) * 0.02).bfloat16(),
        "model.layers.0.mixer.g_proj.weight": (torch.randn(256, 256) * 0.02).bfloat16(),
        "model.layers.0.mixer.sqk": torch.full((256,), 0.0177, dtype=torch.bfloat16),
        "model.layers.0.attn_alpha": torch.full((256,), 0.0177, dtype=torch.bfloat16),
    }
    shard2 = {
        "model.layers.1.mlp.gate_proj.weight": (torch.randn(512, 256) * 0.02).bfloat16(),
        "model.layers.1.mlp.down_proj.weight": (torch.randn(256, 512) * 0.02).bfloat16(),
        "lm_head.weight": (torch.randn(64, 256) * 0.02).bfloat16(),
        "lm_head.logit_scale": torch.full((64,), 0.0177, dtype=torch.bfloat16),
    }
    if single_file:
        all_t = {**shard1, **shard2}
        save_file(all_t, str(src / "model.safetensors"))
    else:
        save_file(shard1, str(src / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(src / "model-00002-of-00002.safetensors"))
        # Deve ser ignorado e substituído por um index.json novo
        (src / "model.safetensors.index.json").write_text('{"stale": true}', encoding="utf-8")

    (src / "config.json").write_text(
        json.dumps({"model_type": "dory", "hidden_size": 256, "dtype": "bfloat16"}),
        encoding="utf-8",
    )
    (src / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (src / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}', encoding="utf-8")
    (src / "chat_template.jinja").write_text("{{ messages[0].content }}", encoding="utf-8")
    return {**shard1, **shard2}


@pytest.mark.parametrize("single_file", [False, True])
def test_cmd_quantize_e2e_cpu(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, single_file: bool) -> None:
    src = tmp_path / "hf_snapshot"
    orig_tensors = _make_fake_dory_checkpoint(src, single_file=single_file)

    out_dir = tmp_path / "out_results"
    int8_dir = tmp_path / "int8_local"
    int8_drive = tmp_path / "drive" / "checkpoints" / "int8_drive"
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda repo: str(src)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "INT8_DIR", int8_dir)
    monkeypatch.setattr(t4, "INT8_DRIVE", int8_drive)

    args = types.SimpleNamespace(out=str(int8_dir), skip=["g_proj"], save_drive=True)
    t4.cmd_quantize(args)

    assert int8_dir.is_dir()
    assert not int8_dir.with_name(int8_dir.name + ".tmp").exists()
    assert int8_drive.is_dir()

    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (int8_dir / fname).read_text(encoding="utf-8") == (src / fname).read_text(encoding="utf-8")

    cfg = json.loads((int8_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["dtype"] == "float16"
    assert cfg["torch_dtype"] == "float16"
    assert cfg["quantization_config"] == {
        "quant_method": "gptq",
        "bits": 8,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        "checkpoint_format": "gptq",
    }

    index = json.loads((int8_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert "stale" not in index
    wmap = index["weight_map"]

    quantized_bases = [
        "model.layers.0.mixer.q_proj",
        "model.layers.0.mixer.k_proj",
        "model.layers.1.mlp.gate_proj",
        "model.layers.1.mlp.down_proj",
    ]
    untouched = [
        "model.embeddings.weight",
        "model.layers.0.mixer.g_proj.weight",  # pulado via --skip g_proj
        "model.layers.0.mixer.sqk",
        "model.layers.0.attn_alpha",
        "lm_head.weight",
        "lm_head.logit_scale",
    ]
    expected_keys = set(untouched)
    for base in quantized_bases:
        for suf in ("qweight", "qzeros", "scales"):
            expected_keys.add(f"{base}.{suf}")
    assert set(wmap.keys()) == expected_keys

    loaded: dict[str, torch.Tensor] = {}
    total_bytes = 0
    for shard_name in set(wmap.values()):
        with safe_open(str(int8_dir / shard_name), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                ten = sf.get_tensor(k)
                loaded[k] = ten
                total_bytes += ten.numel() * ten.element_size()

    assert set(loaded.keys()) == expected_keys
    assert index["metadata"]["total_size"] == total_bytes

    for k in untouched:
        assert loaded[k].dtype == orig_tensors[k].dtype
        assert torch.equal(loaded[k], orig_tensors[k])

    for base in quantized_bases:
        sub = {s: loaded[f"{base}.{s}"] for s in ("qweight", "qzeros", "scales")}
        q, zeros, w_rec = _unpack_gptq_8bit(sub, zero_offset=1)
        assert int(q.min()) >= 1 and int(q.max()) <= 255
        assert torch.all(zeros == 127)
        w_orig = orig_tensors[f"{base}.weight"].float()
        rel = ((w_rec - w_orig).norm() / w_orig.norm()).item()
        assert rel < 0.01

    rep = json.loads((out_dir / "quantize.json").read_text(encoding="utf-8"))
    assert rep["n_quantizados"] == 4
    assert rep["lineares_mantidos_fp16"] == ["model.layers.0.mixer.g_proj.weight"]
    assert rep["zero_offset"] == 1
    assert 0.002 <= rep["erro_relativo_medio"] <= 0.01
    assert rep["erro_relativo_max"] >= rep["erro_relativo_medio"]


def test_cmd_quantize_disk_space_checks(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    src = tmp_path / "hf_snapshot"
    _make_fake_dory_checkpoint(src, single_file=True)

    out_dir = tmp_path / "out_results"
    int8_dir = tmp_path / "local" / "int8_local"
    int8_drive = tmp_path / "drive" / "checkpoints" / "int8_drive"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda _r: str(src)))
    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "INT8_DIR", int8_dir)
    monkeypatch.setattr(t4, "INT8_DRIVE", int8_drive)

    # Caso 1: disco local cheio -> falha antes de quantizar
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: types.SimpleNamespace(total=1000, used=990, free=10))
    args = types.SimpleNamespace(out=str(int8_dir), skip=None, save_drive=True)
    with pytest.raises(SystemExit, match="Espaço local insuficiente"):
        t4.cmd_quantize(args)
    assert not int8_dir.exists()

    # Caso 2: disco local suficiente, mas Drive sem espaço -> avisa e pula cópia ao Drive sem falhar
    def fake_disk_usage(p: Path):
        if "drive" in str(p):
            return types.SimpleNamespace(total=1000, used=990, free=10)
        return types.SimpleNamespace(total=10**11, used=10**9, free=10**11)

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)
    t4.cmd_quantize(args)
    assert (int8_dir / "config.json").is_file()
    assert not int8_drive.exists()
    out_txt = capsys.readouterr().out
    assert "AVISO: espaço insuficiente no Drive" in out_txt


# --------------------------------------------------------------------------- 3. _row_metrics, _paired, cmd_report
def _sample_row(pairs: list[tuple[int, bool, bool]]) -> dict:
    samples = [
        {"n_total_tokens": n, "passed": ok, "truncated": trunc}
        for n, ok, trunc in pairs
    ]
    p_hat = sum(1 for _, ok, _ in pairs if ok) / len(pairs)
    return {"p_hat": p_hat, "samples": samples}


def test_row_metrics_and_paired(t4) -> None:
    m1 = t4._row_metrics(_sample_row([(500, False, True), (300, True, False), (200, True, False)]))
    assert m1["p_hat"] == pytest.approx(2 / 3)
    assert m1["solved"] == 1.0
    assert m1["tokens"] == pytest.approx(1000 / 3)
    assert m1["truncated"] == pytest.approx(1 / 3)
    assert m1["tokens_correct"] == 200

    m2 = t4._row_metrics(_sample_row([(1000, False, True), (1000, False, True)]))
    assert m2["solved"] == 0.0
    assert m2["tokens_correct"] is None

    res = t4._paired([0.8, 0.6, 0.4, 1.0], [0.75, 0.55, 0.35, 0.95], n_boot=200)
    assert res["ref"] == pytest.approx(0.7)
    assert res["t4"] == pytest.approx(0.65)
    assert res["delta"] == pytest.approx(-0.05)
    assert res["ic95"] == [-0.05, -0.05]

    res_mix = t4._paired([0.0, 0.5, 1.0, 0.5], [0.25, 0.5, 0.75, 0.75], n_boot=500)
    assert res_mix["ic95"][0] <= res_mix["delta"] <= res_mix["ic95"][1]


def _write_progress_jsonl(path: Path, rows: dict[str, dict]) -> None:
    from lib.io import ResumableProgress

    prog = ResumableProgress(path)
    for item_id, payload in rows.items():
        prog.record(item_id, payload)


def test_cmd_report_missing_files_does_not_crash(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    out_dir = tmp_path / "out_empty"
    out_dir.mkdir()
    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "REF_SAMPLES", tmp_path / "f1a" / "ref.jsonl")

    t4.cmd_report(types.SimpleNamespace())
    rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert rep["env"] is None
    assert rep["smoke"] is None
    assert rep["equivalencia"] == {}
    assert rep["qualidade"] == {}
    captured = capsys.readouterr().out
    assert "## Equivalência" in captured


def test_cmd_report_full_and_missing_optional_fields(
    t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    out_dir = tmp_path / "out_full"
    out_dir.mkdir()
    ref_path = tmp_path / "f1a" / "samples_ref.jsonl"
    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "REF_SAMPLES", ref_path)

    (out_dir / "equiv_int8_L2.json").write_text(
        json.dumps(
            {
                "overall_divergence_rate": 0.0125,
                "max_divergence_rate_threshold": 0.05,
                "passed": True,
                "overall_mean_abs_logprob_diff": 0.0234,
                "referencia": "cpu-fp32",
            }
        ),
        encoding="utf-8",
    )

    ref_rows = {
        "p1": _sample_row([(100, True, False), (200, False, False)]),
        "p2": _sample_row([(300, False, True), (300, False, True)]),
        "p3": _sample_row([(150, True, False), (250, True, False)]),
    }
    cand_no_overlap = {
        "p1": _sample_row([(120, False, False), (220, False, False)]),
        "p2": _sample_row([(280, True, False), (320, True, False)]),
        "p3": _sample_row([(180, False, False), (260, False, False)]),
    }
    _write_progress_jsonl(ref_path, ref_rows)
    _write_progress_jsonl(out_dir / "quality" / "samples_efficient_int8.jsonl", cand_no_overlap)

    t4.cmd_report(types.SimpleNamespace())
    rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    q = rep["qualidade"]["int8"]
    assert q["n_problemas"] == 3
    assert "menor_trace_correto" not in q
    assert "throughput" not in q
    assert q["aprovado"] is False  # p_hat delta = -1/6 < -0.05

    cand_pass = {
        "p1": _sample_row([(90, True, False), (200, False, False)]),
        "p2": _sample_row([(300, False, True), (300, False, True)]),
        "p3": _sample_row([(140, True, False), (250, True, False)]),
    }
    q_path = out_dir / "quality" / "samples_efficient_int8.jsonl"
    q_path.unlink()
    _write_progress_jsonl(q_path, cand_pass)
    (out_dir / "quality" / "throughput_int8.jsonl").write_text(
        json.dumps({"tokens": 2000, "elapsed_s": 120, "load_s": 20}) + "\n",
        encoding="utf-8",
    )

    t4.cmd_report(types.SimpleNamespace())
    rep2 = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    q2 = rep2["qualidade"]["int8"]
    assert q2["menor_trace_correto"]["t4"] == pytest.approx(115.0)
    assert q2["throughput"]["t4_tok_s"] == pytest.approx(20.0)
    assert q2["throughput"]["estimativa_2060s_tok_s"] == pytest.approx(28.0)
    assert q2["aprovado"] is True
    out_txt = capsys.readouterr().out
    assert "APROVADO" in out_txt


@pytest.mark.parametrize(
    ("ref_p", "cand_p", "expected_ok"),
    [
        ([1.0, 0.0], [0.95, 0.0], True),
        ([0.5, 0.5], [0.40, 0.5], True),
        ([0.5, 0.5], [0.39, 0.5], False),
        ([0.5, 0.5], [0.5, 0.0], False),
    ],
)
def test_cmd_report_verdict_rule(
    t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ref_p: list[float], cand_p: list[float], expected_ok: bool
) -> None:
    out_dir = tmp_path / "out_v"
    ref_path = tmp_path / "ref_v.jsonl"
    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "REF_SAMPLES", ref_path)

    def make_rows(probs: list[float]) -> dict[str, dict]:
        rows = {}
        for idx, p in enumerate(probs):
            n_ok = round(p * 100)
            samples = [{"n_total_tokens": 100, "passed": i < n_ok, "truncated": False} for i in range(100)]
            rows[f"id_{idx}"] = {"p_hat": n_ok / 100, "samples": samples}
        return rows

    _write_progress_jsonl(ref_path, make_rows(ref_p))
    _write_progress_jsonl(out_dir / "quality" / "samples_efficient_int8.jsonl", make_rows(cand_p))
    t4.cmd_report(types.SimpleNamespace())
    rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert rep["qualidade"]["int8"]["aprovado"] is expected_ok


# --------------------------------------------------------------------------- 4. main() / argparse
@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["env", "--help"],
        ["quantize", "--help"],
        ["dump", "--help"],
        ["compare", "--help"],
        ["smoke", "--help"],
        ["quality", "--help"],
        ["report", "--help"],
    ],
)
def test_cli_help_all_subcommands(argv: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower()


def test_kernel_check_removed(t4) -> None:
    assert not hasattr(t4, "cmd_kernel_check")
    assert not hasattr(t4, "_call_gptq_gemm")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "kernel-check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_main_argparse_defaults_and_dispatch(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "cli_out"
    monkeypatch.setattr(t4, "OUT", out_dir)
    seen: list[tuple[str, object]] = []

    for cmd_name, attr in (
        ("env", "cmd_env"),
        ("quantize", "cmd_quantize"),
        ("dump", "cmd_dump"),
        ("compare", "cmd_compare"),
        ("smoke", "cmd_smoke"),
        ("quality", "cmd_quality"),
        ("report", "cmd_report"),
    ):
        monkeypatch.setattr(t4, attr, lambda a, name=cmd_name: seen.append((name, a)))

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "quantize"])
    t4.main()
    assert seen[-1][0] == "quantize"
    assert seen[-1][1].out == str(t4.INT8_DIR)
    assert seen[-1][1].save_drive is True
    assert seen[-1][1].skip is None

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "quantize", "--no-save-drive", "--skip", "g_proj", "k_proj"])
    t4.main()
    assert seen[-1][1].save_drive is False
    assert seen[-1][1].skip == ["g_proj", "k_proj"]

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "dump", "--variant", "int8", "--loops", "2"])
    t4.main()
    assert seen[-1][0] == "dump"
    assert seen[-1][1].loops == 2
    assert seen[-1][1].force is False
    assert seen[-1][1].attn_backend is None
    assert seen[-1][1].enforce_eager is True

    monkeypatch.setattr(
        sys, "argv",
        ["t4_validate.py", "dump", "--variant", "int8", "--loops", "2", "--attn-backend", "FLEX_ATTENTION", "--no-enforce-eager"],
    )
    t4.main()
    assert seen[-1][1].attn_backend == "FLEX_ATTENTION"
    assert seen[-1][1].enforce_eager is False

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "compare"])
    t4.main()
    assert seen[-1][0] == "compare"
    assert seen[-1][1].ref == "auto"

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "smoke", "--variant", "int8"])
    t4.main()
    assert seen[-1][0] == "smoke"
    assert seen[-1][1].variant == "int8"
    assert seen[-1][1].n_problems == 32
    assert seen[-1][1].enforce_eager is True
    assert seen[-1][1].attn_backend is None

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "quality", "--variant", "int8"])
    t4.main()
    assert seen[-1][0] == "quality"
    assert seen[-1][1].n_problems == 32
    assert seen[-1][1].batch == 2
    assert seen[-1][1].enforce_eager is True
    assert seen[-1][1].attn_backend is None

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "quality", "--variant", "int8", "--no-enforce-eager"])
    t4.main()
    assert seen[-1][1].enforce_eager is False

    monkeypatch.setattr(sys, "argv", ["t4_validate.py", "dump", "--variant", "int8", "--loops", "3"])
    with pytest.raises(SystemExit):
        t4.main()


# --------------------------------------------------------------------------- 5. Estática, patch check, smoke & compare fallback
def test_py_compile_and_static_checks() -> None:
    py_files = sorted((T4_ROOT / "scripts").glob("*.py")) + sorted((T4_ROOT / "tests").glob("*.py"))
    assert len(py_files) >= 4
    for path in py_files:
        py_compile.compile(str(path), doraise=True)

    for script in sorted((T4_ROOT / "scripts").glob("*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                kw = {k.arg for k in node.keywords}
                assert "encoding" in kw, f"{script.name}: .open() sem encoding na linha {node.lineno}"


def test_require_dory_patch_fails_early(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import patch_vllm_dory

    monkeypatch.setattr(patch_vllm_dory, "check_installed", lambda: (False, Path("/fake/dory.py"), "PENDENTE"))
    with pytest.raises(SystemExit, match="Patch do Dory"):
        t4.cmd_dump(types.SimpleNamespace(variant="fp16", loops=2, force=False, attn_backend=None, enforce_eager=True))
    with pytest.raises(SystemExit, match="Patch do Dory"):
        t4.cmd_smoke(types.SimpleNamespace(variant="int8", n_problems=32, attn_backend=None, enforce_eager=True))


def test_variant_kwargs_and_build_llm_kwargs(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    int8_dir = tmp_path / "int8"
    int8_dir.mkdir()
    (int8_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(t4, "INT8_DIR", int8_dir)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: types.SimpleNamespace(total_memory=16 * 2**30))

    kw_default = t4._build_llm_kwargs("int8", max_model_len=12288, n_loops=2, enforce_eager=True, attn_backend=None)
    assert kw_default["max_num_seqs"] == 16
    assert kw_default["max_num_batched_tokens"] == 2048
    assert kw_default["enforce_eager"] is True
    assert kw_default["max_model_len"] == 12288
    assert "attention_config" not in kw_default

    kw_attn = t4._build_llm_kwargs("int8", max_model_len=2048, n_loops=2, enforce_eager=False, attn_backend="TRITON_ATTN")
    assert kw_attn["enforce_eager"] is False
    assert kw_attn["attention_config"] == {"backend": "TRITON_ATTN"}


def test_cmd_smoke_writes_json_with_kv_and_cost_estimate(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lib.dory as dory_mod

    out_dir = tmp_path / "smoke_out"
    int8_dir = tmp_path / "int8"
    int8_dir.mkdir()
    (int8_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "INT8_DIR", int8_dir)
    monkeypatch.setattr(t4, "_require_dory_patch", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: types.SimpleNamespace(total_memory=16 * 2**30))
    monkeypatch.setattr(dory_mod, "load_dory_tokenizer", lambda _repo: types.SimpleNamespace(
        apply_chat_template=lambda msgs, **kw: f"user:{msgs[0]['content']}"
    ))
    monkeypatch.setattr(dory_mod, "official_sampling", lambda _repo: {
        "temperature": 1.0, "top_p": 0.95, "top_k": 0, "presence_penalty": 0.0
    })

    captured_sp = {}
    captured_llm_kw = {}

    class DummySamplingParams:
        def __init__(self, **kwargs):
            captured_sp.update(kwargs)

    class DummyLLM:
        def __init__(self, **kwargs):
            captured_llm_kw.update(kwargs)
            logging.getLogger("vllm").info(
                "GPU KV cache size: 49,152 tokens, Maximum concurrency for 12,288 tokens per request: 4.00x"
            )
            self.llm_engine = types.SimpleNamespace(
                vllm_config=types.SimpleNamespace(
                    cache_config=types.SimpleNamespace(num_gpu_blocks=3072, block_size=16),
                    attention_config=types.SimpleNamespace(backend=None),
                )
            )

        def generate(self, prompts, sp):
            assert len(prompts) == 2
            return [
                types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=list(range(512))) for _ in range(sp.get("n", 8) if isinstance(sp, dict) else captured_sp["n"])])
                for _ in prompts
            ]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = DummyLLM
    fake_vllm.SamplingParams = DummySamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    times = iter([0.0, 2.0, 2.0, 12.0])
    monkeypatch.setattr(t4.time, "perf_counter", lambda: next(times))

    t4.cmd_smoke(types.SimpleNamespace(variant="int8", n_problems=32, attn_backend=None, enforce_eager=True))

    smoke_file = out_dir / "smoke_int8.json"
    assert smoke_file.is_file()
    data = json.loads(smoke_file.read_text(encoding="utf-8"))
    assert data["n_seqs"] == 16
    assert data["max_tokens"] == 512
    assert data["tokens_generated"] == 16 * 512
    assert data["num_gpu_blocks"] == 3072
    assert data["kv_cache_size_tokens"] == 49152
    assert data["max_concurrency"] == 4.0
    assert data["tok_s"] == pytest.approx(819.2)
    assert data["est_tokens"] == 32 * 8 * 6300
    assert data["t4_cu_per_hour"] == 2.0
    assert data["est_cu"] > 0 and data["est_hours"] > 0
    assert captured_llm_kw["max_model_len"] == 12288
    assert captured_sp["temperature"] == 1.0 and captured_sp["top_p"] == 0.95


def test_cmd_compare_vllm_runner_and_gpu_fallback_to_cpu_bf16(
    t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    out_dir = tmp_path / "cmp_out"
    dumps_dir = out_dir / "dumps"
    dumps_dir.mkdir(parents=True)
    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "_ram_gib", lambda: 12.7)

    d1 = {"variant": "fp16", "n_loops": 1, "raw_prompts": ["p1"], "items": [{"tag": "dump1"}]}
    d2 = {"variant": "int8", "n_loops": 2, "raw_prompts": ["p2"], "items": [{"tag": "dump2"}]}
    (dumps_dir / "fp16_L1.json").write_text(json.dumps(d1), encoding="utf-8")
    (dumps_dir / "int8_L2.json").write_text(json.dumps(d2), encoding="utf-8")

    import lib.dory as dory_mod

    load_calls = []

    def fake_load_dory(_repo, *, n_loops, torch_dtype, device_map):
        load_calls.append((device_map, torch_dtype))
        return ("fake_model", "fake_tok")

    monkeypatch.setattr(dory_mod, "load_dory", fake_load_dory)
    captured_runners = []

    def fake_verify(**kwargs):
        # Simula erro no primeiro forward quando rodando em cuda/bf16
        if load_calls[-1][0] == "cuda":
            raise RuntimeError("SDPA bf16 unsupported on sm_75")
        captured_runners.append((kwargs["loops"], kwargs["max_divergence_rate"], kwargs["vllm_runner"]))
        return {
            "passed": True,
            "max_divergence_rate_threshold": kwargs["max_divergence_rate"],
            "overall_divergence_rate": 0.0,
            "overall_mean_abs_logprob_diff": 0.0,
        }

    monkeypatch.setattr(dory_mod, "verify_vllm_hf_equivalence", fake_verify)
    t4.cmd_compare(types.SimpleNamespace(ref="auto"))

    assert load_calls == [("cuda", torch.bfloat16), ("cpu", torch.bfloat16)]
    assert len(captured_runners) == 2
    by_loop = {loops[0]: (lim, fn(["p"], loops[0], 32)) for loops, lim, fn in captured_runners}
    assert by_loop[1] == (0.02, [{"tag": "dump1"}])
    assert by_loop[2] == (0.05, [{"tag": "dump2"}])
    saved = json.loads((out_dir / "equiv_int8_L2.json").read_text(encoding="utf-8"))
    assert saved["referencia"] == "cpu-bf16"
    out_txt = capsys.readouterr().out
    assert "caindo para cpu-bf16" in out_txt
    assert "High-RAM" in out_txt


def test_cmd_quality_requires_drive_files(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t4, "PROBLEMS", tmp_path / "missing_problems.jsonl")
    monkeypatch.setattr(t4, "REF_SAMPLES", tmp_path / "missing_ref.jsonl")
    args = types.SimpleNamespace(variant="int8", n_problems=32, batch=2, enforce_eager=True, attn_backend=None)
    with pytest.raises(SystemExit, match="não encontrado"):
        t4.cmd_quality(args)


def test_cmd_quality_new_slice_on_resumed_progress(t4, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lib import f1a
    from lib.io import ResumableProgress
    import lib.dory as dory_mod

    out_dir = tmp_path / "q_out"
    problems_path = tmp_path / "problems.jsonl"
    ref_path = tmp_path / "ref.jsonl"
    int8_dir = tmp_path / "int8"
    int8_dir.mkdir()
    (int8_dir / "config.json").write_text("{}", encoding="utf-8")

    problems = [{"task_id": f"t/{i}", "source_idx": i, "problem": f"p{i}", "tests": []} for i in range(4)]
    problems_path.write_text("\n".join(json.dumps(p) for p in problems) + "\n", encoding="utf-8")
    _write_progress_jsonl(ref_path, {f"t/{i}": _sample_row([(10, True, False)]) for i in range(4)})

    sample_file = out_dir / "quality" / "samples_efficient_int8.jsonl"
    _write_progress_jsonl(
        sample_file,
        {
            "t/0": _sample_row([(999, True, False)]),
            "t/2": _sample_row([(888, False, False)]),
        },
    )

    monkeypatch.setattr(t4, "OUT", out_dir)
    monkeypatch.setattr(t4, "PROBLEMS", problems_path)
    monkeypatch.setattr(t4, "REF_SAMPLES", ref_path)
    monkeypatch.setattr(t4, "INT8_DIR", int8_dir)
    monkeypatch.setattr(t4, "_require_dory_patch", lambda: None)

    fake_vllm = types.ModuleType("vllm")

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            pass

    fake_vllm.LLM = DummyLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: types.SimpleNamespace(total_memory=16 * 2**30))
    monkeypatch.setattr(dory_mod, "load_dory_tokenizer", lambda _repo: object())
    monkeypatch.setattr(dory_mod, "official_sampling", lambda _repo: {"temperature": 1.0, "top_p": 0.95})

    def fake_generate_samples(pending, tok, prog: ResumableProgress, stage: str, *, temperature: float, top_p: float):
        assert len(pending) == 2
        for p in pending:
            prog.record(p["task_id"], _sample_row([(50, True, False), (60, False, False)]))

    monkeypatch.setattr(f1a, "generate_samples", fake_generate_samples)

    args = types.SimpleNamespace(variant="int8", n_problems=4, batch=2, enforce_eager=True, attn_backend=None)
    t4.cmd_quality(args)

    tp = [json.loads(line) for line in (out_dir / "quality" / "throughput_int8.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(tp) == 1
    assert tp[0]["problemas"] == 2
    assert tp[0]["tokens"] == 220
