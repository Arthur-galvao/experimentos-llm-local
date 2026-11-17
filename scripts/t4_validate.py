"""Validação na Turing (Colab T4, sm_75) antes de usar as RTX 2060 SUPER do laboratório.

A T4 é da mesma geração das 2060S (sm_75: sem BF16 nativo, sem FlashAttention 2), então o que
funcionar aqui funciona lá. O teste responde quatro perguntas, em ordem:

1. O Dory roda no vLLM em FP16, e sem mudar as previsões? (`dump --variant fp16` + `compare`)
   Mesmo critério do F0b: top-1 teacher-forced diverge em no máximo 2% das posições, por loop.
2. Um INT8 (GPTQ W8A16, pesos arredondados por RTN, sem calibração) cabe em ~7,4 GiB e roda?
   (`quantize` + `dump --variant int8` + `compare`)
3. Cabe o contexto de 12288 tokens do `quality` com folga de KV e qual o custo estimado na T4?
   (`smoke --variant int8`)
4. O INT8 gera dados da F1a com a mesma qualidade da A100? (`quality` + `report`)
   Mesmos problemas e mesma configuração do estágio `efficient` já rodado na A100, comparação pareada
   por problema. É isso que decide se o lab pode gerar a F1a em escala.

Uso (cada comando que abre a GPU roda num processo próprio, como no F0b):
  python scripts/t4_validate.py env
  python scripts/t4_validate.py quantize
  python scripts/t4_validate.py dump --variant int8 --loops 2   # idem 1 e 4
  python scripts/t4_validate.py dump --variant fp16 --loops 2   # idem 1 e 4
  python scripts/t4_validate.py compare
  python scripts/t4_validate.py smoke --variant int8
  python scripts/t4_validate.py quality --variant int8 --n-problems 32
  python scripts/t4_validate.py report

Tudo vai para `results/t4_validation/` no Drive e retoma de onde parou.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/root/.cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/.cache/huggingface/datasets")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

DRIVE_ROOT = Path(os.environ.get("CODER_DRIVE_ROOT", "/content/drive/MyDrive/coder-comprimido"))
OUT = DRIVE_ROOT / "results" / "t4_validation"
F1A_DIR = DRIVE_ROOT / "data" / "f1a"
REF_SAMPLES = F1A_DIR / "samples_efficient_L2_n8_t1_p0.95_m8192.jsonl"  # A100, BF16
PROBLEMS = F1A_DIR / "problems_v2.jsonl"
INT8_DIR = Path(os.environ.get("DORY_INT8_DIR", "/content/dory_int8_gptq"))
INT8_DRIVE = DRIVE_ROOT / "checkpoints" / "dory_int8_gptq_rtn_g128"

# VRAM que o vLLM pode usar numa 2060S (8 GiB menos o que o Ubuntu/driver ocupa). Na T4 isso vira
# gpu_memory_utilization = LAB_VRAM_GIB / VRAM_total, para o INT8 rodar aqui com o aperto de lá.
LAB_VRAM_GIB = float(os.environ.get("LAB_VRAM_GIB", "7.4"))
GROUP_SIZE = 128
BITS = 8
# Em sm_75, quantization="gptq" é redirecionado para AutoGPTQ e o kernel escolhido é Marlin (uint8b128
# simétrico; ignora qzeros e usa zero fixo 128). O exllama (ops.gptq_gemm) é fallback e soma +1 ao qzeros
# (formato v1). Guardar zero = 127 (zero_offset = 1) atende aos dois caminhos.
ZERO_OFFSET = 1
# Lineares das subcamadas de atenção (q, k, v, gate de saída g, o) e de MLP. Embeddings e lm_head ficam em FP16.
LINEAR_RE = re.compile(r"\.(q|k|v|g|o|gate|up|down)_proj\.weight$")
DIV_LIMIT = {"fp16": 0.02, "int8": 0.05}  # fp16: critério do F0b; int8: só informativo, quem decide é o `quality`
QUALITY_MAX_DROP = 0.05  # queda máxima aceita em pass@1 médio e em fração de problemas resolvidos

QUALITY_MAX_MODEL_LEN = 12288
QUALITY_EST_TOKENS_PER_SAMPLE = 6300
SMOKE_N_SEQS = 16
SMOKE_MAX_TOKENS = 512
T4_CU_PER_HOUR = 2.0


def _load_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _ram_gib() -> float:
    if hasattr(os, "sysconf"):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return 0.0


def _require_dory_patch() -> None:
    """Garante antes de subir o vLLM que o patch GPTQ/SupportsQuant em dory.py está aplicado."""
    from scripts.patch_vllm_dory import check_installed

    try:
        ok, path, msg = check_installed()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Falha ao verificar patch do Dory no vLLM: {e}")
    if not ok:
        sys.exit(
            f"Patch do Dory no vLLM não aplicado em {path} ({msg}). "
            "Rode `python scripts/patch_vllm_dory.py` antes."
        )


# --------------------------------------------------------------------------------------------- env
def cmd_env(_a) -> None:
    import torch

    info = {"torch": torch.__version__, "ram_gib": round(_ram_gib(), 1)}
    try:
        import vllm

        info["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        info["vllm"] = f"ERRO: {e}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            gpu=p.name,
            capability=f"{p.major}.{p.minor}",
            vram_gib=round(p.total_memory / 2**30, 2),
            bf16_nativo=p.major >= 8,
        )
        # BF16 em sm_75 é emulado; o `compare` usa o HF em BF16 como referência se a conta der certo.
        try:
            a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            err = ((a @ a).float() - a.float() @ a.float()).abs().max().item()
            info["bf16_matmul_ok"] = bool(err < 1.0)
        except Exception as e:  # noqa: BLE001
            info["bf16_matmul_ok"] = False
            info["bf16_matmul_erro"] = str(e)[:200]
    _save_json(OUT / "env.json", info)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if info.get("capability") != "7.5":
        print(f"AVISO: GPU {info.get('gpu')} (sm_{info.get('capability')}) não é Turing; o teste perde o sentido.")


# ------------------------------------------------------------------------------------- quantização
def _to_int32(x):
    import torch

    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def quantize_rtn_int8(w, group_size: int = GROUP_SIZE, zero_offset: int = ZERO_OFFSET):
    """RTN simétrico por grupo de `group_size` entradas, no formato GPTQ (desc_act=False, sem g_idx).

    `w` é [out, in]. Valor guardado q ∈ [1, 255] com zero real 128: W ≈ scale · (q − 128).
    `zero_offset=1` (padrão) grava qzeros = 127: no caminho AutoGPTQ/Marlin (sm_75) o tipo uint8b128
    ignora qzeros e usa zero fixo 128, enquanto o fallback exllama (v1) soma +1 ao qzeros guardado.
    Devolve (tensores GPTQ, W dequantizado para conferência).
    """
    import torch

    out_f, in_f = w.shape
    n_groups = in_f // group_size
    wf = w.float().reshape(out_f, n_groups, group_size)
    # [out, G], já arredondado a FP16. O piso vem antes do .half(): 1e-8/127 viraria 0 em FP16 e daria NaN.
    scale = (wf.abs().amax(-1) / 127.0).clamp(min=1e-6).half().float()
    q = (torch.round(wf / scale[..., None]).clamp(-127, 127) + 128).to(torch.int64)  # [out, G, gs]
    deq = ((q - 128).float() * scale[..., None]).reshape(out_f, in_f)
    q = q.reshape(out_f, in_f).t().contiguous()  # [in, out]

    pack = 32 // BITS
    qr = q.reshape(in_f // pack, pack, out_f)
    qweight = sum(qr[:, k] << (BITS * k) for k in range(pack))
    zeros = torch.full((n_groups, out_f // pack, pack), 128 - zero_offset, dtype=torch.int64, device=w.device)
    qzeros = sum(zeros[..., k] << (BITS * k) for k in range(pack))
    tensors = {
        "qweight": _to_int32(qweight),
        "qzeros": _to_int32(qzeros),
        "scales": scale.t().contiguous().half(),
    }
    return tensors, deq


def cmd_quantize(a) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    from lib.dory import DORY_HF_REPO

    zero_offset = ZERO_OFFSET
    skip = set(a.skip or [])
    src = Path(snapshot_download(DORY_HF_REPO))
    safetensor_files = sorted(src.glob("*.safetensors"))
    if not safetensor_files:
        sys.exit(f"Nenhum arquivo .safetensors encontrado em {src}.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src_bytes = sum(f.stat().st_size for f in safetensor_files)
    free_local = shutil.disk_usage(out.parent).free
    need_local = max(1, int(src_bytes * 0.75))
    if free_local < need_local:
        sys.exit(
            f"Espaço local insuficiente em {out.parent}: {free_local / 2**30:.2f} GiB livres, "
            f"necessário ~{need_local / 2**30:.2f} GiB."
        )

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    shard: dict = {}
    state = {"bytes": 0, "n": 0, "total": 0}
    weight_map: dict[str, str] = {}

    def flush() -> None:
        if not shard:
            return
        state["n"] += 1
        name = f"model-{state['n']:05d}.safetensors"
        save_file(shard, str(tmp / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in shard})
        shard.clear()
        state["bytes"] = 0

    def add(name: str, t) -> None:
        t = t.contiguous().cpu()
        nbytes = t.numel() * t.element_size()
        shard[name] = t
        state["bytes"] += nbytes
        state["total"] += nbytes
        if state["bytes"] >= 2 * 2**30:
            flush()

    rel_errs, kept_linear, n_q = [], [], 0
    for f in safetensor_files:
        with safe_open(str(f), framework="pt", device=dev) as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                m = LINEAR_RE.search(name)
                proj = f"{m.group(1)}_proj" if m else None
                quantizable = (
                    m is not None and t.ndim == 2 and proj not in skip
                    and t.shape[1] % GROUP_SIZE == 0 and t.shape[0] % (32 // BITS) == 0
                )
                if not quantizable:
                    if m is not None:
                        kept_linear.append(name)
                    add(name, t)
                    continue
                tensors, deq = quantize_rtn_int8(t, GROUP_SIZE, zero_offset)
                rel_errs.append(((deq - t.float()).norm() / t.float().norm()).item())
                prefix = name[: -len(".weight")]
                for suffix, v in tensors.items():
                    add(f"{prefix}.{suffix}", v)
                n_q += 1
    flush()
    if n_q == 0:
        sys.exit("Nenhum linear casou com LINEAR_RE: confira os nomes dos tensores do checkpoint.")

    _save_json(tmp / "model.safetensors.index.json", {"metadata": {"total_size": state["total"]}, "weight_map": weight_map})
    for p in src.iterdir():  # tokenizer, chat template, generation_config, código remoto
        if p.suffix != ".safetensors" and p.name not in ("model.safetensors.index.json", "config.json") and p.is_file():
            shutil.copy(p, tmp / p.name)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["dtype"] = "float16"
    cfg["torch_dtype"] = "float16"
    cfg["quantization_config"] = {
        "quant_method": "gptq",
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        "checkpoint_format": "gptq",
    }
    _save_json(tmp / "config.json", cfg)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)

    rep = {
        "out": str(out),
        "n_quantizados": n_q,
        "lineares_mantidos_fp16": kept_linear,
        "erro_relativo_medio": round(statistics.mean(rel_errs), 5),
        "erro_relativo_max": round(max(rel_errs), 5),
        "tamanho_gib": round(state["total"] / 2**30, 2),
        "zero_offset": zero_offset,
    }
    _save_json(OUT / "quantize.json", rep)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if a.save_drive:
        INT8_DRIVE.parent.mkdir(parents=True, exist_ok=True)
        free_drive = shutil.disk_usage(INT8_DRIVE.parent).free
        need_drive = max(1, int(state["total"] * 1.05))
        if free_drive < need_drive:
            print(
                f"AVISO: espaço insuficiente no Drive ({free_drive / 2**30:.2f} GiB livres, "
                f"necessário ~{need_drive / 2**30:.2f} GiB). Pulando cópia para {INT8_DRIVE}."
            )
        else:
            print(f"Copiando para {INT8_DRIVE} (é o checkpoint que vai para o lab)...")
            shutil.rmtree(INT8_DRIVE, ignore_errors=True)
            shutil.copytree(out, INT8_DRIVE)


# ------------------------------------------------------------------------------- motor vLLM (T4)
def _variant_kwargs(variant: str, *, enforce_eager: bool = True) -> dict:
    import torch

    from lib.dory import DORY_HF_REPO

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if variant == "fp16":
        # 8 GB de pesos em FP16 não cabem numa 2060S sozinha: aqui só se testa a numérica, com a T4 inteira.
        return {"model": DORY_HF_REPO, "dtype": "float16", "gpu_memory_utilization": 0.88}
    if variant == "int8":
        if not (INT8_DIR / "config.json").exists():
            if (INT8_DRIVE / "config.json").exists():
                print(f"Copiando o INT8 do Drive para {INT8_DIR}...")
                shutil.copytree(INT8_DRIVE, INT8_DIR, dirs_exist_ok=True)
            else:
                sys.exit(f"INT8 não encontrado em {INT8_DIR} nem no Drive: rode `quantize` antes.")
        return {
            "model": str(INT8_DIR),
            "dtype": "float16",
            "quantization": "gptq",
            "gpu_memory_utilization": round(min(0.95, LAB_VRAM_GIB / total), 3),
            # Chunked prefill ativo por padrão no vLLM v1 permite max_num_batched_tokens < max_model_len.
            "max_num_seqs": 16,
            "max_num_batched_tokens": 2048,
            "enforce_eager": bool(enforce_eager),
        }
    raise ValueError(variant)


def _build_llm_kwargs(
    variant: str,
    *,
    max_model_len: int,
    n_loops: int = 2,
    enforce_eager: bool = True,
    attn_backend: str | None = None,
) -> dict:
    from lib.gen import build_dory_hf_overrides

    kw = _variant_kwargs(variant, enforce_eager=enforce_eager)
    llm_kwargs = {
        "trust_remote_code": True,
        "seed": 42,
        "max_model_len": int(max_model_len),
        "hf_overrides": build_dory_hf_overrides(n_loops=n_loops),
        **kw,
    }
    llm_kwargs.setdefault("enforce_eager", bool(enforce_eager))
    if attn_backend:
        llm_kwargs["attention_config"] = {"backend": attn_backend}
    return llm_kwargs


def _detect_attn_backend(llm, requested: str | None) -> str:
    try:
        backend = llm.llm_engine.vllm_config.attention_config.backend
        if backend is not None:
            return str(getattr(backend, "name", backend))
    except Exception:  # noqa: BLE001
        pass
    return requested or "padrão"


def _formatted_prompts(tok, prompts) -> list[str]:
    """Mesma formatação do `verify_vllm_hf_equivalence` (chat template, enable_thinking=True)."""
    return [
        str(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True))
        for p in prompts
    ]


def cmd_dump(a) -> None:
    _require_dory_patch()
    from vllm import LLM, SamplingParams

    from lib.dory import DEFAULT_EQUIVALENCE_PROMPTS, DORY_HF_REPO, load_dory_tokenizer

    k = int(a.loops)
    out = OUT / "dumps" / f"{a.variant}_L{k}.json"
    if out.exists() and not a.force:
        print(f"{out} já existe (use --force para refazer).")
        return
    tok = load_dory_tokenizer(DORY_HF_REPO)
    prompts = list(DEFAULT_EQUIVALENCE_PROMPTS)
    formatted = _formatted_prompts(tok, prompts)
    llm_kwargs = _build_llm_kwargs(
        a.variant,
        max_model_len=2048,
        n_loops=k,
        enforce_eager=a.enforce_eager,
        attn_backend=a.attn_backend,
    )
    print(f"[dump] {a.variant} L{k}: {llm_kwargs}")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    load_s = time.time() - t0
    attn_used = _detect_attn_backend(llm, a.attn_backend)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, logprobs=5, seed=42)
    outs = llm.generate(formatted, sp)
    items = []
    for r in outs:
        comp = r.outputs[0]
        gen = list(comp.token_ids)
        steps = []
        for i, tid in enumerate(gen):
            lp = comp.logprobs[i] if comp.logprobs and i < len(comp.logprobs) else {}
            v = lp.get(tid) if lp else None
            steps.append({"top1_id": int(tid), "top1_logprob": float(getattr(v, "logprob", v)) if v is not None else 0.0})
        items.append({"prompt_token_ids": list(r.prompt_token_ids), "generated_token_ids": gen, "steps": steps,
                      "text": comp.text})
    _save_json(out, {
        "variant": a.variant, "n_loops": k, "raw_prompts": prompts, "items": items, "load_s": round(load_s, 1),
        "attention_backend": attn_used, "llm_kwargs": llm_kwargs,
    })
    for it in items:
        print(repr(it["text"][:120]))
    print(f"[dump] OK -> {out}")


# --------------------------------------------------------------------------------------- smoke
class _LogCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001
            pass


def _extract_kv_capacity(llm, log_lines: list[str], max_model_len: int) -> dict:
    num_gpu_blocks = None
    block_size = None
    kv_tokens = None
    max_conc = None
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        num_gpu_blocks = getattr(cc, "num_gpu_blocks", None)
        block_size = getattr(cc, "block_size", None)
        kv_tokens = getattr(cc, "kv_cache_size_tokens", None)
        max_conc = getattr(cc, "kv_cache_max_concurrency", None)
    except Exception:  # noqa: BLE001
        pass

    joined = "\n".join(log_lines)
    if kv_tokens is None or max_conc is None:
        m = re.search(
            r"KV cache size:\s*([\d,]+)\s*tokens.*?Maximum concurrency for\s*[\d,]+\s*tokens per request:\s*([\d.]+)x",
            joined,
            re.DOTALL,
        )
        if m:
            if kv_tokens is None:
                kv_tokens = int(m.group(1).replace(",", ""))
            if max_conc is None:
                max_conc = float(m.group(2))

    if kv_tokens is None and num_gpu_blocks is not None and block_size is not None:
        kv_tokens = int(num_gpu_blocks) * int(block_size)
    if max_conc is None and kv_tokens is not None and max_model_len > 0:
        max_conc = round(float(kv_tokens) / float(max_model_len), 2)

    return {
        "num_gpu_blocks": int(num_gpu_blocks) if num_gpu_blocks is not None else None,
        "block_size": int(block_size) if block_size is not None else None,
        "kv_cache_size_tokens": int(kv_tokens) if kv_tokens is not None else None,
        "max_concurrency": round(float(max_conc), 2) if max_conc is not None else None,
    }


def cmd_smoke(a) -> None:
    _require_dory_patch()
    from vllm import LLM, SamplingParams

    from lib.dory import DEFAULT_EQUIVALENCE_PROMPTS, DORY_HF_REPO, load_dory_tokenizer, official_sampling

    llm_kwargs = _build_llm_kwargs(
        a.variant,
        max_model_len=QUALITY_MAX_MODEL_LEN,
        n_loops=2,
        enforce_eager=a.enforce_eager,
        attn_backend=a.attn_backend,
    )
    print(f"[smoke] {a.variant}: {llm_kwargs}")

    tok = load_dory_tokenizer(DORY_HF_REPO)
    prompts = _formatted_prompts(tok, list(DEFAULT_EQUIVALENCE_PROMPTS[:2]))
    s = official_sampling(DORY_HF_REPO)
    n_samples = SMOKE_N_SEQS // len(prompts)
    sp = SamplingParams(
        n=n_samples,
        temperature=float(s["temperature"]),
        top_p=float(s["top_p"]),
        top_k=int(s["top_k"]) if int(s["top_k"]) > 0 else -1,
        presence_penalty=float(s["presence_penalty"]),
        max_tokens=SMOKE_MAX_TOKENS,
        seed=42,
    )

    vllm_logger = logging.getLogger("vllm")
    collector = _LogCollector()
    vllm_logger.addHandler(collector)
    try:
        t_load = time.perf_counter()
        llm = LLM(**llm_kwargs)
        load_s = time.perf_counter() - t_load
    finally:
        vllm_logger.removeHandler(collector)

    kv_info = _extract_kv_capacity(llm, collector.lines, QUALITY_MAX_MODEL_LEN)
    attn_used = _detect_attn_backend(llm, a.attn_backend)

    t_gen = time.perf_counter()
    outs = llm.generate(prompts, sp)
    gen_s = max(1e-3, time.perf_counter() - t_gen)
    n_tokens = sum(len(comp.token_ids) for r in outs for comp in r.outputs)
    raw_tok_s = n_tokens / gen_s

    est_tokens = int(a.n_problems) * 8 * QUALITY_EST_TOKENS_PER_SAMPLE
    est_hours = est_tokens / max(1e-3, raw_tok_s) / 3600.0
    est_cu = est_hours * T4_CU_PER_HOUR

    rep = {
        "variant": a.variant,
        "n_seqs": len(prompts) * n_samples,
        "max_tokens": SMOKE_MAX_TOKENS,
        "tokens_generated": n_tokens,
        "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 2),
        "tok_s": round(raw_tok_s, 1),
        **kv_info,
        "n_problems_est": int(a.n_problems),
        "est_tokens_per_sample": QUALITY_EST_TOKENS_PER_SAMPLE,
        "est_tokens": est_tokens,
        "est_hours": round(est_hours, 2),
        "est_cu": round(est_cu, 2),
        "t4_cu_per_hour": T4_CU_PER_HOUR,
        "attention_backend": attn_used,
        "llm_kwargs": llm_kwargs,
    }
    out_path = OUT / f"smoke_{a.variant}.json"
    _save_json(out_path, rep)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[smoke] OK -> {out_path}")


# ------------------------------------------------------------------------------ equivalência
def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cmd_compare(a) -> None:
    import torch

    from lib.dory import DORY_HF_REPO, load_dory, verify_vllm_hf_equivalence

    dumps = sorted((OUT / "dumps").glob("*.json"))
    if not dumps:
        sys.exit("Nenhum dump: rode `dump` antes.")
    env = _load_json(OUT / "env.json", {})
    ref_mode = a.ref
    if ref_mode == "auto":
        ref_mode = "gpu-bf16" if env.get("bf16_matmul_ok", True) else ("cpu-fp32" if _ram_gib() > 30 else "cpu-bf16")

    modes = {
        "gpu-bf16": ("cuda", torch.bfloat16),
        "cpu-fp32": ("cpu", torch.float32),
        "cpu-bf16": ("cpu", torch.bfloat16),
    }

    def run_with_mode(mode: str) -> None:
        device, dtype = modes[mode]
        print(f"[compare] referência = HF `lib/dory.py` em {mode}")
        if device == "cpu":
            ram = _ram_gib()
            min_ram = 24.0 if mode == "cpu-fp32" else 16.0
            if ram > 0 and ram < min_ram:
                print(
                    f"AVISO: RAM={ram:.1f} GiB — compare em CPU ({mode}) precisa de Colab High-RAM "
                    f"(>= {min_ram:.0f} GiB) para evitar OOM."
                )
            torch.set_num_threads(os.cpu_count() or 2)
        model = None
        try:
            model, tok = load_dory(DORY_HF_REPO, n_loops=2, torch_dtype=dtype, device_map=device)
            for d in dumps:
                data = json.loads(d.read_text(encoding="utf-8"))
                k, variant = int(data["n_loops"]), data["variant"]
                rep = verify_vllm_hf_equivalence(
                    prompts=data["raw_prompts"], loops=(k,), max_new_tokens=32,
                    max_divergence_rate=DIV_LIMIT[variant], hf_model=model, tokenizer=tok,
                    vllm_runner=lambda _fp, _k, _n, items=data["items"]: items,
                    output_json_path=str(OUT / f"equiv_{d.stem}.json"), halt_on_failure=False,
                )
                rep["referencia"] = mode
                _save_json(OUT / f"equiv_{d.stem}.json", rep)
        finally:
            if model is not None:
                del model
            if device == "cuda":
                _free_gpu()

    try:
        run_with_mode(ref_mode)
    except Exception as e:  # noqa: BLE001
        if ref_mode != "gpu-bf16":
            raise
        print(f"AVISO: gpu-bf16 falhou ({e}); liberando GPU e caindo para cpu-bf16...")
        _free_gpu()
        run_with_mode("cpu-bf16")


# ------------------------------------------------------------------------------ qualidade F1a
def cmd_quality(a) -> None:
    if not PROBLEMS.exists():
        sys.exit(f"Arquivo de problemas não encontrado no Drive: {PROBLEMS}")
    if not REF_SAMPLES.exists():
        sys.exit(f"Arquivo de referência A100 não encontrado no Drive: {REF_SAMPLES}")

    _require_dory_patch()
    import vllm

    from lib import f1a
    from lib.dory import DORY_HF_REPO, load_dory_tokenizer, official_sampling
    from lib.io import ResumableProgress

    kw = _variant_kwargs(a.variant, enforce_eager=a.enforce_eager)
    timing = {"load_s": 0.0}
    orig_llm = vllm.LLM

    class _LLMT4(orig_llm):  # o lib/gen.py faz `from vllm import LLM` dentro da função: o patch pega
        def __init__(self, *args, **kwargs):
            kwargs.update(kw)
            kwargs["enforce_eager"] = bool(a.enforce_eager)
            if getattr(a, "attn_backend", None):
                kwargs["attention_config"] = {"backend": a.attn_backend}
            print(f"[quality] motor vLLM: {kwargs}")
            t0 = time.time()
            super().__init__(*args, **kwargs)
            timing["load_s"] += time.time() - t0

    vllm.LLM = _LLMT4
    f1a.BATCH_PROBLEMS = a.batch  # lotes menores: sessão que cair perde menos

    problems = {p["task_id"]: p for p in map(json.loads, PROBLEMS.open(encoding="utf-8"))}
    ref = ResumableProgress(REF_SAMPLES)
    ids = sorted(r["item_id"] for r in ref.all_records())
    random.Random(42).shuffle(ids)
    ids = ids[: a.n_problems]
    prog = ResumableProgress(OUT / "quality" / f"samples_efficient_{a.variant}.jsonl")
    pending = [problems[i] for i in ids if not prog.is_done(i)]
    print(f"[quality] {a.variant}: {len(ids)} problemas da referência A100, {len(pending)} a gerar.")
    if not pending:
        return
    tok = load_dory_tokenizer(DORY_HF_REPO)
    s = official_sampling(DORY_HF_REPO)
    before = len(prog)
    t0 = time.time()
    f1a.generate_samples(pending, tok, prog, "efficient", temperature=float(s["temperature"]), top_p=float(s["top_p"]))
    elapsed = time.time() - t0
    new = [r for r in prog.all_records()][before:]
    n_tok = sum(smp["n_total_tokens"] for r in new for smp in r["samples"])
    gen_s = max(1.0, elapsed - timing["load_s"])
    run = {"variant": a.variant, "problemas": len(new), "tokens": n_tok, "elapsed_s": round(elapsed),
           "load_s": round(timing["load_s"]), "tok_s": round(n_tok / gen_s, 1), "llm_kwargs": kw,
           "quando": time.strftime("%Y-%m-%d %H:%M:%S")}
    with (OUT / "quality" / f"throughput_{a.variant}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    print(json.dumps(run, indent=2, ensure_ascii=False))


def _row_metrics(row: dict) -> dict:
    smp = row["samples"]
    ok = [x for x in smp if x["passed"]]
    return {
        "p_hat": row["p_hat"],
        "solved": float(row["p_hat"] > 0),
        "tokens": statistics.mean(x["n_total_tokens"] for x in smp),
        "truncated": statistics.mean(float(x["truncated"]) for x in smp),
        "tokens_correct": min((x["n_total_tokens"] for x in ok), default=None),
    }


def _paired(ref: list[float], cand: list[float], n_boot: int = 4000) -> dict:
    diffs = [c - r for r, c in zip(ref, cand)]
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {"ref": round(statistics.mean(ref), 4), "t4": round(statistics.mean(cand), 4),
            "delta": round(statistics.mean(diffs), 4),
            "ic95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)]}


def cmd_report(_a) -> None:
    from lib.io import ResumableProgress

    rep: dict = {"env": _load_json(OUT / "env.json"), "quantize": _load_json(OUT / "quantize.json"),
                 "smoke": _load_json(OUT / "smoke_int8.json"), "equivalencia": {}, "qualidade": {}}
    for p in sorted(OUT.glob("equiv_*.json")):
        e = _load_json(p)
        rep["equivalencia"][p.stem[len("equiv_"):]] = {
            "divergencia": e["overall_divergence_rate"], "limite": e["max_divergence_rate_threshold"],
            "aprovado": e["passed"], "dlogprob_medio": e["overall_mean_abs_logprob_diff"],
            "referencia": e.get("referencia")}

    ref = ResumableProgress(REF_SAMPLES)
    for f in sorted((OUT / "quality").glob("samples_efficient_*.jsonl")):
        variant = f.stem[len("samples_efficient_"):]
        cand = ResumableProgress(f)
        ids = [r["item_id"] for r in cand.all_records() if ref.is_done(r["item_id"])]
        if not ids:
            continue
        mr = [_row_metrics(ref.get(i)) for i in ids]
        mc = [_row_metrics(cand.get(i)) for i in ids]
        q = {"n_problemas": len(ids)}
        for key in ("p_hat", "solved", "tokens", "truncated"):
            q[key] = _paired([m[key] for m in mr], [m[key] for m in mc])
        both = [(r["tokens_correct"], c["tokens_correct"]) for r, c in zip(mr, mc)
                if r["tokens_correct"] is not None and c["tokens_correct"] is not None]
        if both:
            q["menor_trace_correto"] = _paired([b[0] for b in both], [b[1] for b in both])
        runs = [json.loads(line) for line in (OUT / "quality" / f"throughput_{variant}.jsonl").open(encoding="utf-8")] \
            if (OUT / "quality" / f"throughput_{variant}.jsonl").exists() else []
        if runs:
            tok_s = sum(r["tokens"] for r in runs) / max(1, sum(r["elapsed_s"] - r["load_s"] for r in runs))
            # 2060S: 448 GB/s contra 320 GB/s da T4; o decode é limitado por banda. Estimativa, medir no lab.
            q["throughput"] = {"t4_tok_s": round(tok_s, 1), "estimativa_2060s_tok_s": round(tok_s * 1.4, 1),
                               "estimativa_lab_15pcs_tok_s": round(tok_s * 1.4 * 15)}
        q["aprovado"] = q["p_hat"]["delta"] >= -QUALITY_MAX_DROP and q["solved"]["delta"] >= -QUALITY_MAX_DROP
        rep["qualidade"][variant] = q
    _save_json(OUT / "report.json", rep)

    print("\n## Equivalência (top-1 teacher-forced, 5 prompts × 32 tokens)")
    print("| variante | divergência | limite | |Δlogprob| | resultado |\n|---|---|---|---|---|")
    for k, e in rep["equivalencia"].items():
        print(f"| {k} | {e['divergencia']:.2%} | {e['limite']:.0%} | {e['dlogprob_medio']:.4f} | "
              f"{'APROVADO' if e['aprovado'] else 'REPROVADO'} |")
    for v, q in rep["qualidade"].items():
        print(f"\n## Qualidade F1a `efficient`: {v} na T4 × BF16 na A100 ({q['n_problemas']} problemas, pareado)")
        print("| métrica | A100 | T4 | Δ | IC 95% |\n|---|---|---|---|---|")
        for key in ("p_hat", "solved", "tokens", "truncated", "menor_trace_correto"):
            if key in q:
                m = q[key]
                print(f"| {key} | {m['ref']} | {m['t4']} | {m['delta']:+} | {m['ic95']} |")
        if "throughput" in q:
            t = q["throughput"]
            print(f"\nThroughput: T4 {t['t4_tok_s']} tok/s → 2060S ~{t['estimativa_2060s_tok_s']} tok/s (estimado) "
                  f"→ lab ~{t['estimativa_lab_15pcs_tok_s']} tok/s com 15 PCs.")
        print(f"\n**{v}: {'APROVADO' if q['aprovado'] else 'REPROVADO'} para gerar a F1a** "
              f"(queda máxima aceita: {QUALITY_MAX_DROP} em pass@1 e em resolvidos)")


def _add_eager_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true",
                        help="desliga CUDA graphs (padrão: ligado)")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false",
                        help="habilita CUDA graphs (usa mais VRAM)")
    parser.set_defaults(enforce_eager=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    q = sub.add_parser("quantize")
    q.add_argument("--out", default=str(INT8_DIR))
    q.add_argument("--skip", nargs="*", help="projeções a manter em FP16, ex.: g_proj")
    q.add_argument("--no-save-drive", dest="save_drive", action="store_false")
    d = sub.add_parser("dump")
    d.add_argument("--variant", choices=["fp16", "int8"], required=True)
    d.add_argument("--loops", type=int, choices=[1, 2, 4], required=True)
    d.add_argument("--attn-backend", default=None, help="ex.: TRITON_ATTN ou FLEX_ATTENTION")
    _add_eager_flags(d)
    d.add_argument("--force", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--ref", choices=["auto", "gpu-bf16", "cpu-fp32", "cpu-bf16"], default="auto")
    sm = sub.add_parser("smoke")
    sm.add_argument("--variant", choices=["fp16", "int8"], default="int8")
    sm.add_argument("--n-problems", type=int, default=32, help="N problemas para estimativa de custo do quality")
    sm.add_argument("--attn-backend", default=None)
    _add_eager_flags(sm)
    ql = sub.add_parser("quality")
    ql.add_argument("--variant", choices=["fp16", "int8"], required=True)
    ql.add_argument("--n-problems", type=int, default=32)
    ql.add_argument("--batch", type=int, default=2, help="problemas por chamada ao vLLM (× 8 amostras)")
    ql.add_argument("--attn-backend", default=None)
    _add_eager_flags(ql)
    sub.add_parser("report")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"env": cmd_env, "quantize": cmd_quantize, "dump": cmd_dump,
     "compare": cmd_compare, "smoke": cmd_smoke, "quality": cmd_quality, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
