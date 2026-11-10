"""Hardware detection and profile configuration (Section 1.1 of ESPEC_ANTIGRAVITY.md)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

ProfileName = Literal["a100_80", "a100_40"]


@dataclass(frozen=True)
class HardwareInfo:
    profile: ProfileName
    gpu_name: str
    vram_gb: float
    cuda_available: bool


def _query_nvidia_smi() -> tuple[str, float] | None:
    """Query GPU name and total memory (in GiB) using nvidia-smi."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 2:
                gpu_name = parts[0]
                mem_mib = float(parts[1])
                return gpu_name, mem_mib / 1024.0
    except Exception:
        pass
    return None


def detect_hardware_info() -> HardwareInfo:
    """Detect current GPU and determine PROFILE = 'a100_80' | 'a100_40'.

    Anything >= 65 GiB VRAM selects 'a100_80' (A100-80GB / H100-80GB).
    All smaller GPUs (A100-40GB, L4, T4, local GPU) or CPU fallback select 'a100_40'
    so memory-safe settings remain the default.
    """
    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024.0**3)
        profile: ProfileName = "a100_80" if vram_gb >= 65.0 else "a100_40"
        return HardwareInfo(
            profile=profile,
            gpu_name=gpu_name,
            vram_gb=round(vram_gb, 2),
            cuda_available=True,
        )

    smi = _query_nvidia_smi()
    if smi is not None:
        gpu_name, vram_gb = smi
        profile = "a100_80" if vram_gb >= 65.0 else "a100_40"
        return HardwareInfo(
            profile=profile,
            gpu_name=gpu_name,
            vram_gb=round(vram_gb, 2),
            cuda_available=False,
        )

    return HardwareInfo(
        profile="a100_40",
        gpu_name="CPU / No NVIDIA GPU detected",
        vram_gb=0.0,
        cuda_available=False,
    )


def detect_hardware_profile() -> ProfileName:
    """Return 'a100_80' or 'a100_40' based on detected GPU memory."""
    return detect_hardware_info().profile


PROFILE: ProfileName = detect_hardware_profile()


_HW_CONFIGS: dict[ProfileName, dict[str, dict[str, Any]]] = {
    "a100_80": {
        "sft": {
            "precision": "bf16",
            "load_in_4bit": False,
            "peft_type": "lora",
            "lora_r": 32,
            "lora_alpha": 64,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 8192,
            "gradient_checkpointing": True,
            "optim": "adamw_torch_fused",
        },
        "grpo": {
            "precision": "bf16",
            "load_in_4bit": False,
            "peft_type": "lora",
            "lora_r": 32,
            "lora_alpha": 64,
            "num_generations": 8,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_prompt_length": 1024,
            "max_completion_length": 4096,
            "gradient_checkpointing": True,
        },
        "f3_dsl": {
            "precision": "bf16",
            "load_in_4bit": False,
            "peft_type": "lora",
            "lora_r": 32,
            "lora_alpha": 64,
            "modules_to_save": ["embeddings", "lm_head"],
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 8192,
            "gradient_checkpointing": True,
        },
        "f4_coconut": {
            "precision": "bf16",
            "load_in_4bit": False,
            "peft_type": "lora",
            "lora_r": 32,
            "lora_alpha": 64,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 4096,
            "gradient_checkpointing": True,
        },
        "eval": {
            "precision": "bf16",
            "backend": "auto",
            "batch_size": 8,
            "max_new_tokens": 16000,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 42,
        },
    },
    "a100_40": {
        "sft": {
            "precision": "bf16",
            "load_in_4bit": False,
            "peft_type": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 4096,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
        },
        "grpo": {
            "precision": "bf16",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "peft_type": "qlora",
            "lora_r": 16,
            "lora_alpha": 32,
            "num_generations": 4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_prompt_length": 1024,
            "max_completion_length": 3072,
            "gradient_checkpointing": True,
        },
        "f3_dsl": {
            "precision": "bf16",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "peft_type": "qlora",
            "lora_r": 16,
            "lora_alpha": 32,
            "modules_to_save": ["embeddings", "lm_head"],
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_seq_length": 4096,
            "gradient_checkpointing": True,
        },
        "f4_coconut": {
            "precision": "bf16",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "peft_type": "qlora",
            "lora_r": 16,
            "lora_alpha": 32,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_seq_length": 2048,
            "gradient_checkpointing": True,
        },
        "eval": {
            "precision": "bf16",
            "backend": "auto",
            "batch_size": 4,
            "max_new_tokens": 16000,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 42,
        },
    },
}


def get_hw_config(
    task: Literal["sft", "grpo", "f3_dsl", "f4_coconut", "eval"] = "eval",
    profile: ProfileName | None = None,
) -> dict[str, Any]:
    """Return hardware-adjusted configuration dictionary for the given task."""
    active_profile: ProfileName = profile if profile is not None else detect_hardware_profile()
    if active_profile not in _HW_CONFIGS:
        raise ValueError(f"Unknown hardware profile: {active_profile}")
    if task not in _HW_CONFIGS[active_profile]:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(_HW_CONFIGS[active_profile])}")
    return dict(_HW_CONFIGS[active_profile][task])


def print_hardware_summary() -> HardwareInfo:
    """Print detected GPU and active profile."""
    info = detect_hardware_info()
    print(
        f"[hw] GPU: {info.gpu_name} | VRAM: {info.vram_gb:.2f} GiB | "
        f"CUDA: {info.cuda_available} -> PROFILE = '{info.profile}'"
    )
    return info
