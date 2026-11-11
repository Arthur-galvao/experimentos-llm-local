"""Google Drive persistence, HF_HOME setup, and resumable checkpointing (Section 1.2)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/coder-comprimido"
DEFAULT_LOCAL_HF_HOME = "/root/.cache/huggingface"


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def setup_drive_and_cache(
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    local_fallback_root: str | Path | None = None,
    mount_point: str = "/content/drive",
) -> dict[str, Path]:
    """Mount Google Drive if running on Colab, create standard subdirectories on Drive,
    and configure HuggingFace cache in local disk (`/root/.cache/huggingface`).

    Returns a dictionary of resolved directory paths:
        - root: base persistent directory
        - hf_home: HuggingFace local cache dir (`/root/.cache/huggingface`)
        - checkpoints: model/training checkpoints dir
        - eval_cache: per-problem evaluation resume cache dir
        - results: results dir (containing results.jsonl and plots/)
        - data: dataset splits dir (containing lcb_quick.json)
    """
    if is_colab():
        from google.colab import drive  # type: ignore[import-not-found]

        if not os.path.ismount(mount_point):
            drive.mount(mount_point)
        base_dir = Path(drive_root)
    else:
        if local_fallback_root is not None:
            base_dir = Path(local_fallback_root).resolve()
        else:
            base_dir = Path.cwd().resolve()

    hf_home = Path(DEFAULT_LOCAL_HF_HOME)
    checkpoints_dir = base_dir / "checkpoints"
    eval_cache_dir = base_dir / "eval_cache"
    results_dir = base_dir / "results"
    plots_dir = results_dir / "plots"
    data_dir = base_dir / "data"

    for d in (base_dir, checkpoints_dir, eval_cache_dir, results_dir, plots_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        hf_home.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    os.environ["HF_HOME"] = DEFAULT_LOCAL_HF_HOME
    os.environ["HF_HUB_CACHE"] = f"{DEFAULT_LOCAL_HF_HOME}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{DEFAULT_LOCAL_HF_HOME}/datasets"

    return {
        "root": base_dir,
        "hf_home": hf_home,
        "checkpoints": checkpoints_dir,
        "eval_cache": eval_cache_dir,
        "results": results_dir,
        "plots": plots_dir,
        "data": data_dir,
    }


class CheckpointManager:
    """Manage periodic checkpoints on Drive, keeping only the last `max_to_keep` (default 2)
    and saving every `save_steps` or `save_interval_minutes` (default 15 min).
    """

    def __init__(
        self,
        run_dir: str | Path,
        max_to_keep: int = 2,
        save_steps: int = 50,
        save_interval_minutes: float = 15.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max(1, max_to_keep)
        self.save_steps = save_steps
        self.save_interval_seconds = save_interval_minutes * 60.0
        self._last_save_time = time.monotonic()

    def list_checkpoints(self) -> list[Path]:
        """List existing `checkpoint-<step>` directories sorted by step ascending."""
        if not self.run_dir.exists():
            return []
        ckpts: list[tuple[int, Path]] = []
        for child in self.run_dir.iterdir():
            if child.is_dir() and child.name.startswith("checkpoint-"):
                suffix = child.name.split("checkpoint-", 1)[1]
                if suffix.isdigit():
                    ckpts.append((int(suffix), child))
        ckpts.sort(key=lambda item: item[0])
        return [p for _, p in ckpts]

    def find_latest_checkpoint(self) -> Path | None:
        """Return the latest valid checkpoint directory if one exists."""
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None

    def should_save(self, step: int) -> bool:
        """Return True if `step` is a multiple of `save_steps` or 15 minutes have elapsed."""
        if step > 0 and self.save_steps > 0 and (step % self.save_steps == 0):
            return True
        now = time.monotonic()
        if (now - self._last_save_time) >= self.save_interval_seconds:
            return True
        return False

    def prune_old_checkpoints(self) -> list[Path]:
        """Delete older checkpoints so at most `max_to_keep` remain on Drive."""
        ckpts = self.list_checkpoints()
        removed: list[Path] = []
        while len(ckpts) > self.max_to_keep:
            oldest = ckpts.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
            removed.append(oldest)
        return removed

    def save_state(self, step: int, payload: dict[str, Any], save_fn: Any | None = None) -> Path:
        """Save checkpoint metadata (and optionally invoke `save_fn(ckpt_dir)`) and prune older ones."""
        ckpt_dir = self.run_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if save_fn is not None:
            save_fn(ckpt_dir)
        meta_path = ckpt_dir / "state.json"
        meta = dict(payload)
        meta["step"] = step
        meta["saved_at"] = time.time()
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_save_time = time.monotonic()
        self.prune_old_checkpoints()
        return ckpt_dir


class ResumableProgress:
    """JSONL-backed per-item progress tracker so evaluations and data generation
    automatically resume where they stopped without recomputing finished items.
    """

    def __init__(self, cache_file: str | Path) -> None:
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_file.exists():
            return
        with self.cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    item_id = str(obj["item_id"])
                    self._records[item_id] = obj
                except Exception:
                    continue

    def is_done(self, item_id: str) -> bool:
        return str(item_id) in self._records

    def get(self, item_id: str) -> dict[str, Any] | None:
        return self._records.get(str(item_id))

    def record(self, item_id: str, data: dict[str, Any]) -> None:
        item_key = str(item_id)
        entry = dict(data)
        entry["item_id"] = item_key
        self._records[item_key] = entry
        with self.cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    def all_records(self) -> list[dict[str, Any]]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)
