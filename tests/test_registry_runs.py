"""Registry por run: legado + runs/*.jsonl, ordem cronológica, sem append no legado."""

import importlib.util
import json
import tempfile
from pathlib import Path

# carrega só lib/registry.py (sem torch), para rodar em qualquer máquina
_spec = importlib.util.spec_from_file_location(
    "registry", Path(__file__).resolve().parent.parent / "lib" / "registry.py"
)
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)

KW = dict(
    phase="F1", model="m", checkpoint="c", n_loops=2, eval_tier="quick", bench="b",
    pass1=0.5, tokens_cot_mean=100.0, tokens_total_mean=120.0, latency_mean_s=1.0, git_sha="t",
)


def test_registry_per_run_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        legacy = Path(tmp) / "results" / "results.jsonl"
        legacy.parent.mkdir(parents=True)
        legacy.write_text(
            json.dumps({"run_id": "F0_old", "bench": "b", "eval": "quick", "phase": "F0",
                        "timestamp": "2026-11-12T00:00:00+00:00"}) + "\n"
        )
        reg.append_result(run_id="F1 a/b", results_path=legacy, timestamp="2026-11-17T00:00:02+00:00", **KW)
        reg.append_result(run_id="F1_c", results_path=legacy, timestamp="2026-11-17T00:00:01+00:00", **KW)
        reg.append_result(run_id="F1 a/b", results_path=legacy, timestamp="2026-11-17T00:00:03+00:00", **KW)

        files = sorted(p.name for p in (legacy.parent / "runs").iterdir())
        assert files == ["F1_a_b.jsonl", "F1_c.jsonl"]
        assert legacy.read_text().count("\n") == 1, "legado não pode receber append"
        assert [r["run_id"] for r in reg.load_results(legacy)] == ["F0_old", "F1_c", "F1 a/b", "F1 a/b"]

        assert reg.has_completed_run(run_id="F0_old", bench="b", eval_tier="quick", results_path=legacy)
        assert reg.has_completed_run(run_id="F1_c", bench="b", results_path=legacy)
        assert not reg.has_completed_run(run_id="nada", bench="b", results_path=legacy)

        # linha truncada sem newline no arquivo do run: o próximo append não cola nela
        with reg.run_results_path(legacy, "F1_c").open("a") as f:
            f.write('{"run_id": "quebrada"')
        reg.append_result(run_id="F1_c", results_path=legacy, **KW)
        assert [r["run_id"] for r in reg.load_results(legacy, run_id="F1_c")] == ["F1_c", "F1_c"]

        assert reg.load_results(Path(tmp) / "vazio" / "results.jsonl") == []
