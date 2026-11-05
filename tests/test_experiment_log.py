"""The experiment registry must cover every results file (see scripts/update_experiment_log.py)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text(encoding="utf-8"))
REQUIRED = {"id", "date", "track", "group", "results", "expected_runs", "config", "question", "conclusion"}


def test_entries_are_complete_and_ids_unique():
    entries = REGISTRY["experiments"]
    assert all(REQUIRED <= set(e) for e in entries), [e.get("id") for e in entries if not REQUIRED <= set(e)]
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    assert len({e["results"] for e in entries}) == len(entries)


def test_every_results_file_is_logged():
    registered = {e["results"] for e in REGISTRY["experiments"]} | set(REGISTRY.get("not_experiments") or {})
    on_disk = {str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in (REPO_ROOT / "results" / "metrics").glob("*.csv")}
    assert on_disk - registered == set(), "Unlogged results files; add them to experiments/registry.yaml"
