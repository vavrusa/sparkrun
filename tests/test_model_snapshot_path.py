"""Tests for resolve_model_snapshot_container_path.

The function is the seam that lets a runtime serve a *snapshot directory*
(TensorFold) instead of a Hub repo id: the launcher resolves it post-
distribution and injects it, mirroring the GGUF ``_gguf_model_path`` path
injection.  Its two failure modes are picking the WRONG snapshot when several
exist and silently resolving a stub (config.json-only) directory.
"""

from pathlib import Path

from sparkrun.models.download import (
    CONTAINER_HF_CACHE,
    resolve_model_snapshot_container_path,
)

REV_A = "a" * 40
REV_B = "b" * 40


def _write_snapshot(root: Path, model_id: str, rev: str, weights: bool = True) -> None:
    snap = root / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}")
    if weights:
        (snap / "model.safetensors").write_bytes(b"weights")


def _write_ref(root: Path, model_id: str, ref: str, rev: str) -> None:
    refs = root / "hub" / ("models--" + model_id.replace("/", "--")) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / ref).write_text(rev)


def _resolve(root: Path, model_id: str, revision: str | None = None) -> str | None:
    return resolve_model_snapshot_container_path(model_id, cache_dir=str(root), revision=revision)


def test_resolves_snapshot_dir_under_container_cache(tmp_path):
    _write_snapshot(tmp_path, "org/model", REV_A)
    got = _resolve(tmp_path, "org/model")
    assert got == f"{CONTAINER_HF_CACHE}/hub/models--org--model/snapshots/{REV_A}"


def test_pinned_revision_wins_over_refs_main(tmp_path):
    _write_snapshot(tmp_path, "org/model", REV_A)
    _write_snapshot(tmp_path, "org/model", REV_B)
    _write_ref(tmp_path, "org/model", "main", REV_A)
    assert REV_B in _resolve(tmp_path, "org/model", revision=REV_B)


def test_unpinned_prefers_refs_main(tmp_path):
    _write_snapshot(tmp_path, "org/model", REV_A)
    _write_snapshot(tmp_path, "org/model", REV_B)
    _write_ref(tmp_path, "org/model", "main", REV_B)
    assert REV_B in _resolve(tmp_path, "org/model", revision=None)


def test_unpinned_falls_back_to_any_snapshot(tmp_path):
    """Manually placed cache entries have no refs/ (sha-pinned downloads)."""
    _write_snapshot(tmp_path, "org/model", REV_A)
    assert REV_A in _resolve(tmp_path, "org/model", revision=None)


def test_config_json_only_stub_is_not_resolved(tmp_path):
    _write_snapshot(tmp_path, "org/model", REV_A, weights=False)
    assert _resolve(tmp_path, "org/model") is None


def test_missing_repo_is_none(tmp_path):
    assert _resolve(tmp_path, "org/model") is None


def test_gguf_spec_is_none(tmp_path):
    """GGUF resolution has its own resolver; this one must not grab it."""
    assert _resolve(tmp_path, "org/model:Q4_K_M") is None
