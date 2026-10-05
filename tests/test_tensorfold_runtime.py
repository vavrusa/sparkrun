"""Unit tests for the TensorFold runtime plugin.

The load-bearing property is the **per-node command**: TensorFold needs a
different ``--rank`` on each host, which no recipe ``command:`` template can
express.  ``generate_node_command`` is what makes the engine hostable at all,
so most of these tests pin its shape -- especially that a worker never binds
the HTTP port (getting that wrong has both ranks fight over :8888).
"""

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.tensorfold import (
    TENSORFOLD_DEFAULT_MASTER_PORT,
    TENSORFOLD_DEFAULT_PORT,
    TensorFoldRuntime,
    _TENSORFOLD_FLAG_MAP,
)

HEAD_IP = "10.0.0.1"
MODEL = "example-org/example-model"


def _recipe(**overrides) -> Recipe:
    base = {
        "name": "test-recipe",
        "model": MODEL,
        "runtime": "tensorfold",
    }
    base.update(overrides)
    return Recipe.from_dict(base)


def _node(runtime, recipe, rank, *, num_nodes=2, init_port=TENSORFOLD_DEFAULT_MASTER_PORT):
    return runtime.generate_node_command(
        recipe,
        {},
        head_ip=HEAD_IP,
        num_nodes=num_nodes,
        node_rank=rank,
        init_port=init_port,
    )


# --- Identity ---------------------------------------------------------------


def test_runtime_identity():
    rt = TensorFoldRuntime()
    assert rt.runtime_name == "tensorfold"
    assert rt.cluster_strategy() == "native"


def test_requires_gb10():
    """TensorFold's kernels are sm_121a and its RoCE path assumes CX7."""
    assert TensorFoldRuntime().requires_capability == frozenset({"gb10"})


def test_managed_rendezvous_flags():
    """Declared, not shared: only --master-port overlaps with vLLM's spelling."""
    assert set(TensorFoldRuntime().managed_rendezvous_flags()) == {
        "--rank",
        "--tp",
        "--master",
        "--master-port",
    }


# --- Container / executor ---------------------------------------------------


def test_executor_config_takes_host_ipc():
    """ipc=host (rank bootstrap via /dev/shm); entrypoint left alone.

    The image ENTRYPOINT is the NGC ``exec "$@"`` passthrough sparkrun's probe
    knows, so overriding it would only diverge from upstream's own
    ``docker run ... "$IMAGE" tensorfold serve ...``.
    """
    cfg = TensorFoldRuntime().default_executor_config()
    assert cfg == {"ipc": "host"}


def test_extra_docker_opts_unlock_rdma():
    opts = TensorFoldRuntime().get_extra_docker_opts()
    assert "--cap-add=IPC_LOCK" in opts
    assert "memlock=-1" in opts
    # Upstream RUN_ARGS raises the stack limit for the NCCL bootstrap.
    assert "stack=67108864" in opts


def test_wants_model_snapshot_paths():
    """Opt into the launcher's post-distribution snapshot resolution."""
    assert TensorFoldRuntime().wants_model_snapshot_paths() is True


def test_prepare_adds_drafter_to_distribution():
    """The drafter is a different repo and must ride the distribution config."""
    rt = TensorFoldRuntime()
    recipe = _recipe(
        defaults={
            "draft_model": "example-org/example-drafter",
            "draft_model_revision": "bf582e4eacc1810f76656d1811693ff6c6737d2a",
        }
    )
    rt.prepare(recipe, hosts=["h0", "h1"], dry_run=True)
    entries = {(e.name, e.revision) for e in recipe.distribution_config.models.entries}
    assert ("example-org/example-drafter", "bf582e4eacc1810f76656d1811693ff6c6737d2a") in entries
    # The served checkpoint's (unresolved {model}) entry keeps its own absent pin.
    assert ("{model}", None) in entries


def test_prepare_without_drafter_is_a_noop():
    rt = TensorFoldRuntime()
    recipe = _recipe()
    rt.prepare(recipe, hosts=["h0"], dry_run=True)
    assert [e.name for e in recipe.distribution_config.models.entries] == ["{model}"]


# --- Per-node command: the whole point --------------------------------------


def test_node_command_rank_flags_differ_per_node():
    """--rank is the flag sparkrun's recipe templates cannot vary."""
    rt = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2})
    head = _node(rt, recipe, 0)
    worker = _node(rt, recipe, 1)

    assert "--rank 0" in head
    assert "--rank 1" in worker
    assert "--tp 2" in head and "--tp 2" in worker


def test_both_ranks_agree_on_master():
    """A rendezvous mismatch is a hang, not an error, so pin both ends."""
    rt = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2})
    head = _node(rt, recipe, 0, init_port=29551)
    worker = _node(rt, recipe, 1, init_port=29551)

    for cmd in (head, worker):
        assert f"--master {HEAD_IP}" in cmd
        assert "--master-port 29551" in cmd


def test_worker_does_not_bind_http_port():
    """Regression: leaving host/port to the flag map gave workers --port too."""
    rt = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "port": 8888, "served_model_name": "example-model"})
    worker = _node(rt, recipe, 1)

    assert "--port" not in worker
    assert "--host" not in worker
    # --name is rank 0's: a worker has no HTTP surface to name.
    assert "--name" not in worker


def test_rank0_binds_and_names():
    rt = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "port": 8888, "served_model_name": "example-model"})
    head = _node(rt, recipe, 0)

    assert "--port 8888" in head
    assert "--host 0.0.0.0" in head
    assert "--name example-model" in head


def test_default_port_when_unset():
    rt = TensorFoldRuntime()
    head = _node(rt, _recipe(), 0)
    assert f"--port {TENSORFOLD_DEFAULT_PORT}" in head


def test_default_master_port_matches_upstream():
    """Upstream's MASTER_PORT is 29551, not the 29500/25000 other runtimes use."""
    assert TENSORFOLD_DEFAULT_MASTER_PORT == 29551
    rt = TensorFoldRuntime()
    head = _node(rt, _recipe(), 0, init_port=TENSORFOLD_DEFAULT_MASTER_PORT)
    assert "--master-port 29551" in head


def test_model_arg_defaults_to_recipe_model():
    rt = TensorFoldRuntime()
    assert MODEL in _node(rt, _recipe(), 0)


def test_model_snapshot_path_wins_over_repo_id():
    """tensorfold serve takes a local snapshot path, like llama.cpp's GGUF."""
    rt = TensorFoldRuntime()
    snapshot = "/home/u/.cache/huggingface/hub/models--example-org--example-model/snapshots/078455ff"
    recipe = _recipe(defaults={"_model_snapshot_path": snapshot})
    cmd = _node(rt, recipe, 0)
    assert snapshot in cmd
    assert MODEL not in cmd


def test_drafter_flag_emitted_when_configured():
    rt = TensorFoldRuntime()
    recipe = _recipe(defaults={"_draft_snapshot_path": "/cache/drafter/snapshots/bf582e4e"})
    assert "--drafter /cache/drafter/snapshots/bf582e4e" in _node(rt, recipe, 0)


def test_drafter_absent_when_unset():
    assert "--drafter" not in _node(TensorFoldRuntime(), _recipe(), 0)


# --- Flag mapping -----------------------------------------------------------


def test_mapped_flags_reach_the_command():
    rt = TensorFoldRuntime()
    recipe = _recipe(
        defaults={
            "max_model_len": 1048576,
            "max_num_seqs": 4,
            "max_tokens": 32768,
        }
    )
    head = _node(rt, recipe, 0)
    assert "--context 1048576" in head
    assert "--parallel 4" in head
    assert "--max-tokens 32768" in head


# --- TensorFold v1.5 flags ---------------------------------------------------


def test_v15_flags_reach_the_command():
    """--spill-gib / --snapshot-dir (TensorFold v1.5) render as engine flags."""
    rt = TensorFoldRuntime()
    cmd = _node(rt, _recipe(defaults={"spill_gib": 32, "snapshot_dir": "/cache/prefix-snapshots"}), 0)
    assert "--spill-gib 32" in cmd
    assert "--snapshot-dir /cache/prefix-snapshots" in cmd
    assert "--spill-gib" not in _node(rt, _recipe(), 0)


def test_master_ip_override_wins_over_head_ip():
    """The store may live on the CX7 fabric even when hosts are LAN-named."""
    rt = TensorFoldRuntime()
    cmd = _node(rt, _recipe(defaults={"master_ip": "10.1.0.1"}), 1)
    assert "--master 10.1.0.1" in cmd
    # default stays the cluster's head IP
    assert "--master 10.1.0.1" not in _node(rt, _recipe(), 1)
    assert f"--master {HEAD_IP}" in _node(rt, _recipe(), 1)


def test_master_ip_flows_through_overrides():
    rt = TensorFoldRuntime()
    cmd = rt.generate_node_command(_recipe(), {"master_ip": "10.0.0.7"}, head_ip=HEAD_IP, num_nodes=2, node_rank=0)
    assert "--master 10.0.0.7" in cmd


def test_thinking_false_emits_negated_flag():
    """The engine defaults thinking ON, so `false` must not be a no-op."""
    rt = TensorFoldRuntime()
    assert "--no-thinking" in _node(rt, _recipe(defaults={"thinking": False}), 0)


def test_thinking_true_and_omitted():
    rt = TensorFoldRuntime()
    assert "--thinking" in _node(rt, _recipe(defaults={"thinking": True}), 0)
    assert "--thinking" not in _node(rt, _recipe(), 0)


def test_known_config_keys_covers_every_mapped_key():
    """A defaults: key outside this set is dropped in silence (issue #276)."""
    known = TensorFoldRuntime().known_config_keys()
    assert set(_TENSORFOLD_FLAG_MAP) <= known
    # env-only knobs are consumed too, so they must not be reported as dropped
    assert {"kv_cache_dtype", "tensor_parallel", "init_port"} <= known


def test_serve_flag_map_matches_module_map():
    assert TensorFoldRuntime().serve_flag_map() is _TENSORFOLD_FLAG_MAP
