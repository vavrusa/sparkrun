"""TensorFold runtime for sparkrun.

TensorFold (https://github.com/ashhart/TensorFold) is a
standalone C++/CUDA inference engine with its own launcher -- not a vLLM or
llama.cpp server.  Its multi-node bootstrap is a single ``--rank`` per node::

    tensorfold serve <model-path> --tp 2 --rank <N> \\
        --master <head-ip> --master-port 29551 [engine flags]

Only rank 0 exposes the OpenAI surface (``--name``/``--host``/``--port``);
worker ranks must NOT bind it.  A recipe ``command:`` template cannot express
that -- a command string renders identically on every node -- so this runtime
implements :meth:`TensorFoldRuntime.generate_node_command` and declares the
``"native"`` clustering strategy, exactly as ``atlas`` and ``sglang`` do.
The flag vocabulary is nearly identical to Atlas's
(``--rank``/``--master``/``--master-port``), which is why that file is the
model for this one.

Validated on two DGX Sparks (GB10): 1M context, RoCE all-gathers (the
engine auto-detects the CX7 fabric), cold load 280 s.

Environment-only knobs
----------------------
Several TensorFold knobs are **not** CLI flags: the engine reads ``TF_GLM_*``
environment variables (e.g. ``TF_GLM_KV=fp8`` selects the fp8 DSA latent
cache; the dense-weight, RoCE, split-prefill and draft-policy families are the
same).  Those belong in the recipe's ``env:`` block / ``get_common_env()``, not
in ``_TENSORFOLD_FLAG_MAP``.  Putting one in ``defaults:`` would be silently
dropped, which is what ``known_config_keys()`` exists to make audible.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes._util import default_env_hf_offline
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv

logger = logging.getLogger(__name__)

#: TensorFold's default coordination port.  NOT the 29500/25000 the other
#: runtimes use -- upstream's ``MASTER_PORT`` default is 29551, and the two
#: ranks must agree, so this is the runtime's business to pin.
TENSORFOLD_DEFAULT_MASTER_PORT = 29551

#: TensorFold's default HTTP port for the rank-0 OpenAI surface.
TENSORFOLD_DEFAULT_PORT = 8888

#: Recipe config key -> ``tensorfold serve`` flag.
#:
#: Only keys with a *direct* CLI spelling belong here.  Every key in this map
#: is emitted as ``--flag value`` except those in
#: ``_TENSORFOLD_BOOL_FLAGS``.  Anything an operator puts in ``defaults:``
#: that is neither here nor in ``_TENSORFOLD_SPARKRUN_OWNED_KEYS`` is reported
#: by ``known_config_keys()`` rather than dropped in silence (issue #276).
_TENSORFOLD_FLAG_MAP = {
    # sparkrun-standard keys, where TensorFold has an equivalent
    "port": "--port",
    "host": "--host",
    "served_model_name": "--name",
    "max_model_len": "--context",
    "max_num_seqs": "--parallel",
    "max_tokens": "--max-tokens",
    # TensorFold-specific keys, passed through as-is
    "thinking": "--thinking",
    "vision": "--vision",
    "vision_urls": "--vision-urls",
    # engine-side memory/prefix-cache knobs (TensorFold v1.5)
    "spill_gib": "--spill-gib",
    "snapshot_dir": "--snapshot-dir",
}

#: Presence-only toggles: emitted bare when truthy, omitted when falsy.
_TENSORFOLD_BOOL_FLAGS = frozenset({"thinking", "vision", "vision_urls"})

#: ``thinking`` defaults to ON in the engine, so a recipe's ``thinking: false``
#: must emit ``--no-thinking`` -- omitting the flag would silently serve
#: thinking-on replies.  ``vision``/``vision_urls`` default off, so omitting
#: them is already correct and they stay out of this set.
_TENSORFOLD_NEGATABLE_KEYS = frozenset({"thinking"})

#: Keys sparkrun itself consumes before the engine sees them, declared here so
#: ``known_config_keys()`` does not report them as dropped.
_TENSORFOLD_SPARKRUN_OWNED_KEYS = frozenset(
    {
        # parallelism / rendezvous -- emitted by generate_node_command()
        "tensor_parallel",
        "tp",
        "init_port",
        "master_port",
        "master_ip",
        # model + draft distribution, resolved by the launcher
        "model",
        "model_revision",
        "draft_model",
        "draft_model_revision",
        "_model_snapshot_path",
        "_draft_snapshot_path",
        # recipe plumbing
        "runtime",
        "image",
        "env",
        "command",
        # engine knobs that are ENV vars, not flags (see module docstring)
        "kv_cache_dtype",
        "dense",
        "comm",
        "split",
        "kda_chunked",
        "copy",
        "copy_max",
        "copy_code",
        "draft_policy",
        "multi_prefill",
        "shared_prefix",
        "stream_smooth",
        "stream_smooth_ms",
        "fill_budget_ms",
        "fill_drafts",
        "memory_reserve_gib",
        "kv_pool_gib",
    }
)


class TensorFoldRuntime(RuntimePlugin):
    """Native TensorFold runtime for GB10 clusters.

    TensorFold bootstraps its own ranks over the CX7 fabric, so this runtime
    uses the ``"native"`` clustering strategy and emits the per-node ``--rank``
    itself.  Only rank 0 binds the HTTP listener; workers pass no
    ``--host``/``--port`` at all.
    """

    runtime_name = "tensorfold"

    #: TensorFold's CUDA kernels are built for GB10 (sm_121a) and its RoCE path
    #: assumes the Spark's CX7 fabric, so pin the runtime to GB10 hosts rather
    #: than let a placement land somewhere it cannot run.
    requires_capability = frozenset({"gb10"})

    # --- Clustering ---

    def cluster_strategy(self) -> str:
        """TensorFold bootstraps its own ranks; there is no Ray here."""
        return "native"

    def managed_rendezvous_flags(self) -> tuple[str, ...]:
        """TensorFold spells the world ``--tp``/``--rank``/``--master``.

        Declared per-runtime rather than shared because only
        ``--master-port`` overlaps with vLLM's
        ``--nnodes``/``--node-rank`` vocabulary; Atlas is the close relative
        but still differs (``--world-size`` vs ``--tp``, ``--master-addr`` vs
        ``--master``).
        """
        return ("--rank", "--tp", "--master", "--master-port")

    # --- Environment / container ---

    def get_common_env(self):
        """Resolve the checkpoint from the local cache, never the Hub.

        The launcher distributes the weights to every node before the serve
        command runs, so an online check would only add a failure mode.
        """
        return default_env_hf_offline()

    def default_executor_config(self) -> dict[str, Any]:
        """Take the host IPC namespace; leave the image entrypoint alone.

        * Upstream runs TensorFold with ``--ipc=host`` and the rank bootstrap
          goes through ``/dev/shm``.  With ``ipc: host`` the container sees
          the host's tmpfs and ``shm_size`` is not consulted, so the 25gb
          default is left alone.
        * The image ENTRYPOINT is the NGC ``nvidia_entrypoint.sh`` (``exec
          "$@"``) — a passthrough wrapper, and sparkrun's entrypoint probe
          knows the pattern.  The generated ``bash -c`` command runs through
          it exactly as upstream's ``docker run ... "$IMAGE" tensorfold serve
          ...`` does, so no override is needed.
        """
        return {"ipc": "host"}

    def get_extra_docker_opts(self) -> list[str]:
        """RDMA + rlimit options TensorFold's rank bootstrap needs.

        ``IPC_LOCK`` and an unlimited ``memlock`` rlimit let ``ibv_reg_mr`` pin
        the registered buffers for the RoCE all-gathers — without them the RoCE
        path fails at registration rather than degrading.  The raised stack
        limit matches upstream's ``RUN_ARGS`` (the NCCL bootstrap recurses
        deeply enough to need it).
        """
        return [
            "--cap-add=IPC_LOCK",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
        ]

    # --- Config ---

    def wants_model_snapshot_paths(self) -> bool:
        """``tensorfold serve`` takes a snapshot *directory*, not a repo id.

        Upstream builds ``$HF_CACHE/hub/models--<id>/snapshots/<rev>`` itself
        (its ``snapshot()`` helper) and the engine reads the checkpoint from
        that path.  The launcher therefore resolves it post-distribution — the
        same seam as the GGUF ``_gguf_model_path`` injection.
        """
        return True

    def prepare(
        self,
        recipe: "Recipe",
        hosts: list[str],
        config: "SparkrunConfig | None" = None,
        dry_run: bool = False,
        transfer_mode: str = "auto",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Add the drafter to distribution when configured.

        The drafter is a *different repo* from the served checkpoint and must
        never inherit its revision pin, so it rides the standard
        ``distribution_config`` machinery (as Atlas/vLLM draft models do)
        rather than being resolved from a config default at serve time.
        """
        draft_model = recipe._effective_default("draft_model")
        if draft_model:
            draft_revision = recipe._effective_default("draft_model_revision")
            recipe.distribution_config.add_model(str(draft_model), revision=str(draft_revision) if draft_revision else None)

    def serve_flag_map(self):
        return _TENSORFOLD_FLAG_MAP

    def known_config_keys(self) -> frozenset[str]:
        """Every config key TensorFold consumes, flags *and* env-only knobs.

        See :func:`sparkrun.core.launcher.report_unmapped_config_keys`.  A key
        outside this set reaches nothing, and silence is the failure mode this
        guards against.
        """
        return frozenset(_TENSORFOLD_FLAG_MAP) | _TENSORFOLD_SPARKRUN_OWNED_KEYS

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        cmds["tensorfold"] = "tensorfold --version 2>/dev/null || echo unknown"
        return cmds

    # --- Command generation ---

    @staticmethod
    def _model_arg(config, recipe: "Recipe") -> str:
        """Resolve the checkpoint argument.

        ``tensorfold serve`` takes a **local path**, not a Hub repo id:
        upstream builds ``$HF_CACHE/hub/models--<id>/snapshots/<rev>`` itself.
        sparkrun's launcher resolves that directory at launch time and injects
        it as the private ``_model_snapshot_path`` key -- the same idiom
        ``llama_cpp``/``sglang`` use for ``_gguf_model_path``.  Fall back to
        the raw ``recipe.model`` so a dry run and a hand-written path still
        work.
        """
        return str(config.get("_model_snapshot_path") or recipe.model)

    @staticmethod
    def _drafter_arg(config) -> str | None:
        """Resolve ``--drafter``.

        ``none`` is upstream's documented spelling for "use the checkpoint's
        own MTP head instead of an external drafter", so it is passed through
        rather than treated as a missing value.
        """
        draft = config.get("_draft_snapshot_path") or config.get("draft_model")
        return str(draft) if draft else None

    def _base_parts(
        self,
        recipe: "Recipe",
        config,
        *,
        node_rank: int,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> list[str]:
        """Build everything except the rank-coordination flags.

        Rank 0 owns the HTTP surface (``--host``/``--port``) and the served
        name (``--name``); worker ranks get none of the three.

        ``host``/``port`` are excluded from the flag map **unconditionally**
        and emitted by hand for rank 0 only.  Leaving them to the map for
        ``node_rank > 0`` would hand every worker the same ``--port`` and have
        them fight rank 0 for it -- upstream passes neither on a worker.
        """
        parts = ["tensorfold", "serve", self._model_arg(config, recipe)]

        drafter = self._drafter_arg(config)
        if drafter:
            parts += ["--drafter", drafter]

        # The HTTP surface is rank 0's alone, whichever rank we are.
        skip = set(skip_keys) | {"host", "port"}
        if node_rank == 0:
            parts += [
                "--host",
                str(config.get("host") or "0.0.0.0"),
                "--port",
                str(config.get("port") or TENSORFOLD_DEFAULT_PORT),
            ]
        else:
            # Upstream passes --name on rank 0 only; a worker never binds, so
            # it has no use for it.
            skip.add("served_model_name")

        parts += self.build_flags_from_map(
            config,
            _TENSORFOLD_FLAG_MAP,
            bool_keys=_TENSORFOLD_BOOL_FLAGS,
            negatable_keys=_TENSORFOLD_NEGATABLE_KEYS,
            skip_keys=skip,
        )
        return parts

    def _master_port(self, config) -> int:
        """Coordination port: ``init_port`` (sparkrun's spelling) wins."""
        return int(config.get("init_port") or config.get("master_port") or TENSORFOLD_DEFAULT_MASTER_PORT)

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the solo / rank-0 command.

        In cluster mode this is rank 0's command; workers come from
        :meth:`generate_node_command`.
        """
        config = recipe.build_config_chain(overrides)

        rendered = recipe.render_command(config)
        if rendered:
            return rendered

        parts = self._base_parts(recipe, config, node_rank=0, skip_keys=skip_keys)
        tp = int(config.get("tensor_parallel") or config.get("tp") or 1)
        if is_cluster and head_ip:
            parts += [
                "--tp",
                str(tp),
                "--rank",
                "0",
                "--master",
                head_ip,
                "--master-port",
                str(self._master_port(config)),
            ]
        return " ".join(parts)

    def generate_node_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        head_ip: str,
        num_nodes: int,
        node_rank: int,
        init_port: int = TENSORFOLD_DEFAULT_MASTER_PORT,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        hosts: list[str] | None = None,
        placement=None,
    ) -> str:
        """Generate the per-rank ``tensorfold serve`` command.

        This is the method that makes TensorFold hostable by sparkrun at all:
        ``--rank`` differs per node, which no recipe ``command:`` template can
        express.  ``--master`` is always the **head**, and both ranks must
        agree on ``--master-port``.
        """
        config = recipe.build_config_chain(overrides)
        config.set("init_port", init_port)

        parts = self._base_parts(recipe, config, node_rank=node_rank, skip_keys=skip_keys)
        tp = int(config.get("tensor_parallel") or config.get("tp") or num_nodes)
        # ``--master`` is the rendezvous address BOTH ranks must reach.  The
        # cluster's head IP is the right default, but a recipe may pin an
        # explicit ``master_ip`` -- e.g. a CX7 fabric address when the cluster
        # hosts are named by their management LAN IPs and NCCL_SOCKET_IFNAME is
        # pinned to the fabric interface (the store must live on the same
        # fabric the collectives use).
        master = config.get("master_ip") or head_ip
        parts += [
            "--tp",
            str(tp),
            "--rank",
            str(node_rank),
            "--master",
            str(master),
            "--master-port",
            str(self._master_port(config)),
        ]
        return " ".join(parts)

    # --- Cluster lifecycle ---

    def _stop_cluster(
        self,
        hosts: list[str],
        cluster_id: str,
        config: "SparkrunConfig | None" = None,
        dry_run: bool = False,
    ) -> int:
        """Stop a TensorFold native cluster."""
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)

    def _run_cluster(
        self,
        hosts: list[str],
        image: str,
        serve_command: str = "",
        recipe=None,
        overrides=None,
        *,
        cluster_id: str = "sparkrun0",
        env: dict[str, str] | None = None,
        cache_dir: str | None = None,
        config: "SparkrunConfig | None" = None,
        dry_run: bool = False,
        detached: bool = True,
        comm_env: "ClusterCommEnv | None" = None,
        ib_ip_map: dict[str, str] | None = None,
        init_port: int = TENSORFOLD_DEFAULT_MASTER_PORT,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        **kwargs,
    ) -> int:
        """Orchestrate a multi-rank TensorFold cluster using native bootstrap."""
        return self._run_native_cluster(
            hosts=hosts,
            image=image,
            serve_command=serve_command,
            recipe=recipe,
            overrides=overrides,
            cluster_id=cluster_id,
            env=env,
            cache_dir=cache_dir,
            config=config,
            dry_run=dry_run,
            detached=detached,
            comm_env=comm_env,
            ib_ip_map=ib_ip_map,
            init_port=init_port,
            skip_keys=skip_keys,
            banner_title="TensorFold Cluster Launcher",
            port_label="Master Port",
            node_label="tensorfold node",
            **kwargs,
        )
