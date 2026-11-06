# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three-tier encoder cache storage for the score-based encoder cache (PR7500).

Storage half of the ModelRunner V2 integration, split out of the runner to
mirror the upstream vllm V2 separation (``mm/encoder_cache.py`` storage class
vs ``mm/encoder_runner.py`` executor).  Tier 1 (resident) is the framework
``EncoderCache.encoder_outputs`` this class wraps; tier 2 (tmp) is an NPU
scratch dict refcounted per request; tier 3 (cpu) is host-pinned staging for
cold entries.  All tiering decisions arrive from the scheduler-side
``ScoreEncoderCacheManager`` via ``EncoderCacheManagerMetadata``.
"""

from collections import deque
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


class ScoreEncoderCache:
    """Three-tier encoder output storage (tier 1 wraps the framework cache)."""

    def __init__(
        self,
        encoder_cache: "EncoderCache",
        device: torch.device,
    ) -> None:
        # Tier 1: NPU-resident, framework-owned (injected).
        self.encoder_cache = encoder_cache
        self.device = device
        # Tier 2: NPU scratch entries, evicted once no request references
        # them (refcount decrement on request removal).
        self.tmp_encoder_cache: dict[str, torch.Tensor] = {}
        # Tier 3: host pinned staging for cold entries.
        self.cpu_encoder_cache: dict[str, torch.Tensor] = {}
        # In-flight async H2D copies, polled every scheduler step.
        self._pending_encoder_cache_copies: deque[
            tuple[torch.Tensor, torch.npu.Event]
        ] = deque()
        # Notepad: latest manager metadata (written by apply_metadata, read
        # when encoder outputs are stored).
        self._current_step_metadata: (
            "EncoderCacheManagerMetadata | None") = None
        # mm hash -> req ids referencing it (drives tmp-tier eviction).
        self._req_refs: dict[str, set[str]] = {}
        # req id -> mm hashes (V2 has no per-request state object, so this
        # index replaces V1's req_state.mm_features lookup).
        self._req_hash_index: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Tier traversal (called by the runner's cache hooks).
    # ------------------------------------------------------------------
    def has(self, mm_hash: str) -> bool:
        return (
            mm_hash in self.encoder_cache.encoder_outputs
            or mm_hash in self.tmp_encoder_cache
            or mm_hash in self.cpu_encoder_cache
        )

    def get(self, mm_hash: str) -> torch.Tensor | None:
        encoder_output = self.encoder_cache.encoder_outputs.get(mm_hash, None)
        if encoder_output is not None:
            return encoder_output

        encoder_output = self.tmp_encoder_cache.get(mm_hash, None)
        if encoder_output is not None:
            return encoder_output

        cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
        if cpu_value is None:
            return None

        # The score scheduler asks the worker to stage CPU hits into a
        # device tier before gather.  Keep a defensive synchronous fallback
        # so a stale worker-side temporary entry cannot turn a scheduler
        # CPU-cache hit into an engine-fatal cache miss.
        encoder_output = self._copy_cpu_encoder_cache_to_device(
            cpu_value, non_blocking=False
        )
        self.tmp_encoder_cache[mm_hash] = encoder_output
        return encoder_output

    def store(self, mm_hash: str, output: torch.Tensor) -> None:
        """CPU staging is always written; NPU tier follows the metadata."""
        metadata = self._current_step_metadata
        promoting_mm_hashes = (
            metadata.promoting_mm_hashes if metadata is not None else []
        )
        npu_freed = metadata.npu_freed if metadata is not None else []

        staging = torch.empty_like(output, device="cpu", pin_memory=True)
        staging.copy_(output.detach(), non_blocking=True)
        self.cpu_encoder_cache[mm_hash] = staging

        if mm_hash in promoting_mm_hashes and mm_hash not in npu_freed:
            # Tier 1: NPU-resident.
            self.encoder_cache.encoder_outputs[mm_hash] = output
        else:
            # Tier 2: NPU scratch for this step's requests.
            self.tmp_encoder_cache[mm_hash] = output

    # ------------------------------------------------------------------
    # Scheduler metadata consumption (called by the runner hook).
    # ------------------------------------------------------------------
    def apply_metadata(self, metadata: "EncoderCacheManagerMetadata") -> None:
        self._current_step_metadata = metadata
        self._clear_finished_encoder_cache_copies()

        # Free the cached encoder outputs.
        for mm_hash in metadata.npu_freed:
            self.encoder_cache.encoder_outputs.pop(mm_hash, None)

        for mm_hash in metadata.cpu_freed:
            self.cpu_encoder_cache.pop(mm_hash, None)

        for mm_hash in metadata.promoting_mm_hashes:
            cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
            if cpu_value is None:
                continue
            self.encoder_cache.encoder_outputs[mm_hash] = (
                self._copy_cpu_encoder_cache_to_device(cpu_value)
            )

        for mm_hash in metadata.cpu_get_encoder_mm_hashes:
            if (
                mm_hash in self.encoder_cache.encoder_outputs
                or mm_hash in self.tmp_encoder_cache
            ):
                continue
            cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
            if cpu_value is None:
                continue
            self.tmp_encoder_cache[mm_hash] = (
                self._copy_cpu_encoder_cache_to_device(cpu_value)
            )

    # ------------------------------------------------------------------
    # Request-scoped refcount lifecycle.
    # ------------------------------------------------------------------
    def track_request(
        self,
        req_id: str,
        mm_features: "list[MultiModalFeatureSpec]",
    ) -> None:
        """Build refcounts (and the reverse index) for a new request."""
        hashes = {f.identifier for f in mm_features}
        self._req_hash_index[req_id] = hashes
        for mm_hash in hashes:
            self._req_refs.setdefault(mm_hash, set()).add(req_id)

    def on_request_removed(self, req_id: str) -> None:
        """Drop a request's refcounts; evict tmp entries that hit zero."""
        for mm_hash in self._req_hash_index.pop(req_id, ()):
            refs = self._req_refs.get(mm_hash)
            if refs is None:
                continue
            refs.discard(req_id)
            if not refs:
                del self._req_refs[mm_hash]
                self.tmp_encoder_cache.pop(mm_hash, None)

    def reset(self) -> None:
        """Clear all three tiers and in-flight copies."""
        if (
            self.tmp_encoder_cache
            or self.cpu_encoder_cache
            or self._pending_encoder_cache_copies
        ):
            torch.npu.synchronize()
        self.encoder_cache.encoder_outputs.clear()
        self.tmp_encoder_cache.clear()
        self.cpu_encoder_cache.clear()
        self._req_refs.clear()
        self._req_hash_index.clear()
        self._current_step_metadata = None
        self._pending_encoder_cache_copies.clear()

    # ------------------------------------------------------------------
    # Async H2D copy helpers (V1 verbatim).
    # ------------------------------------------------------------------
    def _clear_finished_encoder_cache_copies(self) -> None:
        while (
            self._pending_encoder_cache_copies
            and self._pending_encoder_cache_copies[0][1].query()
        ):
            self._pending_encoder_cache_copies.popleft()

    def _copy_cpu_encoder_cache_to_device(
        self,
        cpu_value: torch.Tensor,
        *,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        if not non_blocking:
            return cpu_value.to(self.device, non_blocking=False)

        self._clear_finished_encoder_cache_copies()
        source = cpu_value if cpu_value.is_pinned() else cpu_value.pin_memory()
        npu_value = torch.empty_like(source, device=self.device)
        npu_value.copy_(source, non_blocking=True)

        copy_done = torch.npu.Event()
        copy_done.record(torch.npu.current_stream())
        self._pending_encoder_cache_copies.append((source, copy_done))
        npu_value.record_stream(torch.npu.current_stream())
        return npu_value
