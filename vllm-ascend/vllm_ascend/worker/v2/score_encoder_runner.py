# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score-based three-tier encoder cache runner (PR7500) for ModelRunner V2.

Executor half of the integration: extends the framework ``EncoderRunner``
cache hooks to route stores/lookups through the three-tier storage defined
in ``score_encoder_cache.py`` (mirroring the upstream split of
``mm/encoder_runner.py`` executor vs ``mm/encoder_cache.py`` storage).

Ported from the ModelRunner V1 integration in
``vllm_ascend/worker/v1/model_runner.py`` with three deliberate changes:

1. No feature-flag branching inside this class: the runner is instantiated
   only when the score encoder cache manager is configured, so its presence
   alone means the feature is enabled.
2. The per-step ``EncoderCacheManagerMetadata`` reaches the storage-side
   notepad through ``process_scheduler_metadata`` instead of a hook-signature
   parameter, keeping vllm free of any metadata plumbing.
3. ``_req_hash_index`` compensates for the absence of a per-request state
   object in ModelRunner V2: request -> mm-hash bookkeeping is built from
   ``NewRequestData.mm_features`` at add_requests time.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

from vllm_ascend.worker.v2.score_encoder_cache import ScoreEncoderCache

if TYPE_CHECKING:
    from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
    from vllm.multimodal.inputs import MultiModalFeatureSpec


class AscendScoreEncoderRunner(EncoderRunner):
    """EncoderRunner whose cache hooks delegate to a ScoreEncoderCache."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Three-tier storage; tier 1 wraps the injected framework cache.
        self.storage = ScoreEncoderCache(self.encoder_cache, self.device)

    # ------------------------------------------------------------------
    # Framework hook overrides (vllm gpu/mm/encoder_runner.py).
    # ------------------------------------------------------------------
    def cache_encoder_outputs(
        self,
        mm_hashes: list[str],
        outputs: list[torch.Tensor],
    ) -> None:
        """Route every batched store through the tiering hook."""
        for mm_hash, output in zip(mm_hashes, outputs):
            self._cache_encoder_output(mm_hash, output)

    def _cache_encoder_output(
        self,
        mm_hash: str,
        output: torch.Tensor,
    ) -> None:
        self.storage.store(mm_hash, output)

    def _has_cached_encoder_output(self, mm_hash: str) -> bool:
        return self.storage.has(mm_hash)

    def _get_encoder_output_from_cache(
        self, mm_hash: str
    ) -> torch.Tensor | None:
        return self.storage.get(mm_hash)

    # ------------------------------------------------------------------
    # Entry points called by the platform model runner (hook 1/2 overrides).
    # ------------------------------------------------------------------
    def process_scheduler_metadata(
        self,
        metadata: "EncoderCacheManagerMetadata",
    ) -> None:
        """Forward the manager's four lists to the storage."""
        self.storage.apply_metadata(metadata)

    def track_request(
        self,
        req_id: str,
        mm_features: "list[MultiModalFeatureSpec]",
    ) -> None:
        self.storage.track_request(req_id, mm_features)

    def on_request_removed(self, req_id: str) -> None:
        self.storage.on_request_removed(req_id)

    def reset_encoder_cache(self) -> None:
        self.storage.reset()
