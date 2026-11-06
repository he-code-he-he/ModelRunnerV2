# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the score-based three-tier encoder cache (PR7500) under
ModelRunner V2.

Part 1 covers the AscendScoreEncoderRunner core (tier routing, refcount
eviction, four-list metadata consumption, three-tier lookup, reset, batched
entry). Part 2 covers the thin NPUModelRunner hook overrides (A9/A10) that
wire the runner in. CPU tensors only: the async H2D helper is stubbed where
the scheduler path would otherwise touch NPU events.
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu  # noqa: F401  # pin_memory / torch.npu used by the runner

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.score_encoder_runner import AscendScoreEncoderRunner
from vllm_ascend.worker.v2.score_encoder_cache import ScoreEncoderCache


def _make_runner() -> AscendScoreEncoderRunner:
    """Bare runner harness: skip the framework __init__, seed hook state."""
    runner = AscendScoreEncoderRunner.__new__(AscendScoreEncoderRunner)
    runner.encoder_cache = SimpleNamespace(encoder_outputs={})
    runner.device = torch.device("cpu")
    runner.storage = ScoreEncoderCache(runner.encoder_cache, runner.device)
    return runner


def _metadata(npu_freed=(), cpu_freed=(), promoting=(), cpu_get=()):
    return SimpleNamespace(
        promoting_mm_hashes=list(promoting),
        cpu_get_encoder_mm_hashes=list(cpu_get),
        npu_freed=list(npu_freed),
        cpu_freed=list(cpu_freed),
    )


def _mm_feature(mm_hash: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=0, length=4),
    )


def _tensor(value: float) -> torch.Tensor:
    return torch.tensor([value])


# ---------------------------------------------------------------------------
# Part 1: AscendScoreEncoderRunner core (A6).
# ---------------------------------------------------------------------------
def test_track_and_remove_refcounts_evict_tmp():
    runner = _make_runner()
    runner.storage.tmp_encoder_cache = {
        "h1": _tensor(1),
        "h2": _tensor(2),
        "h3": _tensor(3),
    }

    runner.track_request("req-a", [_mm_feature("h1"), _mm_feature("h2")])
    runner.track_request("req-b", [_mm_feature("h1"), _mm_feature("h3")])
    assert runner.storage._req_refs == {
        "h1": {"req-a", "req-b"},
        "h2": {"req-a"},
        "h3": {"req-b"},
    }

    # Removing an unknown request is a no-op.
    runner.on_request_removed("req-unknown")

    # One request finishing keeps the shared entry, evicts its private one.
    runner.on_request_removed("req-a")
    assert "h1" in runner.storage.tmp_encoder_cache
    assert "h2" not in runner.storage.tmp_encoder_cache

    # The last referencing request finishing evicts the shared entry.
    runner.on_request_removed("req-b")
    assert runner.storage.tmp_encoder_cache == {}
    assert runner.storage._req_refs == {}
    assert runner.storage._req_hash_index == {}


def test_cache_encoder_output_routes_by_metadata():
    runner = _make_runner()
    resident, freed, cold = _tensor(1), _tensor(2), _tensor(3)
    runner.storage._current_step_metadata = _metadata(
        promoting=["resident"],
        npu_freed=["freed-and-promoted"],
    )

    runner._cache_encoder_output("resident", resident)
    runner._cache_encoder_output("freed-and-promoted", freed)
    runner.storage._current_step_metadata = None
    runner._cache_encoder_output("cold", cold)

    # Promoted entries become NPU-resident; npu_freed overrides promotion;
    # without metadata everything lands in the per-step tmp tier.
    assert runner.encoder_cache.encoder_outputs == {"resident": resident}
    assert set(runner.storage.tmp_encoder_cache) == {"freed-and-promoted", "cold"}

    # CPU staging is written for every entry regardless of tier.
    assert set(runner.storage.cpu_encoder_cache) == {"resident", "freed-and-promoted", "cold"}
    assert torch.equal(runner.storage.cpu_encoder_cache["resident"], resident)


def test_process_scheduler_metadata_consumes_four_lists():
    runner = _make_runner()
    runner.encoder_cache.encoder_outputs = {"evicted": _tensor(1)}
    runner.storage.tmp_encoder_cache = {"already-staged": _tensor(5)}
    runner.storage.cpu_encoder_cache = {
        "evicted-cpu": _tensor(4),
        "to-promote": _tensor(2),
        "to-stage": _tensor(3),
    }

    # Keep the async H2D helper out of the unit test; observe it instead.
    copies = MagicMock(return_value=_tensor(9))
    polled = MagicMock()
    runner.storage._copy_cpu_encoder_cache_to_device = copies
    runner.storage._clear_finished_encoder_cache_copies = polled

    metadata = _metadata(
        npu_freed=["evicted"],
        cpu_freed=["evicted-cpu"],
        promoting=["to-promote", "missing-on-cpu"],
        cpu_get=["to-stage", "already-staged", "missing-on-cpu"],
    )
    runner.process_scheduler_metadata(metadata)

    polled.assert_called_once_with()
    assert runner.storage._current_step_metadata is metadata
    # npu_freed evicts the resident entry.
    assert "evicted" not in runner.encoder_cache.encoder_outputs
    # promoting stages CPU hits into the resident tier (missing ones skipped).
    assert runner.encoder_cache.encoder_outputs["to-promote"] is copies.return_value
    # cpu_freed drops CPU entries.
    assert "evicted-cpu" not in runner.storage.cpu_encoder_cache
    # cpu_get stages CPU hits into tmp unless already on device.
    assert runner.storage.tmp_encoder_cache["to-stage"] is copies.return_value
    assert copies.call_count == 2


def test_get_encoder_output_three_tier_priority():
    runner = _make_runner()
    resident, tmp, cpu = _tensor(1), _tensor(2), _tensor(3)

    # Resident tier wins; the has-check sees it too.
    runner.encoder_cache.encoder_outputs = {"mm": resident}
    assert runner._get_encoder_output_from_cache("mm") is resident
    assert runner._has_cached_encoder_output("mm")

    # Tmp tier is second.
    runner.encoder_cache.encoder_outputs = {}
    runner.storage.tmp_encoder_cache = {"mm": tmp}
    assert runner._get_encoder_output_from_cache("mm") is tmp
    assert runner._has_cached_encoder_output("mm")

    # CPU tier falls back to a synchronous copy that lands in tmp.
    runner.storage.tmp_encoder_cache = {}
    runner.storage.cpu_encoder_cache = {"mm": cpu}
    assert runner._has_cached_encoder_output("mm")
    staged = runner._get_encoder_output_from_cache("mm")
    assert runner.storage.tmp_encoder_cache["mm"] is staged
    assert torch.equal(staged, cpu)

    # Complete miss (all three tiers drained).
    runner.storage.tmp_encoder_cache = {}
    runner.storage.cpu_encoder_cache = {}
    assert not runner._has_cached_encoder_output("mm")
    assert runner._get_encoder_output_from_cache("mm") is None


def test_reset_encoder_cache_clears_all_tiers(monkeypatch):
    runner = _make_runner()
    runner.encoder_cache.encoder_outputs = {"mm": _tensor(1)}
    runner.storage.tmp_encoder_cache = {"mm2": _tensor(2)}
    runner.storage.cpu_encoder_cache = {"mm3": _tensor(3)}
    runner.storage._current_step_metadata = _metadata()
    runner.storage._req_refs = {"mm2": {"req-a"}}
    runner.storage._req_hash_index = {"req-a": {"mm2"}}
    runner.storage._pending_encoder_cache_copies = deque([(_tensor(4), MagicMock())])

    sync = MagicMock()
    monkeypatch.setattr(torch.npu, "synchronize", sync)

    runner.reset_encoder_cache()
    sync.assert_called_once_with()
    assert runner.encoder_cache.encoder_outputs == {}
    assert runner.storage.tmp_encoder_cache == {}
    assert runner.storage.cpu_encoder_cache == {}
    assert runner.storage._req_refs == {}
    assert runner.storage._req_hash_index == {}
    assert runner.storage._current_step_metadata is None
    assert len(runner.storage._pending_encoder_cache_copies) == 0

    # A fully drained runner resets without touching the device.
    runner.reset_encoder_cache()
    sync.assert_called_once_with()


def test_cache_encoder_outputs_batches_per_item():
    runner = _make_runner()
    hook = MagicMock()
    runner._cache_encoder_output = hook
    a, b = _tensor(1), _tensor(2)

    runner.cache_encoder_outputs(["mm-a", "mm-b"], [a, b])

    assert hook.call_count == 2
    assert hook.call_args_list[0].args == ("mm-a", a)
    assert hook.call_args_list[1].args == ("mm-b", b)


# ---------------------------------------------------------------------------
# Part 2: NPUModelRunner hook overrides (A9/A10 wiring).
# ---------------------------------------------------------------------------
def _make_model_runner(flag: bool) -> NPUModelRunner:
    """Bare NPUModelRunner harness with the score-cache knobs seeded."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.use_score_encoder_cache = flag
    runner.model_state = SimpleNamespace(encoder_runner=MagicMock())
    return runner


def test_add_requests_tracks_new_mm_requests(monkeypatch):
    parent = MagicMock()
    monkeypatch.setattr(GPUModelRunner, "add_requests", parent)
    with_mm = SimpleNamespace(req_id="req-a", mm_features=[_mm_feature("h1")])
    without_mm = SimpleNamespace(req_id="req-b", mm_features=[])
    scheduler_output = SimpleNamespace(scheduled_new_reqs=[with_mm, without_mm])

    runner = _make_model_runner(flag=True)
    runner.add_requests(scheduler_output)

    parent.assert_called_once_with(scheduler_output)
    runner.model_state.encoder_runner.track_request.assert_called_once_with(
        "req-a", with_mm.mm_features)

    # Flag off: bookkeeping skipped, upstream lifecycle untouched.
    parent.reset_mock()
    runner = _make_model_runner(flag=False)
    runner.add_requests(scheduler_output)
    parent.assert_called_once_with(scheduler_output)
    runner.model_state.encoder_runner.track_request.assert_not_called()


def test_process_encoder_cache_scheduler_output_branches(monkeypatch):
    parent = MagicMock()
    monkeypatch.setattr(
        GPUModelRunner, "_process_encoder_cache_scheduler_output", parent)
    metadata = _metadata()
    delegated = SimpleNamespace(ec_manager_metadata=metadata)
    legacy = SimpleNamespace(ec_manager_metadata=None)

    # Flag off: upstream default behavior (old free-list channel).
    runner = _make_model_runner(flag=False)
    runner._process_encoder_cache_scheduler_output(delegated)
    parent.assert_called_once_with(delegated)
    runner.model_state.encoder_runner.process_scheduler_metadata.assert_not_called()

    # Flag on but no manager metadata: fall back to upstream behavior.
    parent.reset_mock()
    runner = _make_model_runner(flag=True)
    runner._process_encoder_cache_scheduler_output(legacy)
    parent.assert_called_once_with(legacy)
    runner.model_state.encoder_runner.process_scheduler_metadata.assert_not_called()

    # Flag on with metadata: delegate to the score runner.
    parent.reset_mock()
    runner = _make_model_runner(flag=True)
    runner._process_encoder_cache_scheduler_output(delegated)
    parent.assert_not_called()
    runner.model_state.encoder_runner.process_scheduler_metadata.assert_called_once_with(
        metadata)


def test_hook1_and_reset_delegate_when_enabled(monkeypatch):
    runner = _make_model_runner(flag=True)
    runner._on_request_state_removed("req-a")
    runner.model_state.encoder_runner.on_request_removed.assert_called_once_with(
        "req-a")

    parent_reset = MagicMock()
    monkeypatch.setattr(GPUModelRunner, "reset_encoder_cache", parent_reset)
    runner.reset_encoder_cache()
    parent_reset.assert_called_once_with()
    runner.model_state.encoder_runner.reset_encoder_cache.assert_called_once_with()

    # Flag off keeps the base-class behavior only. The framework reset is
    # unconditional (one call per runner); the score-tier reset stays gated.
    runner = _make_model_runner(flag=False)
    runner._on_request_state_removed("req-b")
    runner.model_state.encoder_runner.on_request_removed.assert_not_called()
    runner.reset_encoder_cache()
    assert parent_reset.call_count == 2
    runner.model_state.encoder_runner.reset_encoder_cache.assert_not_called()
