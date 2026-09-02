# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from afd_plugin.compat.patches.spec_decode_drafters import configure_afd_drafter


class _Runner:
    def __init__(self, suppress_metadata_send=False):
        self._afd_suppress_metadata_send = suppress_metadata_send


class _RecordingDrafter:
    def __init__(self, runner, *, raises: bool = False):
        self.runner = runner
        self.raises = raises
        self.installs = 0
        self.announces = 0
        self.calls: list[tuple[int, bool, bool, dict[str, str] | None, bool]] = []

    def dummy_run(
        self,
        num_tokens,
        use_cudagraphs=True,
        is_graph_capturing=False,
        slot_mappings=None,
    ):
        self.calls.append(
            (
                num_tokens,
                use_cudagraphs,
                is_graph_capturing,
                slot_mappings,
                self.runner._afd_suppress_metadata_send,
            ),
        )
        # AFD metadata must still install for model/ubatch invariants. Only the
        # unmatched control-plane announce is conditional on the runner flag.
        self.installs += 1
        if not self.runner._afd_suppress_metadata_send:
            self.announces += 1
        if self.raises:
            raise RuntimeError("drafter failed")


class _DFlashDrafter(_RecordingDrafter):
    pass


class _EagleDrafter(_RecordingDrafter):
    pass


@pytest.mark.parametrize("drafter_cls", [_DFlashDrafter, _EagleDrafter])
def test_configured_model_drafter_installs_without_announcing(drafter_cls):
    runner = _Runner()
    drafter = drafter_cls(runner)

    configure_afd_drafter(drafter, runner)
    drafter.dummy_run(
        17,
        use_cudagraphs=False,
        is_graph_capturing=True,
        slot_mappings={"layer": "mapping"},
    )

    assert drafter.calls == [
        (17, False, True, {"layer": "mapping"}, True),
    ]
    assert drafter.installs == 1
    assert drafter.announces == 0
    assert runner._afd_suppress_metadata_send is False


def test_configured_drafter_is_per_runner_and_idempotent():
    runner = _Runner()
    other_runner = _Runner()
    drafter = _DFlashDrafter(runner)
    other_drafter = _DFlashDrafter(other_runner)

    configure_afd_drafter(drafter, runner)
    wrapped_dummy_run = drafter.dummy_run
    configure_afd_drafter(drafter, runner)

    drafter.dummy_run(1)
    other_drafter.dummy_run(1)

    assert drafter.dummy_run == wrapped_dummy_run
    assert drafter.calls[-1][-1] is True
    assert other_drafter.calls[-1][-1] is False
    assert runner._afd_suppress_metadata_send is False
    assert other_runner._afd_suppress_metadata_send is False


def test_configured_drafter_restores_runner_state_after_failure():
    runner = _Runner()
    drafter = _EagleDrafter(runner, raises=True)
    configure_afd_drafter(drafter, runner)

    with pytest.raises(RuntimeError, match="drafter failed"):
        drafter.dummy_run(1)

    assert runner._afd_suppress_metadata_send is False


def test_configured_drafter_preserves_preexisting_suppression():
    runner = _Runner(suppress_metadata_send=True)
    drafter = _DFlashDrafter(runner)
    configure_afd_drafter(drafter, runner)

    drafter.dummy_run(1)

    assert drafter.calls[-1][-1] is True
    assert runner._afd_suppress_metadata_send is True
