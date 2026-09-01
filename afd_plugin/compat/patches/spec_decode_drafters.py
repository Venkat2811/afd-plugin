# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep AFD's control plane out of speculative drafter forwards.

Patch reason: during dummy runs AFD wraps ``create_forward_context`` globally,
so a model-based drafter's forward (DFlash) is instrumented too. The drafter is
a dense model that never crosses the attention/FFN boundary, yet AFD announced
a transaction for it -- the FFN role then waits forever on data that never
arrives (deadlock), or the DP=1 metadata synthesis rejects the drafter's
context shape ("AFD DP=1 fallback only supports one stage").
Patch functionality: wrap ``DFlashProposer.dummy_run`` so the announce is
suppressed while transaction metadata installs stay intact.
Signature: matches upstream; no added parameters.
Upstream: vLLM v0.26.0, vllm/v1/spec_decode/dflash.py
"""

from __future__ import annotations

from functools import wraps

from afd_plugin.v1.worker.attention_metadata import suppress_afd_announce

_WRAPPED_ATTR = "_afd_announce_suppressed"


def _wrap_dummy_run(proposer_cls: type) -> None:
    original = proposer_cls.dummy_run
    if getattr(original, _WRAPPED_ATTR, False):
        return

    @wraps(original)
    def dummy_run(self, *args, **kwargs):
        with suppress_afd_announce():
            return original(self, *args, **kwargs)

    setattr(dummy_run, _WRAPPED_ATTR, True)
    proposer_cls.dummy_run = dummy_run


def _apply() -> None:
    # Eagle-family proposers run their own drafter forwards inside the same
    # dummy-run scope and are expected to need the same treatment; only DFlash
    # is wrapped here because only DFlash has been validated end to end.
    try:
        from vllm.v1.spec_decode.dflash import DFlashProposer
    except Exception:  # pragma: no cover - older vLLM without DFlash
        return
    _wrap_dummy_run(DFlashProposer)


_apply()
