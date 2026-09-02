# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep AFD's control plane out of model-based drafter dummy forwards.

During a target-model dummy run, AFD temporarily instruments vLLM's
``create_forward_context`` factory. Model-based proposers create another
forward context inside that scope, but their local model never crosses AFD's
attention/FFN boundary. Announcing that nested context leaves the FFN role
waiting for data that will never be sent.

The owning AFD runner installs this compatibility wrapper on its own proposer
instance. DFlash and ``SpecDecodeBaseProposer`` implementations (including the
Eagle family, draft-model proposers, Gemma4, and Step3p5 MTP) share the wrapped
``dummy_run`` signature. ``ExtractHiddenStatesProposer`` has the same contract
and is configured by the runner too. Custom proposers are intentionally out of
scope because the plugin cannot assume that their forwards stay local.

Real proposal forwards need no wrapper: AFD's context-factory provider is only
active around target dummy/capture runs.
"""

from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Protocol
from weakref import ref

if TYPE_CHECKING:
    import torch

_WRAPPED_ATTR = "_afd_dummy_run_announce_suppressed"


class _AFDMetadataRunner(Protocol):
    _afd_suppress_metadata_send: bool


class ModelBasedDrafter(Protocol):
    """Model-based vLLM proposer interface used by the AFD wrapper."""

    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None: ...


def configure_afd_drafter(
    proposer: ModelBasedDrafter,
    runner: _AFDMetadataRunner,
) -> None:
    """Suppress AFD announces only for one runner's nested drafter forwards."""

    if proposer.__dict__.get(_WRAPPED_ATTR, False):
        return

    original_dummy_run = type(proposer).dummy_run
    runner_ref = ref(runner)

    # Patch reason: a model-based drafter dummy forward is nested inside AFD's
    # target-runner context provider but does not cross the AFD boundary.
    # Patch functionality: retain AFD transaction installation while suppressing
    # its unmatched control-plane announce on this runner instance only.
    # Signature: matches vLLM v0.26.0 model-based proposer dummy_run methods.
    # Delegation exception: upstream dummy_run implementations own substantial,
    # proposer-specific logic; this wrapper changes only the AFD runner flag.
    # Upstream sources: vllm/v1/spec_decode/{llm_base_proposer,dflash,
    # extract_hidden_states}.py. Remove when vLLM exposes scoped nested-forward
    # metadata providers or retains the proposer-to-runner dependency itself.
    @wraps(original_dummy_run)
    def dummy_run(
        self: ModelBasedDrafter,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        owning_runner = runner_ref()
        if owning_runner is None:
            return original_dummy_run(
                self,
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )

        # ### PATCH START: scope suppression to the owning AFD runner.
        previous_suppress_send = owning_runner._afd_suppress_metadata_send
        owning_runner._afd_suppress_metadata_send = True
        try:
            original_dummy_run(
                self,
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )
        finally:
            owning_runner._afd_suppress_metadata_send = previous_suppress_send
        # ### PATCH END: scope suppression to the owning AFD runner.

    proposer.dummy_run = MethodType(dummy_run, proposer)
    proposer.__dict__[_WRAPPED_ATTR] = True


__all__ = ["configure_afd_drafter"]
