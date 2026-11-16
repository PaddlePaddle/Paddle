# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic publisher for pipeline scheduler events.

The pipeline schedulers know things that nothing outside them can recover:
scheduler-wide micro step identity, the virtual pipeline ``chunk_id``, which of
the two ``(micro, chunk)`` pairs an overlapped window's forward half belongs to,
whether a particular p2p call was posted asynchronously, and where ``dW`` and
early recompute actually ran. A profiler that guesses any of these silently
misattributes time.

So the schedulers publish those facts, and whoever wants them registers a
listener::

    from paddle.distributed.fleet.meta_parallel.pipeline_event_hook import (
        get_pipeline_event_hook,
    )

    hook = get_pipeline_event_hook()
    handle = hook.register(lambda event, payload: ...)
    ...
    hook.unregister(handle)

Paddle does not know what a listener does with the events. No timing, no CUDA
events, no files, no synchronization happens here -- that all belongs to the
listener.

Instrumenting a region takes one line, not two::

    with self._pp_hook.forward(micro=micro_step, chunk=chunk_id):
        output_tensor = self._forward_step(...)

A region cannot be left open: ``__exit__`` publishes the end event even when the
body raises. That removes a whole class of trace defect -- a begin with no end,
which is what a truncated span looks like offline.

With no listener registered, ``span()`` returns a module-level do-nothing
context, so nothing is allocated and the cost is one truth test on a list. That
is why these are safe to leave in the per-microbatch path.

The event vocabulary below is the contract. Adding a field to a payload is
backwards compatible; renaming an event is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "PipelineEventHook",
    "get_pipeline_event_hook",
    "pipeline_event_span",
    "pipeline_event_mark",
    "NULL_SPAN",
]


# ── event vocabulary ────────────────────────────────────────────────────
# Boundaries of one optimizer step. A listener that measures time takes its
# origin here; everything else is relative to it.
STEP_BEGIN = "step_begin"
STEP_END = "step_end"

# warmup / steady / cooldown. payload: phase, micro
PHASE_BEGIN = "phase_begin"
PHASE_END = "phase_end"

# payload: micro, [chunk]
FORWARD_BEGIN = "forward_begin"
FORWARD_END = "forward_end"
BACKWARD_BEGIN = "backward_begin"
BACKWARD_END = "backward_end"

# WeightGradStore.pop(): the weight-grad half deferred out of backward.
# payload: micro, [chunk]
WEIGHT_GRAD_BEGIN = "weight_grad_begin"
WEIGHT_GRAD_END = "weight_grad_end"

# RecomputeStore.run(): recompute run ahead of the backward that needs it.
# payload: micro, [chunk]
RECOMPUTE_BEGIN = "recompute_begin"
RECOMPUTE_END = "recompute_end"

# Posting a p2p transfer. payload: op, micro, [chunk], [is_async]
#
# ``is_async`` has to come from the call site: whether a given call is
# asynchronous depends on the ``overlap_p2p_comm`` argument *it* passes, not on
# any global flag. A scheduler that mixes both (Interleave posts async in
# warmup/steady but calls the cooldown send_backward_recv_backward
# synchronously) cannot be told apart any other way, and getting it wrong turns
# real communication into unexplained idle time.
P2P_ISSUE_BEGIN = "p2p_issue_begin"
P2P_ISSUE_END = "p2p_issue_end"

# The calc stream parked on req.wait(). payload: direction, micro, [chunk], [role]
#
# ``role`` splits the same stall by what was being waited on: one
# ``send_x_recv_x`` returns the isend to the downstream stage *and* the irecv
# from the upstream one in one list, and the two halves belong to microbatches
# one apart, so a single span can only label one of them.
# "OL" means the wait happened inside the model's overlapped_forward_backward
# rather than in the scheduler, so it has no matching issue.
P2P_WAIT_BEGIN = "p2p_wait_begin"
P2P_WAIT_END = "p2p_wait_end"

# Finer-grained spans added from model code. payload: name, [which], [layer]
SUB_SPAN_BEGIN = "sub_span_begin"
SUB_SPAN_END = "sub_span_end"

DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"

# ── span kinds ──────────────────────────────────────────────────────────
# A kind is the pair of events: ``forward`` -> forward_begin / forward_end.
# The wire vocabulary is unchanged; this is only the call-site surface.
SPAN_PHASE = "phase"
SPAN_FORWARD = "forward"
SPAN_BACKWARD = "backward"
SPAN_WEIGHT_GRAD = "weight_grad"
SPAN_RECOMPUTE = "recompute"
SPAN_P2P_ISSUE = "p2p_issue"
SPAN_P2P_WAIT = "p2p_wait"
SPAN_SUB = "sub_span"

SPAN_KINDS = (
    SPAN_PHASE,
    SPAN_FORWARD,
    SPAN_BACKWARD,
    SPAN_WEIGHT_GRAD,
    SPAN_RECOMPUTE,
    SPAN_P2P_ISSUE,
    SPAN_P2P_WAIT,
    SPAN_SUB,
)


class _NullSpan:
    """Do-nothing context used when nobody is listening.

    A module-level singleton on purpose: entering an instrumented region must not
    allocate when profiling is off, and there is nothing to keep per region.
    """

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_NULL_SPAN = _NullSpan()

# Exported so that instrumentation living outside a scheduler (p2p_communication's
# helper methods) can opt out of a span without reaching for a private name: a p2p
# call coming from a path that has no microbatch identity to give -- dualpipev runs
# its own schedule -- must not emit an event no listener can attribute.
NULL_SPAN = _NULL_SPAN


class _Span:
    """Publishes ``<kind>_begin`` on entry and ``<kind>_end`` on exit.

    The end event carries the same payload as the begin one. Listeners pair them
    by identity plus occurrence, so the payload must not drift between the two --
    that is the whole reason this is a context manager and not two calls.
    """

    __slots__ = ("_hook", "_begin", "_end", "_payload")

    def __init__(self, hook, kind, payload):
        self._hook = hook
        self._begin = kind + "_begin"
        self._end = kind + "_end"
        self._payload = payload

    def __enter__(self):
        self._hook.emit(self._begin, **self._payload)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Also on the exception path: a region left open looks exactly like a
        # truncated trace offline, and then nothing can be trusted.
        self._hook.emit(self._end, **self._payload)
        return False


class PipelineEventHook:
    """Listener registry for pipeline scheduler events."""

    def __init__(self):
        self._listeners: dict[int, Callable] = {}
        self._next_handle = 0
        self._context: dict = {}

    # ── static context ─────────────────────────────────────────────────
    def set_context(self, **kwargs) -> None:
        """Publish the facts that do not change during a run.

        Topology (stage id, number of stages, ranks) and the p2p batching mode
        belong here rather than on every event: repeating them a hundred
        thousand times per step buys nothing.
        """
        self._context.update(kwargs)

    @property
    def context(self) -> dict:
        return dict(self._context)

    # ── listeners ──────────────────────────────────────────────────────
    def register(self, listener: Callable) -> int:
        """Register ``listener(event, payload)``; returns a handle to remove it."""
        self._next_handle += 1
        self._listeners[self._next_handle] = listener
        return self._next_handle

    def unregister(self, handle: int) -> None:
        self._listeners.pop(handle, None)

    def clear(self) -> None:
        self._listeners.clear()

    @property
    def enabled(self) -> bool:
        return bool(self._listeners)

    # ── publishing ─────────────────────────────────────────────────────
    def emit(self, event: str, **payload) -> None:
        """Publish one event. Costs one truth test when nobody is listening.

        Use this for the step boundaries, whose begin and end sit in different
        places. Everything that brackets a region should use the span helpers
        below instead, so it cannot be left half-open.
        """
        if not self._listeners:
            return
        for listener in list(self._listeners.values()):
            listener(event, payload)

    # ── regions ────────────────────────────────────────────────────────
    def span(self, kind: str, **payload):
        """``with hook.span(kind, ...)``: publishes begin on entry, end on exit.

        Returns the shared do-nothing context when nobody is listening, so an
        instrumented region costs one truth test and allocates nothing.
        """
        if not self._listeners:
            return _NULL_SPAN
        return _Span(self, kind, payload)

    # The helpers below exist so a call site reads as what it is, and so a typo
    # in a payload key is a TypeError here rather than a silently dropped field
    # a hundred thousand events later.
    def phase(self, name: str, micro: int | None = None):
        """warmup / steady / cooldown.

        ``micro`` is only context for the listener; the phase itself is bounded
        by the region, so begin and end carry the same value.
        """
        return self.span(SPAN_PHASE, phase=name, micro=micro)

    def forward(self, micro: int, chunk: int | None = None):
        return self.span(SPAN_FORWARD, micro=micro, chunk=chunk)

    def backward(self, micro: int, chunk: int | None = None):
        return self.span(SPAN_BACKWARD, micro=micro, chunk=chunk)

    def weight_grad(self, micro: int, chunk: int | None = None):
        """WeightGradStore.pop(): the weight-grad half deferred out of backward."""
        return self.span(SPAN_WEIGHT_GRAD, micro=micro, chunk=chunk)

    def recompute(self, micro: int, chunk: int | None = None):
        """RecomputeStore.run(): recompute run ahead of the backward that needs it."""
        return self.span(SPAN_RECOMPUTE, micro=micro, chunk=chunk)

    def p2p_issue(
        self,
        op: str,
        micro: int,
        chunk: int | None = None,
        is_async: bool = False,
    ):
        """Posting a p2p transfer.

        ``is_async`` has to come from the call site: whether a given call is
        asynchronous depends on the ``overlap_p2p_comm`` argument *it* passes, not
        on any global flag. A scheduler that mixes both (Interleave posts async in
        warmup/steady but calls the cooldown send_backward_recv_backward
        synchronously) cannot be told apart any other way, and getting it wrong
        turns real communication into unexplained idle time.
        """
        return self.span(
            SPAN_P2P_ISSUE, op=op, micro=micro, chunk=chunk, is_async=is_async
        )

    def p2p_wait(
        self,
        direction: str,
        micro: int,
        chunk: int | None = None,
        role: str | None = None,
    ):
        """The calc stream parked on req.wait().

        ``role`` splits the same stall by what was being waited on: one
        ``send_x_recv_x`` returns the isend to the downstream stage *and* the
        irecv from the upstream one in one list, and the two halves belong to
        microbatches one apart, so a single span can only label one of them --
        which is how "the gradient arrived long ago but it is still waiting"
        becomes unexplainable on a trace. "OL" means the wait happened inside
        the model's overlapped_forward_backward rather than in the scheduler, so
        it has no matching issue.
        """
        return self.span(
            SPAN_P2P_WAIT,
            direction=direction,
            micro=micro,
            chunk=chunk,
            role=role,
        )


_HOOK = PipelineEventHook()


def get_pipeline_event_hook() -> PipelineEventHook:
    """The process-wide hook. One pipeline per process, so one hook."""
    return _HOOK


def pipeline_event_span(event, which=None, layer=None):
    """Add a sub-span from model code::

        with pipeline_event_span("attn.L12"):
            ...

    No-op (and no allocation) when nobody is listening.

    The scheduler's identity for the enclosing forward/backward is not passed:
    the listener latches it from the enclosing ``forward_*`` / ``backward_*``
    event, so a sub-span is attributed to the right ``(micro, chunk)`` without
    model code knowing anything about microbatches.

    ``layer`` tags the span with a layer number, needed when one chunk holds
    more than one transformer layer and the same span name fires twice per
    window.

    Same-named spans must be strictly sequential, never nested or interleaved.
    Put a discriminator in the name (e.g. ``f"attn.L{layer_number}"``) if that
    cannot be guaranteed.
    """
    return _HOOK.span(SPAN_SUB, name=event, which=which, layer=layer)


def pipeline_event_mark(event, phase, which=None, layer=None) -> None:
    """Sub-span as two separate calls. Prefer :func:`pipeline_event_span`.

    Only for the places where begin and end genuinely cannot sit in one block.
    A region opened here and never closed looks exactly like a truncated trace
    offline, so the listener will report it rather than guess.
    """
    if not _HOOK.enabled:
        return
    _HOOK.emit(
        SUB_SPAN_BEGIN if phase == "begin" else SUB_SPAN_END,
        name=event,
        which=which,
        layer=layer,
    )
