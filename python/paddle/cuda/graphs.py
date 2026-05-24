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
"""PyTorch-compatible CUDA graph APIs.

Mirrors ``torch.cuda.graphs``: re-exports :class:`CUDAGraph` from
:mod:`paddle.device.cuda.graphs` and adds the :class:`graph` context manager
plus :func:`graph_pool_handle`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle import core, device as _paddle_device
from paddle.device.cuda.graphs import (
    CUDAGraph,
    is_cuda_graph_supported,
)

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    from paddle.device import Stream


__all__ = [
    "CUDAGraph",
    "graph",
    "graph_pool_handle",
    "is_cuda_graph_supported",
]


def graph_pool_handle() -> int:
    """Return an opaque token usable as the ``pool`` argument of
    :meth:`CUDAGraph.capture_begin` or :class:`graph`.

    Mirrors ``torch.cuda.graph_pool_handle``: graphs that share the same pool
    token also share an underlying memory pool, which is what enables their
    captured allocations to alias across replays.
    """
    if not is_cuda_graph_supported():
        raise RuntimeError(
            "CUDA Graph is only supported on PaddlePaddle compiled with "
            "NVIDIA GPU."
        )
    return core.CUDAGraph.gen_new_memory_pool_id()


class graph:
    """Context manager that wraps a CUDA graph capture.

    Mirrors ``torch.cuda.graph``: entering the context switches to
    ``stream`` (via :func:`paddle.cuda.stream`) when provided and calls
    ``cuda_graph.capture_begin(...)``; exiting calls
    ``cuda_graph.capture_end()`` and restores the previous stream.

    Args:
        cuda_graph (CUDAGraph): The :class:`CUDAGraph` instance to capture into.
        pool (int, optional): Memory pool token from :func:`graph_pool_handle`
            or another graph's :meth:`CUDAGraph.pool`. Passed through to
            ``capture_begin``.
        stream (paddle.cuda.Stream, optional): CUDA stream to capture on.
            When provided, the capture runs under
            ``paddle.cuda.stream(stream)`` so the previous stream is restored
            on exit. When ``None``, capture happens on the current stream.
        capture_error_mode (str, optional): Passed through to
            ``capture_begin``; only ``'global'`` is honored.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> g = paddle.cuda.CUDAGraph()
            >>> x = paddle.zeros([2, 3])
            >>> with paddle.cuda.graph(g):
            ...     y = x + 1
            >>> g.replay()
    """

    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: int | None = None,
        stream: Stream | None = None,
        capture_error_mode: str = 'global',
    ) -> None:
        # Attribute names mirror ``torch.cuda.graph`` so user code that
        # introspects ``g.cuda_graph`` / ``g.capture_stream`` / ``g.stream_ctx``
        # / ``g.pool`` / ``g.capture_error_mode`` ports across unchanged.
        self.cuda_graph = cuda_graph
        self.pool = pool
        self.capture_stream = stream
        self.capture_error_mode = capture_error_mode
        # ``paddle.device.stream(None)`` is a documented no-op, so the same
        # guard works whether or not the caller pinned a stream.
        self.stream_ctx = _paddle_device.stream(stream)

    def __enter__(self) -> Self:
        # Drain pending work and release cached memory so the capture pool
        # starts with maximal headroom (mirrors torch.cuda.graph.__enter__).
        # Paddle's C++ ``BeginCUDAGraphCapture`` does neither.
        _paddle_device.synchronize()
        _paddle_device.empty_cache()
        self.stream_ctx.__enter__()
        try:
            self.cuda_graph.capture_begin(
                pool=self.pool, capture_error_mode=self.capture_error_mode
            )
        except BaseException:
            self.stream_ctx.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.cuda_graph.capture_end()
        finally:
            self.stream_ctx.__exit__(exc_type, exc_value, traceback)
