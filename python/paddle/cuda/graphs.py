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

from paddle import core
from paddle.device.cuda.graphs import (
    CUDAGraph,
    is_cuda_graph_supported,
)

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


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

    Mirrors ``torch.cuda.graph``: entering the context calls
    ``cuda_graph.capture_begin(...)``, exiting calls
    ``cuda_graph.capture_end()``.

    Args:
        cuda_graph (CUDAGraph): The :class:`CUDAGraph` instance to capture into.
        pool (int, optional): Memory pool token from :func:`graph_pool_handle`
            or another graph's :meth:`CUDAGraph.pool`. Passed through to
            ``capture_begin``.
        stream (paddle.cuda.Stream, optional): CUDA stream to capture on.
            Accepted for ``torch.cuda.graph`` parity; Paddle currently
            captures on the current stream and ignores this value.
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
        stream=None,
        capture_error_mode: str = 'global',
    ) -> None:
        self.cuda_graph = cuda_graph
        self.pool = pool
        self.stream = stream
        self.capture_error_mode = capture_error_mode

    def __enter__(self) -> Self:
        self.cuda_graph.capture_begin(
            pool=self.pool, capture_error_mode=self.capture_error_mode
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cuda_graph.capture_end()
