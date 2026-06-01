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
"""CUDA graph APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle import core, device as _paddle_device
from paddle.device.cuda.graphs import (
    CUDAGraph,
    is_cuda_graph_supported,
)

if TYPE_CHECKING:
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
    :meth:`CUDAGraph.capture_begin` or :class:`graph`. Graphs sharing the
    same token share an underlying memory pool.
    """
    if not is_cuda_graph_supported():
        raise RuntimeError(
            "CUDA Graph is only supported on PaddlePaddle compiled with "
            "NVIDIA GPU."
        )
    return core.CUDAGraph.gen_new_memory_pool_id()


class graph:
    """Context manager that wraps a CUDA graph capture.

    Args:
        cuda_graph (CUDAGraph): The :class:`CUDAGraph` instance to capture into.
        pool (int, optional): Memory pool token from :func:`graph_pool_handle`
            or another graph's :meth:`CUDAGraph.pool`.
        stream (paddle.cuda.Stream, optional): CUDA stream to capture on.
            When ``None``, capture happens on the current stream.
        capture_error_mode (str, optional): One of ``'global'``,
            ``'thread_local'``, ``'relaxed'``. When ``None`` (default) the
            underlying :class:`CUDAGraph`'s constructor ``mode`` is used.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
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
        capture_error_mode: str | None = None,
    ) -> None:
        self.cuda_graph = cuda_graph
        self.pool = pool
        self.capture_stream = stream
        self.capture_error_mode = capture_error_mode
        self.stream_ctx = _paddle_device.stream(stream)

    def __enter__(self) -> Self:
        # Synchronize the graph's own device, not the process-wide current
        # device which may be CPU (synchronize rejects non-accelerator places).
        _paddle_device.synchronize(self.cuda_graph._place)
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

    def __exit__(self, *args: object) -> None:
        try:
            self.cuda_graph.capture_end()
        finally:
            self.stream_ctx.__exit__(*args)
