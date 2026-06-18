# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import os
import warnings
from typing import NoReturn, overload

from paddle.base.core import (
    CUDAPlace,
    CustomPlace,
    XPUPlace,
    get_all_custom_device_type,
    is_compiled_with_cuda,
    is_compiled_with_custom_device,
    is_compiled_with_rocm,
    is_compiled_with_xpu,
)


def check_compiled_with_custom_device():
    custom_device_flag = False
    custom_devices_types = get_all_custom_device_type()
    for device_type in custom_devices_types:
        if is_compiled_with_custom_device(device_type):
            custom_device_flag = True
            break
    return custom_device_flag


if (
    is_compiled_with_cuda()
    or is_compiled_with_rocm()
    or check_compiled_with_custom_device()
    or is_compiled_with_xpu()
):
    from paddle.base.core import CUDAGraph as CoreCUDAGraph

    def is_cuda_graph_supported():
        return True
else:
    CoreCUDAGraph = None

    def is_cuda_graph_supported():
        return False


def current_expected_place():
    for device in get_all_custom_device_type():
        selected_devices = os.getenv(f"FLAGS_selected_{device}s", "0").split(
            ","
        )
        device_id = int(selected_devices[0])
        return CustomPlace(device, device_id)
    return None


ALL_MODES = ["global", "thread_local", "relaxed"]
cuda_graph_id = 0


class CUDAGraph:
    """
    The native Paddle constructor takes ``place``, ``mode``, ``pool_id`` and
    ``enable_replace``; the PyTorch-compatible ``keep_graph`` keyword is
    accepted as well. ``capture_begin`` additionally accepts the PyTorch
    keywords ``pool`` and ``capture_error_mode`` so the same instance can be
    driven from either API style.
    """

    @overload
    def __init__(self, keep_graph: bool, /) -> None: ...

    @overload
    def __init__(
        self,
        place: CUDAPlace | XPUPlace | CustomPlace | None = None,
        mode: str = "thread_local",
        pool_id: int | None = None,
        enable_replace: bool = False,
        *,
        keep_graph: bool = False,
    ) -> None: ...

    def __init__(
        self,
        place=None,
        mode="thread_local",
        pool_id=None,
        enable_replace=False,
        *,
        keep_graph: bool = False,
    ):
        assert CoreCUDAGraph is not None, (
            "CUDA Graph is only supported on PaddlePaddle compiled with NVIDIA GPU."
        )

        if isinstance(place, bool):
            if keep_graph is not False:
                raise TypeError(
                    "keep_graph is specified both positionally and by keyword"
                )
            keep_graph = place
            place = None

        self._graph = None
        if place is None and check_compiled_with_custom_device():
            place = current_expected_place()
        elif place is None:
            if is_compiled_with_cuda():
                device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = CUDAPlace(device_id)
            elif is_compiled_with_xpu():
                device_id = int(os.environ.get('FLAGS_selected_xpus', 0))
                place = XPUPlace(device_id)
            else:
                raise RuntimeError("Not Supported devices")

        self._place = place
        assert mode in ALL_MODES
        self._mode = ALL_MODES.index(mode)
        self._pool_id = pool_id
        self._enable_replace = enable_replace
        self._keep_graph = keep_graph
        self._debug_mode = False

    def capture_begin(
        self, pool: int | None = None, capture_error_mode: str | None = None
    ) -> None:
        """Begin capturing CUDA work on the current stream.

        Args:
            pool (int, optional): A memory pool token from
                :func:`paddle.cuda.graph_pool_handle` or another graph's
                :meth:`pool`. When provided, this graph shares the indicated
                memory pool. Overrides ``pool_id`` from the constructor.
            capture_error_mode (str, optional): One of ``'global'``,
                ``'thread_local'``, ``'relaxed'`` (see :data:`ALL_MODES`).
                When ``None`` (default) the constructor's ``mode`` is used;
                otherwise it overrides the constructor for this capture and a
                :class:`UserWarning` is emitted to flag the precedence.
                Invalid values raise :class:`ValueError`.
        """
        if pool is not None:
            self._pool_id = pool
        elif self._pool_id is None:
            self._pool_id = CoreCUDAGraph.gen_new_memory_pool_id()

        if capture_error_mode is None:
            mode = self._mode
        else:
            if capture_error_mode not in ALL_MODES:
                raise ValueError(
                    f"capture_error_mode must be one of {ALL_MODES}, "
                    f"but got {capture_error_mode!r}."
                )
            mode = ALL_MODES.index(capture_error_mode)
            if mode != self._mode:
                warnings.warn(
                    f"capture_error_mode={capture_error_mode!r} differs from "
                    f"the constructor mode={ALL_MODES[self._mode]!r}; the "
                    f"explicit capture_error_mode takes precedence for this "
                    f"capture.",
                    stacklevel=2,
                )

        CoreCUDAGraph.begin_capture_with_pool_id(
            self._place, mode, self._pool_id, self._enable_replace
        )

    def capture_end(self):
        self._graph = CoreCUDAGraph.end_capture()

    def _require_captured(self) -> None:
        """Raise a clear error if no graph has been captured yet.

        ``self._graph`` is only populated by :meth:`capture_end`; methods that
        consume it (``replay`` / ``reset`` / ``debug_dump`` / ...) would
        otherwise raise ``AttributeError`` on ``NoneType`` when called too
        early. Centralizing the check produces a single, actionable message.
        """
        if self._graph is None:
            raise RuntimeError(
                "CUDAGraph has not been captured yet. "
                "Call capture_begin/capture_end first."
            )

    def instantiate(self) -> CoreCUDAGraph:
        """Return the instantiated core CUDA graph held by this wrapper.

        Paddle builds the executable graph eagerly inside :meth:`capture_end`,
        so by the time this is called the graph is already instantiated. It is
        kept for source compatibility with ``torch.cuda.CUDAGraph.instantiate``
        and returns the held core :class:`~paddle.base.core.CUDAGraph` produced
        by :meth:`capture_end`.
        """
        self._require_captured()
        return self._graph

    def replay(self):
        self._require_captured()
        self._graph.replay()

    def reset(self):
        self._require_captured()
        self._graph.reset()

    def pool(self) -> int:
        """Return an opaque integer token representing this graph's memory pool.

        The token can be passed as the ``pool`` argument to another graph's
        :meth:`capture_begin` (or to :class:`paddle.cuda.graph`) so the two
        graphs share the same memory pool.
        """
        if self._pool_id is None:
            self._pool_id = CoreCUDAGraph.gen_new_memory_pool_id()
        return self._pool_id

    def enable_debug_mode(self) -> None:
        """Enable debug mode so that :meth:`debug_dump` is permitted."""
        self._debug_mode = True

    def debug_dump(self, debug_path) -> None:
        """Dump the captured graph to ``debug_path`` for inspection.

        :meth:`enable_debug_mode` must be called first.
        """
        if not self._debug_mode:
            raise RuntimeError(
                "debug_dump requires debug mode to be enabled first. "
                "Call enable_debug_mode() before debug_dump()."
            )
        self._require_captured()
        self.print_to_dot_files(debug_path)

    def raw_cuda_graph(self) -> NoReturn:
        """Paddle does not expose the raw ``cudaGraph_t`` handle."""
        raise NotImplementedError(
            "raw_cuda_graph is not yet supported in Paddle CUDAGraph. "
            "The underlying cudaGraph_t handle is not exposed by the Python "
            "binding."
        )

    def raw_cuda_graph_exec(self) -> NoReturn:
        """Paddle does not expose the raw ``cudaGraphExec_t`` handle."""
        raise NotImplementedError(
            "raw_cuda_graph_exec is not yet supported in Paddle CUDAGraph. "
            "The underlying cudaGraphExec_t handle is not exposed by the "
            "Python binding."
        )

    def print_to_dot_files(self, dirname, flags=None):
        if not isinstance(dirname, (str, bytes)):
            dirname = dirname.name
        os.makedirs(name=dirname, exist_ok=True)
        assert os.path.isdir(dirname), (
            f"The dirname {dirname} should be a directory"
        )
        if flags is None:
            flags = 2047  # only all information. It can be any integer inside [1, 2048)
        self._graph.print_to_dot_files(dirname, flags)

    def replace_input_ptrs(self, old_ptrs, new_ptrs):
        self._graph.replace_input_ptrs(old_ptrs, new_ptrs)
