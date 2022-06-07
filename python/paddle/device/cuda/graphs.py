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

import os
import paddle
from paddle.fluid.core import is_compiled_with_cuda, is_compiled_with_rocm, CUDAPlace

if is_compiled_with_cuda() and not is_compiled_with_rocm():
    from paddle.fluid.core import CUDAGraph as CoreCUDAGraph

    def is_cuda_graph_supported():
        return True
else:
    CoreCUDAGraph = None

    def is_cuda_graph_supported():
        return False


ALL_MODES = ["global", "thread_local", "relaxed"]
cuda_graph_id = 0


class CUDAGraph:

    def __init__(self, place=None, mode="thread_local"):
        assert CoreCUDAGraph is not None, "CUDA Graph is only supported on PaddlePaddle compiled with NVIDIA GPU."

        self._graph = None
        if place is None:
            device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            place = CUDAPlace(device_id)
        self._place = place
        assert mode in ALL_MODES
        self._mode = ALL_MODES.index(mode)

    def capture_begin(self):
        CoreCUDAGraph.begin_capture(self._place, self._mode)

    def capture_end(self):
        self._graph = CoreCUDAGraph.end_capture()

    def replay(self):
        self._graph.replay()

    def reset(self):
        self._graph.reset()

    def print_to_dot_files(self, dirname, flags=None):
        if not isinstance(dirname, (str, bytes)):
            dirname = dirname.name
        os.makedirs(name=dirname, exist_ok=True)
        assert os.path.isdir(
            dirname), "The dirname {} should be a directory".format(dirname)
        if flags is None:
            flags = 2047  # only all information. It can be any integer inside [1, 2048)
        self._graph.print_to_dot_files(dirname, flags)


def wrap_cuda_graph(function, mode="thread_local", memory_pool="default"):
    assert mode in ALL_MODES
    if not paddle.in_dynamic_mode():
        # static mode
        from paddle.fluid.framework import _cuda_graph_guard
        global cuda_graph_id
        graph_id = str(cuda_graph_id)
        cuda_graph_id += 1
        if memory_pool == 'default':
            memory_pool_id = 0
        elif memory_pool == 'new':
            memory_pool_id = CoreCUDAGraph.gen_new_memory_pool_id()
        else:
            raise ValueError(
                "memory_pool should be one of default or new under static mode, but got",
                memory_pool)
        return _cuda_graph_guard(
            mode + ';' + str(memory_pool_id) + ';' +
            graph_id)(lambda *args, **kwargs: function(*args, **kwargs))

    from paddle.jit import to_static
    from paddle.nn import Layer
    new_function = to_static(function)
    if isinstance(function, Layer):
        mock_func = new_function.forward
    else:
        mock_func = new_function
    mock_func._cuda_graph_capture_mode = mode
    if memory_pool == "default":
        mock_func._cuda_graph_pool_id = 0
    elif memory_pool == "new":
        mock_func._cuda_graph_pool_id = CoreCUDAGraph.gen_new_memory_pool_id()
    else:
        if isinstance(memory_pool, Layer):
            mock_func._cuda_graph_pool_id = memory_pool.forward._cuda_graph_pool_id
        else:
            mock_func._cuda_graph_pool_id = memory_pool._cuda_graph_pool_id
    return new_function
