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


def find_in_n_out(program):
    # find the all logical inputs and outpus of a program section
    ins = []
    outs = []
    return ins, outs


def get_cuda_graph_sections(program):
    # get all sections that should run under cuda graph and the corresponding idx
    block = program.global_block()
    cuda_graph_sections = []  # record all ops in every cuda graph sections
    sections_idx = []  # idx of all ops in every cuda graph sections
    internal_section = [
    ]  # ops between cuda graph wrapped op, may belong to a section
    internal_idx = [
    ]  # ops' idx between cuda graph wrapped op, may belong to a section
    current_section = []  # current recording cuda graph sections
    current_idx = []  # current recording cuda graph ops' idx
    current_cuda_graph_id = -1  # current recording cuda graph id
    for idx, op in enumerate(block.ops):
        # find cuda graph sections
        if op._cuda_graph_attr is not None:
            assert isinstance(op._cuda_graph_attr,
                              str), "cuda_graph_attr should be a str"
            cuda_graph_attrs = op._cuda_graph_attr.split(';')
            assert len(cuda_graph_attrs) == 3, "cuda graph attr should have three fields: " \
                                               "cuda graph mode, cuda graph memory pool id, cuda graph id"
            local_cuda_graph_id = int(cuda_graph_attrs[2])
            if local_cuda_graph_id == current_cuda_graph_id:
                if len(internal_section) > 0:
                    for internal_op in internal_section:
                        if internal_op.attr(
                                'op_role') == 256 or internal_op.attr(
                                    'op_role') == 257:
                            # The internal section contains loss related ops,
                            # although these ops are between two cuda graph sections with same graph id,
                            # they belong to none of these two sections.
                            # The loss related op should be wrapped by user explicitly.
                            internal_section = []
                            # Beside clear the internal section, a new cuda graph section should be recorded
                            assert len(current_section) == len(current_idx), \
                                "num of section's op is not equal with the idx"
                            if len(current_section) > 0:
                                # store previous section
                                cuda_graph_sections.append(current_section)
                                sections_idx.append(current_idx)
                                current_section = []
                                current_idx = []
                    # some ops inserted by optimizer, should be added to current section
                    for internal_op in internal_section:
                        current_section.append(internal_op)
                internal_section = []
                current_section.append(op)
                current_idx.append(idx)
            else:
                # current graph id is different with previous, start a new section of cuda graph
                internal_section = [
                ]  # internal ops belong to no section, just clear it
                internal_idx = [
                ]  # internal idx belong to no section, just clear it
                current_cuda_graph_id = local_cuda_graph_id  # start record a new section
                assert len(current_section) == len(
                    current_idx
                ), "num of section's op is not equal with num of idx"
                if len(current_section) > 0:
                    # store previous section
                    cuda_graph_sections.append(current_section)
                    sections_idx.append(current_idx)
                    current_section = [op]
                    current_idx = [idx]
        else:
            # recode ops which cuda_graph_attr is None, may belong to a section
            internal_section.append(op)
            internal_idx.append(idx)

    # handle the last section
    assert len(current_section) == len(
        current_idx), "num of section's op is not equal with num of idx"
    if len(current_section) > 0:
        # store previous section
        cuda_graph_sections.append(current_section)
        sections_idx.append(current_idx)

    return cuda_graph_sections, sections_idx


def cuda_graph_transform(program):
    # replace the ops marked with cuda_graph_attr to run_program_op to use cuda graph

    # step 1: get all cuda graph sections
    cuda_graph_sections, sections_idx = get_cuda_graph_sections(program)

    return program
