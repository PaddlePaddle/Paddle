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
from paddle.fluid import core
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


def copy_var_desc(dst, src):
    """
    copy var desc from src to dst

    :param dst: dst var desc, cpp VarDesc instance
    :param src: src var desc, cpp VarDesc instance
    :return: no return
    """
    dst.set_shape(src.shape)
    dst.set_dtype(src.dtype)
    dst.set_lod_level(src.lod_level)
    dst.set_type(src.type)
    dst.set_persistable(src.persistable)
    dst.set_is_parameter(src.is_parameter)
    dst.set_stop_gradient(src.stop_gradient)


def all_inputs_of_later_op(block, begin_idx):
    """
    find all inputs of ops after an idx, used to determine the logical output of a cuda graph section

    :param block: the original block
    :param begin_idx: from which idx (not include) to find the later ins
    :return: a list of inputs names for all ops behind begin_idx
    """
    ins = []
    for idx, op in enumerate(block.ops):
        if idx <= begin_idx:
            continue
        for in_name in op.input_arg_names:
            ins.append(in_name)
    return list(set(ins))


def construct_program_and_find_ins_outs(section, origin_program, section_idx):
    """
    1. Construct a new program for corresponding section
    2. Find all the logical inputs and outputs of a program section

    :param section: one cuda graph section, list of ops
    :param origin_program: origin program
    :param section_idx: the section ops' idx corresponding to the cuda graph section, a list of idx
    :return: a new program for the cuda graph section
             the logical ins and outs of the cuda graph section
    """
    program = paddle.static.Program()
    block = program.global_block()
    origin_block = origin_program.global_block()
    ins = []
    outs = []
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    later_ins = all_inputs_of_later_op(origin_block, section_idx[-1])
    for op in section:
        for in_name in op.input_arg_names:
            var = origin_block.var(in_name)
            new_var_desc = block.desc.var(var.name.encode("ascii"))
            copy_var_desc(new_var_desc, var)
            if outs.count(in_name) == 0 and ins.count(in_name) == 0:
                # This in var is generated from op outside this section
                # Only record once for same input
                ins.append(in_name)
            elif later_ins.count(in_name) == 0:
                # this is var is generated from op inside this section, and only will be used inside this section
                # remove one in_name from the outs
                outs.remove(in_name)
        for out_name in op.output_arg_names:
            var = origin_block.var(out_name)
            new_var_desc = block.desc.var(var.name.encode("ascii"))
            copy_var_desc(new_var_desc, var)
            # for every output, we add it to the section's outs
            if outs.count(out_name) == 0:
                # Only record one out var even if it will be generated by multi ops.
                # For scenario like this:
                # A = op1(a)
                # A = op2(b)
                # B = op3(A)
                outs.append(out_name)
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(op.desc)
        new_op_desc._set_attr(op_role_attr_name, op.attr(op_role_attr_name))
    program._sync_with_cpp()
    return program, [ins, outs]


def get_cuda_graph_sections(program):
    """
    get all sections that should run under cuda graph and the corresponding idx

    :param program: the original program
    :return: a list of cuda graph sections and the corresponding ops' idx in the block
    """
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
    """
    replace the ops marked with cuda_graph_attr to run_program_op to use cuda graph

    :param program: the program to be transformed
    :return: the updated program
    """

    # step 1: get all cuda graph sections
    cuda_graph_sections, sections_idx = get_cuda_graph_sections(program)
    assert len(cuda_graph_sections) == len(sections_idx), \
        "num of cuda graph sections is not equal with num of idx sections"

    # step 2: construct new program for each section and find inputs and outputs of each section
    ins_and_outs = []
    section_programs = []
    for i in range(len(cuda_graph_sections)):
        # creating new program for current section
        section_program, ins_outs = construct_program_and_find_ins_outs(
            cuda_graph_sections[i], program, sections_idx[i])
        ins_and_outs.append(ins_outs)
        section_programs.append(section_program)

    return program
