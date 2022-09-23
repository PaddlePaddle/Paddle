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
from paddle.fluid.layers.utils import _hash_with_id
from paddle.fluid.core import is_compiled_with_cuda, is_compiled_with_rocm, CUDAPlace
import warnings

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

    :param dst: framework.VarDesc(cpp), dst var desc, cpp VarDesc instance
    :param src: framework.VarDesc(cpp), src var desc, cpp VarDesc instance
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

    :param block: framework.Block, the original block
    :param begin_idx: int, from which idx (not include) to find the later ins
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

    :param section: list, one cuda graph section, list of ops
    :param origin_program: framework.Program, origin program
    :param section_idx: list, the section ops' idx corresponding to the cuda graph section, a list of idx
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
            elif later_ins.count(in_name) == 0 and outs.count(in_name) > 0:
                # this is var is generated from op inside this section, and only will be used inside this section
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

    :param program: framework.Program, the original program
    :return: A list of cuda graph sections and the corresponding ops' idx in the block.
             The program is under is test or not.
    """
    block = program.global_block()
    cuda_graph_sections = []  # record all ops in every cuda graph sections
    sections_idx = []  # idx of all ops in every cuda graph sections
    is_test = False  # will be set to True is any op's 'is_test' attr is True

    # ops and it's idx between cuda graph wrapped op, may belong to a section
    internal_section = []
    internal_idx = []

    current_section = []  # current recording cuda graph sections
    current_idx = []  # current recording cuda graph ops' idx
    current_cuda_graph_id = -1  # current recording cuda graph id
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    loss_op_role = int(core.op_proto_and_checker_maker.OpRole.Loss)
    backward_op_role = int(core.op_proto_and_checker_maker.OpRole.Backward)
    loss_grad_op_role = loss_op_role | backward_op_role

    for idx, op in enumerate(block.ops):
        if op.type == 'conditional_block' or op.type == 'while':
            assert op._cuda_graph_attr is None, "Cuda graph not support conditional block op and while op."
        if op.has_attr('is_test') and op.attr('is_test'):
            is_test = True
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
                    assert len(internal_section) == len(
                        internal_idx
                    ), "len of internal section should be equal with len of internal idx"
                    for internal_op in internal_section:
                        loss_related = (int(internal_op.attr(op_role_attr_name))
                                        == loss_op_role) or int(
                                            (internal_op.attr(op_role_attr_name)
                                             ) == loss_grad_op_role)
                        sub_block_related = (op.type == 'conditional_block'
                                             or op.type == 'while')
                        if loss_related or sub_block_related:
                            # If loss_related is True
                            # The internal section contains loss related ops,
                            # although these ops are between two cuda graph sections with same graph id,
                            # they belong to none of these two sections.
                            # The loss related op should be wrapped by user explicitly.

                            # If sub_block_related is True
                            # The internal section contains while op or conditional block op.
                            # These two ops are not supported by cuda graph. Won't extend the section.
                            internal_section = []
                            internal_idx = []
                            # Beside clear the internal section, a new cuda graph section should be recorded
                            assert len(current_section) == len(current_idx), \
                                "num of section's op is not equal with the idx"
                            if len(current_section) > 0:
                                # store previous section
                                cuda_graph_sections.append(current_section)
                                sections_idx.append(current_idx)
                            current_section = []
                            current_idx = []
                            break
                    # some ops inserted by some optimizer, should be added to current section
                    for i in range(len(internal_section)):
                        current_section.append(internal_section[i])
                        current_idx.append(internal_idx[i])
                internal_section = []
                internal_idx = []
                current_section.append(op)
                current_idx.append(idx)
            else:
                # current graph id is different with previous, start a new section of cuda graph
                # internal ops and idx belong to no section, just clear it
                internal_section = []
                internal_idx = []
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

    return cuda_graph_sections, sections_idx, is_test


def replace_cuda_graph_section(ins_and_outs, section_program, section_idx,
                               origin_program, cuda_graph_section, order,
                               is_test):
    """
    Use section_program and ins_and_outs to initialize a run_program_op,
    and replace the section_idx marks ops in the origin program.

    :param ins_and_outs: list, the logical ins and outs of the section program
    :param section_program: framework.Program, the partial program need to run under cuda graph
    :param section_idx: list, the idx need to be removed from origin program
    :param origin_program: framework.Program, the origin program
    :param cuda_graph_section: list, the ops in current sections, used to get the mode, memory pool id and is_test
    :param order: int, the order of current section, used to create unique cuda graph var
    :param is_test: bool, the program is running under is_test or not
    :return: no return
    """
    ins = ins_and_outs[0]
    outs = ins_and_outs[1]
    insert_idx = section_idx[0]
    origin_block = origin_program.global_block()

    for idx in reversed(section_idx):
        # remove all cuda graph marked ops from origin block
        origin_block._remove_op(idx, sync=False)

    mode = None
    memory_pool_id = None

    for op in cuda_graph_section:
        # find the cuda graph mode and memory pool id, determine is test or not
        if op._cuda_graph_attr is not None:
            attrs = op._cuda_graph_attr.split(';')
            mode = attrs[0]
            memory_pool_id = int(attrs[1])
            break

    assert mode is not None and memory_pool_id is not None, \
        "mode and memory pool id should be specified in cuda graph attr"

    cuda_graph_var = origin_block.create_var(
        name="cuda_graph_" + str(order),
        type=core.VarDesc.VarType.RAW,
        persistable=True,
        stop_gradient=True,
    )

    # not used for the run_program_op, just needed by the op, but won't be used
    out_scope_var = origin_block.create_var(
        name="program_out_scope_" + str(order),
        type=core.VarDesc.VarType.STEP_SCOPES,
        persistable=True,
        stop_gradient=True,
    )

    program_id = _hash_with_id(section_program, ins_and_outs)

    # insert the run_program_op into the block
    origin_block._insert_op(
        insert_idx,
        type='run_program',
        inputs={'X': ins},
        outputs={
            'Out': outs,
            'OutScope': out_scope_var,
            'CUDAGraph': cuda_graph_var
        },
        attrs={
            'global_block': section_program.global_block(),
            'start_op_index': 0,
            'end_op_index': len(section_program.global_block().ops),
            'is_test': is_test,
            'program_id': program_id,
            'cuda_graph_capture_mode': mode,
            'cuda_graph_pool_id': memory_pool_id,
            # Todo: now not support use interpretercore
            'use_interpretorcore': False,
            'forward_global_block': section_program.global_block(),
            'backward_global_block': section_program.global_block(),
        })


def cuda_graph_transform(program):
    """
    replace the ops marked with cuda_graph_attr to run_program_op to use cuda graph

    :param program: framework.Program, the program to be transformed
    :return: the cuda graph section program, user should hold these programs!
    """

    if len(program.blocks) > 1:
        # some sub blocks may be inserted by optimizer but will not use during training, just warn here
        warnings.warn(
            "Sub block(s) has been detected in the program. "
            "Cuda graph not support op with sub block, and it will only handle the global block."
        )

    # step 1: get all cuda graph sections.
    # A cuda graph section contains all ops marked with same cuda graph id and
    # some ops inserted by some optimizers (amp, sharding for example) between ops with same id.
    cuda_graph_sections, sections_idx, is_test = get_cuda_graph_sections(
        program)
    assert len(cuda_graph_sections) == len(sections_idx), \
        "num of cuda graph sections is not equal with num of idx sections"

    # step 2: construct new program for each section and find inputs and outputs of each section.
    # The inputs are variables generated outside the section but will be used by this section.
    # The outputs are variables generated by this section and will be used after the end of the section.
    ins_and_outs = []
    section_programs = []
    for i in range(len(cuda_graph_sections)):
        # creating new program for current section
        section_program, ins_outs = construct_program_and_find_ins_outs(
            cuda_graph_sections[i], program, sections_idx[i])
        ins_and_outs.append(ins_outs)
        section_programs.append(section_program)
    assert len(section_programs) == len(cuda_graph_sections), \
        "the num of cuda graph sections should be equal with the num of new program"

    # step 3: replace the ops in original program with run_program_op.
    # Will remove all ops in the section from origin program, and use run_program_op to replace them.
    for i in reversed(range(len(cuda_graph_sections))):
        # carry out the replacement in reversed order, to keep the previous idx intact
        replace_cuda_graph_section(ins_and_outs[i],
                                   section_programs[i],
                                   sections_idx[i],
                                   program,
                                   cuda_graph_sections[i],
                                   order=i,
                                   is_test=is_test)

    # NOTE: user should hold these program, for now just return these program back to caller
    return section_programs
