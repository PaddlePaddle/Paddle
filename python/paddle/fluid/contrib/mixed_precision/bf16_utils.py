#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import struct

from ... import core
from ... import framework
from ... import layers
from ... import global_scope
from ...log_helper import get_logger
from ...wrapped_decorator import signature_safe_contextmanager
from .bf16_lists import AutoMixedPrecisionListsBF16
import collections
import logging
import numpy as np

__all__ = [
    "bf16_guard", "rewrite_program_bf16", "convert_float_to_uint16",
    "convert_uint16_to_float"
]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

_valid_types = [
    core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.SELECTED_ROWS,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY
]

_bf16_guard_pattern = "__use_bf16__"


def convert_float_to_uint16(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack('<I', struct.pack('<f', x))[0] >> 16,
        otypes=[np.uint16])(in_list.flat)
    return np.reshape(out, in_list.shape)


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0],
        otypes=[np.float32])(in_list.flat)
    return np.reshape(out, in_list.shape)


def _rename_arg(op, old_name, new_name):
    """
    If an op has old_name input and output, rename these input
    args new_name.

    Args:
        op (Operator): Current operator.
        old_name (str): The old name of input args.
        new_name (str): The new name of input args.
    """
    op_desc = op.desc
    if isinstance(op_desc, tuple):
        op_desc = op_desc[0]
    op_desc._rename_input(old_name, new_name)
    op_desc._rename_output(old_name, new_name)


def _rename_op_input(program, op_var_rename_map, origin_ops, keep_fp32_ops):
    for block in program.blocks:
        ops = block.ops
        block_id = block.idx
        for op in ops:
            if op not in origin_ops or op in keep_fp32_ops:
                continue
            for name in op.input_arg_names:
                if name in op_var_rename_map[block_id]:
                    op._rename_input(name, op_var_rename_map[block_id][name])


def _dtype_to_str(dtype):
    """
    Convert specific variable type to its corresponding string.

    Args:
        dtype (VarType): Variable type.
    """
    if dtype == core.VarDesc.VarType.BF16:
        return 'bf16'
    else:
        return 'fp32'


def _insert_cast_op(block, op, idx, src_dtype, dest_dtype):
    """
    Insert cast op and rename args of input and output.

    Args:
        block (Program): The block in which the operator is.
        op (Operator): The operator to insert cast op.
        idx (int): The index of current operator.
        src_dtype (VarType): The input variable dtype of cast op.
        dest_dtype (VarType): The output variable dtype of cast op.

    Returns:
        num_cast_op (int): The number of cast ops that have been inserted.
    """
    num_cast_ops = 0

    for in_name in op.input_names:
        if src_dtype == core.VarDesc.VarType.FP32 and op.type in [
                'batch_norm', 'fused_bn_add_activation', 'layer_norm'
        ]:
            if in_name not in {'X', 'Z'}:
                continue
        for in_var_name in op.input(in_name):
            in_var = block.var(in_var_name)
            if in_var.type not in _valid_types or in_var.dtype == dest_dtype:
                continue
            if in_var.dtype == src_dtype:
                cast_name = in_var.name + '.cast_' + _dtype_to_str(dest_dtype)
                out_var = block.vars.get(cast_name)
                if out_var is None or out_var.dtype != dest_dtype:
                    out_var = block.create_var(
                        name=cast_name,
                        dtype=dest_dtype,
                        persistable=False,
                        stop_gradient=in_var.stop_gradient)

                    block._insert_op(
                        idx,
                        type="cast",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={
                            "in_dtype": in_var.dtype,
                            "out_dtype": out_var.dtype
                        })
                    num_cast_ops += 1
                _rename_arg(op, in_var.name, out_var.name)
            else:
                if op.has_attr('in_dtype'):
                    op._set_attr('in_dtype', dest_dtype)
    if src_dtype == core.VarDesc.VarType.FP32 and dest_dtype == core.VarDesc.VarType.BF16:
        for out_name in op.output_names:
            if op.type in [
                    'batch_norm', 'fused_bn_add_activation', 'layer_norm'
            ] and out_name != 'Y':
                continue
            for out_var_name in op.output(out_name):
                out_var = block.var(out_var_name)
                if out_var.type not in _valid_types:
                    continue
                if out_var.dtype == core.VarDesc.VarType.FP32:
                    out_var.desc.set_dtype(core.VarDesc.VarType.BF16)
                    if op.has_attr('out_dtype'):
                        op._set_attr('out_dtype', core.VarDesc.VarType.BF16)
    return num_cast_ops


def find_true_prev_op(ops, cur_op, var_name):
    """
    Find the true prev op that outputs var_name variable.

    Args:
        ops (list): A list of ops.
        cur_op (Operator): Current operator which has var_name variable.
        var_name (string): Variable name.
    """
    prev_op = []
    for op in ops:
        if op == cur_op:
            break
        for out_name in op.output_names:
            for out_var_name in op.output(out_name):
                if out_var_name == var_name:
                    prev_op.append(op)
    if prev_op:
        if not len(prev_op) == 1:
            raise ValueError("There must be only one previous op "
                             "that outputs {0} variable".format(var_name))
        else:
            return prev_op[0]
    return None


def find_true_post_op(ops, cur_op, var_name):
    """
    if there are post ops, return them, if there is no post op,
    return None instead.
    Args:
        ops (list): A list of ops.
        cur_op (Operator): Current operator which has var_name variable.
        var_name (string): Variable name.
    """
    post_op = []
    for idx, op in enumerate(ops):
        if op == cur_op:
            break

    for i in range(idx + 1, len(ops)):
        op = ops[i]
        for in_name in op.input_names:
            for in_var_name in op.input(in_name):
                if in_var_name == var_name:
                    post_op.append(op)

    return post_op


def find_op_index(block_desc, cur_op_desc):
    """
    """
    for idx in range(block_desc.op_size()):
        if cur_op_desc == block_desc.op(idx):
            return idx
    return -1


def _is_in_black_varnames(op, amp_lists):
    for in_name in op.input_arg_names:
        if in_name in amp_lists.black_varnames:
            return True

    for out_name in op.output_arg_names:
        if out_name in amp_lists.black_varnames:
            return True

    return False


def _need_keep_fp32(op, unsupported_op_list, use_bf16_guard):
    if op.type in unsupported_op_list:
        # the highest priority condition: If ops don't have bf16 computing kernels,
        # they must be executed in fp32 calculation pattern.
        return True

    # process ops about learning rate
    in_out_arg_names = []
    in_out_arg_names.extend(list(op.input_arg_names))
    in_out_arg_names.extend(list(op.output_arg_names))
    for name in in_out_arg_names:
        if "learning_rate" in name:
            return True

    if use_bf16_guard:
        if op.has_attr("op_namescope") and \
                (_bf16_guard_pattern in op.attr("op_namescope")):
            # op in bf16 guard
            return False
        else:
            # op not in bf16 guard
            return True
    else:
        return False


@signature_safe_contextmanager
def bf16_guard():
    """
    As for the pure bf16 training, if users set `use_bf16_guard` to True,
    only those ops created in the context manager `bf16_guard` will be
    transformed as float16 type.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F
            paddle.enable_static()
            data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
            conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)

            with paddle.static.amp.bf16_guard():
                bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                hidden = paddle.static.nn.fc(pool, size=10)
                loss = paddle.mean(hidden)
    """
    with framework.name_scope(prefix=_bf16_guard_pattern):
        yield


def rewrite_program_bf16(main_prog, amp_lists=None, use_bf16_guard=False):
    """
    Traverse all ops in current block and insert cast op according to
    which set current op belongs to.

    1. When an op belongs to the black list, add it to black set
    2. When an op belongs to the white list, add it to white set
    3. When an op belongs to the gray list. If one
       of its inputs is the output of black set op or black list op,
       add it to black set. If all of its previous ops are not black
       op and one of its inputs is the output of white set op or
       white list op, add it to white set.
    4. When an op isn't in the lists, add it to black op set.
    5. Add necessary cast ops to make sure that black set op will be
       computed in fp32 mode, while white set op will be computed in
       bf16 mode.

    Args:
        main_prog (Program): The main program for training.
    """
    if amp_lists is None:
        amp_lists = AutoMixedPrecisionListsBF16()
    block = main_prog.global_block()
    ops = block.ops
    white_op_set = set()
    black_op_set = set()
    for op in ops:

        # NOTE(zhiqiu): 'create_py_reader' and 'read' is used in non-iterable DataLoder,
        # we don't need to handle reader op and the input of 'create_py_reader' is not
        # in block, which may result in errors.
        # See GeneratorLoader._init_non_iterable() for details.
        if op.type == 'create_py_reader' or op.type == 'read':
            continue

        if amp_lists.black_varnames is not None and _is_in_black_varnames(
                op, amp_lists):
            black_op_set.add(op)
            continue

        if op.type in amp_lists.black_list or _need_keep_fp32(
                op, amp_lists.unsupported_list, use_bf16_guard):
            black_op_set.add(op)
        elif op.type in amp_lists.white_list:
            white_op_set.add(op)
        elif op.type in amp_lists.gray_list:
            is_black_op = False
            is_white_op = False
            for in_name in op.input_names:
                # if this op has inputs
                if in_name:
                    for in_var_name in op.input(in_name):
                        in_var = block.var(in_var_name)
                        # this in_var isn't the output of other op
                        if in_var.op is None:
                            continue
                        elif in_var.op is op:
                            prev_op = find_true_prev_op(ops, op, in_var_name)
                            if prev_op is None:
                                continue
                        else:
                            prev_op = in_var.op
                        # if it's one of inputs
                        if prev_op in black_op_set or \
                                prev_op.type in amp_lists.black_list:
                            is_black_op = True
                        elif prev_op in white_op_set or \
                                prev_op.type in amp_lists.white_list:
                            is_white_op = True
            if is_black_op:
                black_op_set.add(op)
            elif is_white_op:
                white_op_set.add(op)
            else:
                pass
        else:
            # For numerical safe, we apply fp32 computation on ops that
            # are not determined which list they should stay.
            black_op_set.add(op)

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        num_cast_ops = 0
        if op in black_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.BF16,
                                           core.VarDesc.VarType.FP32)
        elif op in white_op_set:
            if use_bf16_guard:
                if not (op.has_attr('op_namescope') and
                        (_bf16_guard_pattern in op.attr("op_namescope"))):
                    idx += 1
                    continue
            if op.has_attr('use_mkldnn'):
                op._set_attr('use_mkldnn', True)
                op._set_attr('mkldnn_data_type', 'bfloat16')
            elif op.has_attr('dtype') and op.attr(
                    'dtype') == core.VarDesc.VarType.FP32:
                op._set_attr('dtype', core.VarDesc.VarType.BF16)

            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP32,
                                           core.VarDesc.VarType.BF16)
        else:
            pass

        idx += num_cast_ops + 1
