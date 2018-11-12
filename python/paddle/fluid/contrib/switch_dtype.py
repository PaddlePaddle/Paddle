#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import contextlib

from .. import framework
from .. import unique_name

__all__ = ['switch_dtype_block', ]


def _rename_arg_(cur_block, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(cur_block.ops)
    for i in range(begin_idx, end_idx):
        op_desc = cur_block.ops[i].desc
        if isinstance(op_desc, tuple):
            op_desc = op_desc[0]
        op_desc._rename_input(old_name, new_name)
        op_desc._rename_output(old_name, new_name)


def _create_tmp_variable_(cur_block, name, dtype, stop_gradient=False):
    return cur_block.create_var(
        name=unique_name.generate(".".join([name, 'tmp'])),
        dtype=dtype,
        persistable=False,
        stop_gradient=stop_gradient)


@contextlib.contextmanager
def switch_dtype_block(main_program, dtype=np.float16, enable_tensor_core=None):

    pre_op_idx = len(main_program.current_block().ops)
    yield
    cur_op_idx = len(main_program.current_block().ops)

    # By default, the type of the scale, bias, mean,
    # and var tensors should both be float. (For float or float16 input tensor)
    # or double (For double input tensor).
    not_cast_param_ops = ["batch_norm"]

    cur_block = main_program.current_block()
    ops = [
        cur_block.ops[i + pre_op_idx] for i in range(cur_op_idx - pre_op_idx)
    ]

    avoid_convert_var = set()
    inner_inputs = set()
    inner_outputs = set()
    for op in ops:
        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if in_var_name not in inner_outputs:
                    inner_inputs.add(in_var_name)
                    if op.type in not_cast_param_ops and cur_block.var(
                            in_var_name).persistable:
                        avoid_convert_var.add(in_var_name)
        for oname in op.output_names:
            for out_var_name in op.output(oname):
                inner_outputs.add(out_var_name)

    # reset the argument of cur_block.ops and insert them to the current block
    next_op_idx = pre_op_idx
    for in_var_name in inner_inputs:
        if in_var_name in avoid_convert_var:
            continue
        in_var = cur_block.var(in_var_name)
        out_var_dtype = framework.convert_np_dtype_to_dtype_(dtype)
        if in_var.dtype == out_var_dtype:
            continue
        out_var = _create_tmp_variable_(cur_block, "casted_" + in_var.name,
                                        out_var_dtype)
        cur_block._insert_op(
            next_op_idx,
            type="cast",
            inputs={'X': in_var},
            outputs={"Out": out_var},
            attrs={'in_dtype': in_var.dtype,
                   'out_dtype': out_var_dtype})
        next_op_idx += 1
        _rename_arg_(cur_block, in_var.name, out_var.name, next_op_idx)

    cur_op_idx = len(main_program.current_block().ops)
    ops = [
        cur_block.ops[i + pre_op_idx] for i in range(cur_op_idx - pre_op_idx)
    ]

    for op in ops:
        # ReInfer the output var's type and dtype
        op.desc.infer_var_type(cur_block.desc)
        # op.desc.infer_shape(cur_block.desc)
