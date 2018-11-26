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

from . import recommend_fp16

from ... import framework
from ... import unique_name

__all__ = ['switch_to_fp16', ]


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


def _dtype_to_str_(np_dtype):
    dtype = np.dtype(np_dtype)
    if dtype == np.float32:
        return "float32"
    elif dtype == np.float64:
        return "float64"
    elif dtype == np.float16:
        return "float16"
    else:
        raise ValueError("Not supported numpy dtype %s" % dtype)


def _numerical_overflow_check_(op_type):
    if op_type in recommend_fp16.black_list:
        print("%s should not apply fp16 computation, because it's output"
              " maybe overflow or underflow." % (op_type))
    if op_type in recommend_fp16.gray_list:
        print("%s's gradient operator may be overflow or underflow when"
              " using fp16 computation." % (op_type))


@contextlib.contextmanager
def switch_to_fp16(main_program, verbose=True):
    """
    switch_to_fp16 is used to change the dtype of input and output to fp16.

    Args:
        main_program(Program): The current main program.
        verbose(bool): Whether to print warning info of
            numerical_overflow_check and dtype cast.

    Examples:

        >>> import paddle.fluid as fluid
        >>> img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        >>> label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        >>> with fluid.contrib.switch_to_fp16(fluid.default_main_program()):
        >>>    prediction = fluid.layers.fc(input=img, size=200, act='tanh')
        >>> prediction = fluid.layers.cast(prediction, np.float32)
        >>> loss = fluid.layers.cross_entropy(input=prediction, label=label)
    """
    dtype = np.float16

    pre_op_idx = len(main_program.current_block().ops)
    yield
    cur_op_idx = len(main_program.current_block().ops)

    # when we are using fp16 mode to run Batch norm cudnn, the data
    # type of input x and output y will be fp16, but the other input parameters
    # including mean and variance will still be float32 or float64.
    not_cast_param_ops = ["batch_norm"]

    cur_block = main_program.current_block()
    ops = [
        cur_block.ops[i + pre_op_idx] for i in range(cur_op_idx - pre_op_idx)
    ]

    avoid_convert_var = set()
    inner_inputs = set()
    inner_outputs = set()
    for op in ops:
        if verbose:
            _numerical_overflow_check_(op.type)

        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if in_var_name not in inner_outputs:
                    inner_inputs.add(in_var_name)
                    if op.type in not_cast_param_ops \
                       and cur_block.var(in_var_name).persistable:
                        avoid_convert_var.add(in_var_name)
        for oname in op.output_names:
            for out_var_name in op.output(oname):
                inner_outputs.add(out_var_name)

    # insert cast to the current block
    next_op_idx = pre_op_idx
    cast_to_dtype = framework.convert_np_dtype_to_dtype_(dtype)
    for in_var_name in inner_inputs:
        if in_var_name in avoid_convert_var:
            continue
        in_var = cur_block.var(in_var_name)
        if in_var.dtype == cast_to_dtype:
            continue

        out_var = _create_tmp_variable_(
            cur_block, "casted_" + _dtype_to_str_(dtype) + "_" + in_var.name,
            cast_to_dtype)

        cur_block._insert_op(
            next_op_idx,
            type="cast",
            inputs={'X': in_var},
            outputs={"Out": out_var},
            attrs={'in_dtype': in_var.dtype,
                   'out_dtype': cast_to_dtype})

        if verbose:
            print("Add cast, %s(%s) -> %s(%s)" %
                  (in_var.name, in_var.dtype, out_var.name, out_var.dtype))

        next_op_idx += 1
        _rename_arg_(cur_block, in_var.name, out_var.name, next_op_idx)

    # ReInfer the output var's dtype
    cur_op_idx = len(main_program.current_block().ops)
    ops = [
        cur_block.ops[i + pre_op_idx] for i in range(cur_op_idx - pre_op_idx)
    ]
    for op in ops:
        op.desc.infer_var_type(cur_block.desc)
