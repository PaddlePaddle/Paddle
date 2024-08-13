# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import logging

import paddle
from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class OperatorStatsUnit:
    def __init__(self):
        self.op_type = None
        self.fp32_calls = 0
        self.fp16_calls = 0
        self.bf16_calls = 0
        self.other_calls = 0

    def update(self, dtype):
        if dtype is None:
            self.other_calls = self.other_calls + 1
        else:
            if dtype == paddle.float32:
                self.fp32_calls = self.fp32_calls + 1
            elif dtype == paddle.float16:
                self.fp16_calls = self.fp16_calls + 1
            elif dtype == paddle.bfloat16:
                self.bf16_calls = self.bf16_calls + 1
            else:
                self.other_calls = self.other_calls + 1

    def addto(self, another):
        self.fp32_calls += another.fp32_calls
        self.fp16_calls += another.fp16_calls
        self.bf16_calls += another.bf16_calls
        self.other_calls += another.other_calls

    def convert_to_list(self):
        return [
            self.fp16_calls,
            self.bf16_calls,
            self.fp32_calls,
            self.other_calls,
        ]


def _is_floating_point(dtype):
    if dtype in [
        paddle.base.core.VarDesc.VarType.FP64,
        paddle.base.core.VarDesc.VarType.FP32,
        paddle.base.core.VarDesc.VarType.FP16,
        paddle.base.core.VarDesc.VarType.BF16,
    ]:
        return True
    else:
        return False


def _get_var_dtype_from_block(block, op, arg_name, is_input):
    var_names = op.input(arg_name) if is_input else op.output(arg_name)
    assert isinstance(var_names, list)
    if len(var_names) == 0:
        return None

    var_name = var_names[0]
    try:
        var = block._var_recursive(var_name)
        return var.dtype
    except:
        _logger.warning(
            "Operator < {} > gets {} < {} : {} > error!".format(
                op.type, "input" if is_input else "output", arg_name, var_name
            )
        )
        return None


def _extract_compute_dtype(op, block):
    var_name = None
    compute_dtype = None
    for in_name in op.input_names:
        var_dtype = _get_var_dtype_from_block(block, op, in_name, True)
        if var_dtype is None:
            continue

        if compute_dtype is None:
            compute_dtype = var_dtype
        else:
            if compute_dtype != var_dtype:
                if _is_floating_point(compute_dtype) and _is_floating_point(
                    var_dtype
                ):
                    _logger.warning(
                        f"Operator < {op.type} > has different input data types, input_names = {op.input_names}, output_names = {op.output_names}."
                    )
                elif _is_floating_point(var_dtype):
                    # When there are multiple inputs, such as embedding
                    # (ids is integer, w is floating-point), the kernel
                    # dtype is normally decided by the input of floating-point.
                    compute_dtype = var_dtype

    for out_name in op.output_names:
        var_dtype = _get_var_dtype_from_block(block, op, out_name, False)
        if var_dtype is None:
            continue

        if compute_dtype is None:
            # Kernel dtype is mostly decided by the input's dtype.
            # When the operator has no input, it mightly has a attr
            # such as dtype to specify the output's dtype.
            compute_dtype = var_dtype
        else:
            if compute_dtype != var_dtype:
                if _is_floating_point(compute_dtype) and _is_floating_point(
                    var_dtype
                ):
                    _logger.warning(
                        f"Operator < {op.type} > has different input / output data types, input_names = {op.input_names}, output_names = {op.output_names}."
                    )
    return compute_dtype


def _merge_op_stats(op_stats_list):
    merged_op_stats_dict = {}
    for each_op_stats_dict in op_stats_list:
        for op_type, unit in each_op_stats_dict.items():
            if merged_op_stats_dict.get(op_type, None) is None:
                merged_op_stats_dict[op_type] = copy.copy(unit)
            else:
                merged_op_stats_dict[op_type].addto(unit)
    return merged_op_stats_dict


def _get_op_stats_list(program):
    def _is_special_ops_with_input_x(op_type):
        # operators have input X and have inputs different dtypes.
        special_op_list = ['cast', 'batch_norm', 'instance_norm', 'layer_norm']
        if op_type in special_op_list:
            return True
        if op_type.replace("_grad", "") in special_op_list:
            return True
        return False

    op_stats_list = []
    for block in program.blocks:
        block_op_stats_dict = {}
        for op in block.ops:
            if block_op_stats_dict.get(op.type, None) is None:
                unit = OperatorStatsUnit()
                block_op_stats_dict[op.type] = unit
            else:
                unit = block_op_stats_dict[op.type]

            if op.type in [
                'create_py_reader',
                'read',
                'create_double_buffer_reader',
            ]:
                compute_dtype = None
            elif _is_special_ops_with_input_x(op.type):
                # Not check the input and output dtype difference for this operators.
                compute_dtype = _get_var_dtype_from_block(block, op, 'X', True)
            elif "Param" in op.input_names:
                # Specify compute_dtype for optimizers.
                compute_dtype = _get_var_dtype_from_block(
                    block, op, 'Param', True
                )
            else:
                compute_dtype = _extract_compute_dtype(op, block)
            unit.update(dtype=compute_dtype)
        op_stats_list.append(block_op_stats_dict)
    return op_stats_list


def collect_operator_stats(program=None, print_subblocks=False):
    """
    Collect the number of operators for different data types through parsing
    the program. The statistical data are categorized according to four data
    types, namely float32, float16, bfloat16 and others.

    Args:
        program(Program, optional): The program to parse. Default None, and the default main_program will be parsed.
        print_subblocks(bool, optional): Whether to print the operator stats for each subblock. Default False.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> class SimpleConvNet(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3)
            ...         self.linear = paddle.nn.Linear(in_features=26, out_features=10)
            ...
            ...     def forward(self, x):
            ...         out = self.conv(x)
            ...         out = paddle.nn.functional.relu(out)
            ...         out = self.linear(out)
            ...         out = paddle.nn.functional.softmax(out)
            ...         return out

            >>> main_program = paddle.static.Program()
            >>> startup_program = paddle.static.Program()
            >>> with paddle.utils.unique_name.guard():
            ...     with paddle.static.program_guard(main_program, startup_program):
            ...         model = SimpleConvNet()
            ...         x = paddle.static.data(
            ...             name='input', shape=[None, 1, 28, 28], dtype='float32'
            ...         )
            ...         out = model(x)
            ...         loss = paddle.mean(out)
            ...         optimizer = paddle.optimizer.AdamW()
            ...         optimizer = paddle.static.amp.decorate(optimizer)
            ...         optimizer.minimize(loss)
            >>> paddle.static.amp.debugging.collect_operator_stats(main_program)
            <------------------------------------------------ op list of all blocks ------------------------------------------------->
            <------------------------------------------------------- op list -------------------------------------------------------->
            <--------------- Op Name ---------------- | -- FP16 Calls --- | -- BF16 Calls --- | --- FP32 Calls--- | -- Other Calls -->
            adamw                                   |  0                |  0                |  4                |  0
            cast                                    |  5                |  0                |  6                |  0
            check_finite_and_unscale                |  0                |  0                |  1                |  0
            conv2d                                  |  1                |  0                |  0                |  0
            conv2d_grad                             |  1                |  0                |  0                |  0
            elementwise_add                         |  2                |  0                |  0                |  0
            elementwise_add_grad                    |  2                |  0                |  0                |  0
            elementwise_mul                         |  0                |  0                |  1                |  0
            elementwise_mul_grad                    |  0                |  0                |  1                |  0
            fill_constant                           |  0                |  0                |  1                |  0
            matmul_v2                               |  1                |  0                |  0                |  0
            matmul_v2_grad                          |  1                |  0                |  0                |  0
            memcpy                                  |  0                |  0                |  0                |  1
            reduce_mean                             |  0                |  0                |  1                |  0
            reduce_mean_grad                        |  0                |  0                |  1                |  0
            relu                                    |  1                |  0                |  0                |  0
            relu_grad                               |  1                |  0                |  0                |  0
            reshape2                                |  0                |  0                |  1                |  0
            reshape2_grad                           |  0                |  0                |  1                |  0
            softmax                                 |  0                |  0                |  1                |  0
            softmax_grad                            |  0                |  0                |  1                |  0
            update_loss_scaling                     |  0                |  0                |  1                |  0
            <----------------------------------------------------- op count: 22 ----------------------------------------------------->
    """

    def _convert_to_list(op_stats_unit_dict):
        for key, value in op_stats_unit_dict.items():
            op_stats_unit_dict[key] = value.convert_to_list()
        return op_stats_unit_dict

    if program is None:
        program = paddle.static.default_main_program()

    op_stats_list = _get_op_stats_list(program)
    merged_op_stats = _merge_op_stats(op_stats_list)
    if print_subblocks and len(op_stats_list) > 1:
        for i in range(len(op_stats_list)):
            print("<{:-^120}>".format(" op list of block " + str(i) + " "))
            paddle.amp.debugging._print_operator_stats(
                _convert_to_list(op_stats_list[i])
            )
    print("<{:-^120}>".format(" op list of all blocks "))
    paddle.amp.debugging._print_operator_stats(
        _convert_to_list(merged_op_stats)
    )
