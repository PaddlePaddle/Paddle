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

import paddle


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
            if dtype == paddle.fluid.core.VarDesc.VarType.FP32:
                self.fp32_calls = self.fp32_calls + 1
            elif dtype == paddle.fluid.core.VarDesc.VarType.FP16:
                self.fp16_calls = self.fp16_calls + 1
            elif dtype == paddle.fluid.core.VarDesc.VarType.BF16:
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


def _print_operator_stats(op_count_dict):
    """
    Parse and print the stats of operators, mainly including the calls of
    dtypes such as different fp32, fp16, bf16 and others.

    Args:
        op_count_dict(dict): a dict to record the number of calls for different
            operator and dtype.
    """
    print("<{:-^120}>".format(" op list "))
    total_ops = 0
    print(
        "<{:-^40}".format(" Op Name "),
        "|",
        "{:-^17}".format(" FP16 Calls "),
        "|",
        "{:-^17}".format(" BF16 Calls "),
        "|",
        "{:-^17}".format(" FP32 Calls"),
        "|",
        "{:-^17}>".format(" Other Calls "),
    )
    if op_count_dict is not None and isinstance(op_count_dict, dict):
        for op_type in sorted(op_count_dict.keys()):
            # fp16, bf16, fp32, other
            unit = op_count_dict[op_type]
            print(
                "  %-40s|  %-17s|  %-17s|  %-17s|  %-17s"
                % (
                    op_type,
                    unit.fp16_calls,
                    unit.bf16_calls,
                    unit.fp32_calls,
                    unit.other_calls,
                )
            )
            total_ops += 1
    print("<{:-^120}>\n".format(" op count: " + str(total_ops) + " "))


def _is_floating_point(dtype):
    if dtype in [
        paddle.fluid.core.VarDesc.VarType.FP64,
        paddle.fluid.core.VarDesc.VarType.FP32,
        paddle.fluid.core.VarDesc.VarType.FP16,
        paddle.fluid.core.VarDesc.VarType.BF16,
    ]:
        return True
    else:
        return False


def _get_var_from_block(block, op, arg_name, is_input):
    var_names = op.input(arg_name) if is_input else op.output(arg_name)
    assert isinstance(var_names, list)
    if len(var_names) == 0:
        return None

    var_name = var_names[0]
    try:
        var = block._var_recursive(var_name)
        return var
    except:
        print(
            "Operator < {} > gets {} < {} : {} > error!".format(
                op.type, "input" if is_input else "output", arg_name, var_name
            )
        )
        return None


def _extract_compute_dtype(op, block):
    var_name = None
    compute_dtype = None
    for in_name in op.input_names:
        in_var = _get_var_from_block(block, op, in_name, True)
        if in_var is None:
            continue

        var_dtype = in_var.dtype
        if compute_dtype is None:
            compute_dtype = var_dtype
        else:
            if compute_dtype != var_dtype:
                if _is_floating_point(compute_dtype) and _is_floating_point(
                    var_dtype
                ):
                    print(
                        "Operator < {} > has different input data types.".format(
                            op.type
                        )
                    )
                elif _is_floating_point(var_dtype):
                    compute_dtype = var_dtype

    for out_name in op.output_names:
        out_var = _get_var_from_block(block, op, out_name, False)
        if out_var is None:
            continue

        var_dtype = out_var.dtype
        if compute_dtype is None:
            compute_dtype = var_dtype
        else:
            if compute_dtype != var_dtype:
                if _is_floating_point(compute_dtype) and _is_floating_point(
                    var_dtype
                ):
                    print(
                        "Operator < {} > has different input / output data types.".format(
                            op.type
                        )
                    )
                elif _is_floating_point(var_dtype):
                    compute_dtype = var_dtype
    return compute_dtype


def _merge_op_stats(op_stats_list):
    merged_op_stats_dict = {}
    for each_op_stats_dict in op_stats_list:
        for op_type, unit in each_op_stats_dict.items():
            if merged_op_stats_dict.get(op_type, None) is None:
                merged_op_stats_dict[op_type] = unit
            else:
                merged_op_stats_dict[op_type].addto(unit)
    return merged_op_stats_dict


def collect_operator_stats(program=None, print_subblocks=False):
    if program is None:
        program = paddle.static.default_main_program()

    print(
        "====================== Operator Stats Collection ======================"
    )
    print(f"- Program has {program.num_blocks} blocks")

    global_block = program.global_block()
    param_names = [p.name for p in global_block.all_parameters()]

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
            elif op.type in ['cast', 'layer_norm', 'layer_norm_grad']:
                # Not check the input and output dtype difference for this operators.
                in_var = _get_var_from_block(block, op, 'X', True)
                compute_dtype = in_var.dtype
            elif "Param" in op.input_names:
                # Specify compute_dtype for optimizers.
                in_var = _get_var_from_block(block, op, 'Param', True)
                compute_dtype = in_var.dtype
            else:
                compute_dtype = _extract_compute_dtype(op, block)
            unit.update(dtype=compute_dtype)
        op_stats_list.append(block_op_stats_dict)

    if print_subblocks and len(op_stats_list) > 1:
        for i in range(len(op_stats_list)):
            print("<{:-^120}>".format(" op list of block " + str(i) + " "))
            _print_operator_stats(op_stats_list[i])
    print("<{:-^120}>".format(" op list of all blocks "))
    _print_operator_stats(_merge_op_stats(op_stats_list))
