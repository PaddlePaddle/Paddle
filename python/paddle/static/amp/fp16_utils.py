#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import numpy as np

import paddle
from paddle.fluid import core, framework, global_scope
from paddle.fluid.log_helper import get_logger
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

<<<<<<< HEAD
from .fp16_lists import AutoMixedPrecisionLists, get_low_precision_dtypestr, black_list
=======
from .fp16_lists import AutoMixedPrecisionLists, black_list
>>>>>>> unify o1 and o2

_logger = get_logger(
    __name__, logging.DEBUG, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

_valid_types = [
    core.VarDesc.VarType.LOD_TENSOR,
    core.VarDesc.VarType.SELECTED_ROWS,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
]

_fp16_guard_pattern = "__use_fp16__"


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
    if dtype in [core.VarDesc.VarType.FP16, core.VarDesc.VarType.BF16]:
        # TODO(Xreki): change the returned str to "bf16" for BF16 data type.
        # Currently too many codes use "cast_fp16" as key.
        return 'fp16'
    else:
        return 'fp32'


_keep_layer_norm_scale_bias_to_fp32_flag = True


def _keep_layer_norm_scale_bias_to_fp32(*args):
    global _keep_layer_norm_scale_bias_to_fp32_flag
    if len(args) == 0:
        return _keep_layer_norm_scale_bias_to_fp32_flag
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        old_value = _keep_layer_norm_scale_bias_to_fp32_flag
        _keep_layer_norm_scale_bias_to_fp32_flag = args[0]
        return old_value


def _keep_fp32_input(op, in_name):
    op_type = op.type
    if op_type == 'batch_norm':
        # Scale, Bias, Mean, Variance should be float32.
        return in_name != 'X'
    if op_type == 'layer_norm' and _keep_layer_norm_scale_bias_to_fp32():
        return in_name != 'X'
    if op_type == 'instance_norm':
        return in_name != 'X'
    if op_type == 'fused_bn_add_activation':
        return in_name not in {'X', 'Z'}
    if op_type == 'resnet_unit':
        return in_name not in {'X', 'FilterX', 'Z', 'FilterZ'}
    if op_type in ['fused_attention', 'fused_feedforward']:
        return in_name in {
            'LnScale',
            'LnBias',
            'Ln2Scale',
            'Ln2Bias',
            "Ln1Scale",
            "Ln1Bias",
        }
    if op_type == 'fused_multi_transformer':
        return in_name in {'LnScale', 'LnBias', 'FFNLnScale', 'FFNLnBias'}
    return False


def _keep_fp32_output(op, out_name):
    op_type = op.type
    if op_type in ['batch_norm', 'fused_bn_add_activation']:
        return out_name != 'Y'
    if op_type == 'layer_norm' and _keep_layer_norm_scale_bias_to_fp32():
        return out_name != 'Y'
    if op_type == 'resnet_unit':
        return out_name not in {'Y', 'ConvX', 'ConvZ'}
    if op_type in ['fused_attention', 'fused_feedforward']:
        return out_name in {
            'LnMean',
            'LnVariance',
            'Ln2Mean',
            'Ln2Variance',
            'Ln1Mean',
            'Ln1Variance',
        }
    return False


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
        if src_dtype == core.VarDesc.VarType.FP32 and _keep_fp32_input(
            op, in_name
        ):
            continue
        for in_var_name in op.input(in_name):
            in_var = block._find_var_recursive(in_var_name)
            if in_var.type not in _valid_types or in_var.dtype == dest_dtype:
                continue
            if in_var.dtype == src_dtype:
                cast_name = in_var.name + '.cast_' + _dtype_to_str(dest_dtype)
                out_var = block.vars.get(cast_name)
                if out_var is None or out_var.dtype != dest_dtype:
                    op_device = op.attr('op_device')
                    # NOTE(wangxi): optimize for pipeline, reduce one send.
                    # if in_var is stop_gradient and prev_op device is `all`,
                    # set cast_op device to `all`, can reduce send cast_var.
                    # TODO: need remove this after we unified the dynamic
                    # and static pipeline interface.
                    if (
                        src_dtype == core.VarDesc.VarType.FP32
                        and in_var.stop_gradient
                    ):
                        prev_op = None
                        if in_var.op is op:
                            prev_op = find_true_prev_op(
                                block.ops, op, in_var_name
                            )
                        elif in_var.op is not None:
                            prev_op = in_var.op

                        prev_op_device = None
                        if prev_op is not None:
                            prev_op_device = prev_op.attr('op_device')

                        if (
                            prev_op_device is not None
                            and 'all' in prev_op_device
                        ):
                            op_device = prev_op_device

                    out_var = block.create_var(
                        name=cast_name,
                        dtype=dest_dtype,
                        persistable=False,
                        stop_gradient=in_var.stop_gradient,
                    )

                    block._insert_op_without_sync(
                        idx,
                        type="cast",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={
                            "in_dtype": in_var.dtype,
                            "out_dtype": out_var.dtype,
                            "op_device": op_device,
                            "op_role": op.attr("op_role"),
                        },
                    )
                    num_cast_ops += 1
                _rename_arg(op, in_var.name, out_var.name)
            else:
                if op.has_attr('in_dtype'):
                    op._set_attr('in_dtype', dest_dtype)
    if src_dtype == core.VarDesc.VarType.FP32 and dest_dtype in [
        core.VarDesc.VarType.FP16,
        core.VarDesc.VarType.BF16,
    ]:
        for out_name in op.output_names:
            if _keep_fp32_output(op, out_name):
                continue
            for out_var_name in op.output(out_name):
                out_var = block.var(out_var_name)
                if out_var.type not in _valid_types:
                    continue
                if out_var.dtype == core.VarDesc.VarType.FP32:
                    out_var.desc.set_dtype(dest_dtype)
                    if op.has_attr('out_dtype'):
                        op._set_attr('out_dtype', dest_dtype)
    return num_cast_ops


def _insert_cast_post_op(
    block, op, idx, src_dtype, dest_dtype, target_name, op_var_rename_map
):
    num_cast_ops = 0

    target_var = block.var(target_name)
    if target_var.type not in _valid_types or target_var.dtype == dest_dtype:
        return num_cast_ops

    assert (
        target_var.dtype == src_dtype
    ), "The real dtype({}) is not equal to the src dtype({})".format(
        _dtype_to_str(target_var.dtype), _dtype_to_str(src_dtype)
    )

    cast_name = target_var.name + '.cast_' + _dtype_to_str(dest_dtype)
    cast_var = block.vars.get(cast_name)
    if cast_var is None or cast_var.dtype != dest_dtype:
        cast_var = block.create_var(
            name=cast_name,
            dtype=dest_dtype,
            persistable=False,
            stop_gradient=target_var.stop_gradient,
        )
        block._insert_op(
            idx,
            type="cast",
            inputs={"X": target_var},
            outputs={"Out": cast_var},
            attrs={
                "in_dtype": target_var.dtype,
                "out_dtype": cast_var.dtype,
                "op_device": op.attr("op_device"),
                "op_role": op.attr("op_role"),
            },
        )
        num_cast_ops += 1
        op_var_rename_map[block.idx][target_var.name] = cast_var.name

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
            raise ValueError(
                "There must be only one previous op "
                f"that outputs {var_name} variable"
            )
        else:
            return prev_op[0]
    return None


def find_true_post_op(ops, cur_op, var_name, search_all=False):
    """
    if there are post ops, return them, if there is no post op,
    return None instead.
    Args:
        ops (list): A list of ops.
        cur_op (Operator): Current operator which has var_name variable.
        var_name (string): Variable name.
        search_all (bool): The type of operator search. Use if \"cur_op\" is not in the \"ops\" set.
    """
    post_op = []
    if search_all:
        """
        \"cur_op\" do not have to be in list of \"ops\". E.g. \"cur_op\" can come
        from startup_prog block and \"ops\" list from main_prog block.
        By setting idx to -1, we'll start looking for post-ops from the top of the list.
        If search_all is False, assume that \"cur_op\" is in \"ops\" list,
        so to reduce the time of search we can start iterating from \"cur_op\" idx.
        """
        idx = -1
    else:
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
    """ """
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


def _need_keep_fp32(op, unsupported_op_list, use_fp16_guard):
    if op.type in unsupported_op_list:
        # the highest priority condition: If ops don't have fp16 computing kernels,
        # they must be executed in fp32 calculation pattern.
        return True

    # process ops about learning rate
    in_out_arg_names = []
    in_out_arg_names.extend(list(op.input_arg_names))
    in_out_arg_names.extend(list(op.output_arg_names))
    for name in in_out_arg_names:
        if "learning_rate" in name:
            return True

    if use_fp16_guard:
        if op.has_attr("op_namescope") and (
            _fp16_guard_pattern in op.attr("op_namescope")
        ):
            # op in fp16 guard
            return False
        else:
            # op not in fp16 guard
            return True
    else:
        return False


@signature_safe_contextmanager
def fp16_guard():
    """
    As for the pure fp16 training, if users set `use_fp16_guard` to True,
    only those ops created in the context manager `fp16_guard` will be
    transformed as float16 type.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F
            paddle.enable_static()
            data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
            conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)

            with paddle.static.amp.fp16_guard():
                bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                hidden = paddle.static.nn.fc(pool, size=10)
                loss = paddle.mean(hidden)
    """
    with framework.name_scope(prefix=_fp16_guard_pattern):
        yield

<<<<<<< HEAD
def set_var_dst_dtype(op, var_names, block, global_block, need_set_dtype):
    to_fp16_var_names = set()
    for in_var_name in var_names:
        if _keep_fp32_input(op, in_var_name):
            continue
        in_var = None
=======

def set_var_dst_dtype(
    op, var_names, block, global_block, dtype, need_set_dtype
):
    low_precison_var_names = set()
    for var_name in var_names:
        var = None
>>>>>>> unify o1 and o2
        try:
            var = block._var_recursive(var_name)
        except ValueError as e:
            _logger.debug(
                "-- {}, try to get it in the global block --".format(e)
            )
            var = global_block.var(var_name)
            if var is not None:
                _logger.debug(
                    "-- var {} is got in the global block --".format(var_name)
                )

            if var is None or var.type not in _valid_types:
                continue

        def is_float_dtype(dtype):
            return (
                dtype == core.VarDesc.VarType.FP32
                or var.dtype == core.VarDesc.VarType.FP16
                or var.dtype == core.VarDesc.VarType.BF16
                or dtype == core.VarDesc.VarType.FP64
            )

        if is_float_dtype(var.dtype):
            low_precison_var_names.add(var_name)
            if need_set_dtype:
                var.desc.set_dtype(dtype)

        _logger.debug(
            "---- op type: {}, var name: {}, var dtype: {} ----".format(
                op.type, var_name, var.dtype
            )
        )

    return low_precison_var_names


def set_param_dtype(program, dtype, amp_lists, use_fp16_guard):
    keep_fp32_var_names = set()
    all_parameters = []
    for block in program.blocks:
        all_parameters.extend(block.all_parameters())
        ops = block.ops
        for op in ops:
            if op_need_keep_fp32(op, amp_lists, use_fp16_guard):
                for in_name in op.input_names:
                    keep_fp32_var_names = keep_fp32_var_names.union(
                        op.input(in_name)
                    )
            else:
                for in_name in op.input_names:
                    if not core.is_compiled_with_ipu() and _keep_fp32_input(
                        op, in_name
                    ):
                        keep_fp32_var_names = keep_fp32_var_names.union(
                            op.input(in_name)
                        )

    for param in all_parameters:
        if param.name not in keep_fp32_var_names:
            _logger.debug("-- set param {} to {} --.".format(param.name, dtype))
            param.desc.set_dtype(dtype)


def op_need_keep_fp32(op, amp_lists, use_fp16_guard):
    need_keep_fp32 = False
    if _need_keep_fp32(
        op,
        amp_lists.unsupported_list,
        use_fp16_guard,
    ):
        need_keep_fp32 = True
    elif amp_lists.black_varnames is not None and _is_in_black_varnames(
        op, amp_lists
    ):
        need_keep_fp32 = True
    elif op.type in amp_lists.black_list:
        need_keep_fp32 = True

    return need_keep_fp32


def get_promote_dtype(op, amp_dtype, block, amp_lists, black_op_set):
    dst_dtype = amp_dtype
    for in_name in op.input_names:
        # for ipu, all inputs must be converted to fp16
        if not core.is_compiled_with_ipu() and _keep_fp32_input(op, in_name):
            _logger.debug(
                "---- Input {} {} should be kept fp32 ----".format(
                    in_name, op.input(in_name)
                )
            )
            continue
        # if this op has inputs
        if in_name:
            for in_var_name in op.input(in_name):
                in_var = block.var(in_var_name)
                prev_op = None
                _logger.debug(
                    "---- Input {} {} dtype is {} ----".format(
                        in_name, op.input(in_name), in_var.dtype
                    )
                )
                # this in_var isn't the output of other op
                if in_var.op is None:
                    if in_var.dtype == core.VarDesc.VarType.FP32:
                        dst_dtype = core.VarDesc.VarType.FP32
                        break
                # inplace
                elif in_var.op is op:
                    ops = block.ops
                    prev_op = find_true_prev_op(ops, op, in_var_name)
                    if (
                        prev_op is None
                        and in_var.dtype == core.VarDesc.VarType.FP32
                    ):
                        dst_dtype = core.VarDesc.VarType.FP32
                        break
                else:
                    prev_op = in_var.op

                # if it's one of inputs
                if prev_op is not None and (
                    prev_op in black_op_set
                    or prev_op.type in amp_lists.black_list
                ):
                    dst_dtype = core.VarDesc.VarType.FP32
                    break
        else:
            dst_dtype = core.VarDesc.VarType.FP32

    return dst_dtype


def process_op_input_and_outputs(op, block, global_block, dtype):
    low_precison_var_names = set()
    # Get the FP16 input because the low_precison_var_names is required for the parameter casting.
    # The dtype of the input is not set to fp16, because it is done in the step 3 of cast_model_to_fp16.
    for in_name in op.input_names:
        # for ipu, all inputs must be converted to fp16
        if not core.is_compiled_with_ipu() and _keep_fp32_input(op, in_name):
            continue
        in_vars = set_var_dst_dtype(
            op,
            op.input(in_name),
            block,
            global_block,
            dtype,
            need_set_dtype=False,
        )
        low_precison_var_names = low_precison_var_names.union(in_vars)
    # Set the output to FP16 because its consumer OP needs to determine if the dtype needs
    # to be promoted.
    for out_name in op.output_names:
        # for ipu, all outputs must be converted to fp16
        if not core.is_compiled_with_ipu() and _keep_fp32_output(op, out_name):
            continue
        set_var_dst_dtype(
            op,
            op.output(out_name),
            block,
            global_block,
            dtype,
            need_set_dtype=True,
        )
    return low_precison_var_names


def cast_model_to_fp16(
    program, amp_lists=None, use_fp16_guard=True, level='O2'
):
    """
    Traverse all ops in the whole model and set their inputs and outputs
    to the fp16 data type. This function will do some special process for
    the batch normalization, which keeps the computational process of
    batchnorms in FP32.
    Args:
        program (Program): The used program.
        amp_lists (AutoMixedPrecisionLists): An AutoMixedPrecisionLists object.
        use_fp16_guard(bool): Determine whether to use `fp16_guard` when
                              constructing the program. Default True.
        dest_type(core.VarDesc.VarType): the cast type. such as core.VarDesc.VarType.FP16 and core.VarDesc.VarType.BF16.
    """
    _logger.debug("---- before cast model to fp16 ----")
    _logger.debug(program)
    if amp_lists is None:
<<<<<<< HEAD
        dtype = get_low_precision_dtypestr(dest_type)
        amp_lists = AutoMixedPrecisionLists(dtype)
    amp_lists.unsupported_list -= {
        "conditional_block_grad",
        "conditional_block",
        "conditional_block_infer",
        "select_input",
        "while",
        "while_grad",
        "cast",
        "tensor_array_to_tensor",
        "lod_array_length",
        "write_to_array",
    }
    amp_lists.white_list = white_list
    amp_lists.black_list -= black_list
    print(amp_lists.white_list)
    print(amp_lists.black_list)
    print(amp_lists.unsupported_list)
=======
        amp_lists = AutoMixedPrecisionLists()
    # For amp o2 there is no blacklist by default.
    if level == 'O2':
        amp_lists.black_list = amp_lists.black_list - black_list
        if amp_lists._custom_black_list:
            amp_lists.black_list = amp_lists.black_list.union(
                amp_lists._custom_black_list
            )
>>>>>>> unify o1 and o2
    global_block = program.global_block()
    keep_fp32_ops = set()
    keep_fp16_ops = set()
    to_fp16_var_names = set()

    # step 1: set all params to fp16
    param_dtype = (
        core.VarDesc.VarType.FP16
        if level == 'O2'
        else core.VarDesc.VarType.FP32
    )
    set_param_dtype(
        program,
        dtype=param_dtype,
        amp_lists=amp_lists,
        use_fp16_guard=use_fp16_guard,
    )

    # step 2: divide op into different sets according to the black/unsupported and white lists.
    for block in program.blocks:
        ops = block.ops
        for op in ops:
            _logger.debug("-- process op: {}  --".format(op))
            if op.type == 'create_py_reader' or op.type == 'read':
                continue
            if op_need_keep_fp32(op, amp_lists, use_fp16_guard):
                keep_fp32_ops.add(op)
                _logger.debug(
                    "---- Add into keep_fp32_ops because the op needs to be kept fp32 ----"
                )
            elif op.type in amp_lists.white_list:
                keep_fp16_ops.add(op)
<<<<<<< HEAD
                # set all var to fp16 if necessary
                for in_name in op.input_names:
                    # for ipu, all inputs must be converted to fp16
                    if not core.is_compiled_with_ipu() and _keep_fp32_input(
                        op, in_name
                    ):
                        continue

                    if in_var.dtype == core.VarDesc.VarType.FP32:
                        in_var.desc.set_dtype(dest_type)
                        to_fp16_var_names.add(in_var_name)

                    _logger.debug(
                        "-- op type: {}, in var name: {}, in var dtype: {} --".format(
                            op.type, in_var_name, in_var.dtype
                        )
                    fp16_vars = set_var_dst_dtype(
                        op, op.input(in_name), block, global_block, False
                    )
                    print("========= fp16_vars ========= ", fp16_vars)
                    to_fp16_var_names = to_fp16_var_names.union(fp16_vars)
                    print(
                        "=========== to_fp16_var_names =========== ",
                        to_fp16_var_names,
                    )
                for out_name in op.output_names:
                    # for ipu, all outputs must be converted to fp16
                    if not core.is_compiled_with_ipu() and _keep_fp32_output(
                        op, out_name
                    ):
                        continue

                    if out_var.dtype == core.VarDesc.VarType.FP32:
                        out_var.desc.set_dtype(dest_type)

                    _logger.debug(
                        "-- op type: {}, out var name: {}, out var dtype: {} --".format(
                            op.type, out_var_name, out_var.dtype
                        )
                    )
            for attr_name in ['in_dtype', 'out_dtype', 'dtype']:
                if (
                    op.has_attr(attr_name)
                    and op.attr(attr_name) == core.VarDesc.VarType.FP32
                ):
                    op._set_attr(attr_name, dest_type)
                    set_var_dst_dtype(
                        op, op.output(out_name), block, global_block, True
                    )
=======
                # get fp16 inputs and set op's outputs to fp16 for promote judgments
                fp16_var_names = process_op_input_and_outputs(
                    op, block, global_block, core.VarDesc.VarType.FP16
                )
                to_fp16_var_names = to_fp16_var_names.union(fp16_var_names)
                _logger.debug(
                    "---- Add into keep_fp16_ops because the op in white_list ----"
                )
>>>>>>> unify o1 and o2
            else:
                # divide others ops into fp16/fp32 sets according to promoting principle.
                dst_dtype = get_promote_dtype(
                    op,
                    core.VarDesc.VarType.FP16,
                    block,
                    amp_lists,
                    keep_fp32_ops,
                )
                if dst_dtype == core.VarDesc.VarType.FP16:
                    keep_fp16_ops.add(op)
                    fp16_var_names = process_op_input_and_outputs(
                        op, block, global_block, core.VarDesc.VarType.FP16
                    )
                    to_fp16_var_names = to_fp16_var_names.union(fp16_var_names)
                    _logger.debug(
                        "----  Add into keep_fp16_ops because it should be promoted to fp16 ----"
                    )
                else:
                    keep_fp32_ops.add(op)
                    _logger.debug(
                        "----  Add into keep_fp32_ops because it should be promoted to fp32 ----"
                    )

<<<<<<< HEAD
            """
            if (
                op.has_attr('in_dtype')
                and op.attr('in_dtype') == core.VarDesc.VarType.FP32
            ):
                op._set_attr('in_dtype', core.VarDesc.VarType.FP16)
            if (
                op.has_attr('out_dtype')
                and op.attr('out_dtype') == core.VarDesc.VarType.FP32
            ):
                op._set_attr('out_dtype', core.VarDesc.VarType.FP16)
            if (
                op.has_attr('dtype')
                and op.attr('dtype') == core.VarDesc.VarType.FP32
            ):
                op._set_attr('dtype', core.VarDesc.VarType.FP16)
            """

    print("===================== before cast =================")
    print(program)
>>>>>>> support promote dtype for static amp training

=======
    # step 3: insert cast op for op's inputs.
>>>>>>> unify o1 and o2
    for block in program.blocks:
        ops = block.ops
        idx = 0
        while idx < len(ops):
            op = ops[idx]
            num_cast_ops = 0
            if op in keep_fp16_ops:
                in_var_cast_num = _insert_cast_op(
                    block,
                    op,
                    idx,
                    core.VarDesc.VarType.FP32,
                    core.VarDesc.VarType.FP16,
                )
                num_cast_ops += in_var_cast_num
            if op in keep_fp32_ops:
                in_var_cast_num = _insert_cast_op(
                    block,
                    op,
                    idx,
                    dest_type,
                    core.VarDesc.VarType.FP32,
                )
<<<<<<< HEAD
                num_cast_ops += pre_cast_num
                for out_var_name in op.output_arg_names:
                    out_var = block.vars.get(out_var_name)
                    if out_var is None or out_var.type not in _valid_types:
                        continue
                    if out_var.dtype == dest_type:
                        out_var.desc.set_dtype(core.VarDesc.VarType.FP32)
                        post_ops = find_true_post_op(ops, op, out_var_name)
                        for post_op in post_ops:
                            if post_op in keep_fp32_ops:
                                continue
                            post_cast_num = _insert_cast_post_op(
                                block,
                                op,
                                idx + pre_cast_num + 1,
                                core.VarDesc.VarType.FP32,
                                dest_type,
                                out_var_name,
                                op_var_rename_map,
                            )
                            num_cast_ops += post_cast_num
=======
                num_cast_ops += in_var_cast_num

>>>>>>> support promote dtype for static amp training
            idx += num_cast_ops + 1
    _logger.debug("---- after cast model to fp16 ----")
    _logger.debug(program)
    return to_fp16_var_names


def _convert_float_to_bfloat16(place, fp32_array):
    paddle.disable_static()
    framework._set_expected_place(place)
    fp32_tensor = paddle.to_tensor(fp32_array)
    bf16_array = paddle.cast(fp32_tensor, paddle.bfloat16).numpy()
    paddle.enable_static()
    return bf16_array


def cast_parameters_to_fp16(
    place,
    program,
    scope=None,
    to_fp16_var_names=None,
    dest_type=core.VarDesc.VarType.FP16,
):
    """
    Traverse all parameters in the whole model and set them to the FP16 data type.
    Whereas, this function will keep parameters of batchnorms in FP32.
    Args:
        place(fluid.CPUPlace|fluid.CUDAPlace): `place` is used to restore the FP16 weight tensors.
        program (Program): The used program.
        scope(fluid.Scope, optional): `scope` is used to get the FP32 weight tensor values.
                                      Default is None.
        to_fp16_var_names(set|list, optional): The data types of vars in `to_fp16_var_names`
                                               will be set to FP16. Usually, it is the returned
                                               value of `cast_model_to_fp16` API.
        dest_type(core.VarDesc.VarType): the cast type. such as core.VarDesc.VarType.FP16 and core.VarDesc.VarType.BF16.
    """
    all_parameters = []
    for block in program.blocks:
        all_parameters.extend(block.all_parameters())

    dtype_str = get_low_precision_dtypestr(dest_type)
    fp16_var_names = to_fp16_var_names if to_fp16_var_names else set()
    var_scope = scope if scope else global_scope()
    for param in all_parameters:
        if param.name in fp16_var_names:
            _logger.debug(
                f"-- cast {param.name} to {dtype_str}, place is {place}"
            )
            if var_scope.find_var(param.name):
                param_t = var_scope.find_var(param.name).get_tensor()
                data = np.array(param_t)
                if dest_type == core.VarDesc.VarType.BF16:
                    bf16_data = _convert_float_to_bfloat16(place, data)
                    param_t.set(bf16_data, place)
                else:
                    param_t.set(np.float16(data), place)
            else:
                _logger.warning(f"Cannot find {param.name}")


<<<<<<< HEAD
def rewrite_program(main_prog, amp_lists, dest_type=core.VarDesc.VarType.FP16):
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
       fp16 mode.

    Args:
        main_prog (Program): The main program for training.
        dest_type(core.VarDesc.VarType): the cast type. such as core.VarDesc.VarType.FP16 and core.VarDesc.VarType.BF16.
    """
    _logger.debug("---- origin program ----")
    _logger.debug(main_prog)
    block = main_prog.global_block()
    block._sync_with_cpp()
    global_block = main_prog.global_block()
    ops = block.ops
    white_op_set = set()
    black_op_set = set()
    # step 1: set param to fp32
    set_param_dtype(main_prog, dtype=core.VarDesc.VarType.FP32)
    # step 2:
    for op in ops:
        # NOTE(zhiqiu): 'create_py_reader' and 'read' is used in non-iterable DataLoder,
        # we don't need to handle reader op and the input of 'create_py_reader' is not
        # in block, which may result in errors.
        # See GeneratorLoader._init_non_iterable() for details.
        _logger.debug("---- process op: {}  ----".format(op))

        if op.type == 'create_py_reader' or op.type == 'read':
            continue

        if op_need_keep_fp32(op, amp_lists, use_fp16_guard=False):
            _logger.debug(
                "---- Add into fp32_op_set because the op needs to be kept fp32 ----"
            )
            black_op_set.add(op)
        elif op.type in amp_lists.white_list:
            _logger.debug(
                "---- Add into fp16_op_set because the op in white_list ----"
            )
            white_op_set.add(op)
            for out_name in op.output_names:
                set_var_dst_dtype(
                    op, op.output(out_name), block, global_block, True
                )
        else:
            dst_dtype = get_promote_dtype(
                op, core.VarDesc.VarType.FP16, block, amp_lists, black_op_set
            )
            if dst_dtype == core.VarDesc.VarType.FP16:
                _logger.debug(
                    "----  Add into fp16_op_set because it should be promoted to fp16 ----"
                )
                white_op_set.add(op)
                for out_name in op.output_names:
                    set_var_dst_dtype(
                        op, op.output(out_name), block, global_block, True
                    )
            else:
                _logger.debug(
                    "----  Add into fp32_op_set because it should be promoted to fp32 ----"
                )
                black_op_set.add(op)

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        num_cast_ops = 0
        if op in white_op_set:
            num_cast_ops = _insert_cast_op(
<<<<<<< HEAD
                block, op, idx, dest_type, core.VarDesc.VarType.FP32
=======
                block,
                op,
                idx,
                core.VarDesc.VarType.FP32,
                core.VarDesc.VarType.FP16,
>>>>>>> support promote dtype for static amp training
            )
        elif op in black_op_set:
            num_cast_ops = _insert_cast_op(
<<<<<<< HEAD
                block, op, idx, core.VarDesc.VarType.FP32, dest_type
=======
                block,
                op,
                idx,
                core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.FP32,
>>>>>>> support promote dtype for static amp training
            )

        idx += num_cast_ops + 1
    _logger.debug("---- amp program ----")
    _logger.debug(main_prog)


=======
>>>>>>> unify o1 and o2
def update_role_var_grad(main_prog, params_grads):
    """
    Update op_role_var attr for some ops to make sure the gradients
    transferred across GPUs is FP16.
    1. Check whether the op that outputs gradient is cast or not.
    2. If op is cast and gradient is FP32, remove the op_role_var
       and find the prev op which outputs FP16 gradient
    3. Update the op_role_var of the prev op.

    Args:
        main_prog (Program): The main program for training.
        params_grads (list): A list of params and grads.
    """
    block = main_prog.global_block()
    block._sync_with_cpp()
    BACKWARD = core.op_proto_and_checker_maker.OpRole.Backward
    OPTIMIZE = core.op_proto_and_checker_maker.OpRole.Optimize
    for p, g in params_grads:
        op = g.op
        if g.dtype == core.VarDesc.VarType.FP32 and op.type == 'cast':
            role = op.attr('op_role')
            if role & int(BACKWARD) and op.has_attr('op_role_var'):
                op._remove_attr("op_role_var")
            else:
                raise ValueError(
                    f"The cast op {op} must be in BACKWARD role "
                    "and have op_role_var attr."
                )

            fp16_grad_name = op.input(op.input_names[0])[0]
            op_for_fp16_grad = find_true_prev_op(block.ops, op, fp16_grad_name)
            op_role_var_attr_name = (
                core.op_proto_and_checker_maker.kOpRoleVarAttrName()
            )
            attr_val = [p.name, fp16_grad_name]
            if op_for_fp16_grad.has_attr(op_role_var_attr_name):
                attr_val.extend(op_for_fp16_grad.attr(op_role_var_attr_name))
            op_for_fp16_grad._set_attr(op_role_var_attr_name, attr_val)

            # Maximize the all_reduce overlap, and perform the cast
            # operation after gradients transfer.
            op._set_attr('op_role', OPTIMIZE)
            # optimize op should stay behind forward and backward ops
            if op == block.ops[-1]:
                continue
            post_ops = find_true_post_op(block.ops, op, g.name)
            if post_ops:
                raise ValueError(
                    f"The cast op {op}'s output should not be"
                    "used by a non-optimize op, however, it"
                    f"is used by {post_ops[0]}"
                )
            # add new op in the python and cpp at the same time
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            new_op = framework.Operator(
                block=block,
                desc=new_op_desc,
                type=None,
                inputs=None,
                outputs=None,
                attrs=None,
            )
            block.ops.append(new_op)
            op_idx = find_op_index(block.desc, op.desc)
            if op_idx == -1:
                raise ValueError(f"The op {op} is not in program")
            block._remove_op(op_idx, sync=False)
    block._sync_with_cpp()
