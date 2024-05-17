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

import paddle
from paddle.base.data_feeder import check_type, check_variable_and_dtype
from paddle.distributed.auto_parallel.static.dist_attribute import (
    OperatorDistAttr,
)
from paddle.distributed.auto_parallel.static.process_group import (
    get_world_process_group,
)
from paddle.distributed.auto_parallel.static.utils import (
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_var_dist_attr,
)
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole
from paddle.framework import core
from paddle.static.amp.fp16_utils import (
    AutoMixedPrecisionLists,
    _is_in_black_varnames,
    _keep_fp32_input,
    _keep_fp32_output,
    _rename_arg,
    _valid_types,
    find_op_index,
    find_true_post_op,
    find_true_prev_op,
)
from paddle.utils import unique_name

from ..auto_parallel.process_mesh import ProcessMesh
from ..auto_parallel.static.utils import (
    is_backward_op,
    is_forward_op,
    is_loss_grad_op,
    is_loss_op,
    is_optimize_op,
)
from .pass_base import PassBase, register_pass

world_process_group = get_world_process_group()

__amp_skip_ops__ = [
    'create_py_reader',
    'create_double_buffer_reader',
    'while',
]


def _dtype_to_str(dtype):
    if dtype == paddle.float16:
        return 'fp16'
    elif dtype == paddle.bfloat16:
        return 'bf16'
    else:
        return 'fp32'


def _str_to_dtype(dstr):
    if dstr == 'float16':
        return core.VarDesc.VarType.FP16
    elif dstr == 'bfloat16':
        return core.VarDesc.VarType.BF16
    else:
        return core.VarDesc.VarType.FP32


class AMPLists:
    def __init__(
        self,
        white_list=None,
        black_list=None,
        black_varnames=None,
        dtype="float16",
    ):
        self._amp_list = AutoMixedPrecisionLists(
            set(white_list), set(black_list), set(black_varnames), dtype=dtype
        )
        self._dtype = dtype

    @property
    def white_list(self):
        return self._amp_list.white_list

    @property
    def black_list(self):
        return self._amp_list.black_list

    @property
    def gray_list(self):
        return self._amp_list.gray_list

    @property
    def black_varnames(self):
        return self._amp_list.black_varnames

    @property
    def dtype(self):
        return self._dtype

    @property
    def amp_list(self):
        return self._amp_list

    def _is_in_black_fp32_varnames(self, op):
        return _is_in_black_varnames(op, self._amp_list)

    def _op_keep_fp32_input(self, op, in_name):
        if not op.amp_options.enable:
            return True
        return _keep_fp32_input(op, in_name)

    def _op_keep_fp32_output(self, op, out_name):
        if not op.amp_options.enable:
            return True
        return _keep_fp32_output(op, out_name)


class AMPState:
    def __init__(self, program, amp_lists, amp_dtype, dist_context):
        self.program = program
        self.dist_context = dist_context
        self.amp_lists = amp_lists
        self.amp_dtype = amp_dtype
        self.grad_op_to_op_map = (
            dist_context.dist_op_context.grad_op_id_to_op_id
        )

        # op_id --> True/False. 'True' means that the current op is in fp16/bf16 mode.
        self._op_fp16_dict = {}
        # fwd_op_id --> {old_name: cast_name}
        self._var_name_dict = {}
        # out_var_name --> [op_ids]
        self.out_var_op_deps = {}

    def _is_fp16_op(self, op_id):
        return self._op_fp16_dict.get(op_id, None)

    def build_state(self):
        is_train = False
        for block in self.program.blocks:
            for op in block.ops:
                # to record the inplace operation and their outputs
                for name in op.output_arg_names:
                    if name not in self.out_var_op_deps:
                        self.out_var_op_deps[name] = [op.desc.original_id()]
                    else:
                        self.out_var_op_deps[name].extend(
                            [op.desc.original_id()]
                        )

                if is_loss_grad_op(op):
                    is_train = True

                if op.type in __amp_skip_ops__:
                    continue

                if is_forward_op(op):
                    self._mark_black_white_ops(op, block.ops, block)
                elif is_backward_op(op):
                    if op.desc.original_id() in self.grad_op_to_op_map:
                        fwd_op_id = self.grad_op_to_op_map[
                            op.desc.original_id()
                        ]
                        assert fwd_op_id in self._op_fp16_dict, str(op)
                        self._op_fp16_dict[
                            op.desc.original_id()
                        ] = self._is_fp16_op(fwd_op_id)
                elif is_optimize_op(op):
                    break

        # insert cast ops
        for block in self.program.blocks:
            self._cast_block(block)

        return is_train

    def _mark_black_white_ops(self, op, ops, block):
        # deal auto_cast info
        if not op.amp_options.enable:
            self._op_fp16_dict[op.desc.original_id()] = False
            return

        # ernie inference trick
        if op.type == "assign" and "array_" in op.input_arg_names[0]:
            self._op_fp16_dict[op.desc.original_id()] = False
            return

        # If assign op is inplace-operation, assign op exec mode should be same with the created op of output_var.
        if op.type == "assign":
            out_name = op.output_arg_names[0]
            if len(self.out_var_op_deps[out_name]) > 1:
                if not self._is_fp16_op(self.out_var_op_deps[out_name][0]):
                    self._op_fp16_dict[op.desc.original_id()] = False
                else:
                    self._op_fp16_dict[op.desc.original_id()] = True
                return

        if (
            self.amp_lists.black_varnames is not None
            and self.amp_lists._is_in_black_fp32_varnames(op)
        ):
            self._op_fp16_dict[op.desc.original_id()] = False
            return
        if op.type in self.amp_lists.black_list:
            self._op_fp16_dict[op.desc.original_id()] = False
        elif op.type in self.amp_lists.white_list:
            self._op_fp16_dict[op.desc.original_id()] = True
        elif op.type in self.amp_lists.gray_list:
            is_black_op = False
            is_white_op = False
            for in_name in op.input_names:
                # if this op has inputs
                if in_name:
                    for in_var_name in op.input(in_name):
                        in_var = block._var_recursive(in_var_name)
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
                        if (
                            self._is_fp16_op(prev_op.desc.original_id())
                            is False
                            or prev_op.type in self.amp_lists.black_list
                        ):
                            is_black_op = True
                        elif (
                            self._is_fp16_op(prev_op.desc.original_id()) is True
                            or prev_op.type in self.amp_lists.white_list
                        ):
                            is_white_op = True
            if is_black_op:
                self._op_fp16_dict[op.desc.original_id()] = False
            elif is_white_op:
                self._op_fp16_dict[op.desc.original_id()] = True
            else:
                pass
        else:
            # For numerical safe, we apply fp32 computation on ops that
            # are not determined which list they should stay.
            self._op_fp16_dict[op.desc.original_id()] = False

    def _cast_block(self, block):
        idx = 0
        appended_grad_times = 0
        while idx < len(block.ops):
            op = block.ops[idx]
            num_cast_ops = 0

            if op.type in __amp_skip_ops__:
                idx += 1
                continue

            elif is_forward_op(op):
                if self._is_fp16_op(op.desc.original_id()) is False:
                    num_cast_ops = self._insert_cast_op_forward(
                        block,
                        op,
                        idx,
                        _str_to_dtype(self.amp_dtype),
                        core.VarDesc.VarType.FP32,
                        self.dist_context,
                    )
                elif self._is_fp16_op(op.desc.original_id()) is True:
                    # deal with op with attribute 'dtype', such as 'fill_constant'
                    if (
                        op.has_attr('dtype')
                        and op.attr('dtype') == paddle.float32
                    ):
                        op._set_attr('dtype', _str_to_dtype(self.amp_dtype))
                    num_cast_ops = self._insert_cast_op_forward(
                        block,
                        op,
                        idx,
                        core.VarDesc.VarType.FP32,
                        _str_to_dtype(self.amp_dtype),
                        self.dist_context,
                    )
            elif is_backward_op(op):
                # NOTE: the map in `grad_var_to_var` may be changed when the var is casted,
                # which will affect the dist_op to insert allreduce_sum op.
                op_dist_attr = self.dist_context.get_op_dist_attr_for_program(
                    op
                )
                if is_backward_op(op) and (
                    is_forward_op(block.ops[idx - 1])
                    or is_loss_op(block.ops[idx - 1])
                ):
                    if not op_dist_attr.is_recompute:
                        appended_grad_times += 1

                if op.desc.original_id() in self.grad_op_to_op_map:
                    if self._is_fp16_op(op.desc.original_id()) is False:  # fp32
                        num_cast_ops = self._insert_cast_op_backward(
                            block,
                            op,
                            idx,
                            _str_to_dtype(self.amp_dtype),
                            core.VarDesc.VarType.FP32,
                            self.dist_context,
                            appended_grad_times,
                        )
                    elif self._is_fp16_op(op.desc.original_id()) is True:
                        # deal with op with attribute 'dtype', such as 'fill_constant'
                        if (
                            op.has_attr('dtype')
                            and op.attr('dtype') == paddle.float32
                        ):
                            op._set_attr('dtype', _str_to_dtype(self.amp_dtype))
                        num_cast_ops = self._insert_cast_op_backward(
                            block,
                            op,
                            idx,
                            core.VarDesc.VarType.FP32,
                            _str_to_dtype(self.amp_dtype),
                            self.dist_context,
                            appended_grad_times,
                        )
                elif op.type == "sum":
                    # all inputs dtype of sum should be equal and output dtype should follow input
                    out_var_name = op.desc.output_arg_names()[0]
                    in_var_name = op.desc.input_arg_names()[0]
                    out_var = block.var(out_var_name)
                    in_var = block._find_var_recursive(in_var_name)
                    for in_var_name in op.input_arg_names:
                        assert (
                            in_var.dtype == block.var(in_var_name).dtype
                        ), f"{in_var}, {block.var(in_var_name)}, {str(op)}"
                    out_var.desc.set_dtype(in_var.dtype)
                elif int(op.attr('op_role')) == 257:
                    pass
                else:
                    raise ValueError(
                        f"'{op.type}' op is not supported in the complete amp pass."
                    )
            idx += num_cast_ops + 1
        block._sync_with_cpp()

    def _insert_cast_op_forward(
        self, block, op, idx, src_dtype, dst_dtype, dist_context
    ):
        """
        only for forward cast
        modified from paddle.static.amp
        """
        num_cast_ops = 0
        var_name_dict = {}

        if op.type == "cast":
            in_var = block._find_var_recursive(op.input('X')[0])
            out_var = block._find_var_recursive(op.output('Out')[0])
            op._set_attr('in_dtype', in_var.dtype)
            out_var.desc.set_dtype(paddle.dtype(op.attr('out_dtype')))
            return num_cast_ops

        for in_name in op.input_names:
            if (
                src_dtype == paddle.float32
                and self.amp_lists._op_keep_fp32_input(op, in_name)
            ):
                continue
            for in_var_name in op.input(in_name):
                in_var = block._find_var_recursive(in_var_name)
                if in_var.type not in _valid_types or in_var.dtype == dst_dtype:
                    continue
                if in_var.dtype == src_dtype:
                    cast_name = (
                        in_var.name + '.cast_' + _dtype_to_str(dst_dtype)
                    )
                    cast_var = block.vars.get(cast_name)
                    var_name_dict[in_var.name] = cast_name
                    consume_op_attr = dist_context.get_op_dist_attr_for_program(
                        op
                    )
                    assert consume_op_attr is not None
                    if cast_var is None or cast_var.dtype != dst_dtype:
                        # NOTE we make the cast op and var's dist attr as the op that consume the
                        # cast var instead of the op which generates the var
                        in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                            in_var.name
                        )
                        assert in_var_dist_attr is not None
                        ref_mesh = in_var_dist_attr.process_mesh
                        ref_mapping = in_var_dist_attr.dims_mapping
                        ref_chunk_id = consume_op_attr.chunk_id

                        in_var_dist_attr.chunk_id = ref_chunk_id
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr
                        )

                        cast_var = block.create_var(
                            name=cast_name,
                            dtype=dst_dtype,
                            persistable=False,
                            stop_gradient=in_var.stop_gradient,
                        )
                        set_var_dist_attr(
                            dist_context,
                            cast_var,
                            ref_mapping,
                            ref_mesh,
                            chunk_id=ref_chunk_id,
                        )

                        op_namescope = "/"
                        if op.has_attr('op_namescope'):
                            op_namescope = op.attr('op_namescope')
                        cast_op = block._insert_op_without_sync(
                            idx,
                            type="cast",
                            inputs={"X": in_var},
                            outputs={"Out": cast_var},
                            attrs={
                                "in_dtype": in_var.dtype,
                                "out_dtype": cast_var.dtype,
                            },
                        )
                        cast_op._set_attr(
                            'op_namescope', op_namescope
                        )  # for recompute
                        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                            cast_op,
                            ref_mesh,
                            ref_mapping,
                            dist_context,
                            chunk_id=ref_chunk_id,
                        )
                        num_cast_ops += 1
                    else:
                        in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                            in_var.name
                        )
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr
                        )
                    _rename_arg(op, in_var.name, cast_name)
                else:
                    if op.has_attr('in_dtype'):
                        op._set_attr('in_dtype', dst_dtype)
        self._var_name_dict[op.desc.original_id()] = var_name_dict

        if src_dtype == paddle.float32 and dst_dtype == _str_to_dtype(
            self.amp_dtype
        ):
            for out_name in op.output_names:
                if self.amp_lists._op_keep_fp32_output(op, out_name):
                    continue
                for out_var_name in op.output(out_name):
                    out_var = block._var_recursive(out_var_name)
                    if out_var.type not in _valid_types:
                        continue
                    if out_var.dtype == paddle.float32:
                        out_var.desc.set_dtype(_str_to_dtype(self.amp_dtype))
                        if op.has_attr('out_dtype'):
                            op._set_attr(
                                'out_dtype', _str_to_dtype(self.amp_dtype)
                            )

        return num_cast_ops

    def _insert_cast_op_backward(
        self,
        block,
        op,
        idx,
        src_dtype,
        dst_dtype,
        dist_context,
        appended_grad_times,
    ):
        """only for backward cast"""

        def _keep_fp32_input(op, in_name):
            op_type = op.type
            if op_type in ['layer_norm_grad']:
                return in_name not in {'X', 'Y@GRAD'}
            return False

        def _keep_fp32_output(op, out_name):
            op_type = op.type
            if op_type in ['layer_norm_grad']:
                return out_name != 'X@GRAD'
            return False

        num_cast_ops = 0
        original_id = op.desc.original_id()
        dist_op_context = dist_context.dist_op_context
        fwd_op_id = self.grad_op_to_op_map[original_id]

        if op.type == "cast":
            in_name = op.input('X')[0]
            out_name = op.output('Out')[0]
            in_var = block._find_var_recursive(in_name)
            out_var = block._find_var_recursive(out_name)
            in_var_fw = block._find_var_recursive(in_name[: in_name.find("@")])
            out_var_fw = block._find_var_recursive(
                out_name[: out_name.find("@")]
            )
            op._set_attr('in_dtype', in_var_fw.dtype)
            op._set_attr('out_dtype', out_var_fw.dtype)
            in_var.desc.set_dtype(in_var_fw.dtype)
            out_var.desc.set_dtype(out_var_fw.dtype)
            return num_cast_ops

        for in_name in op.input_names:
            if src_dtype == paddle.float32 and _keep_fp32_input(op, in_name):
                for in_var_name in op.input(in_name):
                    in_var = block._var_recursive(in_var_name)
                    assert in_var.dtype == paddle.float32
                continue

            for in_var_name in op.input(in_name):
                in_var = block._var_recursive(in_var_name)
                if in_var.dtype == src_dtype:
                    consume_op_attr = dist_context.get_op_dist_attr_for_program(
                        op
                    )
                    if in_var_name in self._var_name_dict[fwd_op_id]:
                        # NOTE: if in_var of consume grad_op has been casted before,
                        # it should be renamed and reset dist_attr.
                        cast_name = self._var_name_dict[fwd_op_id][in_var_name]
                        op.desc._rename_input(in_var_name, cast_name)
                        in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                            in_var_name
                        )
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr
                        )
                    else:
                        assert (
                            in_var.dtype == dst_dtype
                        ), f"op [{op.type}] expect input [{in_name}] to be dtype [{dst_dtype}] BUT got [{in_var.dtype}]. {str(op)}"

        for out_name in op.output_names:
            if src_dtype == paddle.float32 and _keep_fp32_output(op, out_name):
                for out_var_name in op.output(out_name):
                    out_var = block._var_recursive(out_var_name)
                    assert out_var.dtype == paddle.float32
                continue

            for out_var_name in op.output(out_name):
                out_var = block._var_recursive(out_var_name)
                out_var_name_prefix = out_var_name[: out_var_name.find("@")]
                fwd_var = block._var_recursive(out_var_name_prefix)
                # NOTE: the out_var's dtype of consume grad_op should equal to the fwd_var's dtype
                if out_var.dtype != fwd_var.dtype:
                    out_var.desc.set_dtype(fwd_var.dtype)

                if out_var.dtype == src_dtype:
                    if out_var_name_prefix in self._var_name_dict[fwd_op_id]:
                        # NOTE: if out_var of consume grad_op has been casted before,
                        # it should be renamed and reset dist_attr, then we insert cast op to
                        # convert the cast_var to original dtype
                        consume_op_attr = (
                            dist_context.get_op_dist_attr_for_program(op)
                        )
                        fwd_cast_name = self._var_name_dict[fwd_op_id][
                            out_var_name_prefix
                        ]
                        suffix = ""
                        if "@RENAME" in out_var_name:
                            suffix = out_var_name[
                                out_var_name.find("@RENAME") :
                            ]
                        cast_name = fwd_cast_name + "@GRAD" + suffix
                        cast_var = block.vars.get(cast_name)
                        if cast_var is None or cast_var.dtype != dst_dtype:
                            op.desc._rename_output(out_var_name, cast_name)
                            out_var_dist_attr = (
                                consume_op_attr.get_output_dist_attr(
                                    out_var_name
                                )
                            )
                            ref_mesh = out_var_dist_attr.process_mesh
                            ref_mapping = out_var_dist_attr.dims_mapping
                            ref_chunk_id = consume_op_attr.chunk_id

                            out_var_dist_attr.chunk_id = ref_chunk_id
                            consume_op_attr.set_output_dist_attr(
                                cast_name, out_var_dist_attr
                            )
                            assert ref_mapping is not None
                            cast_var = block.create_var(
                                name=cast_name,
                                shape=out_var.shape,
                                dtype=dst_dtype,
                                persistable=False,
                                stop_gradient=out_var.stop_gradient,
                            )
                            set_var_dist_attr(
                                dist_context,
                                cast_var,
                                ref_mapping,
                                ref_mesh,
                                chunk_id=ref_chunk_id,
                            )
                            dist_op_context.grad_var_to_var[
                                appended_grad_times
                            ][cast_name] = fwd_cast_name

                            cast_op = block._insert_op(
                                idx + 1,
                                type="cast",
                                inputs={"X": cast_var},
                                outputs={"Out": out_var},
                                attrs={
                                    "in_dtype": cast_var.dtype,
                                    "out_dtype": out_var.dtype,
                                    "op_role": OpRole.Backward,
                                },
                            )
                            cast_op._remove_attr("op_role_var")
                            cast_op._remove_attr("op_namescope")
                            cast_op._remove_attr("with_quant_attr")
                            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                                cast_op,
                                ref_mesh,
                                ref_mapping,
                                dist_context,
                                chunk_id=ref_chunk_id,
                            )
                            num_cast_ops += 1
                else:
                    assert out_var.dtype == dst_dtype

        if op.has_attr('dtype') and op.attr('dtype') == paddle.float32:
            op._set_attr('dtype', _str_to_dtype(self.amp_dtype))

        return num_cast_ops


@register_pass("auto_parallel_amp")
class AMPPass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("dtype", "")  # fp16/bf16
        self.set_attr("loss", None)
        self.set_attr("dist_context", None)
        self.set_attr("custom_white_list", None)
        self.set_attr("custom_black_list", None)
        self.set_attr("custom_black_varnames", None)
        self.set_attr("init_loss_scaling", 32768.0)
        self.set_attr("incr_every_n_steps", 1000)
        self.set_attr("decr_every_n_nan_or_inf", 2)
        self.set_attr("incr_ratio", 2.0)
        self.set_attr("decr_ratio", 0.8)
        self.set_attr("use_dynamic_loss_scaling", False)
        self.set_attr("input_data", [])
        self.set_attr("params_grads", [])
        self.set_attr("dtype", "")  # fp16/bf16
        self._loss = None
        self._loss_scaling = None
        self._num_good_steps = None
        self._num_bad_steps = None

    def _check_self(self):
        if self.get_attr("dtype") not in ["float16", "bfloat16"]:
            return False
        if self.get_attr("init_loss_scaling") < 0:
            return False
        if self.get_attr("incr_every_n_steps") < 0:
            return False
        if self.get_attr("decr_every_n_nan_or_inf") < 0:
            return False
        if self.get_attr("incr_ratio") < 0:
            return False
        if self.get_attr("decr_ratio") < 0:
            return False
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    # NOTE: why AMPBackwardPass can override apply_single_impl instead of
    # apply_impl? AMP is an optimization pass for serial program,
    # in distributed scenario, all ranks should have the same modification.
    def _apply_single_impl(self, main_program, startup_program, context):
        self.dist_context = self.get_attr("dist_context")
        self.params_grads = self.get_attr("params_grads")
        self.amp_dtype = self.get_attr("dtype")

        amp_lists = AMPLists(
            set(self.get_attr("custom_white_list")),
            set(self.get_attr("custom_black_list")),
            set(self.get_attr("custom_black_varnames")),
            self.amp_dtype,
        )

        with paddle.static.program_guard(main_program, startup_program):
            amp_state = AMPState(
                main_program, amp_lists, self.amp_dtype, self.dist_context
            )
            is_train = amp_state.build_state()

            if is_train:
                self._update_backward_cast_ops()
                self._cast_loss(self.amp_dtype)

            if is_train and self.amp_dtype == "float16":
                self._init_amp_var()
                self._scale_loss()
                if (
                    self.get_attr("use_dynamic_loss_scaling")
                    or self.get_attr("init_loss_scaling") != 1.0
                ):
                    grads, found_inf = self._check_and_update_gradient()

                if self.get_attr("use_dynamic_loss_scaling"):
                    self._update_loss_scaling(grads, found_inf)

    def _update_backward_cast_ops(self):
        """
        move param grad cast to the end of backward segment
        in order to enable fp16 allreduce
        """
        # TODO filter optimize ops in future

        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()

        for p, g in self.params_grads:
            op = g.op
            if g.dtype == paddle.float32 and op.type == 'cast':
                if int(op.attr('op_role')) == int(
                    OpRole.Backward
                ) and op.has_attr('op_role_var'):
                    op._remove_attr("op_role_var")

                post_ops = find_true_post_op(main_block.ops, op, g.name)
                if post_ops:
                    raise ValueError(
                        f"The cast op {op}'s output should not be"
                        "used by a non-optimize op, however, it"
                        f"is used by {post_ops[0]}"
                    )

                if op == main_block.ops[-1]:
                    continue

                # add new op in the python and cpp at the same time
                new_op_desc = main_block.desc.append_op()
                new_op_desc.copy_from(op.desc)
                new_op = paddle.static.Operator(
                    block=main_block,
                    desc=new_op_desc,
                    type=None,
                    inputs=None,
                    outputs=None,
                    attrs=None,
                )
                main_block.ops.append(new_op)

                # dist attr
                param_dist_attr = (
                    self.dist_context.get_tensor_dist_attr_for_program(p)
                )
                output_dist_attr = (
                    self.dist_context.get_tensor_dist_attr_for_program(
                        main_block.var(op.output_arg_names[0])
                    )
                )
                assert param_dist_attr is not None
                assert output_dist_attr is not None
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    new_op,
                    param_dist_attr.process_mesh,
                    param_dist_attr.dims_mapping,
                    self.dist_context,
                    chunk_id=param_dist_attr.chunk_id,
                )

                output_dist_attr.process_mesh = param_dist_attr.process_mesh
                output_dist_attr.dims_mapping = param_dist_attr.dims_mapping
                output_dist_attr.chunk_id = param_dist_attr.chunk_id

                op_idx = find_op_index(main_block.desc, op.desc)
                if op_idx == -1:
                    raise ValueError(f"The op {op} is not in program")
                main_block._remove_op(op_idx, sync=False)

        main_block._sync_with_cpp()

    def _check_and_update_gradient(self):
        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()

        grads = [g for _, g in self.params_grads]
        check_type(grads, 'x', (tuple, list), 'check_finite_and_unscale')
        for e in grads:
            check_variable_and_dtype(
                e,
                "x",
                ['float16', 'float32', 'float64'],
                'check_finite_and_unscale',
            )

        found_inf = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(['find_infinite_scale', 'tmp'])
            ),
            shape=[1],
            dtype='bool',
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False,
        )
        set_var_dist_attr(
            self.dist_context,
            found_inf,
            [-1],
            world_process_group.ranks,
            chunk_id=0,
        )

        inputs = {'X': grads, 'Scale': self._loss_scaling}
        outputs = {'Out': grads, 'FoundInfinite': found_inf}
        attrs = {'op_role': OpRole.Optimize}
        new_op = main_block.append_op(
            type='check_finite_and_unscale',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        # Constructing dist attr from op_desc can
        # give all inputs and outputs default dist attrs
        new_op_dist_attr = OperatorDistAttr(new_op.desc)
        new_op_dist_attr.process_mesh = ProcessMesh(world_process_group.ranks)
        new_op_dist_attr.impl_idx = 0
        new_op_dist_attr.chunk_id = 0
        if len(world_process_group.ranks) > 1:
            new_op_dist_attr.impl_type = "check_finite_and_unscale"
        for g in grads:
            g_dist_attr = self.dist_context.get_tensor_dist_attr_for_program(g)
            assert g_dist_attr is not None
            new_op_dist_attr.set_input_dims_mapping(
                g.name, g_dist_attr.dims_mapping
            )
            new_op_dist_attr.set_output_dims_mapping(
                g.name, g_dist_attr.dims_mapping
            )
        self.dist_context.set_op_dist_attr_for_program(new_op, new_op_dist_attr)
        return grads, found_inf

    def _init_amp_var(self):
        self._loss_scaling = paddle.static.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=self.get_attr("init_loss_scaling"),
            dtype='float32',
            persistable=True,
        )
        set_var_dist_attr(
            self.dist_context,
            self._loss_scaling,
            [-1],
            world_process_group.ranks,
            chunk_id=0,
        )

        if self.get_attr("use_dynamic_loss_scaling"):
            self._num_good_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True,
            )
            set_var_dist_attr(
                self.dist_context,
                self._num_good_steps,
                [-1],
                world_process_group.ranks,
                chunk_id=0,
            )

            self._num_bad_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True,
            )
            set_var_dist_attr(
                self.dist_context,
                self._num_bad_steps,
                [-1],
                world_process_group.ranks,
                chunk_id=0,
            )

    def _cast_loss(self, target_dtype):
        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()

        loss = self.get_attr("loss")
        assert loss is not None
        loss_op = loss.op
        loss_op_dist_attr = self.dist_context.get_op_dist_attr_for_program(
            loss_op
        )

        if loss.dtype != core.VarDesc.VarType.FP32:
            tmp_name = unique_name.generate(loss.name + ".cast_fp32")
            cast_loss = main_block.create_var(
                name=tmp_name, dtype=core.VarDesc.VarType.FP32
            )
            loss_dist_attr = self.dist_context.get_tensor_dist_attr_for_program(
                loss
            )
            ref_mesh = loss_op_dist_attr.process_mesh
            ref_chunk_id = loss_op_dist_attr.chunk_id

            loss_dist_attr.chunk_id = ref_chunk_id
            self.dist_context.set_tensor_dist_attr_for_program(
                cast_loss, loss_dist_attr
            )

            # forward
            loss_op_idx = find_op_index(main_block.desc, loss_op.desc)
            cast_op = main_block._insert_op(
                loss_op_idx + 1,
                type='cast',
                inputs={'X': [loss]},
                outputs={'Out': [cast_loss]},
                attrs={
                    "in_dtype": loss.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32,
                    "op_role": loss_op.all_attrs()[OP_ROLE_KEY],
                },
            )

            loss_op._set_attr(OP_ROLE_KEY, OpRole.Forward)
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                cast_op,
                ref_mesh,
                [-1 for i in loss.shape],
                self.dist_context,
                chunk_id=ref_chunk_id,
            )

            # backward
            first_backward_op = None
            insert_op_offset = 3
            for idx, op in enumerate(main_block.ops[loss_op_idx:]):
                if op.type == "fill_constant" and is_loss_grad_op(op):
                    first_backward_op = op
                    insert_op_offset = idx + 1
                    break
                if is_backward_op(op):
                    break

            assert first_backward_op is not None, "There is not loss_grad op."

            cast_loss_grad = main_block.create_var(
                name=unique_name.generate(tmp_name + "@GRAD"),
                shape=loss.shape,
                dtype=core.VarDesc.VarType.FP32,
                persistable=loss.persistable,
            )
            set_var_dist_attr(
                self.dist_context,
                cast_loss_grad,
                [-1] * len(loss.shape),
                ref_mesh,
                chunk_id=ref_chunk_id,
            )

            pre_grad_name = first_backward_op.output_arg_names[0]
            first_backward_op._rename_output(pre_grad_name, cast_loss_grad.name)
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                first_backward_op,
                ref_mesh,
                [-1] * len(loss.shape),
                self.dist_context,
                chunk_id=ref_chunk_id,
            )
            cast_grad_op = main_block._insert_op(
                loss_op_idx + insert_op_offset,
                type='cast',
                inputs={'X': [cast_loss_grad]},
                outputs={'Out': [pre_grad_name]},
                attrs={
                    "in_dtype": core.VarDesc.VarType.FP32,
                    "out_dtype": _str_to_dtype(target_dtype),
                    "op_role": OpRole.Backward,
                },
            )
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                cast_grad_op,
                ref_mesh,
                [-1 for i in loss.shape],
                self.dist_context,
                chunk_id=ref_chunk_id,
            )
            loss_op = cast_op
            loss = cast_loss
            self.set_attr("loss", loss)
        self._loss = loss
        main_block._sync_with_cpp()

    def _scale_loss(self):
        main_block = paddle.static.default_main_program().global_block()
        loss = self.get_attr("loss")
        assert loss is not None
        loss_op = loss.op
        loss_op_dist_attr = self.dist_context.get_op_dist_attr_for_program(
            loss_op
        )

        if (
            self.get_attr("use_dynamic_loss_scaling")
            or self.get_attr("init_loss_scaling") != 1.0
        ):
            loss_op_idx = find_op_index(main_block.desc, loss_op.desc)

            # forward
            ref_mesh = loss_op_dist_attr.process_mesh
            ref_chunk_id = loss_op_dist_attr.chunk_id

            scaled_loss = main_block.create_var(
                name=unique_name.generate("scaled_loss"),
                shape=loss.shape,
                dtype=loss.dtype,
                persistable=loss.persistable,
            )
            set_var_dist_attr(
                self.dist_context,
                scaled_loss,
                [-1 for i in loss.shape],
                ref_mesh,
                chunk_id=ref_chunk_id,
            )

            elementwise_mul_op = main_block._insert_op(
                loss_op_idx + 1,
                type='elementwise_mul',
                inputs={'X': [loss], 'Y': [self._loss_scaling]},
                outputs={'Out': [scaled_loss]},
                attrs={
                    'op_role': loss_op.all_attrs()[OP_ROLE_KEY],
                },
            )
            loss_op._set_attr(OP_ROLE_KEY, OpRole.Forward)
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                elementwise_mul_op,
                ref_mesh,
                [-1 for i in loss.shape],
                self.dist_context,
                chunk_id=ref_chunk_id,
            )

            # backward
            first_backward_op = None
            for op in main_block.ops[loss_op_idx:]:
                if op.type == "fill_constant" and is_loss_grad_op(op):
                    first_backward_op = op
                    break
                if is_backward_op(op):
                    break

            assert first_backward_op is not None, "There is not loss_grad op."

            scaled_loss_grad = main_block.create_var(
                name=unique_name.generate("scaled_loss") + "@GRAD",
                shape=loss.shape,
                dtype=loss.dtype,
                persistable=loss.persistable,
            )
            set_var_dist_attr(
                self.dist_context,
                scaled_loss_grad,
                [-1] * len(loss.shape),
                ref_mesh,
                chunk_id=ref_chunk_id,
            )
            pre_grad_name = first_backward_op.output_arg_names[0]
            first_backward_op._rename_output(
                pre_grad_name, scaled_loss_grad.name
            )
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                first_backward_op,
                ref_mesh,
                [-1] * len(loss.shape),
                self.dist_context,
                chunk_id=ref_chunk_id,
            )
            scaled_loss_grad.op = first_backward_op
            # FIXME(JZ-LIANG) a trick to insert backward op
            main_block._sync_with_cpp()
            elementwise_mul_grad_op_desc = main_block.desc._insert_op(
                loss_op_idx + 3
            )
            elementwise_mul_grad_op_desc.set_type("elementwise_mul_grad")
            elementwise_mul_grad_op_desc.set_input(
                'Out@GRAD', [scaled_loss_grad.name]
            )
            elementwise_mul_grad_op_desc.set_input('X', [loss.name])
            elementwise_mul_grad_op_desc.set_input(
                'Y', [self._loss_scaling.name]
            )
            elementwise_mul_grad_op_desc.set_output('X@GRAD', [pre_grad_name])
            elementwise_mul_grad_op_desc.set_output('Y@GRAD', [])
            elementwise_mul_grad_op_desc._set_attr(OP_ROLE_KEY, OpRole.Backward)
            elementwise_mul_grad_op_desc._set_attr('axis', -1)
            elementwise_mul_grad_op = paddle.static.Operator(
                main_block, elementwise_mul_grad_op_desc
            )
            main_block.ops.insert(loss_op_idx + 3, elementwise_mul_grad_op)
            main_block._sync_with_cpp()
            elementwise_mul_grad_op = main_block.ops[loss_op_idx + 3]
            assert elementwise_mul_grad_op.type == "elementwise_mul_grad"
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                elementwise_mul_grad_op,
                ref_mesh,
                [-1 for i in loss.shape],
                self.dist_context,
                chunk_id=ref_chunk_id,
            )
        else:
            scaled_loss = loss
        self._loss = scaled_loss
        main_block._sync_with_cpp()

    def _update_loss_scaling(self, grads, found_inf):
        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()

        check_variable_and_dtype(
            self._loss_scaling,
            "prev_loss_scaling",
            ['float32', 'float64'],
            "update_loss_scaling",
        )
        check_type(grads, 'x', (tuple, list), 'update_loss_scaling')
        for e in grads:
            check_variable_and_dtype(
                e, "x", ['float16', 'float32', 'float64'], 'update_loss_scaling'
            )
            if e.dtype == paddle.float16:
                assert (
                    self._loss_scaling.dtype == paddle.float32
                ), "The dtype of prev_loss_scaling should be float32 when the dtype of x is float16."
            else:
                assert (
                    self._loss_scaling.dtype == e.dtype
                ), "The dtype of prev_loss_scaling should be equal to the dtype of x."

        inputs = {
            'X': grads,
            'FoundInfinite': found_inf,
            'PrevLossScaling': self._loss_scaling,
            'InGoodSteps': self._num_good_steps,
            'InBadSteps': self._num_bad_steps,
        }

        outputs = {
            'Out': grads,
            'LossScaling': self._loss_scaling,
            'OutGoodSteps': self._num_good_steps,
            'OutBadSteps': self._num_bad_steps,
        }

        attrs = {
            'incr_every_n_steps': self.get_attr("incr_every_n_steps"),
            'decr_every_n_nan_or_inf': self.get_attr("decr_every_n_nan_or_inf"),
            'incr_ratio': self.get_attr("incr_ratio"),
            'decr_ratio': self.get_attr("decr_ratio"),
            'stop_update': self.get_attr("stop_update"),
            'op_role': OpRole.Optimize,
        }

        new_op = main_block.append_op(
            type='update_loss_scaling',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        # Constructing dist attr from op_desc can
        # give all inputs and outputs default dist attrs
        new_op_dist_attr = OperatorDistAttr(new_op.desc)
        new_op_dist_attr.process_mesh = ProcessMesh(world_process_group.ranks)
        new_op_dist_attr.impl_idx = 0
        new_op_dist_attr.chunk_id = 0
        if len(world_process_group.ranks) > 1:
            new_op_dist_attr.impl_type = "update_loss_scaling"
        for g in grads:
            g_dist_attr = self.dist_context.get_tensor_dist_attr_for_program(g)
            assert g_dist_attr is not None
            new_op_dist_attr.set_input_dims_mapping(
                g.name, g_dist_attr.dims_mapping
            )
            new_op_dist_attr.set_output_dims_mapping(
                g.name, g_dist_attr.dims_mapping
            )
        self.dist_context.set_op_dist_attr_for_program(new_op, new_op_dist_attr)

        main_block._sync_with_cpp()

    def get_loss(self):
        # the amp might change the effective loss variable for network and
        # therefore would affect the subsequent passes that rely on the loss.
        # return the effective loss after amp pass.

        if self._loss:
            return self._loss
        else:
            return self.get_attr("loss")
