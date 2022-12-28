# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import static
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed.auto_parallel.process_group import (
    get_world_process_group,
)
from paddle.distributed.auto_parallel.utils import (
    get_loss_op,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_var_dist_attr,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.distributed.passes.pass_base import PassBase, register_pass
from paddle.fluid import unique_name
from paddle.fluid.contrib.mixed_precision.bf16 import (
    AutoMixedPrecisionListsBF16,
)
from paddle.fluid.contrib.mixed_precision.bf16.amp_utils import (
    _dtype_to_str,
    _is_in_fp32_varnames,
    _valid_types,
    find_op_index,
    find_true_post_op,
)
from paddle.fluid.contrib.mixed_precision.fp16_utils import (
    _rename_arg,
    find_true_prev_op,
)
from paddle.fluid.framework import Block
from paddle.framework import core

from ..auto_parallel.utils import is_backward_op, is_forward_op, is_loss_op

world_process_group = get_world_process_group()


class BF16State(object):
    def __init__(self, block):
        self._block: Block = block
        self._op_bf16_dict = {}
        self._var_name_dict = {}

    def _is_bf16_op(self, op_id):
        return self._op_bf16_dict.get(op_id, None)

    def _build_state(self, amp_lists, dist_context):
        ops = self._block.ops
        dist_op_context = dist_context.dist_op_context
        training = False
        for op in ops:
            if int(op.attr("op_role")) == 257:
                training = True

            if int(op.attr("op_role")) == int(OpRole.Forward):
                self._mark_black_white_op(amp_lists, op, ops)
            elif int(op.attr("op_role")) == int(OpRole.Backward):
                if op.desc.original_id() in dist_op_context.grad_op_id_to_op_id:
                    fwd_op_id = dist_op_context.grad_op_id_to_op_id[
                        op.desc.original_id()
                    ]
                    if self._is_bf16_op(fwd_op_id) is True:
                        self._op_bf16_dict[op.desc.original_id()] = True
                    elif self._is_bf16_op(fwd_op_id) is False:
                        self._op_bf16_dict[op.desc.original_id()] = False
            elif int(op.attr("op_role")) == int(OpRole.Optimize):
                break
        return training

    def _mark_black_white_op(self, amp_lists, op, ops):
        if op.type == "create_py_reader" or op.type == "read":
            return
        if amp_lists.fp32_varnames is not None and _is_in_fp32_varnames(
            op, amp_lists
        ):
            self._op_bf16_dict[op.desc.original_id()] = False
            return
        if op.type in amp_lists.bf16_list:
            self._op_bf16_dict[op.desc.original_id()] = True
        elif op.type in amp_lists.gray_list:
            is_fp32_op = False
            is_bf16_op = False
            for in_name in op.input_names:
                if in_name:
                    for in_var_name in op.input(in_name):
                        in_var = self._block.var(in_var_name)
                        if in_var.op is None:
                            continue
                        elif in_var.op is op:
                            prev_op = find_true_prev_op(ops, op, in_var_name)
                            if prev_op is None:
                                continue
                        else:
                            prev_op = in_var.op
                        if (
                            self._op_bf16_dict.get(
                                prev_op.desc.original_id(), False
                            )
                            is False
                            or prev_op.type in amp_lists.fp32_list
                        ):
                            is_fp32_op = True
                        elif (
                            self._op_bf16_dict.get(
                                prev_op.desc.original_id(), False
                            )
                            is True
                            or prev_op.type in amp_lists.bf16_list
                        ):
                            is_bf16_op = True
            if is_fp32_op:
                self._op_bf16_dict[op.desc.original_id()] = False
            elif is_bf16_op:
                self._op_bf16_dict[op.desc.original_id()] = True
            else:
                pass
        else:
            self._op_bf16_dict[op.desc.original_id()] = False

    def cast_forward_program(self, dist_context):
        ops = self._block.ops
        idx = 0
        while idx < len(ops):
            num_cast_ops = 0
            op = ops[idx]
            if int(op.attr('op_role')) == int(OpRole.Backward):
                break
            if self._is_bf16_op(op.desc.original_id()) is False:
                num_cast_ops = self._insert_cast_op_forward(
                    op,
                    idx,
                    core.VarDesc.VarType.BF16,
                    core.VarDesc.VarType.FP32,
                    dist_context,
                )
            elif self._is_bf16_op(op.desc.original_id()) is True:
                if op.has_attr('use_mkldnn'):
                    op._set_attr('use_mkldnn', True)
                    op._set_attr('mkldnn_data_type', 'bfloat16')
                elif (
                    op.has_attr('dtype')
                    and op.attr('dtype') == core.VarDesc.VarType.FP32
                ):
                    op._set_attr('dtype', core.VarDesc.VarType.BF16)

                num_cast_ops = self._insert_cast_op_forward(
                    op,
                    idx,
                    core.VarDesc.VarType.FP32,
                    core.VarDesc.VarType.BF16,
                    dist_context,
                )
            else:
                pass

            idx += num_cast_ops + 1
        self._block._sync_with_cpp()

    def _insert_cast_op_forward(
        self, op, idx, src_dtype, dst_dtype, dist_context: DistributedContext
    ):
        num_cast_ops = 0
        var_name_dict = {}
        for in_name in op.input_names:
            if src_dtype == core.VarDesc.VarType.FP32 and op.type in [
                'batch_norm',
                'fused_bn_add_activation',
                'layer_norm',
            ]:
                if in_name not in {'X', 'Z'}:
                    continue
            for in_var_name in op.input(in_name):
                in_var = self._block.var(in_var_name)
                if in_var.type not in _valid_types or in_var.dtype == dst_dtype:
                    continue
                if in_var.dtype == src_dtype:
                    cast_name = (
                        in_var.name + '.cast_' + _dtype_to_str(dst_dtype)
                    )
                    var_name_dict[in_var.name] = cast_name
                    out_var = self._block.vars.get(cast_name)
                    consume_op_attr = dist_context.get_op_dist_attr_for_program(
                        op
                    )
                    assert consume_op_attr is not None
                    in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                        in_var_name
                    )
                    if out_var is None or out_var.dtype != dst_dtype:
                        assert in_var_dist_attr is not None
                        ref_mesh = in_var_dist_attr.process_mesh
                        ref_mapping = in_var_dist_attr.dims_mapping
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr
                        )

                        out_var = self._block.create_var(
                            name=cast_name,
                            dtype=dst_dtype,
                            persistable=False,
                            stop_gradient=in_var.stop_gradient,
                        )
                        set_var_dist_attr(
                            dist_context, out_var, ref_mapping, ref_mesh
                        )

                        cast_op = self._block._insert_op_without_sync(
                            idx,
                            type="cast",
                            inputs={"X": in_var},
                            outputs={"Out": out_var},
                            attrs={
                                "in_dtype": in_var.dtype,
                                "out_dtype": out_var.dtype,
                            },
                        )
                        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                            cast_op, ref_mesh, ref_mapping, dist_context
                        )
                        num_cast_ops += 1
                    else:
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr
                        )
                    _rename_arg(op, in_var_name, out_var.name)
                else:
                    if op.has_attr('in_dtype'):
                        op._set_attr('in_dtype', dst_dtype)
        self._var_name_dict[op.desc.original_id()] = var_name_dict

        if (
            src_dtype == core.VarDesc.VarType.FP32
            and dst_dtype == core.VarDesc.VarType.BF16
        ):
            for out_name in op.output_names:
                if (
                    op.type
                    in ['batch_norm', 'fused_bn_add_activation', 'layer_norm']
                    and out_name != 'Y'
                ):
                    continue
                for out_var_name in op.output(out_name):
                    out_var = self._block.var(out_var_name)
                    if out_var.type not in _valid_types:
                        continue
                    if out_var.dtype == core.VarDesc.VarType.FP32:
                        out_var.desc.set_dtype(core.VarDesc.VarType.BF16)
                        if op.has_attr('out_dtype'):
                            op._set_attr('out_dtype', core.VarDesc.VarType.BF16)
        return num_cast_ops

    def cast_backward_program(self, params_grads, dist_context):
        self._block._sync_with_cpp()
        ops = self._block.ops
        appended_grad_times = 0
        dist_op_context = dist_context.dist_op_context
        loss_op = get_loss_op(self._block)
        idx = find_op_index(self._block.desc, loss_op.desc) + 1
        while idx < len(ops):
            num_cast_ops = 0
            grad_op = ops[idx]
            op_dist_attr = dist_context.get_op_dist_attr_for_program(grad_op)
            if is_backward_op(grad_op) and (
                is_forward_op(ops[idx - 1]) or is_loss_op(ops[idx - 1])
            ):
                if not op_dist_attr.is_recompute:
                    appended_grad_times += 1
            if (
                grad_op.desc.original_id()
                in dist_op_context.grad_op_id_to_op_id
            ):
                if self._is_bf16_op(grad_op.desc.original_id()) is False:
                    num_cast_ops = self._insert_cast_op_backward(
                        grad_op,
                        idx,
                        core.VarDesc.VarType.BF16,
                        core.VarDesc.VarType.FP32,
                        dist_context,
                        appended_grad_times,
                    )
                elif self._is_bf16_op(grad_op.desc.original_id()) is True:
                    if grad_op.has_attr('use_mkldnn'):
                        grad_op._set_attr('use_mkldnn', True)
                        grad_op._set_attr('mkldnn_data_type', 'bfloat16')
                    elif (
                        grad_op.has_attr('dtype')
                        and grad_op.attr('dtype') == core.VarDesc.VarType.FP32
                    ):
                        grad_op._set_attr('dtype', core.VarDesc.VarType.BF16)
                    num_cast_ops = self._insert_cast_op_backward(
                        grad_op,
                        idx,
                        core.VarDesc.VarType.FP32,
                        core.VarDesc.VarType.BF16,
                        dist_context,
                        appended_grad_times,
                    )
            elif grad_op.type == "sum":
                in_var_name = grad_op.desc.input_arg_names()[0]
                src_dtype = self._block.var(in_var_name).dtype
                for in_var_name in grad_op.desc.input_arg_names():
                    assert src_dtype == self._block.var(in_var_name).dtype
                out_var_name = grad_op.desc.output_arg_names()[0]
                out_var = self._block.var(out_var_name)
                if out_var.dtype != src_dtype:
                    out_var.desc.set_dtype(src_dtype)
            elif int(grad_op.attr("op_role")) == 257:
                pass
            else:
                raise ValueError(
                    "'{}' op is not supported in the complete amp pass.".format(
                        grad_op.type
                    )
                )
            idx += num_cast_ops + 1
        self._block._sync_with_cpp()
        _update_backward_cast_ops(params_grads, dist_context)

    def _insert_cast_op_backward(
        self,
        grad_op,
        idx,
        src_dtype,
        dst_dtype,
        dist_context,
        appended_grad_times,
    ):
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
        original_id = grad_op.desc.original_id()
        dist_op_context = dist_context.dist_op_context
        fwd_op_id = dist_op_context.grad_op_id_to_op_id[original_id]
        for in_name in grad_op.input_names:
            if src_dtype == core.VarDesc.VarType.FP32 and _keep_fp32_input(
                grad_op, in_name
            ):
                for in_var_name in grad_op.input(in_name):
                    in_var = self._block._find_var_recursive(in_var_name)
                    assert in_var.dtype == core.VarDesc.VarType.FP32
                continue
            for in_var_name in grad_op.input(in_name):
                in_var = self._block._find_var_recursive(in_var_name)
                if in_var.dtype == src_dtype:
                    consume_op_attr = dist_context.get_op_dist_attr_for_program(
                        grad_op
                    )
                    if in_var_name in self._var_name_dict[fwd_op_id]:
                        cast_name = self._var_name_dict[fwd_op_id][in_var_name]
                        grad_op.desc._rename_input(in_var_name, cast_name)
                        in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                            in_var_name
                        )
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr
                        )
                    else:
                        assert (
                            in_var.dtype == dst_dtype
                        ), "op [{}] expect input [{}] to be dtype [{}] BUT got [{}]. {}".format(
                            grad_op.type,
                            in_name,
                            dst_dtype,
                            in_var.dtype,
                            str(grad_op),
                        )

        for out_name in grad_op.output_names:
            if src_dtype == core.VarDesc.VarType.FP32 and _keep_fp32_output(
                grad_op, out_name
            ):
                for out_var_name in grad_op.output(out_name):
                    out_var = self._block._find_var_recursive(out_var_name)
                    assert out_var.dtype == core.VarDesc.VarType.FP32
                continue

            for out_var_name in grad_op.output(out_name):
                out_var = self._block._find_var_recursive(out_var_name)
                out_var_name_prefix = out_var_name[: out_var_name.find('@')]
                fwd_var = self._block._find_var_recursive(out_var_name_prefix)
                if out_var.dtype != fwd_var.dtype:
                    out_var.desc.set_dtype(fwd_var.dtype)

                if out_var.dtype == src_dtype:
                    if out_var_name_prefix in self._var_name_dict[fwd_op_id]:
                        consume_op_attr = (
                            dist_context.get_op_dist_attr_for_program(grad_op)
                        )
                        fwd_cast_name = self._var_name_dict[fwd_op_id][
                            out_var_name_prefix
                        ]
                        suffix = ''
                        if "@RENAME" in out_var_name:
                            suffix = out_var_name[
                                out_var_name.find("@RENAME") :
                            ]
                        cast_name = fwd_cast_name + "@GRAD" + suffix
                        cast_var = self._block.vars.get(cast_name)
                        if cast_var is None or cast_var.dtype != dst_dtype:
                            grad_op.desc._rename_output(out_var_name, cast_name)
                            out_var_dist_attr = (
                                consume_op_attr.get_output_dist_attr(
                                    out_var_name
                                )
                            )
                            ref_mesh = out_var_dist_attr.process_mesh
                            ref_mapping = out_var_dist_attr.dims_mapping
                            consume_op_attr.set_output_dist_attr(
                                cast_name, out_var_dist_attr
                            )
                            assert ref_mapping is not None
                            cast_var = self._block.create_var(
                                name=cast_name,
                                shape=out_var.shape,
                                dtype=dst_dtype,
                                persistable=False,
                                stop_gradient=out_var.stop_gradient,
                            )
                            set_var_dist_attr(
                                dist_context, cast_var, ref_mapping, ref_mesh
                            )
                            dist_op_context.grad_var_to_var[
                                appended_grad_times
                            ][cast_name] = fwd_cast_name

                            cast_op = self._block._insert_op(
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
                                cast_op, ref_mesh, ref_mapping, dist_context
                            )
                            num_cast_ops += 1
                else:
                    assert out_var.dtype == dst_dtype
        return num_cast_ops


def _update_backward_cast_ops(params_grads, dist_context):
    """
    move param grad cast to the end of backward segment
    in order to enabel fp16 allreduce
    """
    # TODO filter optimize ops in future

    main_block = paddle.static.default_main_program().global_block()
    main_block._sync_with_cpp()

    for p, g in params_grads:
        op = g.op
        if g.dtype == core.VarDesc.VarType.FP32 and op.type == 'cast':
            if int(op.attr('op_role')) == int(OpRole.Backward) and op.has_attr(
                'op_role_var'
            ):
                op._remove_attr("op_role_var")

            post_ops = find_true_post_op(main_block.ops, op, g.name)
            if post_ops:
                raise ValueError(
                    "The cast op {0}'s output should not be"
                    "used by a non-optimize op, however, it"
                    "is used by {1}".format(op, post_ops[0])
                )

            if op == main_block.ops[-1]:
                continue

            # add new op in the python and cpp at the same time
            new_op_desc = main_block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            new_op = paddle.fluid.framework.Operator(
                block=main_block,
                desc=new_op_desc,
                type=None,
                inputs=None,
                outputs=None,
                attrs=None,
            )
            main_block.ops.append(new_op)

            # dist attr
            param_dist_attr = dist_context.get_tensor_dist_attr_for_program(p)
            output_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                main_block.var(op.output_arg_names[0])
            )
            assert param_dist_attr is not None
            assert output_dist_attr is not None
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                new_op,
                param_dist_attr.process_mesh,
                param_dist_attr.dims_mapping,
                dist_context,
            )

            output_dist_attr.process_mesh = param_dist_attr.process_mesh
            output_dist_attr.dims_mapping = param_dist_attr.dims_mapping

            op_idx = find_op_index(main_block.desc, op.desc)
            if op_idx == -1:
                raise ValueError("The op {0} is not in program".format(op))
            main_block._remove_op(op_idx, sync=False)

    main_block._sync_with_cpp()


@register_pass("auto_parallel_bf16")
class BF16Pass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("dist_context", None)
        self.set_attr("custom_bf16_list", None)
        self.set_attr("custom_fp32_list", None)
        self.set_attr("custom_fp32_varnames", None)
        self.set_attr("input_data", [])
        self.set_attr("loss", None)
        self.set_attr("params_grads", [])
        self.set_attr("use_bf16_guard", False)
        self._loss = None

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self.dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")

        amp_lists = AutoMixedPrecisionListsBF16(
            self.get_attr("custom_bf16_list"),
            self.get_attr("custom_fp32_list"),
            self.get_attr("custom_fp32_varnames"),
        )

        with static.program_guard(main_program, startup_program):
            amp_state = BF16State(main_program.global_block())
            training = amp_state._build_state(amp_lists, self.dist_context)
            amp_state.cast_forward_program(self.dist_context)

        if training:
            with paddle.static.program_guard(main_program, startup_program):
                amp_state.cast_backward_program(params_grads, self.dist_context)
                self._scale_loss()

    def _scale_loss(self):

        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()

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
            self.dist_context.set_tensor_dist_attr_for_program(
                cast_loss, loss_dist_attr
            )

            loss_op_idx = find_op_index(main_block.desc, loss_op.desc)
            cast_op = main_block._insert_op(
                loss_op_idx + 1,
                type='cast',
                inputs={"X": [loss]},
                outputs={"Out": [cast_loss]},
                attrs={
                    "in_dtype": loss.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32,
                    "op_role": loss_op.all_attrs()[OP_ROLE_KEY],
                },
            )

            loss_op._set_attr(
                OP_ROLE_KEY, core.op_proto_and_checker_maker.OpRole.Forward
            )
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                cast_op, ref_mesh, [-1], self.dist_context
            )
            first_backward_op = main_block.ops[loss_op_idx + 2]
            assert (
                first_backward_op.type == "fill_constant"
                and int(first_backward_op.all_attrs()[OP_ROLE_KEY]) == 257
            )
            cast_loss_grad = main_block.create_var(
                name=unique_name.generate(tmp_name + "@GRAD"),
                shape=loss.shape,
                dtype=core.VarDesc.VarType.FP32,
                persistable=loss.persistable,
            )
            set_var_dist_attr(self.dist_context, cast_loss_grad, [-1], ref_mesh)
            pre_grad_name = first_backward_op.output_arg_names[0]
            first_backward_op._rename_output(pre_grad_name, cast_loss_grad.name)
            cast_grad_op = main_block._insert_op(
                loss_op_idx + 3,
                type='cast',
                inputs={'X': [cast_loss_grad]},
                outputs={'Out': [pre_grad_name]},
                attrs={
                    "in_dtype": core.VarDesc.VarType.FP32,
                    "out_dtype": core.VarDesc.VarType.FP16,
                    'op_role': core.op_proto_and_checker_maker.OpRole.Backward,
                },
            )
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                cast_grad_op, ref_mesh, [-1], self.dist_context
            )
            loss = cast_loss
        self._loss = loss
        main_block._sync_with_cpp()

    def get_loss(self):
        if self._loss:
            return self._loss
        else:
            return self.get_attr("loss")
