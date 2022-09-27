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
from paddle.framework import core
from paddle.fluid import unique_name
from .pass_base import PassBase, register_pass
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.distributed.auto_parallel.utils import get_loss_op, set_var_dist_attr
from paddle.distributed.auto_parallel.utils import naive_set_dist_op_attr_for_program_by_mesh_and_mapping
from paddle.distributed.auto_parallel.process_group import get_world_process_group
from paddle.fluid.contrib.mixed_precision.fp16_utils import AutoMixedPrecisionLists
from paddle.fluid.contrib.mixed_precision.fp16_utils import _keep_fp32_input, _keep_fp32_output, find_op_index
from paddle.fluid.contrib.mixed_precision.fp16_utils import _valid_types, find_true_post_op, find_true_prev_op
from paddle.fluid.contrib.mixed_precision.fp16_utils import _is_in_black_varnames, _dtype_to_str, _rename_arg
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
from ..auto_parallel.utils import is_forward_op, is_backward_op, is_loss_op

world_process_group = get_world_process_group()


class AMPState(object):

    def __init__(self, block):
        self._block = block
        self._op_fp16_dict = {
        }  # op_id --> True/False. 'True' means that the current op is in fp16 mode.
        self._var_name_dict = {}  # fwd_op_id --> {old_name: cast_name}

    def _is_fp16_op(self, op_id):
        return self._op_fp16_dict.get(op_id, None)

    def _build_stats(self, amp_lists, dist_context):
        ops = self._block.ops
        dist_op_context = dist_context.dist_op_context
        for op in ops:
            if int(op.attr('op_role')) == int(OpRole.Forward):
                self._mark_black_white_ops(amp_lists)
            elif int(op.attr('op_role')) == int(OpRole.Backward):
                if op.desc.original_id() in dist_op_context.grad_op_id_to_op_id:
                    fwd_op_id = dist_op_context.grad_op_id_to_op_id[
                        op.desc.original_id()]
                    if self._is_fp16_op(fwd_op_id) == True:
                        self._op_fp16_dict[op.desc.original_id()] = True
                    elif self._is_fp16_op(fwd_op_id) == False:
                        self._op_fp16_dict[op.desc.original_id()] = False
            elif int(op.attr('op_role')) == int(OpRole.Optimize):
                break

    def _mark_black_white_ops(self, amp_lists):
        """
        this function is modified from paddle.fluid.contrib.mixed_precision
        """
        self._block._sync_with_cpp()
        ops = self._block.ops

        for op in ops:
            if int(op.attr('op_role')) == int(OpRole.Backward):
                break
            if op.type == 'create_py_reader' or op.type == 'read':
                continue
            if amp_lists.black_varnames is not None and _is_in_black_varnames(
                    op, amp_lists):
                self._op_fp16_dict[op.desc.original_id()] = False
                continue
            if op.type in amp_lists.black_list:
                self._op_fp16_dict[op.desc.original_id()] = False
            elif op.type in amp_lists.white_list:
                self._op_fp16_dict[op.desc.original_id()] = True
            elif op.type in amp_lists.gray_list:
                is_black_op = False
                is_white_op = False
                for in_name in op.input_names:
                    # if this op has inputs
                    if in_name:
                        for in_var_name in op.input(in_name):
                            in_var = self._block.var(in_var_name)
                            # this in_var isn't the output of other op
                            if in_var.op is None:
                                continue
                            elif in_var.op is op:
                                prev_op = find_true_prev_op(
                                    ops, op, in_var_name)
                                if prev_op is None:
                                    continue
                            else:
                                prev_op = in_var.op
                            # if it's one of inputs
                            if self._is_fp16_op(prev_op.desc.original_id()) == False or \
                                    prev_op.type in amp_lists.black_list:
                                is_black_op = True
                            elif self._is_fp16_op(prev_op.desc.original_id()) == True or \
                                    prev_op.type in amp_lists.white_list:
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

    def cast_forward_program(self, dist_context):
        ops = self._block.ops
        idx = 0
        while idx < len(ops):
            op = ops[idx]
            num_cast_ops = 0
            if int(op.attr('op_role')) == int(OpRole.Backward):
                break
            if self._is_fp16_op(op.desc.original_id()) == False:
                num_cast_ops = self._insert_cast_op_forward(
                    op, idx, core.VarDesc.VarType.FP16,
                    core.VarDesc.VarType.FP32, dist_context)
            elif self._is_fp16_op(op.desc.original_id()) == True:
                num_cast_ops = self._insert_cast_op_forward(
                    op, idx, core.VarDesc.VarType.FP32,
                    core.VarDesc.VarType.FP16, dist_context)
            else:
                pass
            idx += num_cast_ops + 1
        self._block._sync_with_cpp()

    def _insert_cast_op_forward(self, op, idx, src_dtype, dst_dtype,
                                dist_context):
        """
        only for forward cast
        modified from paddle.fluid.contrib.mixed_precision
        """
        num_cast_ops = 0
        var_name_dict = {}
        for in_name in op.input_names:
            if src_dtype == core.VarDesc.VarType.FP32 and _keep_fp32_input(
                    op, in_name):
                continue
            for in_var_name in op.input(in_name):
                in_var = self._block._find_var_recursive(in_var_name)
                if in_var.type not in _valid_types or in_var.dtype == dst_dtype:
                    continue
                if in_var.dtype == src_dtype:
                    cast_name = in_var.name + '.cast_' + _dtype_to_str(
                        dst_dtype)
                    out_var = self._block.vars.get(cast_name)
                    var_name_dict[in_var.name] = cast_name
                    consume_op_attr = dist_context.get_op_dist_attr_for_program(
                        op)
                    assert consume_op_attr is not None
                    if out_var is None or out_var.dtype != dst_dtype:
                        # NOTE we make the cast op and var's dist attr as the op that consume the
                        # cast var instead of the op which generates the var
                        in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                            in_var.name)
                        assert in_var_dist_attr is not None
                        ref_mesh = in_var_dist_attr.process_mesh
                        ref_mapping = in_var_dist_attr.dims_mapping
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr)

                        out_var = self._block.create_var(
                            name=cast_name,
                            dtype=dst_dtype,
                            persistable=False,
                            stop_gradient=in_var.stop_gradient)
                        set_var_dist_attr(dist_context, out_var, ref_mapping,
                                          ref_mesh)

                        cast_op = self._block._insert_op_without_sync(
                            idx,
                            type="cast",
                            inputs={"X": in_var},
                            outputs={"Out": out_var},
                            attrs={
                                "in_dtype": in_var.dtype,
                                "out_dtype": out_var.dtype,
                            })
                        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                            cast_op, ref_mesh, ref_mapping, dist_context)
                        num_cast_ops += 1
                    else:
                        in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                            in_var.name)
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr)
                    _rename_arg(op, in_var.name, cast_name)
                else:
                    if op.has_attr('in_dtype'):
                        op._set_attr('in_dtype', dst_dtype)
        self._var_name_dict[op.desc.original_id()] = var_name_dict

        if src_dtype == core.VarDesc.VarType.FP32 and dst_dtype == core.VarDesc.VarType.FP16:
            for out_name in op.output_names:
                if _keep_fp32_output(op, out_name):
                    continue
                for out_var_name in op.output(out_name):
                    out_var = self._block.var(out_var_name)
                    if out_var.type not in _valid_types:
                        continue
                    if out_var.dtype == core.VarDesc.VarType.FP32:
                        out_var.desc.set_dtype(core.VarDesc.VarType.FP16)
                        if op.has_attr('out_dtype'):
                            op._set_attr('out_dtype', core.VarDesc.VarType.FP16)
        return num_cast_ops

    def cast_backward_program(self, params_grads, dist_context):
        self._block._sync_with_cpp()
        ops = self._block.ops

        loss_op = get_loss_op(self._block)
        loss_op_index = find_op_index(self._block.desc, loss_op.desc)

        appended_grad_times = 0
        idx = loss_op_index + 1
        while idx < len(ops):
            num_cast_ops = 0
            grad_op = ops[idx]

            # NOTE: the map in `grad_var_to_var` may be changed when the var is casted,
            # which will affect the dist_op to insert allreduce_sum op.
            op_dist_attr = dist_context.get_op_dist_attr_for_program(grad_op)
            if is_backward_op(grad_op) and (is_forward_op(ops[idx - 1])
                                            or is_loss_op(ops[idx - 1])):
                if not op_dist_attr.is_recompute:
                    appended_grad_times += 1

            grad_op_orig_id = grad_op.desc.original_id()
            dist_op_context = dist_context.dist_op_context
            if grad_op_orig_id in dist_op_context.grad_op_id_to_op_id:
                if self._is_fp16_op(grad_op_orig_id) == False:  # fp32
                    num_cast_ops = self._insert_cast_op_backward(
                        grad_op, idx, core.VarDesc.VarType.FP16,
                        core.VarDesc.VarType.FP32, dist_context,
                        appended_grad_times)
                elif self._is_fp16_op(grad_op_orig_id) == True:  # fp16
                    num_cast_ops = self._insert_cast_op_backward(
                        grad_op, idx, core.VarDesc.VarType.FP32,
                        core.VarDesc.VarType.FP16, dist_context,
                        appended_grad_times)
            elif grad_op.type == "sum":
                in_var_name = grad_op.desc.input_arg_names()[0]
                src_dtype = self._block.var(in_var_name).dtype
                for in_var_name in grad_op.desc.input_arg_names():
                    assert src_dtype == self._block.var(in_var_name).dtype
                out_var_name = grad_op.desc.output_arg_names()[0]
                out_var = self._block.var(out_var_name)
                if out_var.dtype != src_dtype:
                    out_var.desc.set_dtype(src_dtype)
            elif int(grad_op.attr('op_role')) == 257:
                pass
            else:
                raise ValueError(
                    "'{}' op is not supported in the complete amp pass.".format(
                        grad_op.type))
            idx += num_cast_ops + 1

        self._block._sync_with_cpp()
        _update_backward_cast_ops(params_grads, dist_context)

    def _insert_cast_op_backward(self, grad_op, idx, src_dtype, dst_dtype,
                                 dist_context, appended_grad_times):
        """ only for backward cast """

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
                    grad_op, in_name):
                for in_var_name in grad_op.input(in_name):
                    in_var = self._block._find_var_recursive(in_var_name)
                    assert in_var.dtype == core.VarDesc.VarType.FP32
                continue

            for in_var_name in grad_op.input(in_name):
                in_var = self._block._find_var_recursive(in_var_name)
                if in_var.dtype == src_dtype:
                    consume_op_attr = dist_context.get_op_dist_attr_for_program(
                        grad_op)
                    if in_var_name in self._var_name_dict[fwd_op_id]:
                        # NOTE: if in_var of consume grad_op has been casted before,
                        # it should be renamed and reset dist_attr.
                        cast_name = self._var_name_dict[fwd_op_id][in_var_name]
                        grad_op.desc._rename_input(in_var_name, cast_name)
                        in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                            in_var_name)
                        consume_op_attr.set_input_dist_attr(
                            cast_name, in_var_dist_attr)
                    else:
                        assert in_var.dtype == dst_dtype, "op [{}] expect input [{}] to be dtype [{}] BUT got [{}]. {}".format(
                            grad_op.type, in_name, dst_dtype, in_var.dtype,
                            str(grad_op))

        for out_name in grad_op.output_names:
            if src_dtype == core.VarDesc.VarType.FP32 and _keep_fp32_output(
                    grad_op, out_name):
                for out_var_name in grad_op.output(out_name):
                    out_var = self._block._find_var_recursive(out_var_name)
                    assert out_var.dtype == core.VarDesc.VarType.FP32
                continue

            for out_var_name in grad_op.output(out_name):
                out_var = self._block._find_var_recursive(out_var_name)
                out_var_name_prefix = out_var_name[:out_var_name.find("@")]
                fwd_var = self._block._find_var_recursive(out_var_name_prefix)
                # NOTE: the out_var's dtype of consume grad_op should equal to the fwd_var's dtype
                if out_var.dtype != fwd_var.dtype:
                    out_var.desc.set_dtype(fwd_var.dtype)

                if out_var.dtype == src_dtype:
                    if out_var_name_prefix in self._var_name_dict[fwd_op_id]:
                        # NOTE: if out_var of consume grad_op has been casted before,
                        # it should be renamed and reset dist_attr, then we insert cast op to
                        # convert the cast_var to original dtype
                        consume_op_attr = dist_context.get_op_dist_attr_for_program(
                            grad_op)
                        fwd_cast_name = self._var_name_dict[fwd_op_id][
                            out_var_name_prefix]
                        suffix = ""
                        if "@RENAME" in out_var_name:
                            suffix = out_var_name[out_var_name.find("@RENAME"):]
                        cast_name = fwd_cast_name + "@GRAD" + suffix
                        cast_var = self._block.vars.get(cast_name)
                        if cast_var is None or cast_var.dtype != dst_dtype:
                            grad_op.desc._rename_output(out_var_name, cast_name)
                            out_var_dist_attr = consume_op_attr.get_output_dist_attr(
                                out_var_name)
                            ref_mesh = out_var_dist_attr.process_mesh
                            ref_mapping = out_var_dist_attr.dims_mapping
                            consume_op_attr.set_output_dist_attr(
                                cast_name, out_var_dist_attr)
                            assert ref_mapping is not None
                            cast_var = self._block.create_var(
                                name=cast_name,
                                shape=out_var.shape,
                                dtype=dst_dtype,
                                persistable=False,
                                stop_gradient=out_var.stop_gradient)
                            set_var_dist_attr(dist_context, cast_var,
                                              ref_mapping, ref_mesh)
                            dist_op_context.grad_var_to_var[
                                appended_grad_times][cast_name] = fwd_cast_name

                            cast_op = self._block._insert_op(
                                idx + 1,
                                type="cast",
                                inputs={"X": cast_var},
                                outputs={"Out": out_var},
                                attrs={
                                    "in_dtype": cast_var.dtype,
                                    "out_dtype": out_var.dtype,
                                    "op_role": OpRole.Backward
                                })
                            cast_op._remove_attr("op_role_var")
                            cast_op._remove_attr("op_namescope")
                            cast_op._remove_attr("with_quant_attr")
                            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                                cast_op, ref_mesh, ref_mapping, dist_context)
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
            if int(op.attr('op_role')) == int(
                    OpRole.Backward) and op.has_attr('op_role_var'):
                op._remove_attr("op_role_var")

            post_ops = find_true_post_op(main_block.ops, op, g.name)
            if post_ops:
                raise ValueError("The cast op {0}'s output should not be"
                                 "used by a non-optimize op, however, it"
                                 "is used by {1}".format(op, post_ops[0]))

            if op == main_block.ops[-1]:
                continue

            # add new op in the python and cpp at the same time
            new_op_desc = main_block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            new_op = paddle.fluid.framework.Operator(block=main_block,
                                                     desc=new_op_desc,
                                                     type=None,
                                                     inputs=None,
                                                     outputs=None,
                                                     attrs=None)
            main_block.ops.append(new_op)

            # dist attr
            param_dist_attr = dist_context.get_tensor_dist_attr_for_program(p)
            output_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                main_block.var(op.output_arg_names[0]))
            assert param_dist_attr is not None
            assert output_dist_attr is not None
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                new_op, param_dist_attr.process_mesh,
                param_dist_attr.dims_mapping, dist_context)

            output_dist_attr.process_mesh = param_dist_attr.process_mesh
            output_dist_attr.dims_mapping = param_dist_attr.dims_mapping

            op_idx = find_op_index(main_block.desc, op.desc)
            if op_idx == -1:
                raise ValueError("The op {0} is not in program".format(op))
            main_block._remove_op(op_idx, sync=False)

    main_block._sync_with_cpp()


def _check_and_update_gradient(params_grads, loss_scaling, dist_context):

    main_block = paddle.static.default_main_program().global_block()
    main_block._sync_with_cpp()

    grads = [g for _, g in params_grads]
    check_type(grads, 'x', (tuple, list), 'check_finite_and_unscale')
    for e in grads:
        check_variable_and_dtype(e, "x", ['float16', 'float32', 'float64'],
                                 'check_finite_and_unscale')

    found_inf = main_block.create_var(
        name=unique_name.generate_with_ignorable_key(".".join(
            ['find_infinite_scale', 'tmp'])),
        shape=[1],
        dtype='bool',
        type=core.VarDesc.VarType.LOD_TENSOR,
        persistable=False,
        stop_gradient=False)
    set_var_dist_attr(dist_context, found_inf, [-1], world_process_group.ranks)

    inputs = {'X': grads, 'Scale': loss_scaling}
    outputs = {'Out': grads, 'FoundInfinite': found_inf}
    attrs = {'op_role': OpRole.Optimize}
    new_op = main_block.append_op(type='check_finite_and_unscale',
                                  inputs=inputs,
                                  outputs=outputs,
                                  attrs=attrs)

    new_op_dist_attr = OperatorDistributedAttribute()
    new_op_dist_attr.process_mesh = world_process_group.ranks
    new_op_dist_attr.impl_idx = 0
    if len(world_process_group.ranks) > 1:
        new_op_dist_attr.impl_type = "check_finite_and_unscale"
    for g in grads:
        g_dist_attr = dist_context.get_tensor_dist_attr_for_program(g)
        assert g_dist_attr is not None
        new_op_dist_attr.set_input_dims_mapping(g.name,
                                                g_dist_attr.dims_mapping)
        new_op_dist_attr.set_output_dims_mapping(g.name,
                                                 g_dist_attr.dims_mapping)
    dist_context.set_op_dist_attr_for_program(new_op, new_op_dist_attr)
    return grads, found_inf


@register_pass("auto_parallel_amp")
class AMPPass(PassBase):

    def __init__(self):
        super(AMPPass, self).__init__()
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
        self._loss_scaling = None
        self._num_good_steps = None
        self._num_bad_steps = None

    def _check_self(self):
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
        params_grads = self.get_attr("params_grads")

        amp_lists = AutoMixedPrecisionLists(
            set(self.get_attr("custom_white_list")),
            set(self.get_attr("custom_black_list")),
            set(self.get_attr("custom_black_varnames")))

        amp_state = AMPState(main_program.global_block())
        amp_state._build_stats(amp_lists, self.dist_context)

        with paddle.static.program_guard(main_program, startup_program):
            amp_state.cast_forward_program(self.dist_context)
            amp_state.cast_backward_program(params_grads, self.dist_context)
            # TODO (JZ-LIANG)support cast forward program only when inference
            self._init_amp_var()
            self._scale_loss()

            if self.get_attr("use_dynamic_loss_scaling"
                             ) or self.get_attr("init_loss_scaling") != 1.0:
                grads, found_inf = _check_and_update_gradient(
                    params_grads, self._loss_scaling, self.dist_context)

            if self.get_attr("use_dynamic_loss_scaling"):
                self._update_loss_scaling(grads, found_inf)

    def _init_amp_var(self):
        self._loss_scaling = paddle.static.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=self.get_attr("init_loss_scaling"),
            dtype='float32',
            persistable=True)
        set_var_dist_attr(self.dist_context, self._loss_scaling, [-1],
                          world_process_group.ranks)

        if self.get_attr("use_dynamic_loss_scaling"):
            self._num_good_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            set_var_dist_attr(self.dist_context, self._num_good_steps, [-1],
                              world_process_group.ranks)

            self._num_bad_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            set_var_dist_attr(self.dist_context, self._num_bad_steps, [-1],
                              world_process_group.ranks)

    def _scale_loss(self):

        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()

        loss = self.get_attr("loss")
        assert loss is not None
        loss_op = loss.op
        loss_op_dist_attr = self.dist_context.get_op_dist_attr_for_program(
            loss_op)

        if loss.dtype != core.VarDesc.VarType.FP32:
            # cast loss here will change the effective loss tensor for the computation graph
            # and therefore will effect all following passes whose logic is based on the loss tensor(Recompute & Gradient Merge),
            # so we it is not allowed by now. fixed it in future.
            raise NotImplementedError(
                "Loss's generator op is not support in FP16 in Auto Parallel by now, please put that op into your black-list."
            )

            tmp_name = unique_name.generate(loss.name + ".cast_fp32")
            cast_loss = main_block.create_var(name=tmp_name, dtype=dtype)
            loss_dist_attr = self.dist_context.get_tensor_dist_attr_for_program(
                loss)
            ref_mesh = loss_op_dist_attr.process_mesh
            self.dist_context.set_tensor_dist_attr_for_program(
                cast_loss, loss_dist_attr)

            loss_op_idx = find_op_index(main_block.desc, loss_op.desc)
            cast_op = main_block._insert_op(
                loss_op_idx + 1,
                type='cast',
                inputs={'X': [loss]},
                outputs={'Out': [cast_loss]},
                attrs={
                    "in_dtype": loss.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32,
                    'op_role': loss_op.all_attrs()[OP_ROLE_KEY],
                })

            loss_op._set_attr(OP_ROLE_KEY,
                              core.op_proto_and_checker_maker.OpRole.Forward)
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                cast_op, ref_mesh, [-1], self.dist_context)
            loss = loss.astype('float32')

        if self.get_attr("use_dynamic_loss_scaling"
                         ) or self.get_attr("init_loss_scaling") != 1.0:

            loss_op_idx = find_op_index(main_block.desc, loss_op.desc)

            # forward
            ref_mesh = loss_op_dist_attr.process_mesh
            self._scaled_loss = main_block.create_var(
                name=unique_name.generate("scaled_loss"),
                shape=loss.shape,
                dtype=loss.dtype,
                persistable=loss.persistable)
            set_var_dist_attr(self.dist_context, self._scaled_loss, [-1],
                              ref_mesh)

            elementwise_mul_op = main_block._insert_op(
                loss_op_idx + 1,
                type='elementwise_mul',
                inputs={
                    'X': [loss],
                    'Y': [self._loss_scaling]
                },
                outputs={'Out': [self._scaled_loss]},
                attrs={
                    'op_role': loss_op.all_attrs()[OP_ROLE_KEY],
                })
            loss_op._set_attr(OP_ROLE_KEY,
                              core.op_proto_and_checker_maker.OpRole.Forward)
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                elementwise_mul_op, ref_mesh, [-1], self.dist_context)

            # backward
            first_backward_op = main_block.ops[loss_op_idx + 2]
            assert first_backward_op.type == "fill_constant" and int(
                first_backward_op.all_attrs()[OP_ROLE_KEY]) == 257
            self._scaled_loss_grad = main_block.create_var(
                name=unique_name.generate("scaled_loss") + "@GRAD",
                shape=loss.shape,
                dtype=loss.dtype,
                persistable=loss.persistable)
            set_var_dist_attr(self.dist_context, self._scaled_loss_grad, [-1],
                              ref_mesh)
            pre_grad_name = first_backward_op.output_arg_names[0]
            first_backward_op._rename_output(pre_grad_name,
                                             self._scaled_loss_grad.name)
            # FIXME(JZ-LIANG) a trick to insert backward op
            main_block._sync_with_cpp()
            elementwise_mul_grad_op_desc = main_block.desc._insert_op(
                loss_op_idx + 3)
            elementwise_mul_grad_op_desc.set_type("elementwise_mul_grad")
            elementwise_mul_grad_op_desc.set_input(
                'Out@GRAD', [self._scaled_loss_grad.name])
            elementwise_mul_grad_op_desc.set_input('X', [loss.name])
            elementwise_mul_grad_op_desc.set_input('Y',
                                                   [self._loss_scaling.name])
            elementwise_mul_grad_op_desc.set_output('X@GRAD', [pre_grad_name])
            elementwise_mul_grad_op_desc.set_output('Y@GRAD', [])
            elementwise_mul_grad_op_desc._set_attr(
                OP_ROLE_KEY, core.op_proto_and_checker_maker.OpRole.Backward)
            elementwise_mul_grad_op_desc._set_attr('axis', -1)
            elementwise_mul_grad_op = paddle.fluid.framework.Operator(
                main_block, elementwise_mul_grad_op_desc)
            main_block.ops.insert(loss_op_idx + 3, elementwise_mul_grad_op)
            main_block._sync_with_cpp()
            elementwise_mul_grad_op = main_block.ops[loss_op_idx + 3]
            assert elementwise_mul_grad_op.type == "elementwise_mul_grad"
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                elementwise_mul_grad_op, ref_mesh, [-1], self.dist_context)

        else:
            self._scaled_loss = loss

        main_block._sync_with_cpp()

    def _update_loss_scaling(self, grads, found_inf):

        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()

        check_variable_and_dtype(self._loss_scaling, "prev_loss_scaling",
                                 ['float32', 'float64'], "update_loss_scaling")
        check_type(grads, 'x', (tuple, list), 'update_loss_scaling')
        for e in grads:
            check_variable_and_dtype(e, "x", ['float16', 'float32', 'float64'],
                                     'update_loss_scaling')
            if e.dtype == core.VarDesc.VarType.FP16:
                assert self._loss_scaling.dtype == core.VarDesc.VarType.FP32, \
                    "The dtype of prev_loss_scaling should be float32 when the dtype of x is float16."
            else:
                assert self._loss_scaling.dtype == e.dtype, "The dtype of prev_loss_scaling should be equal to the dtype of x."

        inputs = {
            'X': grads,
            'FoundInfinite': found_inf,
            'PrevLossScaling': self._loss_scaling,
            'InGoodSteps': self._num_good_steps,
            'InBadSteps': self._num_bad_steps
        }

        outputs = {
            'Out': grads,
            'LossScaling': self._loss_scaling,
            'OutGoodSteps': self._num_good_steps,
            'OutBadSteps': self._num_bad_steps
        }

        attrs = {
            'incr_every_n_steps': self.get_attr("incr_every_n_steps"),
            'decr_every_n_nan_or_inf': self.get_attr("decr_every_n_nan_or_inf"),
            'incr_ratio': self.get_attr("incr_ratio"),
            'decr_ratio': self.get_attr("decr_ratio"),
            'stop_update': self.get_attr("stop_update"),
            'op_role': OpRole.Optimize
        }

        new_op = main_block.append_op(type='update_loss_scaling',
                                      inputs=inputs,
                                      outputs=outputs,
                                      attrs=attrs)

        new_op_dist_attr = OperatorDistributedAttribute()
        new_op_dist_attr.process_mesh = world_process_group.ranks
        new_op_dist_attr.impl_idx = 0
        if len(world_process_group.ranks) > 1:
            new_op_dist_attr.impl_type = "update_loss_scaling"
        for g in grads:
            g_dist_attr = self.dist_context.get_tensor_dist_attr_for_program(g)
            assert g_dist_attr is not None
            new_op_dist_attr.set_input_dims_mapping(g.name,
                                                    g_dist_attr.dims_mapping)
            new_op_dist_attr.set_output_dims_mapping(g.name,
                                                     g_dist_attr.dims_mapping)
        self.dist_context.set_op_dist_attr_for_program(new_op, new_op_dist_attr)

        main_block._sync_with_cpp()
