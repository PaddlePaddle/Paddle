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
from paddle.fluid.contrib.mixed_precision.fp16_utils import AutoMixedPrecisionLists, _keep_fp32_input, _keep_fp32_output, _valid_types, find_true_post_op, find_op_index, _is_in_black_varnames, _dtype_to_str, _rename_arg
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
from paddle.distributed.auto_parallel.utils import get_loss_op, set_var_dist_attr
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type

from collections import OrderedDict
import numpy as np

BACKWARD = core.op_proto_and_checker_maker.OpRole.Backward
# FIXME
global_process_mesh = [0, 1]


def _mark_black_white_ops(main_prog, amp_lists):
    """
    this function is modified from paddle.fluid.contrib.mixed_precision

    """
    block = main_prog.global_block()
    block._sync_with_cpp()
    ops = block.ops
    white_op_set = set()
    black_op_set = set()
    # TODO just parse forward ops
    for op in ops:

        if op.type == 'create_py_reader' or op.type == 'read':
            continue

        if amp_lists.black_varnames is not None and _is_in_black_varnames(
                op, amp_lists):
            black_op_set.add(op)
            continue

        if op.type in amp_lists.black_list:
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
    return white_op_set, black_op_set


def _insert_cast_ops(main_program, black_op_set, white_op_set, dist_context):

    block = main_program.global_block()
    ops = block.ops

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        num_cast_ops = 0
        if op in black_op_set:
            num_cast_ops = _insert_cast_op(
                block, op, idx, core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.FP32, dist_context)
        elif op in white_op_set:
            num_cast_ops = _insert_cast_op(
                block, op, idx, core.VarDesc.VarType.FP32,
                core.VarDesc.VarType.FP16, dist_context)
        else:
            pass

        idx += num_cast_ops + 1


def _insert_cast_op(block, op, idx, src_dtype, dest_dtype, dist_context):
    """
    modified from paddle.fluid.contrib.mixed_precision
    """
    num_cast_ops = 0

    # TODO only for forward cast
    for in_name in op.input_names:
        if src_dtype == core.VarDesc.VarType.FP32 and _keep_fp32_input(op,
                                                                       in_name):
            continue
        for in_var_name in op.input(in_name):
            in_var = block._find_var_recursive(in_var_name)
            if in_var.type not in _valid_types or in_var.dtype == dest_dtype:
                continue
            if in_var.dtype == src_dtype:
                cast_name = in_var.name + '.cast_' + _dtype_to_str(dest_dtype)
                out_var = block.vars.get(cast_name)
                consume_op_attr = dist_context.get_op_dist_attr_for_program(op)
                assert consume_op_attr is not None
                if out_var is None or out_var.dtype != dest_dtype:

                    # NOTE we make the cast op and var's dist attr as the op that consume the
                    # cast var instead of the op which generates the var
                    ref_mesh = consume_op_attr.process_mesh
                    ref_mapping = consume_op_attr.get_input_dims_mapping(
                        in_var.name)
                    assert ref_mapping is not None

                    out_var = block.create_var(
                        name=cast_name,
                        dtype=dest_dtype,
                        persistable=False,
                        stop_gradient=in_var.stop_gradient)
                    out_var_dist_attr = set_var_dist_attr(dist_context, out_var,
                                                          ref_mapping, ref_mesh)

                    cast_op = block._insert_op_without_sync(
                        idx,
                        type="cast",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={
                            "in_dtype": in_var.dtype,
                            "out_dtype": out_var.dtype,
                        })
                    cast_op_dist_attr = OperatorDistributedAttribute()
                    cast_op_dist_attr.process_mesh = ref_mesh
                    cast_op_dist_attr.set_input_dims_mapping(in_var.name,
                                                             ref_mapping)
                    cast_op_dist_attr.set_output_dims_mapping(out_var.name,
                                                              ref_mapping)
                    dist_context.set_op_dist_attr_for_program(cast_op,
                                                              cast_op_dist_attr)

                    num_cast_ops += 1
                else:
                    out_var_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        out_var)
                _rename_arg(op, in_var.name, out_var.name)
                consume_op_attr.set_input_dist_attr(out_var.name,
                                                    out_var_dist_attr)
            else:
                if op.has_attr('in_dtype'):
                    op._set_attr('in_dtype', dest_dtype)

    if src_dtype == core.VarDesc.VarType.FP32 and dest_dtype == core.VarDesc.VarType.FP16:
        for out_name in op.output_names:
            if _keep_fp32_output(op, out_name):
                continue
            for out_var_name in op.output(out_name):
                out_var = block.var(out_var_name)
                if out_var.type not in _valid_types:
                    continue
                if out_var.dtype == core.VarDesc.VarType.FP32:
                    out_var.desc.set_dtype(core.VarDesc.VarType.FP16)
                    if op.has_attr('out_dtype'):
                        op._set_attr('out_dtype', core.VarDesc.VarType.FP16)
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
            role = op.attr('op_role')
            if role & int(BACKWARD) and op.has_attr('op_role_var'):
                op._remove_attr("op_role_var")
            else:
                raise ValueError("The cast op {0} must be in BACKWARD role "
                                 "and have op_role_var attr.".format(op))

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
            new_op = paddle.fluid.framework.Operator(
                block=main_block,
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
            ref_process_mesh = param_dist_attr.process_mesh
            ref_mapping = param_dist_attr.dims_mapping
            new_op_dist_attr = OperatorDistributedAttribute()
            new_op_dist_attr.process_mesh = ref_process_mesh
            new_op_dist_attr.set_input_dims_mapping(new_op.input_arg_names[0],
                                                    ref_mapping)
            new_op_dist_attr.set_output_dims_mapping(new_op.output_arg_names[0],
                                                     ref_mapping)
            dist_context.set_op_dist_attr_for_program(new_op, new_op_dist_attr)

            output_dist_attr.process_mesh = ref_process_mesh
            output_dist_attr.dims_mapping = ref_mapping

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
    set_var_dist_attr(dist_context, found_inf, [-1], global_process_mesh)

    inputs = {'X': grads, 'Scale': loss_scaling}
    outputs = {'Out': grads, 'FoundInfinite': found_inf}
    attrs = {'op_role': BACKWARD}
    new_op = main_block.append_op(
        type='check_finite_and_unscale',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)

    new_op_dist_attr = OperatorDistributedAttribute()
    new_op_dist_attr.process_mesh = global_process_mesh
    new_op_dist_attr.impl_idx = 0
    for g in grads:
        g_dist_attr = dist_context.get_tensor_dist_attr_for_program(g)
        assert g_dist_attr is not None
        new_op_dist_attr.set_input_dims_mapping(g.name,
                                                g_dist_attr.dims_mapping)
        new_op_dist_attr.set_output_dims_mapping(g.name,
                                                 g_dist_attr.dims_mapping)
    dist_context.set_op_dist_attr_for_program(new_op, new_op_dist_attr)
    return grads, found_inf


# TODO (JZ-LIANG) merge forward & backward pass 
# NOTE we add the "auto_parallel" prefix to the pass in order to 
# indicate that this pass should obey some constrains by auto_parallel
# for example all ops and vars should has dist attr before and after pass
# should use dist op instead of custom comm op 
@register_pass("auto_parallel_amp_forward")
class AMPForwardPass(PassBase):
    def __init__(self):
        super(AMPForwardPass, self).__init__()
        self.set_attr("dist_context", None)
        self.set_attr("custom_white_list", None)
        self.set_attr("custom_white_list", None)
        self.set_attr("custom_black_varnames", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    # NOTE: why AMPForwardPass can override apply_single_impl instead of 
    # apply_impl? AMP is an optimization pass for serial program, 
    # in distributed scenario, all ranks should have the same modification.
    def _apply_single_impl(self, main_program, startup_program, context):

        self._dist_context = self.get_attr("dist_context")
        amp_lists = AutoMixedPrecisionLists(
            set(self.get_attr("custom_white_list")),
            set(self.get_attr("custom_black_list")),
            set(self.get_attr("custom_black_varnames")))

        white_op_set, black_op_set = _mark_black_white_ops(main_program,
                                                           amp_lists)

        _insert_cast_ops(main_program, black_op_set, white_op_set,
                         self._dist_context)

        main_program.global_block()._sync_with_cpp()


@register_pass("auto_parallel_amp_backward")
class AMPBackwardPass(PassBase):
    def __init__(self):
        super(AMPBackwardPass, self).__init__()
        self.set_attr(
            "init_loss_scaling",
            32768.0, )
        self.set_attr("incr_every_n_steps", 1000)
        self.set_attr("decr_every_n_nan_or_inf", 2)
        self.set_attr("incr_ratio", 2.0)
        self.set_attr("decr_ratio", 0.8)
        self.set_attr("use_dynamic_loss_scaling", False)
        self.set_attr("params_grads", [])
        self.set_attr("dist_context", None)
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
        if len(self.get_attr("params_grads")) <= 0:
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

        self._dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")

        with paddle.static.program_guard(main_program, startup_program):
            self._init_amp_var()
            self._scale_loss()
            _update_backward_cast_ops(params_grads, self._dist_context)

            if self.get_attr("use_dynamic_loss_scaling") or self.get_attr(
                    "init_loss_scaling") != 1.0:
                grads, found_inf = _check_and_update_gradient(
                    params_grads, self._loss_scaling, self._dist_context)

            if self.get_attr("use_dynamic_loss_scaling"):
                self._update_loss_scaling(grads, found_inf)

        main_program.global_block()._sync_with_cpp()

    def _init_amp_var(self):
        self._loss_scaling = paddle.static.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=self.get_attr("init_loss_scaling"),
            dtype='float32',
            persistable=True)
        set_var_dist_attr(self._dist_context, self._loss_scaling, [-1],
                          global_process_mesh)

        if self.get_attr("use_dynamic_loss_scaling"):
            self._num_good_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            set_var_dist_attr(self._dist_context, self._num_good_steps, [-1],
                              global_process_mesh)

            self._num_bad_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            set_var_dist_attr(self._dist_context, self._num_bad_steps, [-1],
                              global_process_mesh)

    def _scale_loss(self):

        main_block = paddle.static.default_main_program().global_block()
        main_block._sync_with_cpp()
        loss_op = get_loss_op(main_block)
        loss = main_block.var(loss_op.output_arg_names[0])

        if loss.dtype != core.VarDesc.VarType.FP32:
            loss = loss.astype('float32')

        if self.get_attr("use_dynamic_loss_scaling") or self.get_attr(
                "init_loss_scaling") != 1.0:

            loss_op_idx = find_op_index(main_block.desc, loss_op.desc)

            # forward
            self._scaled_loss = main_block.create_var(
                name=unique_name.generate("scaled_loss"),
                shape=loss.shape,
                dtype=loss.dtype,
                persistable=loss.persistable)
            set_var_dist_attr(self._dist_context, self._scaled_loss, [-1],
                              global_process_mesh)

            OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
            elementwise_mul_op = main_block._insert_op(
                loss_op_idx + 1,
                type='elementwise_mul',
                inputs={'X': [loss],
                        'Y': [self._loss_scaling]},
                outputs={'Out': [self._scaled_loss]},
                attrs={'op_role': loss_op.all_attrs()[OP_ROLE_KEY], })
            loss_op._set_attr(OP_ROLE_KEY,
                              core.op_proto_and_checker_maker.OpRole.Forward)

            elementwise_mul_op_dist_attr = OperatorDistributedAttribute()
            elementwise_mul_op_dist_attr.process_mesh = global_process_mesh
            elementwise_mul_op_dist_attr.set_input_dims_mapping(loss.name, [-1])
            elementwise_mul_op_dist_attr.set_input_dims_mapping(
                self._loss_scaling.name, [-1])
            elementwise_mul_op_dist_attr.set_output_dims_mapping(
                self._scaled_loss.name, [-1])
            self._dist_context.set_op_dist_attr_for_program(
                elementwise_mul_op, elementwise_mul_op_dist_attr)

            # backward
            first_backward_op = main_block.ops[loss_op_idx + 2]
            assert first_backward_op.type == "fill_constant" and int(
                first_backward_op.all_attrs()[OP_ROLE_KEY]) == 257
            self._scaled_loss_grad = main_block.create_var(
                name=unique_name.generate("scaled_loss") + "@GRAD",
                shape=loss.shape,
                dtype=loss.dtype,
                persistable=loss.persistable)
            set_var_dist_attr(self._dist_context, self._scaled_loss_grad, [-1],
                              global_process_mesh)
            pre_grad_name = first_backward_op.output_arg_names[0]
            first_backward_op._rename_output(pre_grad_name,
                                             self._scaled_loss_grad.name)

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
            elementwise_mul_grad_op_dist_attr = OperatorDistributedAttribute()
            elementwise_mul_grad_op_dist_attr.process_mesh = global_process_mesh
            elementwise_mul_grad_op_dist_attr.set_input_dist_attr(
                loss.name,
                self._dist_context.get_tensor_dist_attr_for_program(loss))
            elementwise_mul_grad_op_dist_attr.set_input_dist_attr(
                self._loss_scaling.name,
                self._dist_context.get_tensor_dist_attr_for_program(
                    self._loss_scaling))
            elementwise_mul_grad_op_dist_attr.set_input_dist_attr(
                self._scaled_loss_grad.name,
                self._dist_context.get_tensor_dist_attr_for_program(
                    self._scaled_loss_grad))
            elementwise_mul_grad_op_dist_attr.set_output_dist_attr(
                pre_grad_name,
                self._dist_context.get_tensor_dist_attr_for_program(
                    main_block.var(pre_grad_name)))
            self._dist_context.set_op_dist_attr_for_program(
                elementwise_mul_grad_op, elementwise_mul_grad_op_dist_attr)

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
            'op_role': BACKWARD
        }

        new_op = main_block.append_op(
            type='update_loss_scaling',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        new_op_dist_attr = OperatorDistributedAttribute()
        new_op_dist_attr.process_mesh = global_process_mesh
        for g in grads:
            g_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(g)
            assert g_dist_attr is not None
            new_op_dist_attr.set_input_dims_mapping(g.name,
                                                    g_dist_attr.dims_mapping)
            new_op_dist_attr.set_output_dims_mapping(g.name,
                                                     g_dist_attr.dims_mapping)
        self._dist_context.set_op_dist_attr_for_program(new_op,
                                                        new_op_dist_attr)

        main_block._sync_with_cpp()
