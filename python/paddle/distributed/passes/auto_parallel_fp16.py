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

from collections import defaultdict

import paddle
from paddle.framework import core
from paddle.fluid.framework import default_main_program, default_startup_program
from paddle.fluid import unique_name
from .pass_base import register_pass
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.distributed.auto_parallel.utils import set_var_dist_attr, naive_set_dist_op_attr_for_program_by_mesh_and_mapping
from paddle.distributed.auto_parallel.process_group import get_world_process_group
from paddle.fluid.contrib.mixed_precision.fp16_utils import AutoMixedPrecisionLists
from paddle.fluid.contrib.mixed_precision.fp16_utils import _keep_layer_norm_scale_bias_to_fp32, _need_keep_fp32, _valid_types, _dtype_to_str
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
from paddle.distributed.auto_parallel.utils import is_forward_op, is_backward_op, OP_ROLE_KEY, OpRole
from .auto_parallel_amp import AMPPass

world_process_group = get_world_process_group()
# if user use python "+, -, * /" for network, there might be cast in vanilla program
__amp_skip_ops__ = [
    'create_py_reader',
    'create_double_buffer_reader',
    'while',
    'cast',
]


def set_op_dtype_to_fp16(op):
    if op.has_attr('in_dtype') and op.attr(
            'in_dtype') == core.VarDesc.VarType.FP32:
        op._set_attr('in_dtype', core.VarDesc.VarType.FP16)
    if op.has_attr('out_dtype') and op.attr(
            'out_dtype') == core.VarDesc.VarType.FP32:
        op._set_attr('out_dtype', core.VarDesc.VarType.FP16)
    if op.has_attr('dtype') and op.attr('dtype') == core.VarDesc.VarType.FP32:
        op._set_attr('dtype', core.VarDesc.VarType.FP16)


# adapot for backward op
def _keep_fp32_input(op, in_name):
    op_type = op.type
    if op_type == 'batch_norm':
        # Scale, Bias, Mean, Variance should be float32.
        return in_name != 'X'
    if op_type == 'layer_norm' and _keep_layer_norm_scale_bias_to_fp32():
        return in_name != 'X'
    if op_type == 'fused_bn_add_activation':
        return in_name not in {'X', 'Z'}
    if op_type == 'resnet_unit':
        return in_name not in {'X', 'FilterX', 'Z', 'FilterZ'}
    if op_type in ['fused_attention', 'fused_feedforward']:
        return in_name in {
            'LnScale', 'LnBias', 'Ln2Scale', 'Ln2Bias', "Ln1Scale", "Ln1Bias"
        }
    # backward
    if op_type in ['batch_norm_grad']:
        return in_name not in {'X', 'Y@GRAD'}
    if op_type in ['layer_norm_grad']:
        return in_name not in {'X', 'Y@GRAD'}
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
            'LnMean', 'LnVariance', 'Ln2Mean', 'Ln2Variance', 'Ln1Mean',
            'Ln1Variance'
        }
    # backward
    if op_type in ['layer_norm_grad']:
        return out_name != 'X@GRAD'
    if op_type in ['batch_norm_grad']:
        return out_name != 'X@GRAD'
    return False


class FP16State(object):

    def __init__(self,
                 program,
                 amp_list,
                 dist_context,
                 use_fp16_guard,
                 input_data_var_names=None):
        self.program = program
        self.amp_list = amp_list
        self.use_fp16_guard = use_fp16_guard
        self.dist_context = dist_context
        self.grad_op_to_op_map = self.dist_context.dist_op_context.grad_op_id_to_op_id
        if input_data_var_names:
            self.input_data_var_names = input_data_var_names
        else:
            self.input_data_var_names = []
        self._op_fp16_dict = {
        }  # op_id --> True/False. 'True' means that the op is should run in fp16 mode.
        # a trick to determine leaf tensor node in program {varname: generator_op_id}
        self.forward_non_leaf_tensors = {}
        # record the cast ops that are inserted for a forward
        self.forward_input_cast_ops = defaultdict(
            list
        )  # {forward_op_id: [(output_name, input_name, out_dtype, in_dtype, slot_name), ]}
        self.is_train = False

    def _is_fp16_op(self, op_id):
        return self._op_fp16_dict.get(op_id, None)

    def _build_state(self):
        """
        mark the execution mode (fp16 or fp32) for ops in all blocks
        include forward ops & backward ops
        """
        # mark op dtype
        # assume all backward block are behind forward blocks
        for block in self.program.blocks:
            for op in block.ops:
                self._mark_op(op)

        # set forward tensor dtype
        for block in self.program.blocks:
            self.resolute_tensor_dtype(block)

        # insert cast ops
        for block in self.program.blocks:
            self.cast_block(block)

        return self.is_train

    def _mark_op(self, op):

        if op.type in __amp_skip_ops__:
            return

        if is_forward_op(op):

            # ernie inference trick
            if op.type == "assign" and "array_" in op.input_arg_names[0]:
                self._op_fp16_dict[op.desc.original_id()] = False
                return
            if _need_keep_fp32(op, self.amp_list.unsupported_list,
                               self.use_fp16_guard):
                self._op_fp16_dict[op.desc.original_id()] = False
            else:
                self._op_fp16_dict[op.desc.original_id()] = True
            for var_name in op.output_arg_names:
                # assert var_name not in self.forward_non_leaf_tensors, "{}".format(var_name)
                self.forward_non_leaf_tensors[var_name] = op.desc.id()

        elif is_backward_op(op) == int(OpRole.Backward):

            if op.desc.original_id() in self.grad_op_to_op_map:
                fwd_op_id = self.grad_op_to_op_map[op.desc.original_id()]
                assert fwd_op_id in self._op_fp16_dict, "{}".format(str(op))
                self._op_fp16_dict[
                    op.desc.original_id()] = self._op_fp16_dict[fwd_op_id]

        if int(op.attr('op_role')) == 257:
            self.is_train = True

    def set_var_to_fp16(self, var_name, block):
        var = None
        try:
            var = block.var(var_name)
        except ValueError as e:
            var = self.program.global_block().var(var_name)

        # NOTE(JZ-LIANG) "array_" is a hack to adopt for ernie3.0 inference, since there is
        # a trick which make the LOD_TENSOR_ARRAY to the float32 in while block to reset the LOD_TENSOR_ARRAY
        if var is None or var.type not in _valid_types or "array_" in var_name:
            return

        if var.dtype == core.VarDesc.VarType.FP32:
            var.desc.set_dtype(core.VarDesc.VarType.FP16)

    def resolute_tensor_dtype(self, block):

        for op in block.ops:
            if is_forward_op(op):
                # NOTE (JZ-LIANG) un-expected cast op when user call "+, -, *, /" in python
                if self._is_fp16_op(op.desc.original_id()) == True \
                    or op.type == "cast":
                    for in_name in op.input_names:
                        if _keep_fp32_input(op, in_name):
                            continue
                        for in_var_name in op.input(in_name):
                            if in_var_name not in self.forward_non_leaf_tensors and in_var_name not in self.input_data_var_names:
                                self.set_var_to_fp16(in_var_name, block)
                    for out_name in op.output_names:
                        if _keep_fp32_output(op, out_name):
                            continue
                        for out_var_name in op.output(out_name):
                            self.set_var_to_fp16(out_var_name, block)
                    set_op_dtype_to_fp16(op)
                # NOTE (JZ-LIANG) un-expected cast op when user call "+, -, *, /" in python
                elif self._is_fp16_op(op.desc.original_id()) == False:
                    for out_var_name in op.output_arg_names:
                        out_var = block.vars.get(out_var_name)
                        if out_var is None or out_var.type not in _valid_types:
                            continue
                        if out_var.dtype == core.VarDesc.VarType.FP16:
                            out_var.desc.set_dtype(core.VarDesc.VarType.FP32)
            elif is_backward_op(op):
                if self._is_fp16_op(op.desc.original_id()) == True:
                    for out_name in op.output_names:
                        if _keep_fp32_output(op, out_name):
                            continue
                        for out_var_name in op.output(out_name):
                            self.set_var_to_fp16(out_var_name, block)
                    set_op_dtype_to_fp16(op)
                # NOTE (JZ-LIANG) un-expected cast op when user call "+, -, *, /" in python
                elif self._is_fp16_op(op.desc.original_id()) == False:
                    for out_var_name in op.output_arg_names:
                        out_var = block.vars.get(out_var_name)
                        if out_var is None or out_var.type not in _valid_types:
                            continue
                        if out_var.dtype == core.VarDesc.VarType.FP16:
                            out_var.desc.set_dtype(core.VarDesc.VarType.FP32)

    def cast_block(self, block):
        dist_op_context = self.dist_context.dist_op_context
        idx = 0
        while idx < len(block.ops):
            op = block.ops[idx]
            num_cast_ops = 0

            if op.type in __amp_skip_ops__:
                idx += 1
                continue
            elif is_forward_op(op):
                if self._is_fp16_op(op.desc.original_id()) == False:
                    num_cast_ops = self._insert_forward_cast_ops(
                        op, idx, block, core.VarDesc.VarType.FP16,
                        core.VarDesc.VarType.FP32, self.dist_context)
                elif self._is_fp16_op(op.desc.original_id()) == True:
                    num_cast_ops = self._insert_forward_cast_ops(
                        op, idx, block, core.VarDesc.VarType.FP32,
                        core.VarDesc.VarType.FP16, self.dist_context)
            elif is_backward_op(op):
                if op.desc.original_id() in dist_op_context.grad_op_id_to_op_id:
                    if self._is_fp16_op(op.desc.original_id()) == False:
                        num_cast_ops = self._insert_backward_cast_ops(
                            op, idx, block, core.VarDesc.VarType.FP16,
                            core.VarDesc.VarType.FP32, self.dist_context)
                    elif self._is_fp16_op(op.desc.original_id()) == True:
                        num_cast_ops = self._insert_backward_cast_ops(
                            op, idx, block, core.VarDesc.VarType.FP32,
                            core.VarDesc.VarType.FP16, self.dist_context)
                elif op.type == "sum":
                    # all inputs dtype of sum should be equal and output dtype should follow input
                    out_var_name = op.output_arg_names[0]
                    in_var_name = op.input_arg_names[0]
                    out_var = block.var(out_var_name)
                    in_var = block._find_var_recursive(in_var_name)
                    for in_var_name in op.input_arg_names:
                        assert in_var.dtype == block.var(
                            in_var_name).dtype, "{}, {}, {}".format(
                                in_var, block.var(in_var_name), str(op))
                    out_var.desc.set_dtype(in_var.dtype)

            idx += num_cast_ops + 1
        block._sync_with_cpp()

    def _insert_forward_cast_ops(self, op, idx, block, src_dtype, dst_dtype,
                                 dist_context):

        num_cast_ops = 0

        for in_name in op.input_names:
            if src_dtype == core.VarDesc.VarType.FP32 and _keep_fp32_input(
                    op, in_name):
                continue

            consume_op_attr = dist_context.get_op_dist_attr_for_program(op)
            assert consume_op_attr is not None
            for in_var_name in op.input(in_name):
                in_var = block._find_var_recursive(in_var_name)
                if in_var is None or in_var.type not in _valid_types or in_var.dtype == dst_dtype:
                    continue

                if in_var.dtype == src_dtype:
                    cast_name = in_var.name + '.cast_' + _dtype_to_str(
                        dst_dtype)
                    cast_var = block.vars.get(cast_name)
                    self.forward_input_cast_ops[op.desc.original_id()] += [
                        (cast_name, in_var.name, dst_dtype, src_dtype, in_name)
                    ]

                    in_var_dist_attr = consume_op_attr.get_input_dist_attr(
                        in_var.name)
                    assert in_var_dist_attr is not None
                    # truly insert cast op
                    if cast_var is None or cast_var.dtype != dst_dtype:
                        # NOTE we make the cast op and var's dist attr as the op that consume the
                        # cast var instead of the op which generates the var
                        # refine op's dist_attr
                        ref_mesh = in_var_dist_attr.process_mesh
                        ref_mapping = in_var_dist_attr.dims_mapping

                        cast_var = block.create_var(
                            name=cast_name,
                            dtype=dst_dtype,
                            persistable=False,
                            stop_gradient=in_var.stop_gradient)
                        set_var_dist_attr(dist_context, cast_var, ref_mapping,
                                          ref_mesh)

                        cast_op = block._insert_op_without_sync(
                            idx,
                            type="cast",
                            inputs={"X": in_var},
                            outputs={"Out": cast_var},
                            attrs={
                                "in_dtype": in_var.dtype,
                                "out_dtype": cast_var.dtype,
                                OP_ROLE_KEY: OpRole.Forward
                            })
                        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                            cast_op, ref_mesh, ref_mapping, dist_context)
                        num_cast_ops += 1

                    op._rename_input(in_var.name, cast_name)
                    consume_op_attr.set_input_dist_attr(cast_name,
                                                        in_var_dist_attr)

        if op.has_attr('out_dtype') and op.attr('out_dtype') != -1:
            assert op.attr('out_dtype') == dst_dtype

        return num_cast_ops

    def _insert_backward_cast_ops(self, op, idx, block, src_dtype, dst_dtype,
                                  dist_context):

        num_cast_ops = 0
        op_id = op.desc.id()
        original_id = op.desc.original_id()
        dist_op_context = dist_context.dist_op_context
        forward_op_id = dist_op_context.grad_op_id_to_op_id[original_id]

        grad_op_attr = dist_context.get_op_dist_attr_for_program(op)
        assert grad_op_attr is not None

        for out_var_name in op.output_arg_names:
            out_var = block.var(out_var_name)
            if _keep_fp32_output(op, out_var.name):
                continue
            assert out_var.dtype == dst_dtype, "{}, {}".format(
                str(out_var), dst_dtype)

        for cast_name, src_name, dst_dtype, src_dtype, slot_name in self.forward_input_cast_ops[
                forward_op_id]:

            # some forward output is not need by backward computation, e.g. logit in softmax_with_cross_entropy
            if slot_name not in op.input_names:
                continue

            # rename input
            assert src_name in op.input(
                slot_name), "var: {} not in op's {}. {}".format(
                    src_name, slot_name, str(op))
            src_var_dist_attr = grad_op_attr.get_input_dist_attr(src_name)
            assert src_var_dist_attr is not None
            op._rename_input(src_name, cast_name)
            grad_op_attr.set_input_dist_attr(cast_name, src_var_dist_attr)

            # create cast grad
            grad_slot_name = slot_name + "@GRAD"
            assert grad_slot_name in op.output_names, "[{}], Current Op: {}".format(
                grad_slot_name, str(op))

            # some forward input maybe stop_gradient=True, e.g. input_mask
            if len(op.output(grad_slot_name)) == 0:
                continue
            assert len(
                op.output(grad_slot_name)) == 1, "[{}], Current Op: {}".format(
                    grad_slot_name, str(op))
            grad_name = op.output(grad_slot_name)[0]
            grad = block.var(grad_name)
            grad_dist_attr = grad_op_attr.get_output_dist_attr(grad_name)
            assert grad_dist_attr is not None, "{}".format(grad_name)
            ref_mesh = grad_dist_attr.process_mesh
            ref_mapping = grad_dist_attr.dims_mapping

            cast_grad = block.create_var(
                name=unique_name.generate_with_ignorable_key("".join(
                    [cast_name, '@GRAD'])),
                dtype=dst_dtype,
                shape=grad.shape,
                type=grad.type,
                persistable=grad.persistable,
                stop_gradient=grad.stop_gradient)
            dist_context.set_tensor_dist_attr_for_program(
                cast_grad, grad_dist_attr)
            op._rename_output(grad_name, cast_grad.name)
            grad_op_attr.set_output_dist_attr(cast_grad.name, grad_dist_attr)

            # add cast
            cast_op = block._insert_op_without_sync(
                idx + 1,
                type="cast",
                inputs={"X": [cast_grad.name]},
                outputs={"Out": [grad.name]},
                attrs={
                    "in_dtype": dst_dtype,
                    "out_dtype": src_dtype,
                    OP_ROLE_KEY: OpRole.Backward
                })
            grad.desc.set_dtype(src_dtype)

            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                cast_op, ref_mesh, ref_mapping, dist_context)
            num_cast_ops += 1

        return num_cast_ops


def _check_and_update_gradient(grads, loss_scaling, name, dist_context):

    main_block = paddle.static.default_main_program().global_block()
    main_block._sync_with_cpp()

    check_type(grads, 'x', (tuple, list), 'check_finite_and_unscale')
    for e in grads:
        check_variable_and_dtype(e, "x", ['float16', 'float32', 'float64'],
                                 'check_finite_and_unscale')

    found_inf = main_block.create_var(
        name=unique_name.generate_with_ignorable_key(".".join(
            ['find_infinite_scale', name])),
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


def _split_grads(params_grads):
    grads = [g for _, g in params_grads]
    fp32_grads = [g for g in grads if g.dtype == core.VarDesc.VarType.FP32]
    fp16_grads = [g for g in grads if g.dtype == core.VarDesc.VarType.FP16]
    assert len(fp32_grads) + len(fp16_grads) == len(grads), \
        "Data types of all grads must be either fp16 or fp32."
    return grads, fp32_grads, fp16_grads


def _set_op_dist_attr_with_ranks(new_op, ranks, block, dist_context):
    new_op_dist_attr = OperatorDistributedAttribute()
    new_op_dist_attr.process_mesh = ranks
    new_op_dist_attr.impl_idx = 0
    for var_name in new_op.input_arg_names:
        var = block.var(var_name)
        var_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
        assert var_dist_attr is not None
        new_op_dist_attr.set_input_dims_mapping(var_name,
                                                var_dist_attr.dims_mapping)
    for var_name in new_op.output_arg_names:
        var = block.var(var_name)
        var_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
        assert var_dist_attr is not None
        new_op_dist_attr.set_output_dims_mapping(var_name,
                                                 var_dist_attr.dims_mapping)
    dist_context.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def _get_memcopy_idx(block, found_inf_var):
    # use reduce_any op for check_nan_inf as the anchor for now
    for idx, op in enumerate(block.ops):
        if op.type == 'reduce_any' and op.output_arg_names[
                0] == found_inf_var.name:
            return idx + 1

    raise RuntimeError(
        "not found the correct location for memcopy for found_inf_var.")


def _insert_memcopy(block, idx, src_var, dist_context, direction="D2H"):
    src_name = src_var.name
    output_var = block.create_var(name=unique_name.generate_with_ignorable_key(
        src_name.join(['memcopy_'])),
                                  dtype=src_var.dtype,
                                  shape=src_var.shape,
                                  type=core.VarDesc.VarType.LOD_TENSOR,
                                  persistable=False,
                                  stop_gradient=src_var.stop_gradient)

    set_var_dist_attr(dist_context, output_var, [-1], world_process_group.ranks)

    # TODO to support CUDAPinned/NPU/XPU Places
    if direction == "D2H":
        dst_place_type = 0
    elif direction == "D2H":
        dst_place_type = 1
    else:
        raise NotImplementedError(
            "direction [{}] is not supported yet.".format(direction))

    attrs = {'dst_place_type': dst_place_type}
    new_op = block._insert_op_without_sync(index=idx,
                                           type='memcpy',
                                           inputs={'X': [src_var]},
                                           outputs={'Out': [output_var]},
                                           attrs=attrs)
    _set_op_dist_attr_with_ranks(new_op, world_process_group.ranks, block,
                                 dist_context)
    block._sync_with_cpp()
    return output_var


def cast_startup_program():
    main_program = default_main_program()
    startup_program = default_startup_program()

    param_to_dtype = {}
    for block in main_program.blocks:
        for p in block.all_parameters():
            param_to_dtype[p.name] = p.dtype

    def is_initialization_op(op):
        comm_op_prefix = "c_"
        op_type = op.type
        if op_type.startswith(comm_op_prefix):
            return False

        if len(op.output_arg_names) != 1 and len(op.input_arg_names) != 0:
            return False

        return True

    for op in startup_program.global_block().ops:
        if is_initialization_op(op):
            output_name = op.output_arg_names[0]
            if param_to_dtype.get(output_name,
                                  None) == core.VarDesc.VarType.FP16:
                assert op.has_attr(
                    'dtype'
                ), "initialization op is supported to has dtype attribute but got {}.".format(
                    str(op))
                if op.attr('dtype') == core.VarDesc.VarType.FP32:
                    op._set_attr('dtype', core.VarDesc.VarType.FP16)


@register_pass("auto_parallel_fp16")
class FP16Pass(AMPPass):

    def __init__(self):
        super(FP16Pass, self).__init__()

    # NOTE: why FP16Pass can override apply_single_impl instead of
    # apply_impl? AMP is an optimization pass for serial program,
    # in distributed scenario, all ranks should have the same modification.
    def _apply_single_impl(self, main_program, startup_program, context):
        self.dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")

        amp_list = AutoMixedPrecisionLists(
            set(self.get_attr("custom_white_list")),
            set(self.get_attr("custom_black_list")), None)

        # NOTE don't not change input data dtype, since it is controled by dataloader
        # and which is out of control of FP16 Pass
        input_data_var_names = [var.name for var in self.get_attr("input_data")]

        with paddle.static.program_guard(main_program, startup_program):
            fp16_state = FP16State(main_program, amp_list, self.dist_context,
                                   self.get_attr("use_fp16_guard"),
                                   input_data_var_names)
            is_train = fp16_state._build_state()

            cast_startup_program()

        if is_train:
            with paddle.static.program_guard(main_program, startup_program):
                # TODO (JZ-LIANG)support cast forward program only when inference
                self._init_amp_var()
                self._scale_loss()

                grads, fp32_grads, fp16_grads = _split_grads(params_grads)

                if self.get_attr("use_dynamic_loss_scaling"
                                 ) or self.get_attr("init_loss_scaling") != 1.0:
                    found_infs = []
                    if fp32_grads:
                        with main_program._optimized_guard([]):
                            _, found_inf_fp32 = _check_and_update_gradient(
                                fp32_grads, self._loss_scaling, "@fp32",
                                self.dist_context)
                        found_infs.append(found_inf_fp32)
                    if fp16_grads:
                        with main_program._optimized_guard([]):
                            _, found_inf_fp16 = _check_and_update_gradient(
                                fp16_grads, self._loss_scaling, "@fp16",
                                self.dist_context)
                        found_infs.append(found_inf_fp16)
                    with main_program._optimized_guard([]):
                        block = main_program.global_block()

                        all_infs = paddle.fluid.layers.concat(found_infs)
                        set_var_dist_attr(self.dist_context, all_infs, [-1],
                                          world_process_group.ranks)
                        new_op = block.ops[-1]
                        assert new_op.type == "concat"
                        _set_op_dist_attr_with_ranks(new_op,
                                                     world_process_group.ranks,
                                                     block, self.dist_context)

                        found_inf = paddle.fluid.layers.reduce_any(all_infs)
                        set_var_dist_attr(self.dist_context, found_inf, [-1],
                                          world_process_group.ranks)
                        new_op = block.ops[-1]
                        assert new_op.type == "reduce_any"
                        _set_op_dist_attr_with_ranks(new_op,
                                                     world_process_group.ranks,
                                                     block, self.dist_context)

                if self.get_attr("use_dynamic_loss_scaling"):
                    with main_program._optimized_guard([]):
                        if fp32_grads:
                            self._update_loss_scaling(fp32_grads, found_inf)
                        if fp16_grads:
                            self._update_loss_scaling(fp16_grads, found_inf)

            # modify optimizer
            base_opt = self.get_attr("base_opt")
            base_opt._multi_precision = True
            if self.get_attr("use_optimizer_fp16"):
                base_opt._multi_precision = False
            if isinstance(
                    base_opt,
                (paddle.fluid.optimizer.Adam, paddle.optimizer.AdamW)):
                with main_program._optimized_guard([]):
                    # found_inf = paddle.tensor.creation._memcpy(
                    #     found_inf, paddle.CPUPlace())
                    insert_idx = _get_memcopy_idx(block, found_inf)
                    found_inf = _insert_memcopy(block, insert_idx, found_inf,
                                                self.dist_context)
                base_opt._set_auxiliary_var('found_inf', found_inf.name)
            elif hasattr(base_opt, "_set_auxiliary_var"):
                base_opt._set_auxiliary_var('found_inf', found_inf.name)
