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
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.static.operators.common import (
    is_data_parallel_reduce_op,
    is_data_parallel_scale_op,
)
from paddle.distributed.auto_parallel.static.process_group import (
    get_world_process_group,
)
from paddle.distributed.auto_parallel.static.utils import (
    is_backward_op,
    is_optimize_op,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_var_dist_attr,
)
from paddle.distributed.fleet.meta_optimizers.common import (
    OP_ROLE_KEY,
    OP_ROLE_VAR_KEY,
    OpRole,
)
from paddle.framework import core

from .pass_base import PassBase, PassType, register_pass

world_process_group = get_world_process_group()


def is_gradient_clip_op(op_desc):
    return op_desc.has_attr("op_namescope") and op_desc.attr(
        "op_namescope"
    ).startswith("/gradient_clip")


def _remove_and_get_ops(main_program, dist_context):
    # 1 create tmp block
    # 2 mv optimizer op from global program to tmp block
    # 3 del the op from dist_context
    main_block = main_program.global_block()
    temp_block = main_program._create_block()
    removed_op_idx = []
    optimize_ops_desc = []
    allreduce_sum_desc = []
    for idx, op in enumerate(main_block.ops):
        if is_optimize_op(op):
            # append optimizer op to tmp block
            new_op_desc = temp_block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            optimize_ops_desc.append(new_op_desc)
            removed_op_idx.append(idx)
            dist_context.del_dist_op_for_program(op)

        # append allreduce_op and scale_op to tmp block
        if is_backward_op(op):
            if is_data_parallel_reduce_op(op) or is_data_parallel_scale_op(op):
                assert len(op.desc.output_arg_names()) == 1
                new_op_desc = temp_block.desc.append_op()
                new_op_desc.copy_from(op.desc)
                allreduce_sum_desc.append(new_op_desc)
                removed_op_idx.append(idx)
                dist_context.del_dist_op_for_program(op)

    for idx in removed_op_idx[::-1]:
        main_block._remove_op(idx, sync=False)
    main_block._sync_with_cpp()

    return optimize_ops_desc, allreduce_sum_desc


def _get_gm_cond_var(main_program, k_steps, dist_context):
    main_block = main_program.global_block()
    # Add const var
    k_step_var = paddle.static.create_global_var(
        name="gradient_merge_k",
        shape=[1],
        value=int(k_steps),
        dtype='int32',
        persistable=True,
        force_cpu=True,
    )
    set_var_dist_attr(dist_context, k_step_var, [-1], world_process_group.ranks)

    zero_var = paddle.static.create_global_var(
        name="gradient_merge_zero",
        shape=[1],
        value=int(0),
        dtype='int32',
        persistable=True,
        force_cpu=True,
    )
    set_var_dist_attr(dist_context, zero_var, [-1], world_process_group.ranks)

    # Add step var & cond var
    step_var = paddle.static.create_global_var(
        name="gradient_merge_step",
        shape=[1],
        value=int(0),
        dtype='int32',
        persistable=True,
        force_cpu=True,
    )
    set_var_dist_attr(dist_context, step_var, [-1], world_process_group.ranks)

    cond_var = main_block.create_var(
        name="gradient_merge_cond", shape=[1], dtype='bool'
    )
    set_var_dist_attr(dist_context, cond_var, [-1], world_process_group.ranks)

    with paddle.static.device_guard("cpu"):
        # step_var += 1
        increment_op = main_block.append_op(
            type='increment',
            inputs={'X': [step_var]},
            outputs={'Out': [step_var]},
            attrs={'step': float(1.0), OP_ROLE_KEY: OpRole.Backward},
        )
        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
            increment_op,
            ProcessMesh(world_process_group.ranks),
            [-1],
            dist_context,
        )
        # step_var %= k_step
        elementwise_mod_op = main_block.append_op(
            type='elementwise_mod',
            inputs={'X': step_var, 'Y': k_step_var},
            outputs={'Out': step_var},
            attrs={
                'axis': -1,
                'use_mkldnn': False,
                OP_ROLE_KEY: OpRole.Backward,
            },
        )
        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
            elementwise_mod_op,
            ProcessMesh(world_process_group.ranks),
            [-1],
            dist_context,
        )
        # cond_var = (step_var == 0)
        equal_op = main_block.append_op(
            type='equal',
            inputs={'X': step_var, 'Y': zero_var},
            outputs={'Out': cond_var},
            attrs={OP_ROLE_KEY: OpRole.Backward},
        )
        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
            equal_op, ProcessMesh(world_process_group.ranks), [-1], dist_context
        )

    return cond_var


def _append_gradient_merge_backward_op(
    main_program,
    startup_program,
    params_grads,
    master_grad,
    dist_context,
):
    main_block = main_program.global_block()
    startup_block = startup_program.global_block()

    # step1: remove grad.op's op_role_var
    for param, grad in params_grads:
        assert (
            param.type != core.VarDesc.VarType.SELECTED_ROWS
        ), "SELECTED_ROWS is not supported in GradientMergeOptimizer for now"

    # {grad.name: gradient_merge_var.name} to rename opt inputs
    grad_to_gradient_merge = {}
    # {param: gradient_merge_var} to insert scale op and fill_constant op
    new_params_to_grads = []
    # step2: create gradient_merge var and init with 0
    for param, grad in params_grads:
        param_name = param.name
        param_var = main_block.var(param_name)
        assert param_var is not None

        dst_dtype = (
            core.VarDesc.VarType.FP32 if master_grad else param_var.dtype
        )

        # 2.1 crate param@GRAD@MERGE var in startup_block
        startup_gradient_merge_var = startup_block.create_var(
            name=param_name + "@GRAD@MERGED",
            shape=param_var.shape,
            dtype=dst_dtype,
            persistable=True,
        )
        startup_block.append_op(
            type="fill_constant",
            outputs={"Out": startup_gradient_merge_var},
            attrs={
                "shape": param_var.shape,
                "dtype": dst_dtype,
                "value": float(0),
            },
        )

        # 2.2 crate param@GRAD@MERGE var in main_block
        ref_dist_attr = dist_context.get_tensor_dist_attr_for_program(param_var)
        assert ref_dist_attr is not None
        gradient_merge_var = main_block.create_var(
            name=param_name + "@GRAD@MERGED",
            shape=param_var.shape,
            dtype=dst_dtype,
            persistable=True,
        )
        ref_process_mesh = ref_dist_attr.process_mesh
        ref_dims_mapping = ref_dist_attr.dims_mapping

        set_var_dist_attr(
            dist_context, gradient_merge_var, ref_dims_mapping, ref_process_mesh
        )

        # 2.3 grad_merge += grad
        grad_name = grad.name
        if grad.dtype != dst_dtype:
            cast_grad_name = grad_name + "@TMP"
            cast_grad_var = main_block.create_var(
                name=cast_grad_name,
                shape=grad.shape,
                dtype=dst_dtype,
                persistable=False,
                stop_gradient=grad.stop_gradient,
            )
            set_var_dist_attr(
                dist_context, cast_grad_var, ref_dims_mapping, ref_process_mesh
            )
            cast_op = main_block.append_op(
                type="cast",
                inputs={"X": grad},
                outputs={"Out": cast_grad_var},
                attrs={
                    "in_dtype": grad.dtype,
                    "out_dtype": dst_dtype,
                    OP_ROLE_KEY: OpRole.Backward,
                },
            )
            naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                cast_op, ref_process_mesh, ref_dims_mapping, dist_context
            )
            grad = cast_grad_var
        new_grad_op = main_block.append_op(
            type="elementwise_add",
            inputs={'X': grad, 'Y': gradient_merge_var},
            outputs={'Out': gradient_merge_var},
            attrs={
                'axis': -1,
                'use_mkldnn': False,
                OP_ROLE_KEY: OpRole.Backward,
            },
        )
        new_params_to_grads.append([param, gradient_merge_var])
        grad_to_gradient_merge[grad_name] = gradient_merge_var.name
        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
            new_grad_op, ref_process_mesh, ref_dims_mapping, dist_context
        )
    return new_params_to_grads, grad_to_gradient_merge


def _rename_arg_names(op_desc, var_name_dict):
    for input_name in op_desc.input_arg_names():
        if input_name in var_name_dict:
            op_desc._rename_input(input_name, var_name_dict[input_name])

    for output_name in op_desc.output_arg_names():
        if output_name in var_name_dict:
            op_desc._rename_output(output_name, var_name_dict[output_name])


def _create_cond_block_and_update_optimizer(
    main_program,
    cond_var,
    params_grads,
    new_params_to_grads,
    grad_to_gradient_merge,
    optimize_ops_desc,
    allreduce_sum_desc,
    k_steps,
    avg,
    master_grad,
):
    def true_apply_gradient():
        cur_block_idx = main_program.current_block_idx
        cur_block = main_program.current_block()

        # cur_block's forward_block & backward_block is itself
        cur_block._set_forward_block_idx(cur_block_idx)

        # record grads_name to insert c_allreduce_sum op
        grads_name = [grad.name for _, grad in params_grads]
        # append c_allreduce_sum ops and scale ops
        for op_desc in allreduce_sum_desc:
            outputs_name = op_desc.output_arg_names()
            assert len(outputs_name) == 1
            if outputs_name[0] in grads_name:
                new_op_desc = cur_block.desc.append_op()
                new_op_desc.copy_from(op_desc)
                _rename_arg_names(new_op_desc, grad_to_gradient_merge)
                new_op_desc._set_attr(OP_ROLE_KEY, OpRole.Optimize)
        cur_block._sync_with_cpp()

        if avg:
            for _, new_grad in new_params_to_grads:
                # grad /= k_steps
                cur_block.append_op(
                    type='scale',
                    inputs={'X': new_grad},
                    outputs={'Out': new_grad},
                    attrs={
                        'scale': 1.0 / k_steps,
                        'bias': 0.0,
                        'bias_after_scale': False,
                    },
                )
                new_grad.op._set_attr(OP_ROLE_KEY, OpRole.Optimize)

        cast_name_dict = {}
        # append optimizer ops
        for op_desc in optimize_ops_desc:
            if master_grad and is_gradient_clip_op(op_desc):
                # the passes' order is amp --> gradient_clip --> gradient_merge,
                # When master_grad is True, the gradient_clip ops' vars dtype must be fp32.
                # Then the cast_ops should be removed, and the relevant ops' varname need to be renamed.
                if op_desc.type() == "cast":
                    if (
                        op_desc.attr('out_dtype') in [4, 22]
                        and op_desc.attr('in_dtype') == 5
                    ):
                        cast_name_dict[
                            op_desc.output_arg_names()[0]
                        ] = op_desc.input_arg_names()[0]
                    elif (
                        op_desc.attr('in_dtype') in [4, 22]
                        and op_desc.attr('out_dtype') == 5
                    ):
                        cast_name_dict[
                            op_desc.output_arg_names()[0]
                        ] = op_desc.input_arg_names()[0]
                    continue

                for out_name in op_desc.output_arg_names():
                    out_var = cur_block._var_recursive(out_name)
                    out_var.desc.set_dtype(core.VarDesc.VarType.FP32)

                _rename_arg_names(op_desc, cast_name_dict)

            new_op_desc = cur_block.desc.append_op()
            new_op_desc.copy_from(op_desc)

            # update input/output
            _rename_arg_names(new_op_desc, grad_to_gradient_merge)

            # remove op_role_var
            if new_op_desc.has_attr(OP_ROLE_VAR_KEY):
                new_op_desc.remove_attr(OP_ROLE_VAR_KEY)

        cur_block._sync_with_cpp()

        # clear gradient_merge_vars
        for _, new_grad in new_params_to_grads:
            paddle.tensor.fill_constant(
                shape=new_grad.shape,
                dtype=new_grad.dtype,
                value=0.0,
                out=new_grad,
            )
            new_grad.op._set_attr(OP_ROLE_KEY, OpRole.Optimize)

    paddle.static.nn.cond(cond_var, true_fn=true_apply_gradient, false_fn=None)
    cond_op = main_program.global_block().ops[-1]
    cond_op._set_attr(OP_ROLE_KEY, OpRole.Optimize)


def parse_program(
    main_program,
    startup_program,
    params_grads,
    k_steps,
    avg,
    master_grad,
    dist_context,
):
    # 1 remove optimizer_op, allreduce_sum_op and scale_op from main_program
    optimize_ops_desc, allreduce_sum_desc = _remove_and_get_ops(
        main_program, dist_context
    )

    # back to block 0
    main_program._rollback()

    # 2 append gradient merge backward op to main_program
    (
        new_params_to_grads,
        grad_to_gradient_merge,
    ) = _append_gradient_merge_backward_op(
        main_program, startup_program, params_grads, master_grad, dist_context
    )

    # 3 create gradient_merge_cond
    cond_var = _get_gm_cond_var(main_program, k_steps, dist_context)

    # 4 create ConditionalBlock and append gradient merge optimizer ops
    _create_cond_block_and_update_optimizer(
        main_program,
        cond_var,
        params_grads,
        new_params_to_grads,
        grad_to_gradient_merge,
        optimize_ops_desc,
        allreduce_sum_desc,
        k_steps,
        avg,
        master_grad,
    )


@register_pass("auto_parallel_gradient_merge_pass")
class GradientMergePass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("k_steps", -1)
        self.set_attr("avg", True)
        self.set_attr("master_grad", False)

    def _check_self(self):
        if self.get_attr("k_steps") < 1:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _type(self):
        return PassType.COMM_OPT

    def _apply_single_impl(self, main_program, startup_program, context):
        k_steps = self.get_attr("k_steps", -1)
        avg = self.get_attr("avg", False)
        master_grad = self.get_attr("master_grad", False)
        dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")
        # TODO(zyl): make master_grad configurable
        master_grad = True
        with paddle.static.program_guard(main_program, startup_program):
            parse_program(
                main_program,
                startup_program,
                params_grads,
                k_steps,
                avg,
                master_grad,
                dist_context,
            )

        main_program._sync_with_cpp()
