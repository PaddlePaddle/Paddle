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

from typing import Any, Dict, List, Tuple

import paddle
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.static.process_group import (
    get_world_process_group,
)
from paddle.distributed.auto_parallel.static.utils import (
    is_forward_op,
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
from paddle.static import device_guard

from .pass_base import PassBase, PassType, register_pass

world_process_group = get_world_process_group()


def _remove_and_get_optimizer_op(main_program, dist_context):
    # 1 create tmp block
    # 2 mv optimizer op from global program to tmp block
    # 3 del the op from dist_context
    main_block = main_program.global_block()
    optimize_ops_block = paddle.static.Program().global_block()
    removed_op_idx = []
    for idx, op in enumerate(main_block.ops):
        if is_optimize_op(op):
            # append optimizer op to tmp block
            new_op_desc = optimize_ops_block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            removed_op_idx.append(idx)

            # del op from dist_context
            # if dist_context:
            #     dist_context.del_dist_op_for_program(op)

    for idx in removed_op_idx[::-1]:
        main_block._remove_op(idx, sync=False)
    main_block._sync_with_cpp()

    return optimize_ops_block


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
        value=0,
        dtype='int32',
        persistable=True,
        force_cpu=True,
    )
    set_var_dist_attr(dist_context, zero_var, [-1], world_process_group.ranks)

    # Add step var & cond var
    step_var = paddle.static.create_global_var(
        name="gradient_merge_step",
        shape=[1],
        value=0,
        dtype='int32',
        persistable=True,
        force_cpu=True,
    )
    set_var_dist_attr(dist_context, step_var, [-1], world_process_group.ranks)

    cond_var = paddle.static.create_global_var(
        name="gradient_merge_cond",
        shape=[1],
        value=bool(0),
        dtype='bool',
        persistable=True,
        force_cpu=True,
    )
    set_var_dist_attr(dist_context, cond_var, [-1], world_process_group.ranks)

    with device_guard("cpu"):
        # step_var += 1
        increment_op = main_block.append_op(
            type='increment',
            inputs={'X': [step_var]},
            outputs={'Out': [step_var]},
            attrs={'step': 1.0, OP_ROLE_KEY: OpRole.Backward},
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
    params_grads: List[Tuple[Any, Any]],
    dist_context,
) -> Tuple[List[Tuple[Any, Any]], Dict[str, Any]]:
    main_block = main_program.global_block()
    startup_block = startup_program.global_block()

    # step1: remove grad.op's op_role_var
    grad_to_params_grads = {}
    for param, grad in params_grads:
        assert (
            param.type != core.VarDesc.VarType.SELECTED_ROWS
        ), "SELECTED_ROWS is not supported in GradientMergeOptimizer for now"
        grad_to_params_grads[grad.name] = (param, grad)

    # {grad.name: gradient_merge_var.name} to rename opt inputs
    grad_to_gradient_merge = {}
    # {param: gradient_merge_var} to insert scale op and fill_constant op
    new_params_grads = []

    for index, op in reversed(list(enumerate(main_block.ops))):
        if len(grad_to_params_grads) == 0:
            break
        if is_forward_op(op):
            break

        for out_name in op.desc.output_arg_names():
            if out_name in grad_to_params_grads:
                param = grad_to_params_grads[out_name][0]
                assert param is not None
                ref_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                    param
                )
                assert ref_dist_attr is not None

                # step2: create gradient_merge var and init with 0
                # Add persistable gradient variables in main_program
                gradient_merge_var = main_block.create_var(
                    name=param.name + "@GRAD@MERGE",
                    shape=param.shape,
                    dtype=param.dtype,
                    persistable=True,
                )
                ref_process_mesh = ref_dist_attr.process_mesh
                ref_dims_mapping = ref_dist_attr.dims_mapping
                set_var_dist_attr(
                    dist_context,
                    gradient_merge_var,
                    ref_dims_mapping,
                    ref_process_mesh,
                )

                # Add persistable gradient variables in startup_program
                startup_gradient_merge_var = startup_block.create_var(
                    name=param.name + "@GRAD@MERGE",
                    shape=param.shape,
                    dtype=param.dtype,
                    persistable=True,
                )
                # Initial persistable gradient variables in startup_program
                startup_block.append_op(
                    type="fill_constant",
                    outputs={"Out": startup_gradient_merge_var},
                    attrs={
                        "shape": param.shape,
                        "dtype": param.dtype,
                        "value": float(0),
                    },
                )

                # step3: Accumulate persistable gradient variables in main_program
                grad = grad_to_params_grads[out_name][1]
                assert grad is not None
                # NOTE(zhaoyingli): inplace operation must be 'a = a + b', cannot be 'a = b + a'
                new_grad_op = main_block._insert_op_without_sync(
                    index + 1,
                    type="elementwise_add",
                    inputs={'X': gradient_merge_var, 'Y': grad},
                    outputs={'Out': gradient_merge_var},
                    attrs={
                        'axis': -1,
                        'use_mkldnn': False,
                        OP_ROLE_KEY: OpRole.Backward,
                    },
                )

                # Construct new_params_grads and grad_to_gradient_merge
                new_params_grads.append([param, gradient_merge_var])
                grad_to_gradient_merge[grad.name] = gradient_merge_var.name
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    new_grad_op,
                    ref_process_mesh,
                    ref_dims_mapping,
                    dist_context,
                )

                del grad_to_params_grads[out_name]

    assert (
        len(grad_to_params_grads) == 0
    ), "grad_to_param_names must be empty right now, but it has {} items".format(
        len(grad_to_params_grads)
    )
    main_block._sync_with_cpp()

    return new_params_grads, grad_to_gradient_merge


def _create_cond_block_and_update_optimizer(
    main_program,
    cond_var,
    new_params_to_grads: List[Tuple[Any, Any]],
    grad_to_gradient_merge: Dict[str, str],
    optimize_ops_block,
    k_steps,
    avg,
    dist_context,
):
    def true_apply_gradient():
        cur_block_idx = main_program.current_block_idx
        cur_block = main_program.current_block()

        if avg:
            for _, new_grad in new_params_to_grads:
                # grad /= k_steps
                scale_op = cur_block.append_op(
                    type='scale',
                    inputs={'X': new_grad},
                    outputs={'Out': new_grad},
                    attrs={
                        'scale': 1.0 / k_steps,
                        'bias': 0.0,
                        'bias_after_scale': False,
                    },
                )
                scale_op._set_attr(OP_ROLE_KEY, OpRole.Optimize)

        # append optimizer ops
        for opt_op_idx in range(optimize_ops_block.desc.op_size()):
            op_desc = optimize_ops_block.desc.op(opt_op_idx)
            new_op_desc = cur_block.desc.append_op()
            new_op_desc.copy_from(op_desc)

            # update input/output
            for input_name in new_op_desc.input_arg_names():
                if input_name in grad_to_gradient_merge:
                    new_op_desc._rename_input(
                        input_name, grad_to_gradient_merge[input_name]
                    )

            for output_name in new_op_desc.output_arg_names():
                if output_name in grad_to_gradient_merge:
                    new_op_desc._rename_output(
                        output_name, grad_to_gradient_merge[output_name]
                    )

            # remove op_role_var
            if new_op_desc.has_attr(OP_ROLE_VAR_KEY):
                new_op_desc.remove_attr(OP_ROLE_VAR_KEY)

            # op's update Grad
            if core.grad_var_suffix() in new_op_desc.input_arg_names():
                grad_value = new_op_desc.input("Grad")[0]
                # TODO FIXME(xym) support fp16
                grad_merge_value = grad_value + '@MERGE'
                new_op_desc.set_input("Grad", [grad_merge_value])

        main_program.global_block()._sync_with_cpp()
        cur_block._sync_with_cpp()

        # update serial op
        for op in cur_block.ops:
            if is_optimize_op(op):
                dist_op = dist_context.get_dist_op_for_program(op)
                if dist_op:
                    dist_op._serial_op = op

        # clear gradient_merge_vars
        # NOTE(zhaoyingli): Must use 'set_value' op in pir to assign 0-value for persistable var.
        for _, new_grad in new_params_to_grads:
            cur_block.append_op(
                type="set_value",
                inputs={"Input": [new_grad]},
                outputs={"Out": [new_grad]},
                attrs={
                    "values": [float(0)],
                    "dtype": new_grad.dtype,
                    "shape": [1],
                    "axes": [],
                    "starts": [],
                    "ends": [],
                    "steps": [],
                    OP_ROLE_KEY: OpRole.Optimize,
                },
            )

    paddle.static.nn.cond(cond_var, true_fn=true_apply_gradient, false_fn=None)
    cond_op = main_program.global_block().ops[-1]
    cond_op._set_attr(OP_ROLE_KEY, OpRole.Optimize)


def parse_program(
    main_program, startup_program, params_grads, k_steps, avg, dist_context
):
    # 1 remove optimizer_op from main_program
    optimize_ops_block = _remove_and_get_optimizer_op(
        main_program, dist_context
    )

    # 2 append gradient merge backward op to main_program
    (
        new_params_to_grads,
        grad_to_gradient_merge,
    ) = _append_gradient_merge_backward_op(
        main_program, startup_program, params_grads, dist_context
    )

    # 3 create gradient_merge_cond
    cond_var = _get_gm_cond_var(main_program, k_steps, dist_context)

    # 4 create ConditionalBlock and append gradient merge optimizer ops
    _create_cond_block_and_update_optimizer(
        main_program,
        cond_var,
        new_params_to_grads,
        grad_to_gradient_merge,
        optimize_ops_block,
        k_steps,
        avg,
        dist_context,
    )


@register_pass("auto_parallel_gradient_merge_pass")
class GradientMergePass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("k_steps", -1)
        self.set_attr("avg", True)

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
        dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")
        with paddle.static.program_guard(main_program, startup_program):
            parse_program(
                main_program,
                startup_program,
                params_grads,
                k_steps,
                avg,
                dist_context,
            )

        main_program._sync_with_cpp()
