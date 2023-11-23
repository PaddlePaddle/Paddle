#  Copyright (c) 2023 Enflame. All Rights Reserved.
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

import os

import paddle
from paddle.fluid import core, framework
from paddle.fluid.framework import (
    Variable,
    device_guard,
    in_dygraph_mode,
    name_scope,
    program_guard,
)
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.optimizer import Optimizer as FluidOptimizer
from paddle.optimizer import Optimizer


class GcuDistributedOptimizer(Optimizer):
    def __init__(self, optimizer):
        self.inner_opt = optimizer
        self.fluid_base = isinstance(self.inner_opt, FluidOptimizer)
        self._learning_rate = self.inner_opt._learning_rate
        self._learning_rate_map = self.inner_opt._learning_rate_map
        os.environ['_gcu_enable_distributed'] = "true"
        print(
            'GcuDistributedOptimizer inner optimizer fluid_base:{}'.format(
                self.fluid_base
            )
        )

    def _global_learning_rate(self, program=None):
        return self.inner_opt._global_learning_rate(program)

    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):

        assert isinstance(loss, Variable), "The loss should be an Tensor."

        parameter_list = (
            parameters if parameters else self.inner_opt._parameter_list
        )

        if self.fluid_base:
            params_grads = self.inner_opt.backward(
                loss,
                startup_program=startup_program,
                parameter_list=parameter_list,
                no_grad_set=no_grad_set,
            )
        else:
            params_grads = self.inner_opt.backward(
                loss,
                startup_program=startup_program,
                parameters=parameter_list,
                no_grad_set=no_grad_set,
            )

        optimize_ops = self._apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads
        )

        return optimize_ops, params_grads

    def _apply_optimize(
        self, loss, startup_program, params_grads, param_group_idx=0
    ):
        if framework.in_dygraph_mode():
            with program_guard(
                framework.default_main_program(),
                framework.default_startup_program(),
            ):
                if isinstance(params_grads, list):
                    if self.inner_opt._grad_clip is not None:
                        params_grads = self.inner_opt._grad_clip(params_grads)
                    params_grads = self.inner_opt.append_regularization_ops(
                        params_grads, self.inner_opt.regularization
                    )
                else:
                    grad_clip = params_grads['grad_clip']
                    if grad_clip is not None:
                        params_grads['params'] = grad_clip(
                            params_grads['params']
                        )

                    params_grads[
                        'params'
                    ] = self.inner_opt.append_regularization_ops(
                        params_grads['params'], self.inner_opt.regularization
                    )
                optimize_ops = self._create_optimization_pass(
                    params_grads, param_group_idx=param_group_idx
                )
        else:
            assert param_group_idx == 0
            program = loss.block.program
            with program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def apply_gradients(self, params_grads):
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        # NOTE(zhiqiu): currently, only support ClipGradByGlobalNorm and without regularization.
        if (
            self.fluid_base
            and self.inner_opt._flatten_param_grads
            and self.inner_opt.regularization is None
        ):
            if self.inner_opt._grad_clip is None or isinstance(
                self.inner_opt._grad_clip, paddle.nn.ClipGradByGlobalNorm
            ):
                params_grads = self.inner_opt.flatten_param_grads(params_grads)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self.inner_opt._grad_clip is not None:
            params_grads = self.inner_opt._grad_clip(params_grads)
        else:
            params_grads = paddle.nn.clip.append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = self.inner_opt.append_regularization_ops(
            params_grads, self.inner_opt.regularization
        )

        if self.fluid_base:
            optimize_ops = self._create_optimization_pass_fluid_base(
                params_grads
            )
        else:
            optimize_ops = self._create_optimization_pass(params_grads)
        return optimize_ops

    def _create_optimization_pass(
        self, parameters_and_grads, param_group_idx=0
    ):
        """Add optimization operators to update gradients to tensors.

        Args:
          parameters_and_grads(list(tuple(Tensor, Tensor))):
            a list of (tensor, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Allways called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert (
                current_block.backward_block_idx != -1
            ), "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx
            ]

        start = len(target_block.ops)
        self.inner_opt.helper = LayerHelper(self.inner_opt.__class__.__name__)

        self.inner_opt._create_global_learning_rate()

        # NOTE: Multi Tensor support [ Momentum, Adam ] for dygraph mode
        if (
            self.inner_opt._use_multi_tensor
            and self.inner_opt.__class__.__name__
            in [
                'Momentum',
                'Adam',
            ]
        ):
            if (
                len(
                    self.inner_opt._param_dict['FP32_LODTensor'][
                        param_group_idx
                    ]
                )
                == 0
                and len(
                    self.inner_opt._param_dict['FP16_LODTensor'][
                        param_group_idx
                    ]
                )
                == 0
            ):
                if isinstance(parameters_and_grads, list):
                    assert param_group_idx == 0
                    self.inner_opt._multi_tensor_init(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads
                            if not p[0].stop_gradient
                        ],
                        param_group_idx,
                    )
                else:
                    self.inner_opt._update_param_group(parameters_and_grads)
                    self.inner_opt._multi_tensor_init(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads['params']
                            if not p[0].stop_gradient
                        ],
                        param_group_idx,
                    )
            if framework.in_dygraph_mode():
                self._append_optimize_multi_tensor_op(
                    target_block,
                    parameters_and_grads,
                    param_group_idx=param_group_idx,
                )
            else:
                self.inner_opt._update_param_device_map(
                    parameters_and_grads, target_block
                )
                # NOTE: Multi Tensor requires all parameters to be in the same device and program.
                # param_grad_list = [p_0,g_0,p_1,g_1,....]
                param_grad_list = []
                for param_and_grad in parameters_and_grads:
                    if (
                        not param_and_grad[0].stop_gradient
                        and param_and_grad[1] is not None
                    ):
                        param_grad_list.append(param_and_grad[0])
                        param_grad_list.append(param_and_grad[1])
                with param_grad_list[0].block.program._optimized_guard(
                    param_grad_list
                ), name_scope("optimizer"):
                    device = self.inner_opt._get_device_for_param(
                        param_grad_list[0].name
                    )
                    with device_guard(device):
                        self._append_optimize_multi_tensor_op(
                            target_block,
                            parameters_and_grads,
                            param_group_idx=param_group_idx,
                        )
        else:
            if not framework.in_dygraph_mode():
                params_grads_device_map = (
                    parameters_and_grads['params']
                    if isinstance(parameters_and_grads, dict)
                    else parameters_and_grads
                )
                self.inner_opt._update_param_device_map(
                    params_grads_device_map, target_block
                )

            if isinstance(parameters_and_grads, list):
                with paddle.fluid.framework.dygraph_guard_if_declarative():
                    self.inner_opt._create_accumulators(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads
                            if not p[0].stop_gradient
                        ],
                    )
            else:
                params_acc_dict = parameters_and_grads.copy()
                params_acc_dict['params'] = [
                    p[0]
                    for p in params_acc_dict['params']
                    if not p[0].stop_gradient
                ]
                with paddle.fluid.framework.dygraph_guard_if_declarative():
                    self.inner_opt._create_accumulators(
                        target_block, params_acc_dict
                    )

            if framework.in_dygraph_mode():
                found_inf = self.inner_opt._get_auxiliary_var('found_inf')
                if found_inf:
                    if isinstance(found_inf, core.eager.Tensor):
                        self.inner_opt._set_auxiliary_var('found_inf', True)
                else:
                    if isinstance(found_inf, core.eager.Tensor):
                        self.inner_opt._set_auxiliary_var('found_inf', False)
                    if isinstance(parameters_and_grads, list):
                        for param_and_grad in parameters_and_grads:
                            if param_and_grad[1] is None:
                                continue
                            if param_and_grad[0].stop_gradient is False:
                                self._append_optimize_op(
                                    target_block, param_and_grad
                                )
                    else:
                        for param_and_grad in parameters_and_grads['params']:
                            if param_and_grad[1] is None:
                                continue
                            if param_and_grad[0].stop_gradient is False:
                                param_grad_dict = {}
                                param_grad_dict['params'] = param_and_grad
                                param_grad_dict.update(
                                    {
                                        k: v
                                        for k, v in parameters_and_grads.items()
                                        if k != 'params'
                                    }
                                )
                                self._append_optimize_op(
                                    target_block, param_grad_dict
                                )
            else:
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    with param_and_grad[0].block.program._optimized_guard(
                        param_and_grad
                    ), name_scope("optimizer"):
                        if param_and_grad[0].stop_gradient is False:
                            device = self.inner_opt._get_device_for_param(
                                param_and_grad[0].name
                            )
                            with device_guard(device):
                                optimize_op = self._append_optimize_op(
                                    target_block, param_and_grad
                                )

        self._append_gcu_attrs(target_block)
        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self.inner_opt._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _create_optimization_pass_fluid_base(self, parameters_and_grads):
        """Add optimization operators to update gradients to variables.

        Args:
          parameters_and_grads(list(tuple(Variable, Variable))):
            a list of (variable, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Allways called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert (
                current_block.backward_block_idx != -1
            ), "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx
            ]

        start = len(target_block.ops)

        self.inner_opt._update_param_device_map(
            parameters_and_grads, target_block
        )
        self.inner_opt._create_accumulators(
            target_block, [p[0] for p in parameters_and_grads if p[0].trainable]
        )
        self.inner_opt._create_global_learning_rate()

        if in_dygraph_mode():
            found_inf = self.inner_opt._get_auxiliary_var('found_inf')
            if found_inf:
                if isinstance(found_inf, core.eager.Tensor):
                    self.inner_opt._set_auxiliary_var('found_inf', True)
            else:
                if isinstance(found_inf, core.eager.Tensor):
                    self.inner_opt._set_auxiliary_var('found_inf', False)
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].trainable is True:
                        self._append_optimize_op(target_block, param_and_grad)
        else:
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                with param_and_grad[0].block.program._optimized_guard(
                    param_and_grad
                ), name_scope("optimizer"):
                    if param_and_grad[0].trainable is True:
                        device = self.inner_opt._get_device_for_param(
                            param_and_grad[0].name
                        )
                        with device_guard(device):
                            optimize_op = self._append_optimize_op(
                                target_block, param_and_grad
                            )

        self._append_gcu_attrs(target_block)
        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self.inner_opt._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _append_gcu_attrs(self, block):
        for op in block.ops:
            if not op.has_attr('_gcu_graph_op_category'):
                op._set_attr('_gcu_graph_op_category', 'gcu_fp_bp')
                if op.type in [
                    'merged_momentum',
                    'merged_adam',
                    'merged_adamw',
                ]:
                    op._set_attr('_gcu_graph_op_category', 'gcu_optimizer')

    def _append_distributed_op(self, block, param_and_grad):
        # insert temp var
        allreduce_out_var = block.create_var(
            name=param_and_grad[1].name + "_gcu_all_reduce",
            shape=param_and_grad[1].shape,
            dtype=param_and_grad[1].dtype,
            type=param_and_grad[1].type,
        )
        # insert all_reduce
        allreduce_op = block.append_op(
            type='eccl_allreduce',
            inputs={'InputList': [param_and_grad[1]]},
            outputs={'OutputList': [allreduce_out_var]},
            attrs={
                'op_role': 8,  # 0: Forward   1: Backward   2: Optimize   4: RPC   8: Dist   16: LRSched   256: Loss
                'reduce_type': 0,  # 0: sum   1: prod   2: max   3: min
                'sync_mode': True,
            },
        )
        allreduce_op._set_attr('_gcu_graph_op_category', 'gcu_allreduce')

        new_param_and_grad = (param_and_grad[0], allreduce_out_var)
        return new_param_and_grad

    def _append_optimize_op(self, block, param_and_grad):
        new_param_and_grad = self._append_distributed_op(block, param_and_grad)
        # insert optimize op
        optimize_op = self.inner_opt._append_optimize_op(
            block=block, param_and_grad=new_param_and_grad
        )
        optimize_op._set_attr('_gcu_graph_op_category', 'gcu_optimizer')
        return optimize_op

    def _append_optimize_multi_tensor_op(
        self, target_block, parameters_and_grads, param_group_idx
    ):
        new_parameters_and_grads = []
        for param_and_grad in parameters_and_grads:
            new_param_and_grad = self._append_distributed_op(
                target_block, param_and_grad
            )
            new_parameters_and_grads.append(new_param_and_grad)
        return self.inner_opt._append_optimize_multi_tensor_op(
            target_block, new_parameters_and_grads, param_group_idx
        )


def get_rank():
    rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    return rank


def get_world_size():
    world_size = int(os.getenv('PADDLE_TRAINERS_NUM', "1"))
    return world_size
