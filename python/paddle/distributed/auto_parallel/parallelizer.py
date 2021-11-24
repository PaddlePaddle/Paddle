#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.fleet import cloud_utils
import paddle.fluid.core as core
from paddle.fluid import program_guard
from .dist_context import DistributedContext
from .dist_context import get_default_distributed_context
from .dist_context import set_default_distributed_context
from .completion import complete_annotation, complete_backward_annotation, complete_update_annotation
from .partitioner import Partitioner
from .process_group import get_all_process_groups
from .utils import make_data_unshard
from .reshard import reshard


class AutoParallelizer:
    """
    AutoParallelizer is the main controller class to do the auto parallel process.
    And the auto parallel process will be triggered in the wrapped parallelize function.
    To facilitate the auto parallelization, it will contain information about program, cluster and the
    related context. In this basic version, the program information will be retrevied from 
    Fleet object, and the cluster information can be retrevied in the new created Cluster object,
    and the context information can be retrevied in the new created DistributedContext. 
    """

    def __init__(self, fleet):
        self._fleet = fleet
        self._optimizer = self._fleet.user_defined_optimizer
        self._dist_strategy = self._fleet._user_defined_strategy
        self._dist_context = DistributedContext()
        # self._pass_context = PassContext()
        self._pass_context = None

    def _remove_distributed_attrs(self, main_program):
        suffix = core.kAutoParallelSuffix()
        # distributed attributes for variable have been removed
        # in previous process.
        for block in main_program.blocks:
            for op in block.ops:
                for attr_name in op.attr_names:
                    if suffix in attr_name:
                        op._remove_attr(attr_name)

    def _apply_serial_forward_pass(self, main_program, startup_program):

        # apply amp forward pass
        if self._dist_strategy.amp:
            auto_parallel_amp_pass = new_pass("auto_parallel_amp_pass",
                                              self._dist_strategy.amp_configs)
            auto_parallel_amp_pass.apply_forward(main_program, startup_program,
                                                 self._pass_context)

        # apply recompute forward pass
        if self._dist_strategy.recompute:
            auto_parallel_recompute_pass = new_pass(
                "auto_parallel_recompute_pass",
                self._dist_strategy.recompute_configs)
            auto_parallel_recompute_pass.apply_forward(
                main_program, startup_program, self._pass_context)

    def _generate_backward(self, main_program, startup_program, loss,
                           parameter_list, no_grad_set, callbacks):

        # apply recompute backward pass
        if self._dist_strategy.recompute:
            assert auto_parallel_recompute_pass
            auto_parallel_recompute_pass.apply_forward(
                main_program, startup_program, parameter_list, no_grad_set,
                self._pass_context)
        else:
            from paddle.fluid.backward import append_backward
            with program_guard(main_program, startup_program):
                params_grads = append_backward(
                    loss,
                    parameter_list,
                    no_grad_set,
                    callbacks,
                    distop_context=self._dist_context.dist_op_context)
            complete_backward_annotation(
                main_program, dist_context=self._dist_context)

        # apply amp forward pass
        if self._dist_strategy.amp:
            assert auto_parallel_amp_pass
            auto_parallel_amp_pass.apply_backward(main_program, startup_program,
                                                  self._pass_context)

        return params_grads

    def _apply_optimize(self, main_program, startup_program, params_grads):

        if self._dist_strategy.sharding:
            auto_parallel_sharding_pass = new_pass(
                "auto_parallel_sharding_pass", self._dist_strategy)
            params_grads = auto_parallel_sharding_pass.apply(
                main_program, startup_program, params_grads, self._pass_context)

        if self._dist_strategy.gradient_merge:
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass",
                self._dist_strategy.gradient_merge_configs)
            auto_parallel_gradient_merge_pass.apply(
                main_program, startup_program, params_grads, self._pass_context)

        else:
            with program_guard(main_program, startup_program):
                optimize_ops = self._optimizer.apply_gradients(params_grads)

        # update completion 
        complete_update_annotation(
            main_program, dist_context=self._dist_context)

        return optimize_ops

    def parallelize(self,
                    loss,
                    startup_program,
                    parameter_list=None,
                    no_grad_set=None):
        assert startup_program is not None
        main_program = loss.block.program

        # Annotation completion
        completed_main_program = complete_annotation(main_program,
                                                     self._dist_context)

        # serial forward pass
        self._apply_serial_forward_pass(completed_main_program, startup_program)

        # serial backward pass
        params_grads = self._generate_backward(
            completed_main_program, startup_program, loss, parameter_list,
            no_grad_set, callbacks)

        # Logical partition 
        rank = paddle.distributed.get_rank()
        partitioner = Partitioner(self._dist_context, rank)
        partitioned_main_prog, partitioned_startup_prog, partitioned_params_grads = partitioner.partition(
            completed_main_program, startup_program, params_grads)

        # generate optimize program
        partitioned_optimize_ops = self._apply_optimize(
            partitioned_main_prog, partitioned_startup_prog,
            partitioned_params_grads)

        # Traverse different rank programs and traverse each op of them,
        # instantiate communication by process_mapping.
        all_process_groups = get_all_process_groups()
        for process_group in all_process_groups:
            if rank not in process_group._ranks:
                continue
            process_group.instantiate()

        # The last step: remove all distributed attributes to be compatiable
        # with inference.
        self._remove_distributed_attrs(partitioned_main_prog)
        make_data_unshard(partitioned_main_prog, partitioned_startup_prog,
                          self._dist_context)

        reshard(partitioned_main_prog, partitioned_startup_prog, rank,
                self._dist_context)

        # Copy distributed info to the default context
        set_default_distributed_context(self._dist_context)

        return partitioned_optimize_ops, partitioned_params_grads, partitioned_startup_prog, partitioned_main_prog
