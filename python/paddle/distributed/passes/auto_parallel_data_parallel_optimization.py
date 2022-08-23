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

from collections import OrderedDict

import paddle
from paddle.fluid.framework import default_main_program
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.distributed.auto_parallel.operators.common import is_data_parallel_scale_op, is_data_parallel_reduce_op
from paddle.distributed.auto_parallel.utils import is_loss_grad_op, is_optimize_op, ring_id_to_process_group
from .pass_base import PassBase, PassType, register_pass

# add new optimizers supporting rescale_grad here
__rescale_grad_supported_opts__ = [
    'lars_momentum', 'sparse_momentum', 'dgc_momentum', 'momentum',
    'merge_momentum'
]

# a heuristic number
__max_stream_num_allow__ = 16


@register_pass("auto_parallel_data_parallel_optimization")
class DataParallelOptimizationPass(PassBase):
    """
    Apply Optimizations that specialized for data parallelism in Auto Parallel.
    1. prune grad scaling 
    2. overlap comm and calc
    3. fuse allreduce
    """

    def __init__(self):
        super(DataParallelOptimizationPass, self).__init__()
        # NOTE not use depence on loss and param_grads
        self.set_attr("dist_context", None)
        self.set_attr("global_rank", -1)
        # {grad1: group1, grad2: group1, grad3: group2}
        # record the order for fuse grad data memory
        self._grad_name_to_group_map = OrderedDict()
        # {group1:[grad1, grad2] , group2:[grad3]}
        self._group_to_grad_name_map = OrderedDict()
        self._support_rescale_grad = False

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if (not isinstance(self.get_attr("global_rank"),
                           int)) or self.get_attr("global_rank") < 0:
            return False

        return True

    def _check_conflict(self, other_pass):
        return True

    def _type(self):
        return PassType.COMM_OPT

    def _apply_single_impl(self, main_program, startup_program, context):

        self.dist_context = self.get_attr("dist_context")
        self.global_rank = int(self.get_attr("global_rank"))

        with paddle.static.program_guard(main_program, startup_program):
            self._analyze_program()
            self._prune_grad_scaling()
            self._calc_comm_overlap()
            self._fuse_allreduce()

    def _prune_grad_scaling(self):

        if not self._could_be_prune():
            return

        if self._all_dp_groups_same_degree():
            self._scale_backward_initial_grad()
        else:
            self._update_opt_rescale_grad()

        self._remove_grad_scaling()

    def _calc_comm_overlap(self):
        if not self._could_be_overlap():
            return
        self._calc_overlap_comms()
        self._update_wait_comms()

    def _fuse_allreduce(self):
        pass

    def _analyze_program(self):
        """
        build two maps
        {param_grad_name: data_parallel_group}
        {pdata_parallel_group: aram_grad_name}
        """

        block = default_main_program().global_block()
        ops = block.ops
        scaled_grads = []

        for op in ops:
            grad_name = op.output_arg_names[0]

            if is_data_parallel_reduce_op(op):
                if grad_name in self._grad_name_to_group_map:
                    continue
                assert op.has_attr(
                    "ring_id"
                ), "Unexception: comm op [{}] has NOT ring id.".format(str(op))
                group = ring_id_to_process_group(op.attr("ring_id"))

                assert group is not None, "Unexception: data parallel group of [{}] from op [{}] is None".format(
                    grad_name, str(op))

                self._grad_name_to_group_map[grad_name] = group

                if group not in self._group_to_grad_name_map:
                    self._group_to_grad_name_map[group] = [grad_name]
                else:
                    self._group_to_grad_name_map[group].append(grad_name)

            elif is_data_parallel_scale_op(op):
                scaled_grads.append(grad_name)

            # TODO support multiple optimizers in on network in future.
            # here we assume that the optimizer is unique in network.
            elif is_optimize_op(
                    op) and op.type in __rescale_grad_supported_opts__:
                self._support_rescale_grad = True

        not_synchronized_grads = []
        for grad_name in scaled_grads:
            if grad_name not in self._grad_name_to_group_map:
                not_synchronized_grads.append(grad_name)
        assert len(
            not_synchronized_grads
        ) == 0, "Unexception: gradients [{}] is scaled BUT NOT synchronized.".format(
            not_synchronized_grads)

    def _could_be_prune(self):

        return self.dist_context._gradient_scale and (
            self._support_rescale_grad or self._all_dp_groups_same_degree())

    def _all_dp_groups_same_degree(self):
        return len(
            set([
                len(group.ranks)
                for group in self._group_to_grad_name_map.keys()
            ])) == 1

    def _scale_backward_initial_grad(self):

        block = default_main_program().global_block()
        dp_degree = len(list(self._group_to_grad_name_map.keys())[0].ranks)

        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                assert op.type == 'fill_constant', \
                    "loss_grad_op must be fill_constant op, " \
                    "but this op is {}".format(op.type)
                assert op.has_attr('value')
                loss_scale = float(op.attr('value'))
                loss_scale = loss_scale / dp_degree
                op._set_attr('value', loss_scale)
                break

    def _remove_grad_scaling(self):
        block = default_main_program().global_block()

        for op_idx, op in reversed(list(enumerate(block.ops))):
            if is_data_parallel_scale_op(op):
                block._remove_op(op_idx, False)

        block._sync_with_cpp()

    def _update_opt_rescale_grad(self):

        block = default_main_program().global_block()
        scaled_grads = set()

        for idx, op in reversed(list(enumerate(block.ops))):
            if is_optimize_op(
                    op) and op.type in __rescale_grad_supported_opts__:
                assert op.has_attr(
                    'rescale_grad'
                ), "Unexception: op [{}] is supported to have [rescale_grad] attribute.".format(
                    str(op))
                assert len(
                    op.input("Grad")
                ) == 1, "Unexception: op [{}] is supported to have only one input grad var.".format(
                    str(op))

                grad_name = op.input("Grad")[0]
                dp_degree = len(
                    list(self._grad_name_to_group_map[grad_name].ranks))
                scaled_grads.add(grad_name)

                rescale_grad = float(op.attr('rescale_grad')) / dp_degree
                op._set_attr('rescale_grad', rescale_grad)

        assert scaled_grads == set(self._grad_name_to_group_map.keys(
        )), "Unexception: gradients [{}] are unscaled.".format(
            set(self._grad_name_to_group_map.keys()) - scaled_grads)

    def _could_be_overlap(self):
        # NOTE current different nccl comm will use different cuda stream
        # so if there too many dp group there will be too many stream need to be
        # created and sync.
        # revise here when framework support custom stream in static mode.
        num_dp_comm_stream = len(set(self._group_to_grad_name_map.keys()))
        if num_dp_comm_stream > __max_stream_num_allow__:
            return False

        return True

    def _calc_overlap_comms(self):
        # TODO support InterpreterCore executor for overlap.
        # InterpreterCore has a different logic for overlapping
        # which is different from use_calc_stream
        block = default_main_program().global_block()
        ops = block.ops

        # comm wait calc to finish
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_data_parallel_reduce_op(op):
                assert op.has_attr('use_calc_stream')
                assert op.has_attr('ring_id')

                op._set_attr('use_calc_stream', False)
                ring_id = op.attr("ring_id")

                block._insert_op_without_sync(idx,
                                              type='c_wait_compute',
                                              inputs={'X': []},
                                              outputs={'Out': []},
                                              attrs={
                                                  'op_role': OpRole.Backward,
                                                  'ring_id': ring_id
                                              })

        block._sync_with_cpp()

    def _update_wait_comms(self):

        block = default_main_program().global_block()
        ops = block.ops

        # update wait comm to finish
        first_optimize_op_idx = -1
        for idx, op in enumerate(ops):
            if is_optimize_op(op):
                first_optimize_op_idx = idx
                break

        assert first_optimize_op_idx > -1, "Unexception: not found optimizer op in program"

        for group in self._group_to_grad_name_map.keys():
            ring_id = group.id
            block._insert_op_without_sync(first_optimize_op_idx,
                                          type='c_wait_comm',
                                          inputs={'X': []},
                                          outputs={'Out': []},
                                          attrs={
                                              'op_role': OpRole.Backward,
                                              'ring_id': ring_id
                                          })
