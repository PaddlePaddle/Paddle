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

from __future__ import print_function
from __future__ import division
import os
import collections
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from paddle.fluid.dygraph import Layer, LayerList
from ..base.private_helper_function import wait_server_ready
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, is_loss_grad_op, is_backward_op, is_optimizer_op


class RawProgramOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(RawProgramOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self.global_ring_id = 0

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(RawProgramOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        self.without_graph_optimization = user_defined_strategy.without_graph_optimization
        self.fuse_all_reduce_ops = user_defined_strategy.fuse_all_reduce_ops
        if self.fuse_all_reduce_ops:
            self.fuse_grad_size_in_num = user_defined_strategy.fuse_grad_size_in_num
            self.calc_comm_same_stream = user_defined_strategy._calc_comm_same_stream

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.without_graph_optimization == True:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.without_graph_optimization = False

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.without_graph_optimization = True

    def _broadcast_params(self, ring_id):
        block = self.startup_program.global_block()
        param = None
        for param in block.iter_parameters():
            if param.is_distributed:
                continue

            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })

        if not param: return  # no parameter on this device
        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': param},
            outputs={'Out': param},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Forward})

    def _get_process_group_info(self):
        # global ring info
        self.global_endpoints = self.endpoints
        self.global_rank = self.rank
        self.global_nranks = self.nranks

    def _init_process_group(self):
        self._get_process_group_info()
        collective_helper = CollectiveHelper(self.role_maker, wait_port=False)
        # Create global ring for all gpus (ring_id = 0)
        collective_helper._init_communicator(
            self.startup_program, self.current_endpoint, self.global_endpoints,
            self.global_rank, self.global_ring_id, True, self.global_ring_id,
            True)
        self._broadcast_params(self.global_ring_id)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.endpoints[self.role_maker._worker_index()]
        self.rank = self.role_maker._worker_index()
        self.nranks = self.role_maker._worker_num()
        if startup_program is None:
            startup_program = fluid.default_startup_program()
        self.startup_program = startup_program

        block = loss.block
        program = block.program
        self.main_program = program

        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        if self.nranks == 1:
            return optimize_ops, params_grads
        self._init_process_group()

        self.main_program = program
        if self.nranks > 1:
            self._transpile_main_program(loss)
        return optimize_ops, params_grads

    def _transpile_main_program(self, loss):
        self._insert_loss_grad_ops(loss)
        if self.fuse_all_reduce_ops and self.fuse_grad_size_in_num > 1:
            self._allreduce_fusion_program()
        else:
            self._insert_allreduce_ops()

    def _insert_loss_grad_ops(self, loss):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = self.main_program.global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / self.nranks,
                        OP_ROLE_KEY: OpRole.Backward
                    })

    def _insert_allreduce_ops(self):
        block = self.main_program.global_block()
        ring_id = self.global_ring_id
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.attr(OP_ROLE_VAR_KEY)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = 1
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    param = block.var(param_name)
                    grad_name = op_role_var[i + 1]
                    grad = block.var(grad_name)
                    if param.is_distributed:
                        continue

                    block._insert_op(
                        idx + offset,
                        type='c_sync_calc_stream',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={OP_ROLE_KEY: OpRole.Backward, })
                    offset += 1
                    block._insert_op(
                        idx + offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                block._insert_op(
                    idx,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
                break

    # This function helps reduce the number of allreduce by integrating op, which can save communication time.
    # to use allreduce fuse, follow these codes:
    # strategy = paddle.distributed.fleet.DistributedStrategy()
    # strategy.without_graph_optimization = True
    # strategy.fuse_all_reduce_ops = True
    # strategy.calc_comm_same_stream = False
    # strategy.fuse_grad_size_in_num = 8
    def _allreduce_fusion_program(self):
        block = self.main_program.global_block()
        ring_id = self.global_ring_id
        param_grads = []

        # find all grad params
        for op in reversed(block.ops):
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.attr(OP_ROLE_VAR_KEY)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0, "vars need to be one param var followed by one grad var, " \
                                                  "but got odd number of vars"
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    param = block.var(param_name)
                    grad_name = op_role_var[i + 1]
                    grad = block.var(grad_name)
                    if param.is_distributed:
                        continue
                    param_grads.append(grad)

        segments = []
        last_dtype = None
        # split the grad based on dtype and fused size
        for var in param_grads:
            if len(segments) == 0 \
                    or len(segments[-1]) == self.fuse_grad_size_in_num \
                    or var.dtype != last_dtype:
                segments.append([var])
                last_dtype = var.dtype
            else:
                segments[-1].append(var)

        fused_vars = []
        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                for segment in segments:
                    # insert coalesce tensor
                    tmp_var = block.create_var(
                        name=unique_name.generate('FusedOutput_{}'.format(
                            segment[0].name)),
                        dtype=segment[0].dtype,
                        persistable=True,
                        stop_gradient=True)
                    fused_vars.append(tmp_var)
                    block._insert_op_without_sync(
                        idx,
                        type="coalesce_tensor",
                        inputs={"Input": segment},
                        outputs={"Output": segment,
                                 "FusedOutput": tmp_var},
                        attrs={
                            "copy_data": True,
                            "use_align": True,
                            "dtype": segment[0].dtype,
                            OP_ROLE_KEY: OpRole.Backward
                        })
                break

        # insert the allreduce_sum op
        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                for fused_var in fused_vars:
                    block._insert_op_without_sync(
                        idx,
                        type='c_allreduce_sum',
                        inputs={'X': fused_var},
                        outputs={'Out': fused_var},
                        attrs={
                            'ring_id': ring_id,
                            'use_calc_stream': self.calc_comm_same_stream,
                            OP_ROLE_KEY: OpRole.Backward
                        })
                    if not self.calc_comm_same_stream:
                        block._insert_op_without_sync(
                            idx,
                            type='c_sync_calc_stream',
                            inputs={'X': fused_var},
                            outputs={'Out': fused_var},
                            attrs={OP_ROLE_KEY: OpRole.Backward})
                break

        if len(fused_vars) == 0:
            block._sync_with_cpp()
            return

        # insert the sync comm op
        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                block._insert_op_without_sync(
                    idx,
                    type='c_sync_comm_stream',
                    inputs={'X': fused_vars[0]},
                    outputs={'Out': fused_vars[0]},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
                break
        block._sync_with_cpp()
