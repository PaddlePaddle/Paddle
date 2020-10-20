#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from ..base.private_helper_function import wait_server_ready
from paddle.fluid.optimizer import PipelineOptimizer as PO
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, is_update_op, is_loss_grad_op, is_backward_op, is_optimizer_op


class PipelineHelper(CollectiveHelper):
    def __init__(self, role_maker, nrings=1, wait_port='6174'):
        super(PipelineHelper, self).__init__(role_maker, nrings, wait_port)

    def _init_communicator(self, program, current_endpoint, endpoints, rank,
                           ring_id, wait_port):
        nranks = len(endpoints)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            wait_server_ready(other_endpoints)

        block = program.global_block()
        nccl_id_var = block.create_var(
            name=unique_name.generate('nccl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)
        block.append_op(
            type='c_gen_nccl_id',
            inputs={},
            outputs={'Out': nccl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints,
                OP_ROLE_KEY: OpRole.Forward
            })

        block.append_op(
            type='c_comm_init',
            inputs={'X': nccl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': ring_id,
                OP_ROLE_KEY: OpRole.Forward,
                'device_id': OpRole.Forward
            })

    def _broadcast_params(self):
        block = self.startup_program.global_block()
        ring_id = 0
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

        for ring_id in range(self.nrings):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id,
                       OP_ROLE_KEY: OpRole.Forward})


class PipelineOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(PipelineOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(PipelineOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        num_microbatches = user_defined_strategy.pipeline_configs['micro_batch']
        self.wrapped_opt = PO(self.inner_opt, num_microbatches=num_microbatches)

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.pipeline == True:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.pipeline = False
        dist_strategy.pipeline_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        # we do not support enable pipeline automatically right now
        return

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        optimize_ops, params_grads, prog_list = \
            self.wrapped_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)
        if self.role_maker._worker_num() == 1:
            return optimize_ops, params_grads

        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        self.startup_program = startup_program
        if startup_program is None:
            self.startup_program = fluid.default_startup_program()

        assert prog_list
        self.main_program_list = prog_list
        self.main_program = loss.block.program
        nranks = len(endpoints)
        self.nranks = nranks
        self.nrings = len(self.main_program_list)

        self.rank = self.role_maker._worker_index()
        self.endpoints = endpoints
        self.current_endpoint = current_endpoint

        pipeline_helper = PipelineHelper(self.role_maker, nrings=self.nrings)
        pipeline_helper.update_startup_program(self.startup_program)

        self._transpile_main_program()
        return optimize_ops, params_grads

    def _transpile_main_program(self):
        self._insert_loss_grad_ops()
        for ring_id in range(self.nrings):
            self._insert_allreduce_ops(ring_id)

    def _insert_loss_grad_ops(self):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = self.main_program_list[self.nrings - 1]['program'].global_block(
        )
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

    def _insert_allreduce_ops(self, ring_id):
        block = self.main_program_list[ring_id]['program'].global_block()
        origin_block = self.main_program.global_block()
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    origin_param = origin_block.vars[op_role_var[i]]
                    if origin_param.is_distributed:
                        continue
                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={OP_ROLE_KEY: OpRole.Backward})
                        offset += 1

                    block._insert_op(
                        offset,
                        type='c_sync_calc_stream',
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
                    idx + ring_id,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
            break
