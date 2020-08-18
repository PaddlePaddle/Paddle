# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

from paddle.fluid import program_guard, layers, default_main_program
from paddle.fluid.optimizer import Momentum, SGD
from .meta_optimizer_base import MetaOptimizerBase
from . import common as cmn


class DataParallelHelper(cmn.CollectiveHelper):
    def __init__(self, role_maker, nrings=1, wait_port='6174'):
        super(DataParallelHelper, self).__init__(role_maker, nrings, wait_port)

    def update_main_program(self, program):
        self._insert_scale_loss_grad_ops(program)
        self._insert_all_reduce_ops(program)

    def _insert_scale_loss_grad_ops(self, program):
        '''
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        '''
        block = program.global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if cmn.is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / self.role_maker.worker_num(),
                        cmn.OP_ROLE_KEY: cmn.OpRole.Backward
                    })

    def _insert_all_reduce_ops(self, program):
        block = program.global_block()
        ring_id = -1
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if cmn.is_backward_op(op) and \
                    cmn.OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[cmn.OP_ROLE_VAR_KEY]

                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0

                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    if param.is_distributed:
                        continue

                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={cmn.OP_ROLE_KEY: cmn.OpRole.Backward})
                        offset += 1

                    # As we search ops reversedly, we should insert c_allreduce_sum
                    # op in the same way to keep the ring_id alternate
                    ring_id = (ring_id + 1) % self.nrings
                    block._insert_op(
                        offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            cmn.OP_ROLE_KEY: cmn.OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if cmn.is_optimizer_op(op):
                for ring_id in range(self.nrings):
                    block._insert_op(
                        idx + ring_id,
                        type='c_sync_comm_stream',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            cmn.OP_ROLE_KEY: cmn.OpRole.Backward
                        })
                break


class DataParallelOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(DataParallelOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = []

    def _can_apply(self):
        if not self.user_defined_strategy.data_parallel:
            return False

        if self.role_maker.worker_num() <= 1:
            return False

        return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.data_parallel = False

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        minimized = self.inner_opt.minimize(
            loss, startup_program=startup_program)

        if startup_program is None:
            startup_program = paddle.static.default_startup_program()

        self.nrings = 1
        helper = DataParallelHelper(self.role_maker, self.nrings)
        helper.update_startup_program(startup_program)
        helper.update_main_program(loss.block.program)

        return minimized
