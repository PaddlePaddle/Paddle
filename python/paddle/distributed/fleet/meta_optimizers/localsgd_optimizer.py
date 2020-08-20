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
from .common import OpRole, OP_ROLE_KEY, CollectiveHelper, is_update_op


class LocalSGDOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(LocalSGDOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = []
        self.snapshot_key = '@SNAPSHOT'

    def _can_apply(self):
        if not self.user_defined_strategy.localsgd:
            return False

        if self.role_maker.worker_num() <= 1:
            return False

        return isinstance(self.inner_opt, Momentum) \
                or isinstance(self.inner_opt, SGD)

    def _disable_strategy(self, dist_strategy):
        dist_strategy.localsgd = False
        dist_strategy.localsgd_configs = {}

    def snapshot_name(self, param_name):
        return param_name + self.snapshot_key

    def create_snapshot_vars(self, program):
        block = program.global_block()

        non_dist_params = []
        for param in block.iter_parameters():
            if not param.is_distributed:
                non_dist_params.append(param)

        p2s = []
        for param in non_dist_params:
            snapshot = block.create_var(
                name=self.snapshot_name(param.name),
                shape=param.shape,
                persistable=True,
                stop_gradient=True,
                dtype=param.dtype)
            p2s.append([param, snapshot])
        return p2s

    def init_snapshot_vars(self, startup_program, param2snapshot):
        with program_guard(startup_program):
            for param, snapshot in param2snapshot:
                layers.assign(param, snapshot)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        minimized = self.inner_opt.minimize(
            loss, startup_program=startup_program)

        init_k_steps = self.user_defined_strategy.localsgd_configs['k_steps']
        auto_steps = self.user_defined_strategy.auto

        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block

        self.nrings = 2
        collective_helper = CollectiveHelper(self.role_maker, self.nrings)
        collective_helper.update_startup_program(startup_program)
        p2s = self.create_snapshot_vars(startup_program)
        self.init_snapshot_vars(startup_program, p2s)

        p2s = self.create_snapshot_vars(main_block.program)
        with program_guard(main_block.program, startup_program):
            step = layers.autoincreased_step_counter(begin=0)
            k_steps = layers.create_global_var(
                name="k_steps",
                shape=[1],
                value=init_k_steps,
                dtype='int64',
                persistable=True)
            last_step = layers.create_global_var(
                name="last_step",
                shape=[1],
                value=int(0),
                dtype='int64',
                persistable=True)

            if auto_steps:
                avg_loss = layers.collective._c_allreduce(
                    loss) / self.role_maker.worker_num()

                lr_0 = layers.create_global_var(
                    name="lr_0",
                    shape=[1],
                    value=float(0),
                    dtype='float32',
                    persistable=True)
                loss_0 = layers.create_global_var(
                    name="loss_0",
                    shape=[1],
                    value=float(0),
                    dtype='float32',
                    persistable=True)

                global_lr = self.inner_opt._global_learning_rate()

                def initialize():
                    layers.assign(loss, loss_0)
                    layers.assign(global_lr, lr_0)

                layers.cond(step == 0, initialize)

            def communicate():
                sub_block = default_main_program().current_block()
                ring_id = -1
                for param, snapshot in p2s:
                    sub_block.append_op(
                        type='elementwise_sub',
                        inputs={'X': [snapshot],
                                'Y': [param]},
                        outputs={'Out': [param]},
                        attrs={OP_ROLE_KEY: OpRole.Optimize})
                    sub_block.append_op(
                        type='c_sync_calc_stream',
                        inputs={'X': param},
                        outputs={'Out': param},
                        attrs={OP_ROLE_KEY: OpRole.Optimize})
                    ring_id = (ring_id + 1) % self.nrings
                    sub_block.append_op(
                        type='c_allreduce_sum',
                        inputs={'X': [param]},
                        outputs={'Out': [param]},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Optimize
                        })

                for ring_id in range(self.nrings):
                    sub_block.append_op(
                        type='c_sync_comm_stream',
                        inputs={'X': param},
                        outputs={'Out': param},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Optimize
                        })

                for param, snapshot in p2s:
                    sub_block.append_op(
                        type='scale',
                        inputs={'X': [param]},
                        outputs={'Out': [param]},
                        attrs={
                            'scale': 1.0 / self.role_maker.worker_num(),
                            OP_ROLE_KEY: OpRole.Optimize
                        })
                    sub_block.append_op(
                        type='elementwise_sub',
                        inputs={'X': [snapshot],
                                'Y': [param]},
                        outputs={'Out': [param]},
                        attrs={OP_ROLE_KEY: OpRole.Optimize})
                    sub_block.append_op(
                        type='assign',
                        inputs={'X': [param]},
                        outputs={'Out': [snapshot]},
                        attrs={OP_ROLE_KEY: OpRole.Optimize})

                if auto_steps:
                    next_local_steps = layers.cast(
                        layers.ceil(
                            layers.sqrt(lr_0 * loss / (global_lr * loss_0) *
                                        float(init_k_steps))),
                        dtype='int64')
                    max_local_steps = layers.fill_constant(
                        shape=[1], dtype='int64', value=16)
                    next_local_steps = layers.elementwise_min(next_local_steps,
                                                              max_local_steps)
                    layers.assign(next_local_steps, k_steps)
                layers.assign(step, last_step)

            layers.cond(step - last_step == k_steps, communicate)

        return minimized
