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

import paddle
from .public import *
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready
from paddle.distributed.passes import new_pass, PassContext


class PsProgramBuilder(object):
    def __init__(self, context):
        self.context = context
        self.cloned_main = self.context['cloned_main']
        self.cloned_startup = self.context['cloned_startup']

        self.use_ps_gpu = self.context['use_ps_gpu']
        self.use_heter_ps = self.context['is_heter_ps_mode']
        self.is_worker = self.context['is_worker']
        self.is_heter_worker = self.context['is_heter_worker']
        self.ps_mode = self.context['ps_mode']

        self.launch_barrier = self.context['launch_barrier']
        self.launch_barrier_flag = self.context['launch_barrier_flag']
        self.server_endpoints = self.context[
            'role_maker']._get_pserver_endpoints()

    def _optimize_programs(self):
        pass

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        is_sgd_adam = False
        ops = get_optimize_ops(self.context['origin_main_program'])
        if len(ops) == 0:
            return
        add_lr_decay_table_pass = new_pass('add_lr_decay_table_pass',
                                           self.context)
        add_lr_decay_table_pass.apply([], [], self.context)
        for op in ops:
            if op.type in ["sgd", "adam"]:
                is_sgd_adam = True
                break
        if is_sgd_adam:
            return

    def _build_programs(self):
        if self.context['is_worker']:
            self._build_trainer_programs()
            loss.block.program = self.cloned_main
            fluid.framework.switch_startup_program(self.cloned_startup)

        elif self.context['is_server']:
            self._build_pserver_programs()
            loss.block.program = self.context['_main_server']
            fluid.framework.switch_startup_program(self.context[
                '_startup_server'])


class GeoPsProgramBuilder(PsProgramBuilder):  # 仅 CPU 模式
    def __init__(self, context):
        super(GeoPsProgramBuilder, self).__init__(context)
        if self.ps_mode != DistributedMode.GEO:
            raise ValueError("ps mode: {} not matched {}",
                             format(ps_mode, "GeoPsProgramBuilder"))

    def _build_trainer_programs(self):
        append_send_ops_pass = new_pass("append_send_ops_pass", self.context)
        append_send_ops_pass.apply([self.cloned_main], [], self.context)

        context['origin_main_program'] = self.cloned_main

        if launch_barrier and launch_barrier_flag:
            wait_server_ready(server_endpoints)

        return


class CpuSyncPsProgramBuilder(PsProgramBuilder):
    def __init__(self, context):
        super(CpuSyncPsProgramBuilder, self).__init__(context)
        if self.ps_mode == DistributedMode.GEO:
            raise ValueError("ps mode: {} not matched {}",
                             format(ps_mode, "CpuSyncPsProgramBuilder"))

    def _build_trainer_programs(self):
        add_lr_decay_table_pass = new_pass("add_lr_decay_table_pass",
                                           self.context)
        add_lr_decay_table_pass.apply([], [], self.context)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.context)
        distributed_ops_pass.apply([self.cloned_main], [], self.context)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.context)
        delete_optimizer_pass.apply([self.cloned_main], [], self.context)

        append_send_ops_pass = new_pass("append_send_ops_pass", self.context)
        append_send_ops_pass.apply([self.cloned_main], [], self.context)

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.context)
        fake_init_ops_pass.apply([], [self.cloned_startup], self.context)

        if launch_barrier and launch_barrier_flag:
            wait_server_ready(server_endpoints)  # why need?

        return


class CpuAsyncPsProgramBuilder(CpuSyncPsProgramBuilder):
    def __init__(self, context):
        super(CpuAsyncPsProgramBuilder, self).__init__(context)


class GpuPsProgramBuilder(PsProgramBuilder):  # 和 geo、sync、async 等无关 
    def __init__(self, context):
        super(GpuPsProgramBuilder, self).__init__(context)

    def _build_trainer_programs(self):
        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.context)
        delete_optimizer_pass.apply([_main], [_startup], self.context)

        add_lr_decay_table_pass = new_pass("add_lr_decay_table_pass",
                                           self.context)
        add_lr_decay_table_pass.apply([_main], [], self.context)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.context)
        distributed_ops_pass.apply([_main], [], self.context)

        ps_fake_init_ops_pass = new_pass("fake_init_ops_pass", self.context)
        ps_fake_init_ops_pass.apply([], [_startup], self.context)

        ps_gpu_pass = new_pass("ps_gpu_pass", self.context)
        ps_gpu_pass.apply([_main], [], self.context)

        ps_transpile_pass = new_pass("ps_transpile_pass", self.context)
        ps_transpile_pass.apply([_main], [_startup], self.context)

        if launch_barrier and launch_barrier_flag:
            wait_server_ready(server_endpoints)  # why need?

        return


class HeterAsyncPsProgramBuilder(PsProgramBuilder):
    def __init__(self, context):
        super(HeterAsyncPsProgramBuilder, self).__init__(context)
        if self.use_ps_gpu or self.ps_mode == DistributedMode.GEO or self.context[
                'is_heter_ps_mode'] == False:
            raise ValueError("ps mode: {} not matched {}",
                             format(ps_mode, "HeterAsyncPsProgramBuilder"))

    def _build_trainer_programs(self):
        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.context)
        delete_optimizer_pass.apply([_main], [_startup], self.context)

        add_lr_decay_table_pass = new_pass("add_lr_decay_table_pass",
                                           self.context)
        add_lr_decay_table_pass.apply([_main], [], self.context)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.context)
        distributed_ops_pass.apply([_main], [], self.context)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.context)
        delete_optimizer_pass.apply([], [_startup], self.context)

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.context)
        fake_init_ops_pass.apply([], [_startup], self.context)

        if is_heter_worker:
            split_heter_worker_ops_pass = new_pass(
                "split_heter_worker_ops_pass", self.context)
            split_heter_worker_ops_pass.apply([_main], [], self.context)
        else:
            # for default worker
            split_trainer_ops_pass = new_pass("split_trainer_ops_pass",
                                              self.context)
            split_trainer_ops_pass([_main], [], self.context)

        if launch_barrier and launch_barrier_flag:
            wait_server_ready(server_endpoints)  # why need?

        return

    def _build_programs(self):
        if self.context['is_worker'] or self.context['is_heter_worker']:
            self._build_trainer_programs()
            ps_set_heter_pipeline_opt_pass = new_pass(
                "set_heter_pipeline_opt_pass", self.context)
            ps_set_heter_pipeline_opt_pass.apply(
                [loss.block.program], [startup_program], self.context)

        elif self.context['is_server']:
            self._build_pserver_programs()
            loss.block.program = self.context['_main_server']
            fluid.framework.switch_startup_program(self.context[
                '_startup_server'])


class FlPsProgramBuilder(PsProgramBuilder):
    def __init__(self, context):
        super(FlPsProgramBuilder, self).__init__(context)

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        pass

    def _build_programs(self):
        pass
