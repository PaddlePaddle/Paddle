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
    def __init__(self, pass_ctx):
        self.pass_ctx = pass_ctx
        self.attrs = self.pass_ctx._attrs
        self.loss = self.attrs['loss']
        self.cloned_main = self.attrs['cloned_main']
        self.cloned_startup = self.attrs['cloned_startup']

        self.use_ps_gpu = self.attrs['use_ps_gpu']
        self.use_heter_ps = self.attrs['is_heter_ps_mode']
        self.is_worker = self.attrs['is_worker']
        self.is_heter_worker = self.attrs['is_heter_worker']
        self.ps_mode = self.attrs['ps_mode']

        self.launch_barrier = self.attrs['launch_barrier']
        self.launch_barrier_flag = self.attrs['launch_barrier_flag']
        self.server_endpoints = self.attrs['role_maker']._get_pserver_endpoints(
        )

    def _optimize_programs(self):
        pass

    def _build_trainer_programs(self):
        raise NotImplementedError

    def _build_pserver_programs(self):
        is_sgd_adam = False
        ops = get_optimize_ops(self.attrs['origin_main_program'])
        if len(ops) == 0:
            return
        add_lr_decay_table_pass = new_pass('add_lr_decay_table_pass',
                                           self.attrs)
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)
        for op in ops:
            if op.type in ["sgd", "adam"]:
                is_sgd_adam = True
                break
        if is_sgd_adam:
            return

    def _build_programs(self):
        if self.attrs['is_worker']:
            self._build_trainer_programs()
            fluid.framework.switch_startup_program(self.cloned_startup)
            self.loss.block.program = self.cloned_main

        elif self.attrs['is_server']:
            self._build_pserver_programs()
            self.loss.block.program = self.attrs['_main_server']
            fluid.framework.switch_startup_program(self.attrs[
                '_startup_server'])


class GeoPsProgramBuilder(PsProgramBuilder):  # 仅 CPU 模式
    def __init__(self, pass_ctx):
        super(GeoPsProgramBuilder, self).__init__(pass_ctx)
        if self.ps_mode != DistributedMode.GEO:
            raise ValueError("ps mode: {} not matched {}",
                             format(self.ps_mode, "GeoPsProgramBuilder"))

    def _build_trainer_programs(self):
        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        self.attrs['origin_main_program'] = self.cloned_main

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

    def _build_pserver_programs(self):
        add_listen_and_serv_pass = new_pass('add_listen_and_serv_pass',
                                            self.attrs)
        add_listen_and_serv_pass.apply([self.attrs['_main_server']], [None],
                                       self.pass_ctx)
        return


class CpuSyncPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_ctx):
        super(CpuSyncPsProgramBuilder, self).__init__(pass_ctx)
        if self.ps_mode != DistributedMode.SYNC and self.ps_mode != DistributedMode.ASYNC:
            raise ValueError("ps mode: {} not matched {}",
                             format(self.ps_mode, "PsProgramBuilder"))

    def _build_trainer_programs(self):
        add_lr_decay_table_pass = new_pass("add_lr_decay_table_pass",
                                           self.attrs)
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.attrs)
        delete_optimizer_pass.apply([self.cloned_main], [None], self.pass_ctx)

        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_extra_optimizer_pass = new_pass("delete_extra_optimizer_pass",
                                               self.attrs)
        delete_extra_optimizer_pass.apply([self.attrs['origin_main_program']],
                                          [self.cloned_startup], self.pass_ctx)

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        self.attrs['origin_main_program'] = self.cloned_main
        self.attrs['origin_startup_program'] = self.cloned_startup

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

        return


class CpuAsyncPsProgramBuilder(CpuSyncPsProgramBuilder):
    def __init__(self, pass_ctx):
        super(CpuAsyncPsProgramBuilder, self).__init__(pass_ctx)


class GpuPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_ctx):
        super(GpuPsProgramBuilder, self).__init__(pass_ctx)

    def _build_trainer_programs(self):

        add_lr_decay_table_pass = new_pass("add_lr_decay_table_pass",
                                           self.attrs)
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        ps_gpu_pass = new_pass("ps_gpu_pass", self.attrs)
        ps_gpu_pass.apply([self.cloned_main], [None], self.pass_ctx)

        ps_transpile_pass = new_pass("ps_transpile_pass", self.attrs)
        ps_transpile_pass.apply([self.cloned_main], [self.cloned_startup],
                                self.pass_ctx)

        self.attrs['origin_main_program'] = self.cloned_main
        self.attrs['origin_startup_program'] = self.cloned_startup

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

        return


class HeterAsyncPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_ctx):
        super(HeterAsyncPsProgramBuilder, self).__init__(pass_ctx)

    def _build_trainer_programs(self):
        add_lr_decay_table_pass = new_pass("add_lr_decay_table_pass",
                                           self.attrs)
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.attrs)
        delete_optimizer_pass.apply([self.cloned_main], [None], self.pass_ctx)

        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_extra_optimizer_pass = new_pass("delete_extra_optimizer_pass",
                                               self.attrs)
        delete_extra_optimizer_pass.apply([self.attrs['origin_main_program']],
                                          [self.cloned_startup], self.pass_ctx)

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        if self.is_heter_worker:
            split_heter_worker_ops_pass = new_pass(
                "split_heter_worker_ops_pass", self.attrs)
            split_heter_worker_ops_pass.apply([self.cloned_main], [None],
                                              self.pass_ctx)
        else:
            split_trainer_ops_pass = new_pass("split_trainer_ops_pass",
                                              self.attrs)
            split_trainer_ops_pass.apply([self.cloned_main], [None],
                                         self.pass_ctx)

        set_heter_pipeline_opt_pass = new_pass('set_heter_pipeline_opt_pass',
                                               self.attrs)
        set_heter_pipeline_opt_pass.apply([self.cloned_main],
                                          [self.cloned_startup], self.pass_ctx)

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

        return

    def _build_programs(self):
        if self.attrs['is_worker'] or self.attrs['is_heter_worker']:
            self._build_trainer_programs()
            ps_set_heter_pipeline_opt_pass = new_pass(
                "set_heter_pipeline_opt_pass", self.attrs)
            ps_set_heter_pipeline_opt_pass.apply(
                [self.cloned_main], [self.cloned_startup], self.pass_ctx)

        elif self.attrs['is_server']:
            self._build_pserver_programs()
            self.loss.block.program = self.attrs['_main_server']
            fluid.framework.switch_startup_program(self.attrs[
                '_startup_server'])


class FlPsProgramBuilder(HeterAsyncPsProgramBuilder):
    def __init__(self, pass_ctx):
        super(FlPsProgramBuilder, self).__init__(pass_ctx)

    def _build_trainer_programs(self):
        _main_file = ps_log_root_dir + '0_fl_worker_main_program.prototxt'
        #debug_program(_main_file, self.cloned_main)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        _main_file = ps_log_root_dir + '1_fl_worker_main_program.prototxt'
        #debug_program(_main_file, self.cloned_main)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.attrs)
        delete_optimizer_pass.apply([self.cloned_main], [None], self.pass_ctx)

        _main_file = ps_log_root_dir + '2_fl_worker_main_program.prototxt'
        #debug_program(_main_file, self.cloned_main)

        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        _main_file = ps_log_root_dir + '3_fl_worker_main_program.prototxt'
        #debug_program(_main_file, self.cloned_main)

        delete_extra_optimizer_pass = new_pass("delete_extra_optimizer_pass",
                                               self.attrs)
        delete_extra_optimizer_pass.apply([self.attrs['origin_main_program']],
                                          [self.cloned_startup], self.pass_ctx)

        _main_file = ps_log_root_dir + '4_fl_worker_main_program.prototxt'
        #debug_program(_main_file, self.cloned_main)

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        _main_file = ps_log_root_dir + '5_fl_worker_main_program.prototxt'
        #debug_program(_main_file, self.cloned_main)

        split_trainer_ops_pass = new_pass("split_fl_ops_pass", self.attrs)
        split_trainer_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        if not self.is_heter_worker:
            self.part_a_program = self.pass_ctx._attrs['part_a_main_program']
            self.cloned_main = self.part_a_program
            _main_file = ps_log_root_dir + '8_fl_A_main_program.prototxt'
            debug_program(_main_file, self.cloned_main)
        else:
            self.part_b_program = self.pass_ctx._attrs['part_b_main_program']
            self.cloned_main = self.part_b_program
            _main_file = ps_log_root_dir + '8_fl_B_main_program.prototxt'
            debug_program(_main_file, self.cloned_main)

        set_heter_pipeline_opt_pass = new_pass('set_heter_pipeline_opt_pass',
                                               self.attrs)
        set_heter_pipeline_opt_pass.apply([self.cloned_main],
                                          [self.cloned_startup], self.pass_ctx)

        if not self.is_heter_worker:
            _main_file = ps_log_root_dir + 'final_fl_A_main_program.prototxt'
            debug_program(_main_file, self.attrs['origin_main_program']
                          ._heter_pipeline_opt['section_program'])
        else:
            _main_file = ps_log_root_dir + 'final_fl_B_main_program.prototxt'
            debug_program(_main_file, self.attrs['origin_main_program']
                          ._heter_pipeline_opt['section_program'])

        return

    def _build_pserver_programs(self):
        pass

    def _build_programs(self):
        self._build_trainer_programs()
        self._build_pserver_programs()
