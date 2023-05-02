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
from paddle import fluid
from paddle.distributed.fleet.base.private_helper_function import (
    wait_server_ready,
)
from paddle.distributed.passes import new_pass

from .public import *  # noqa: F403


class PsProgramBuilder:
    def __init__(self, pass_ctx):
        self.pass_ctx = pass_ctx
        self.attrs = self.pass_ctx._attrs
        self.loss = self.attrs['loss']
        self.origin_startup_program = self.attrs['origin_startup_program']
        self.main_program = self.attrs['origin_main_programs']

        self.cloned_main = self.attrs['cloned_main']
        self.cloned_startup = self.attrs['cloned_startup']

        self.use_ps_gpu = self.attrs['use_ps_gpu']
        self.use_heter_ps = self.attrs['is_heter_ps_mode']
        self.is_worker = self.attrs['is_worker']
        self.is_heter_worker = self.attrs['is_heter_worker']
        self.is_server = self.attrs['is_server']
        self.ps_mode = self.attrs['ps_mode']

        self.launch_barrier = self.attrs['launch_barrier']
        self.launch_barrier_flag = self.attrs['launch_barrier_flag']
        self.server_endpoints = self.attrs[
            'role_maker'
        ]._get_pserver_endpoints()

    def _build_trainer_desc(self):
        opt_info = self.loss.block.program._fleet_opt
        opt_info = {} if opt_info is None else opt_info
        opt_info["trainer"] = opt_info.get("trainer", "MultiTrainer")
        opt_info["device_worker"] = opt_info.get("device_worker", "Hogwild")
        self.cloned_main._fleet_opt = opt_info

    def _optimize_programs(self):
        pass

    def _build_trainer_programs(self):
        raise NotImplementedError

    def _build_pserver_programs(self):
        is_sgd_adam = False
        ops = get_optimize_ops(self.attrs['origin_main_program'])
        if len(ops) == 0:
            return
        add_lr_decay_table_pass = new_pass(
            'add_lr_decay_table_pass', self.attrs
        )
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
            print(
                "paddle.static.default_startup_program: {}".format(
                    paddle.static.default_startup_program
                )
            )
            # print("ps_program_build before =", id(self.loss.block.program))
            self._build_trainer_desc()
            self.loss.block.program = self.cloned_main
            # print("ps_program_build after =", id(self.loss.block.program))
            # print("ps_program_build clone after =", id(self.cloned_main))
            # print("ps_program_build after trainer_desc",
            #       id(self.loss.block.program))
            # print("ps_program build trainer desc",
            #       self.loss.block.program._fleet_opt)

        elif self.attrs['is_server']:
            self._build_pserver_programs()
            self.loss.block.program = self.attrs['_main_server']
            fluid.framework.switch_startup_program(
                self.attrs['_startup_server']
            )


class GeoPsProgramBuilder(PsProgramBuilder):  # 仅 CPU 模式
    def __init__(self, pass_ctx):
        super().__init__(pass_ctx)
        if self.ps_mode != DistributedMode.GEO:
            raise ValueError(
                "ps mode: {} not matched {}",
                format(self.ps_mode, "GeoPsProgramBuilder"),
            )

    def _build_trainer_programs(self):
        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        self.attrs['origin_main_program'] = self.cloned_main

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

    def _build_pserver_programs(self):
        add_listen_and_serv_pass = new_pass(
            'add_listen_and_serv_pass', self.attrs
        )
        add_listen_and_serv_pass.apply(
            [self.attrs['_main_server']], [None], self.pass_ctx
        )
        return


class NuPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_ctx):
        super().__init__(pass_ctx)
        if not self.attrs['local_sparse']:
            raise ValueError("No local sparse params")

    def _build_trainer_programs(self):
        add_lr_decay_table_pass = new_pass(
            "add_lr_decay_table_pass", self.attrs
        )
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.attrs)
        delete_optimizer_pass.apply([self.cloned_main], [None], self.pass_ctx)

        append_send_ops_pass = new_pass(
            "append_send_ops_pass", self.attrs
        )  # fleet->PushDenseVarsAsync
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_extra_optimizer_pass = new_pass(
            "delete_extra_optimizer_pass", self.attrs
        )
        delete_extra_optimizer_pass.apply(
            [self.attrs['origin_main_program']],
            [self.cloned_startup],
            self.pass_ctx,
        )

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        append_send_ops_pass = new_pass(
            "append_send_ops_pass", self.attrs
        )  # communicator->Send
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        self.attrs['origin_main_program'] = self.cloned_main
        self.attrs['origin_startup_program'] = self.cloned_startup

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

        return


class CpuSyncPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_ctx):
        super().__init__(pass_ctx)
        if (
            self.ps_mode != DistributedMode.SYNC
            and self.ps_mode != DistributedMode.ASYNC
        ):
            raise ValueError(
                "ps mode: {} not matched {}",
                format(self.ps_mode, "PsProgramBuilder"),
            )

    def _build_trainer_programs(self):
        # print("build trainer program entry")
        # print("before ps program builder program:", self.cloned_main)
        add_lr_decay_table_pass = new_pass(
            "add_lr_decay_table_pass", self.attrs
        )
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)

        # print("before distributed op pass")
        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.attrs)
        delete_optimizer_pass.apply([self.cloned_main], [None], self.pass_ctx)

        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_extra_optimizer_pass = new_pass(
            "delete_extra_optimizer_pass", self.attrs
        )
        delete_extra_optimizer_pass.apply(
            [self.attrs['origin_main_program']],
            [self.cloned_startup],
            self.pass_ctx,
        )

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        self.attrs['origin_main_program'] = self.cloned_main
        self.attrs['origin_startup_program'] = self.cloned_startup
        # print("after ps program builder program:", self.cloned_main)

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

        return


class CpuAsyncPsProgramBuilder(CpuSyncPsProgramBuilder):
    def __init__(self, pass_ctx):
        super().__init__(pass_ctx)

    def _build_trainer_desc(self):
        opt_info = self.loss.block.program._fleet_opt
        opt_info = {} if opt_info is None else opt_info
        opt_info["trainer"] = opt_info.get("trainer", "DistMultiTrainer")
        opt_info["device_worker"] = opt_info.get(
            "device_worker", "DownpourLite"
        )
        pid = str(id(self.cloned_main))
        program_configs = {
            pid: {
                'pull_dense': [],
                'push_dense': [],
                'pull_sparse': [],
                'push_sparse': [],
            }
        }
        dense_table_config = {}
        send_ctx = get_the_one_send_context(self.attrs)
        recv_ctx = get_the_one_recv_context(self.attrs)
        for name, ctx in send_ctx.items():
            if ctx.program_id() != id(self.loss.block.program):
                continue
            if ctx.is_sparse():
                continue
            if not ctx.is_tensor_table():
                program_configs[pid]['pull_dense'].append(ctx.table_id())
                program_configs[pid]['push_dense'].append(ctx.table_id())
            dense_table_config[ctx.table_id()] = recv_ctx[ctx.table_id()]
        opt_info['program_configs'] = program_configs
        opt_info['dense_table_config'] = dense_table_config
        self.cloned_main._fleet_opt = opt_info


class GpuPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_ctx):
        super().__init__(pass_ctx)

    def _build_trainer_programs(self):

        add_lr_decay_table_pass = new_pass(
            "add_lr_decay_table_pass", self.attrs
        )
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        ps_gpu_pass = new_pass("ps_gpu_pass", self.attrs)
        ps_gpu_pass.apply([self.cloned_main], [None], self.pass_ctx)

        ps_transpile_pass = new_pass("ps_transpile_pass", self.attrs)
        ps_transpile_pass.apply(
            [self.cloned_main], [self.cloned_startup], self.pass_ctx
        )

        self.attrs['origin_main_program'] = self.cloned_main
        self.attrs['origin_startup_program'] = self.cloned_startup

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

        return


class HeterAsyncPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_ctx):
        super().__init__(pass_ctx)

    def _build_trainer_programs(self):
        add_lr_decay_table_pass = new_pass(
            "add_lr_decay_table_pass", self.attrs
        )
        add_lr_decay_table_pass.apply([], [], self.pass_ctx)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.attrs)
        delete_optimizer_pass.apply([self.cloned_main], [None], self.pass_ctx)

        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        delete_extra_optimizer_pass = new_pass(
            "delete_extra_optimizer_pass", self.attrs
        )
        delete_extra_optimizer_pass.apply(
            [self.attrs['origin_main_program']],
            [self.cloned_startup],
            self.pass_ctx,
        )

        fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        if self.is_heter_worker:
            split_heter_worker_ops_pass = new_pass(
                "split_heter_worker_ops_pass", self.attrs
            )
            split_heter_worker_ops_pass.apply(
                [self.cloned_main], [None], self.pass_ctx
            )
        else:
            split_trainer_ops_pass = new_pass(
                "split_trainer_ops_pass", self.attrs
            )
            split_trainer_ops_pass.apply(
                [self.cloned_main], [None], self.pass_ctx
            )

        set_heter_pipeline_opt_pass = new_pass(
            'set_heter_pipeline_opt_pass', self.attrs
        )
        set_heter_pipeline_opt_pass.apply(
            [self.cloned_main], [self.cloned_startup], self.pass_ctx
        )

        if self.launch_barrier and self.launch_barrier_flag:
            wait_server_ready(self.server_endpoints)

        return

    def _build_programs(self):
        if self.attrs['is_worker'] or self.attrs['is_heter_worker']:
            self._build_trainer_programs()
            ps_set_heter_pipeline_opt_pass = new_pass(
                "set_heter_pipeline_opt_pass", self.attrs
            )
            ps_set_heter_pipeline_opt_pass.apply(
                [self.cloned_main], [self.cloned_startup], self.pass_ctx
            )

        elif self.attrs['is_server']:
            self._build_pserver_programs()
            self.loss.block.program = self.attrs['_main_server']
            fluid.framework.switch_startup_program(
                self.attrs['_startup_server']
            )


class FlPsProgramBuilder(HeterAsyncPsProgramBuilder):
    def __init__(self, pass_ctx):
        super().__init__(pass_ctx)

    def _build_trainer_programs(self):
        _main_file = ps_log_root_dir + '0_fl_worker_main_program.prototxt'
        # debug_program(_main_file, self.cloned_main)

        distributed_ops_pass = new_pass("distributed_ops_pass", self.attrs)
        distributed_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        _main_file = ps_log_root_dir + '1_fl_worker_main_program.prototxt'
        # debug_program(_main_file, self.cloned_main)

        delete_optimizer_pass = new_pass("delete_optimizer_pass", self.attrs)
        delete_optimizer_pass.apply([self.cloned_main], [None], self.pass_ctx)

        _main_file = ps_log_root_dir + '2_fl_worker_main_program.prototxt'
        # debug_program(_main_file, self.cloned_main)

        append_send_ops_pass = new_pass("append_send_ops_pass", self.attrs)
        append_send_ops_pass.apply([self.cloned_main], [None], self.pass_ctx)

        _main_file = ps_log_root_dir + '3_fl_worker_main_program.prototxt'
        # debug_program(_main_file, self.cloned_main)

        delete_extra_optimizer_pass = new_pass(
            "delete_extra_optimizer_pass", self.attrs
        )
        delete_extra_optimizer_pass.apply(
            [self.attrs['origin_main_program']],
            [self.cloned_startup],
            self.pass_ctx,
        )

        _main_file = ps_log_root_dir + '4_fl_worker_main_program.prototxt'
        # debug_program(_main_file, self.cloned_main)

        # fake_init_ops_pass = new_pass("fake_init_ops_pass", self.attrs)
        # fake_init_ops_pass.apply([None], [self.cloned_startup], self.pass_ctx)

        _main_file = ps_log_root_dir + '5_fl_worker_main_program.prototxt'
        # debug_program(_main_file, self.cloned_main)

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

        set_heter_pipeline_opt_pass = new_pass(
            'set_heter_pipeline_opt_pass', self.attrs
        )
        set_heter_pipeline_opt_pass.apply(
            [self.cloned_main], [self.cloned_startup], self.pass_ctx
        )

        self.attrs['origin_startup_program'] = self.cloned_startup
        self.attrs['origin_main_program'] = self.cloned_main

        if not self.is_heter_worker:
            _main_file = ps_log_root_dir + 'final_fl_A_main_program.prototxt'
            debug_program(
                _main_file,
                self.attrs['origin_main_program']._heter_pipeline_opt[
                    'section_program'
                ],
            )
        else:
            _main_file = ps_log_root_dir + 'final_fl_B_main_program.prototxt'
            debug_program(
                _main_file,
                self.attrs['origin_main_program']._heter_pipeline_opt[
                    'section_program'
                ],
            )

        return

    def _build_pserver_programs(self):
        self.loss.block.program = self.attrs['_main_server']

    def _build_programs(self):
        if not self.is_server:
            self._build_trainer_programs()
            fluid.framework.switch_startup_program(self.cloned_startup)
            paddle.framework.switch_main_program(self.cloned_main)
            print(
                "paddle.static.default_startup_program: {}".format(
                    paddle.static.default_startup_program()._heter_pipeline_opt
                )
            )
        else:
            self._build_pserver_programs()
            fluid.framework.switch_startup_program(
                self.attrs['_startup_server']
            )
            paddle.framework.switch_main_program(self.attrs['_main_server'])
