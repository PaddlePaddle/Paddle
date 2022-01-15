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


class PsProgramBuilder(object):
    def __init__(self, context):
        self.context = context

    def _optimize_programs(self):
        pass

    def _build_trainer_programs(self):
        pass_context = self.context
        _main = self.context['_main']
        _startup = self.context['_startup']

        use_ps_gpu = self.context['use_ps_gpu']
        use_heter_ps = self.context['is_heter_ps_mode']
        is_worker = self.context['is_worker']
        is_heter_worker = self.context['is_heter_worker']
        ps_mode = self.context['ps_mode']

        launch_barrier = self.context['launch_barrier']
        launch_barrier_flag = self.context['launch_barrier_flag']
        server_endpoints = self.context['role_maker']._get_pserver_endpoints()

        ps_delete_optimizer_pass = new_pass("ps_delete_optimizer_pass",
                                            pass_context)
        ps_delete_optimizer_pass.apply([_main], [_startup], pass_context)

        if ps_mode == DistributedMode.GEO:
            ps_append_send_ops_pass = new_pass("ps_append_send_ops_pass",
                                               pass_context)
            ps_append_send_ops_pass.apply([_main], [_startup], pass_context)
        else:
            ps_add_lr_decay_table_pass = new_pass("ps_add_lr_decay_table_pass",
                                                  pass_context)
            ps_add_lr_decay_table_pass.apply([_main], [], pass_context)

            ps_distributed_ops_pass = new_pass("ps_distributed_ops_pass",
                                               pass_context)
            ps_distributed_ops_pass.apply([_main], [], pass_context)

            if not use_ps_gpu:
                ps_delete_optimizer_pass = new_pass("ps_delete_optimizer_pass",
                                                    pass_context)
                ps_delete_optimizer_pass.apply([], [_startup], pass_context)

            ps_fake_init_ops_pass = new_pass("fake_init_ops_pass", pass_context)
            ps_fake_init_ops_pass.apply([], [_startup], pass_context)

            if use_ps_gpu:
                ps_gpu_pass = new_pass("ps_gpu_pass", pass_context)
                ps_gpu_pass.apply([_main], [], pass_context)

                ps_transpile_pass = new_pass("ps_transpile_pass", pass_context)
                ps_transpile_pass.apply([_main], [_startup], pass_context)

            if use_heter_ps:
                if is_heter_worker:
                    ps_split_heter_worker_ops_pass = new_pass(
                        "split_heter_worker_ops_pass", pass_context)
                    ps_split_heter_worker_ops_pass.apply([_main], [],
                                                         pass_context)
                else:
                    # for default worker
                    ps_split_trainer_ops_pass = new_pass(
                        "ps_split_trainer_ops_pass", pass_context)
                    ps_split_trainer_ops_pass([_main], [], pass_context)

        if launch_barrier and launch_barrier_flag:
            wait_server_ready(server_endpoints)  # why need?

        return

    def _build_pserver_programs(self):
        main_program = self.context['origin_main_program']
        optimize_ops = _get_optimize_ops(main_program)
        is_sgd_adam = False
        for op in ops:
            if op.type in ["sgd", "adam"]:
                is_sgd_adam = True
                break

        ps_mode = self.context['ps_mode']
        if ps_mode != DistributedMode.GEO and len(optimize_ops) == 0:
            return

        pass_context = self.context
        _main = self.context['_main_server']
        _startup = self.context['_startup_server']

        if ps_mode != DistributedMode.GEO and is_sgd_adam == True:
            ps_add_lr_decay_table_pass = new_pass("ps_add_lr_decay_table_pass",
                                                  pass_context)
            ps_add_lr_decay_table_pass.apply([main_program], [], pass_context)
            return

        if ps_mode == DistributedMode.GEO:
            ps_geo_server_pass = new_pass("ps_geo_server_pass", pass_context)
            ps_geo_server_pass.apply([_main], [_startup], pass_context)
        else:
            ps_not_geo_server_pass = new_pass("ps_not_geo_server_pass",
                                              pass_context)
            ps_not_geo_server_pass.apply([_main], [_startup], pass_context)
        return

    def _build_programs():
        if self.context['is_worker'] or self.context['is_heter_worker']:
            self._build_trainer_programs()
            if use_heter_ps:
                ps_set_heter_pipeline_opt_pass = new_pass(
                    "ps_set_heter_pipeline_opt_pass", pass_context)
                ps_set_heter_pipeline_opt_pass.apply(
                    [loss.block.program], [startup_program], pass_context)
            else:
                loss.block.program = self.context['_main']
                fluid.framework.switch_startup_program(self.context['_statup'])

        elif self.context['is_server']:
            self._build_pserver_programs()
            loss.block.program = self.context['_main_server']
            fluid.framework.switch_startup_program(self.context[
                '_startup_server'])


class GeoPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_context):
        pass

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        pass


class NotGeoPsProgramBuilder(PsProgramBuilder):
    def __init__(self, pass_context):
        pass

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        pass


class NotGeoCpuPsProgramBuilder(NotGeoPsProgramBuilder):
    def __init__(self, pass_context):
        pass

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        pass


class NotGeoGpuPsProgramBuilder(NotGeoPsProgramBuilder):
    def __init__(self, pass_context):
        pass

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        pass


class NotGeoHeterPsProgramBuilder(NotGeoPsProgramBuilder):
    def __init__(self, pass_context):
        pass

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        pass


class NotGeoFlPsProgramBuilder(NotGeoPsProgramBuilder):
    def __init__(self, pass_context):
        pass

    def _build_trainer_programs(self):
        pass

    def _build_pserver_programs(self):
        pass
