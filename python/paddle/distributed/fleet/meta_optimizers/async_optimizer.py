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

from paddle import fluid
from .meta_optimizer_base import MetaOptimizerBase


class AsyncMetaOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(AsyncMetaOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _is_graph_out(self):
        return False

    def _can_apply(self):
        if self.role_maker._is_collective:
            return False
        k_steps = self.user_defined_strategy.a_sync_configs["k_steps"]
        return True if k_steps >= 0 else False

    def _get_distributed_strategy(self):
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

        k_steps = self.user_defined_strategy.a_sync_configs["k_steps"]
        strategy = None

        if not self.user_defined_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_sync_strategy()

        if self.user_defined_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_async_strategy()

        if self.user_defined_strategy.a_sync and k_steps > 0:
            strategy = StrategyFactory.create_geo_strategy(k_steps)

        if not strategy:
            raise ValueError("k_steps must be invalid value, please check")

        return strategy

    def _build_trainer_programs(self, compiled_config):
        from paddle.fluid.incubate.fleet.parameter_server.ir import trainer_pass as worker

        _main = compiled_config.origin_main_program.clone()
        _startup = compiled_config.origin_startup_program.clone()

        if not compiled_config.is_geo_mode():
            # for main program
            _main = worker.delete_optimizer_pass(_main, compiled_config)
            _main = worker.distributed_ops_pass(_main, compiled_config)
            _main = worker.append_send_ops_pass(_main, compiled_config)

            # for startup program
            _startup = worker.fake_init_ops_pass(_startup, compiled_config)
            _startup = worker.init_from_server_pass(_startup, compiled_config)
            _startup = worker.delet_extra_optimizes_pass(_startup,
                                                         compiled_config)
        else:
            _main = worker.append_send_ops_pass(_main, compiled_config)
            _startup = _startup

        return _main, _startup

    def _build_pserver_programs(self, compiled_config):
        from paddle.fluid.incubate.fleet.parameter_server.ir import pserver_pass as server

        _main = fluid.Program()
        _startup = fluid.Program()

        if not compiled_config.is_geo_mode():
            _main = server.add_listen_and_serv_pass(_main, compiled_config)
            _main = server.add_rpc_global_flags_pass(_main, compiled_config)
            _main = server.add_optimizer_pass(_main, compiled_config)
            _main = server.large_scale_sparse_pass(_main, _main,
                                                   compiled_config, False)
            _startup = server.build_pserver_startup_program_pass(
                _startup, _main, compiled_config)
            _startup = server.large_scale_sparse_pass(_startup, _main,
                                                      compiled_config, True)

            if not compiled_config.is_sync_mode():
                _main = server.delete_unused_in_main_pass(_main,
                                                          compiled_config)

            _startup = server.delete_unused_in_startup_pass(_startup, _main,
                                                            compiled_config)
        else:
            _main = server.add_listen_and_serv_pass(_main, compiled_config)
            _main = server.add_rpc_global_flags_pass(_main, compiled_config)
            _main = server.add_geo_optimizer_pass(_main, compiled_config)
            _main = server.large_scale_sparse_pass(_main, _main,
                                                   compiled_config, False)
            _startup = server.build_pserver_startup_program_pass(
                _startup, _main, compiled_config)
            _startup = server.large_scale_sparse_pass(_startup, _main,
                                                      compiled_config, True)
            _startup = server.delete_unused_in_startup_pass(_startup, _main,
                                                            compiled_config)

        return _main, _startup

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.inner_opt.minimize(loss, startup_program, parameter_list,
                                no_grad_set)
        strategy = self._get_distributed_strategy()

        _origin_main_program = loss.block.program
        _origin_startup_program = startup_program
        from paddle.fluid.incubate.fleet.parameter_server.ir import public as public

        compiled_config = public.CompileTimeStrategy(_origin_main_program,
                                                     _origin_startup_program,
                                                     strategy, self.role_maker)

        main_program, startup_program = \
            self._build_trainer_programs(compiled_config) if self.role_maker.is_worker() \
                else self._build_pserver_programs(compiled_config)

        loss.block.program = main_program
        fluid.framework.switch_startup_program(startup_program)

        return None, None

    def _disable_strategy(self, dist_strategy):
        self.user_defined_strategy.a_sync_configs = {}
