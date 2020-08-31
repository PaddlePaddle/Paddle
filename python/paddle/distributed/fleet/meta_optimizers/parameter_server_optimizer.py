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
from paddle.fluid import core

dtype_to_size = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}


class ParameterServerOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ParameterServerOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _is_graph_out(self):
        return False

    def _can_apply(self):
        if self.role_maker._is_collective:
            return False
        if self.user_defined_strategy.auto == True:
            return True

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

            # for heter program
            if self.role_maker._is_heter_parameter_server_mode:
                from paddle.fluid.incubate.fleet.parameter_server.ir import heter_trainer_pass as heter_worker
                if self.role_maker._is_heter_worker():
                    # for heter worker
                    _main = heter_worker.split_heter_worker_ops_pass(
                        _main, compiled_config)
                else:
                    # for default worker
                    _main = heter_worker.split_trainer_ops_pass(_main,
                                                                compiled_config)
                # for startup change
                _startup = heter_worker.delete_startup_useless_ops_var_pass(
                    _startup, _main, compiled_config)
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

    def _try_auto_apply_geo(self, program, compiled_config):
        if self.user_defined_strategy.auto == False:
            return

        a_sync_configs = self.user_defined_strategy.a_sync_configs
        if a_sync_configs["k_steps"] >= 0:
            return

        self.user_defined_strategy.a_sync = True
        if not isinstance(self.inner_opt, fluid.optimizer.SGDOptimizer):
            # auto async
            a_sync_configs["k_steps"] = 0
            self.user_defined_strategy.a_sync_configs = a_sync_configs
            return

        import psutil
        free = psutil.virtual_memory().free

        param_grad_pairs = compiled_config.origin_sparse_pairs + compiled_config.origin_dense_pairs
        processed_var_names = set(["@EMPTY@"])

        param_memory_size = 0
        for param_grad_pair in param_grad_pairs:
            param, grad = param_grad_pair
            param_memory_size += param.m_size
            param_memory_size += grad.m_size
            processed_var_names.add(param.name)
            processed_var_names.add(grad.name)
        print("param_grads memory: %d" % param_memory_size)

        upper_mem_use = param_memory_size * 2.5

        _tmp_vars = dict()
        batch_size = 1024
        for op in program.global_block().ops:
            for var_name in op.output_arg_names:
                if var_name in processed_var_names:
                    continue
                processed_var_names.add(var_name)
                var = program.global_block().vars[var_name]

                if var.desc.type() != core.VarDesc.VarType.LOD_TENSOR:
                    continue

                data_count = 1
                neg_dim_count = 0
                for x in var.shape:
                    if x < 0:
                        if neg_dim_count >= 1:
                            raise ValueError(
                                "Var %s has more than one negative dim." %
                                (var_name))
                        neg_dim_count += 1
                        data_count *= (-x)
                    else:
                        data_count *= x
                _tmp_vars[var_name] = (data_count, neg_dim_count,
                                       dtype_to_size[var.dtype])

        for varname in _tmp_vars:
            data_count, neg_dim_count, _ = _tmp_vars[varname]
            if neg_dim_count == 1:
                data_count *= batch_size
            var_memory = data_count * _
            upper_mem_use += var_memory
        print("upper mem: %d" % (upper_mem_use))

        if upper_mem_use < psutil.virtual_memory().free:
            # auto geo
            a_sync_configs["k_steps"] = 400
        else:
            # auto async
            a_sync_configs["k_steps"] = 0
        self.user_defined_strategy.a_sync_configs = a_sync_configs

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.inner_opt.minimize(loss, startup_program, parameter_list,
                                no_grad_set)

        _origin_main_program = loss.block.program
        _origin_startup_program = startup_program
        from paddle.fluid.incubate.fleet.parameter_server.ir import public as public

        compiled_config = public.CompileTimeStrategy(_origin_main_program,
                                                     _origin_startup_program,
                                                     None, self.role_maker)

        self._try_auto_apply_geo(_origin_main_program, compiled_config)

        strategy = self._get_distributed_strategy()
        compiled_config.strategy = strategy

        if self.role_maker.is_worker() or self.role_maker._is_heter_worker():
            main_program, startup_program = self._build_trainer_programs(
                compiled_config)
        elif self.role_maker.is_server():
            main_program, startup_program = self._build_pserver_programs(
                compiled_config)

        loss.block.program = main_program
        fluid.framework.switch_startup_program(startup_program)

        return None, None

    def _disable_strategy(self, dist_strategy):
        self.user_defined_strategy.a_sync_configs = {}
