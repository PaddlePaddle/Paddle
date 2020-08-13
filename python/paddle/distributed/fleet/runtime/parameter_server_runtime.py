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

import os
import logging
import warnings

import paddle.fluid as fluid
from paddle.fluid import core

from .runtime_base import RuntimeBase


class ParameterServerRuntime(RuntimeBase):
    def __init__(self):
        super(ParameterServerRuntime, self).__init__()
        self._communicator = None

    def _set_basic_info(self, context):
        self.context = context
        self.role_maker = context["role_maker"]
        self.origin_main_program = context["origin_main_program"]
        self.origin_startup_program = context["origin_startup_program"]
        self.async_strategy = self._get_distributed_strategy()
        self.compiled_strategy = self.build_compiled_startegy()

    def _get_distributed_strategy(self):
        strategy = None

        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

        dist_strategy = self.context["valid_strategy"]
        k_steps = dist_strategy.a_sync_configs["k_steps"]

        if not dist_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_sync_strategy()

        if dist_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_async_strategy()

        if dist_strategy.a_sync and k_steps > 0:
            strategy = StrategyFactory.create_geo_strategy(k_steps)

        if not strategy:
            raise ValueError("k_steps must be invalid value, please check")

        return strategy

    def build_compiled_startegy(self):
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import CompileTimeStrategy

        compiled_config = CompileTimeStrategy(
            self.origin_main_program, self.origin_main_program,
            self.async_strategy, self.role_maker)
        return compiled_config

    def _load_sparse_params(self, dirname, varnames):
        from paddle.fluid.communicator import LargeScaleKV
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_varname_parts

        scale_kv = LargeScaleKV()
        for varname in varnames:
            origin_varname, _, _ = _get_varname_parts(varname)
            sparse_dir = os.path.join(dirname, origin_varname, varname)
            scale_kv.load(varname, sparse_dir)

    @staticmethod
    def __exclude_vars(exclude_var_names=[]):
        def is_valid(var):
            if var.name in exclude_var_names:
                return False

            from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_varname_parts

            origin_varname, _, _ = _get_varname_parts(var.name)
            if origin_varname.endswith("@GRAD"):
                return False

            if origin_varname == "learning_rate_0":
                return False

            if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                            var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                            var.desc.type() == core.VarDesc.VarType.READER:
                return False
            return var.persistable

        return is_valid

    def _init_worker(self):
        def sync_strategy_envs():
            kwargs = {}
            kwargs["pserver_endpoints"] = self.role_maker.get_pserver_endpoints(
            )
            kwargs["trainer_id"] = self.role_maker.worker_index()
            return kwargs

        def geo_strategy_envs():
            from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames

            def get_sparse_attrs():
                opt_init_map = {}
                opt_init_map["gaussian_random"] = ["seed", "mean", "std"]
                opt_init_map["fill_constant"] = ["value"]
                opt_init_map["uniform_random"] = ["seed", "min", "max"]
                opt_init_map[
                    "truncated_gaussian_random"] = ["seed", "mean", "std"]

                dist_varnames = get_sparse_tablenames(self.origin_main_program,
                                                      True)
                sparse_varnames = get_sparse_tablenames(
                    self.origin_main_program, False)

                if len(dist_varnames) != 0:
                    raise ValueError(
                        "GeoStrategy can not support large scale embeding now, please use fluid.layers.embedding"
                    )

                init_attrs = []
                for value_name in sparse_varnames:
                    value_var = self.origin_main_program.global_block().vars[
                        value_name]
                    value_attr = [
                        value_name,
                        ",".join([str(dim) for dim in value_var.shape])
                    ]
                    for op in self.origin_startup_program.global_block().ops:
                        if op.type in opt_init_map.keys(
                        ) and value_name == op.output("Out")[0]:
                            init_attr = [op.type]
                            for attr in opt_init_map[op.type]:
                                init_attr.append(str(op.attr(attr)))
                            value_attr.append("&".join(init_attr))
                            init_attrs.append(":".join(value_attr))
                            break
                return "#".join(init_attrs)

            kwargs = {}
            kwargs["trainers"] = self.role_maker.worker_num()
            kwargs["sparse_attrs"] = get_sparse_attrs()
            return kwargs

        from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_lr_ops

        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import \
            SyncStrategy, GeoStrategy

        trainer_config = self.async_strategy.get_trainer_runtime_config()
        lrs = _get_lr_ops(self.origin_main_program)

        if len(lrs) > 0:
            kwargs = {"need_global_step": "1"}
        else:
            kwargs = {"need_global_step": "0"}

        if isinstance(self.async_strategy, GeoStrategy):
            geo_kwargs = geo_strategy_envs()
            kwargs.update(geo_kwargs)
        if isinstance(self.async_strategy, SyncStrategy):
            sync_kwargs = sync_strategy_envs()
            kwargs.update(sync_kwargs)

        kwargs = kwargs if kwargs else None

        send_ctx = self.compiled_strategy.get_communicator_send_context()

        if self.compiled_strategy.is_geo_mode():
            recv_ctx = self.compiled_strategy.get_communicator_recv_context(
                recv_type=4)
        else:
            recv_ctx = self.compiled_strategy.get_communicator_recv_context(
                recv_type=1)

        from paddle.fluid.communicator import Communicator
        self._communicator = Communicator(
            trainer_config.mode, kwargs,
            trainer_config.get_communicator_flags())
        self._communicator.init_with_ctx(send_ctx, recv_ctx)

        if not self._communicator.is_running():
            self._communicator.start()
        else:
            warnings.warn("communicator has been initialized, skip")

    def _init_server(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("init server can only accept 1 args: `dirname`")
        elif len(args) == 1:
            model_dirname = args[0]
        else:
            model_dirname = None

        executor = fluid.Executor(fluid.CPUPlace())
        executor.run(fluid.default_startup_program())

        if not model_dirname:
            return

        if not os.path.isdir(model_dirname):
            raise ValueError("There is no directory named '%s'", model_dirname)

        sparse_varnames = self.compiled_strategy.get_sparse_varname_on_ps(True)

        distribtued_varnames = self.compiled_strategy.get_sparse_varname_on_ps(
            False)

        remaining_vars = list(
            filter(
                ParameterServerRuntime.__exclude_vars(sparse_varnames +
                                                      distribtued_varnames),
                fluid.default_main_program().list_vars()))

        fluid.io.load_vars(
            executor,
            main_program=fluid.default_main_program(),
            dirname=model_dirname,
            vars=remaining_vars)

        self._load_sparse_params(
            dirname=model_dirname, varnames=sparse_varnames)

        # todo(tangwei12) load distributed vars
        # self._load_sparse_params(dirname=model_dir, varnames=distribtued_varnames)

    def _run_server(self):
        executor = fluid.Executor(fluid.CPUPlace())
        executor.run(fluid.default_main_program())

    def _stop_worker(self):
        self._communicator.stop()
        executor = fluid.Executor(fluid.CPUPlace())
        executor.close()
