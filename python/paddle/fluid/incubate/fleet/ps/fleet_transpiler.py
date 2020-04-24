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
"""
Convert the fluid program to distributed data-parallelism programs.
"""

import os
import sys
import warnings
from google.protobuf import text_format

from collections import OrderedDict
from .node import DownpourWorker, DownpourServer
from . import ps_pb2 as pslib

from paddle import fluid
from paddle.fluid import core
from paddle.fluid.communicator import Communicator
from paddle.fluid.framework import default_main_program
from paddle.fluid.framework import default_startup_program
from paddle.fluid.framework import Program
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.executor import Executor
from paddle.fluid.parallel_executor import ParallelExecutor
from paddle.fluid.optimizer import Optimizer

from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspiler as OriginTranspiler
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig, ServerRuntimeConfig, \
    DistributedMode

from paddle.fluid.incubate.fleet.base.fleet_base import Fleet
from paddle.fluid.incubate.fleet.base.fleet_base import Mode
from paddle.fluid.incubate.fleet.base.role_maker import MPISymetricRoleMaker

from paddle.fluid.incubate.fleet.ps import version
from paddle.fluid.incubate.fleet.ps.ir.checkport import wait_server_ready
from paddle.fluid.incubate.fleet.ps.distributed_strategy import TrainerRuntimeConfig, DistributedStrategy, \
    SyncStrategy, AsyncStrategy, HalfAsyncStrategy, GeoStrategy, StrategyFactory

from paddle.fluid.incubate.fleet.ps.mode import PSMode
from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer

import paddle.fluid.incubate.fleet.ps.ir.trainer_pass as worker
import paddle.fluid.incubate.fleet.ps.ir.pserver_pass as server
import paddle.fluid.incubate.fleet.ps.ir.public as public


class FleetTranspiler(Fleet):
    """
    A subclass for compatibility with fluid.transpiler.DistributeTranspiler.
    """

    def __init__(self):
        super(FleetTranspiler, self).__init__(Mode.PS)

        self._inner_mode = None

        if version.is_transpiler():
            self._inner_mode = PSMode.TRANSPILER
        else:
            self._inner_mode = PSMode.PSLIB

        self._strategy = None
        self._transpiler = None
        self._origin_main_program = None
        self._origin_startup_program = None
        self._communicator = None
        self.startup_program = None
        self.main_program = None

        self._opt_info = None
        self._local_ip = 0
        self._fleet_ptr = None
        self._main_programs = []
        self._scopes = []
        self._client2client_request_timeout_ms = 500000
        self._client2client_connect_timeout_ms = 10000
        self._client2client_max_retry = 3

    def init(self, role_maker=None):
        if role_maker is None:
            role_maker = MPISymetricRoleMaker()
        super(FleetTranspiler, self).init(role_maker)
        self._fleet_ptr = core.Fleet()

    def _init_transpiler_worker(self):
        """
        `init_worker` has many many functions to do before training,
        first, wait for all parameter servers launch completely.
        second, run executor to initialize startup program
        third, wait for all worker initialize completely.

        Returns:
            None
        """

        def geo_strategy_envs():
            kwargs = {}
            kwargs["push_vars"] = self.vars_info
            kwargs["trainers"] = self.worker_num()
            kwargs["push_nums"] = self._strategy.get_program_config(
            ).geo_sgd_need_push_nums
            return kwargs

        def sync_strategy_envs():
            kwargs = {}
            kwargs[
                "pserver_endpoints"] = self._role_maker.get_pserver_endpoints()
            kwargs["trainer_id"] = self._role_maker.worker_id()
            return kwargs

        # if MPISymetricRoleMaker is defined
        # we suppose a user wants to submit job on mpi cluster
        if isinstance(self._role_maker, MPISymetricRoleMaker):
            # check whether server has been initialized
            wait_server_ready(self.server_endpoints(to_string=False))

        trainer_config = self._strategy.get_trainer_runtime_config()

        print(trainer_config)

        kwargs = None

        if isinstance(self._strategy, GeoStrategy):
            kwargs = geo_strategy_envs()
        if isinstance(self._strategy, GeoStrategy):
            kwargs = sync_strategy_envs()

        if trainer_config.mode in [
                DistributedMode.SYNC, DistributedMode.ASYNC,
                DistributedMode.HALF_ASYNC
        ]:
            send_ctx = fleet.compiled_config.get_communicator_send_context()
            for k, v in send_ctx.items():
                print("k: {}".format(k))

            recv_ctx = fleet.compiled_config.get_communicator_recv_context()
            #recv_ctx = {}
            for k, v in recv_ctx.items():
                print("k: {}".format(k))

            self._communicator = Communicator(
                trainer_config.mode, kwargs,
                trainer_config.get_communicator_flags())
            self._communicator.init_with_ctx(send_ctx, recv_ctx)

        else:
            self._communicator = Communicator(
                trainer_config.mode, kwargs,
                trainer_config.get_communicator_flags())
            self._communicator.init_with_program(self.main_program)

        if not self._communicator.is_running():
            self._communicator.start()
        else:
            warnings.warn("communicator has been initialized, skip")

    def _init_pslib_worker(self):
        if len(self._main_programs) == 0 or not self._opt_info:
            raise ValueError(
                "You should run DistributedOptimizer.minimize() first")

        if "fleet_desc" in self._opt_info:
            self._dist_desc_str = text_format.MessageToString(self._opt_info[
                "fleet_desc"])
            self._dist_desc = self._opt_info["fleet_desc"]
        else:
            raise Exception(
                "You should run DistributedOptimizer.minimize() first")
        # barrier_all for init_server, wait for server starts
        self._role_maker._barrier_all()
        self.all_ips_ = self._role_maker._all_gather(self._local_ip)
        # worker_id * 2 is for compatible with older versions of pslib
        self._fleet_ptr.init_worker(self._dist_desc_str, self.all_ips_,
                                    self._role_maker._get_size(),
                                    self._role_maker.worker_id() * 2)
        # barrier_all for init_worker
        self._role_maker._barrier_all()
        # prepare for client to client communication
        info = self._fleet_ptr.get_clients_info()
        all_info = self._role_maker._worker_gather(info[0])
        self._fleet_ptr.gather_clients(all_info)
        self._fleet_ptr.set_client2client_config(
            self._client2client_request_timeout_ms,
            self._client2client_connect_timeout_ms,
            self._client2client_max_retry)
        self._fleet_ptr.create_client2client_connection()
        # barrier for init model
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            tables = []
            for tp in self._dist_desc.trainer_param:
                for i in tp.dense_table:
                    tables.append(i)
            for prog, scope in zip(self._main_programs, self._scopes):
                prog_id = str(id(prog))
                prog_conf = self._opt_info['program_configs'][prog_id]
                prog_tables = {}
                for key in prog_conf:
                    if "dense" not in key:
                        continue
                    for table_id in prog_conf[key]:
                        prog_tables[int(table_id)] = 0
                for table in tables:
                    if int(table.table_id) not in prog_tables:
                        continue
                    var_name_list = []
                    for i in range(0, len(table.dense_variable_name)):
                        var_name = table.dense_variable_name[i]
                        if scope.find_var(var_name) is None:
                            raise ValueError(
                                "var " + var_name + " not found in scope, " +
                                "you should run startup program first")
                        var_name_list.append(var_name)
                    self._fleet_ptr.init_model(scope,
                                               int(table.table_id),
                                               var_name_list)
        # barrier for init model done
        self._role_maker._barrier_worker()

    def init_worker(self):
        """
        `init_worker` has many many functions to do before training,
        first, wait for all parameter servers launch completely.
        second, run executor to initialize startup program
        third, wait for all worker initialize completely.

        Returns:
            None
        """
        if self._inner_mode == PSMode.TRANSPILER:
            self._init_transpiler_worker()
        else:
            self._init_pslib_worker()

    def _init_transpiler_server(self, model_dir=None):
        if not self.startup_program:
            raise ValueError(
                "startup_program is None, need invoke DistributedOptimizer.minimize first"
            )

        self._executor.run(self.startup_program)

        if model_dir:
            if not os.path.isdir(model_dir):
                raise ValueError("There is no directory named '%s'", model_dir)

            fluid.io.load_persistables(self._executor, model_dir,
                                       self.main_program)

    def _init_pslib_server(self, model_dir=None, **kwargs):
        mode = kwargs.get("mode", 0)
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            self._fleet_ptr.load_model(model_dir, mode)
        self._role_maker._barrier_worker()

    def init_server(self, model_dir=None, **kwargs):
        """
        `init_server` has many many functions to do before start pserver,
        first, run executor to initialize startup program,
        second, if the `model_dir` is not empty, it will load parameters from it for increment training.

        Args:
            model_dir(str): The directory path.

        Returns:
            None
        """

        if self._inner_mode == PSMode.TRANSPILER:
            self._init_transpiler_server(model_dir)
        else:
            self._init_pslib_server(model_dir, **kwargs)

    def run_server(self):
        """
        `run_server` execute executor to start pserver main program.

        Returns:
            None
        """

        if self._inner_mode == PSMode.TRANSPILER:
            if not self.main_program:
                raise ValueError(
                    "main_program is None, need invoke DistributedOptimizer.minimize first"
                )

            self._executor.run(self.main_program)
        else:
            """
             init_pserver(): will be called by user. When a user knows current process is_worker(), he/she
                 should call init_pserver() to initialize global information about parameter server
            """
            if self._opt_info:
                if "fleet_desc" in self._opt_info:
                    self._dist_desc_str = text_format.MessageToString(
                        self._opt_info["fleet_desc"])
                    self._dist_desc = self._opt_info["fleet_desc"]
                else:
                    raise Exception(
                        "You should run DistributedOptimizer.minimize() first")
                # server_id * 2 is for compatible with older versions of pslib
                self._fleet_ptr.init_server(self._dist_desc_str,
                                            self._role_maker.server_id() * 2)
                if isinstance(self._role_maker, MPISymetricRoleMaker):
                    self._local_ip = self._fleet_ptr.run_server()
                else:
                    local_endpoint = self._role_maker.get_local_endpoint()
                    local_endpoint = local_endpoint.split(":")
                    self._local_ip = self._fleet_ptr.run_server(
                        str(local_endpoint[0]), int(local_endpoint[1]))

                # barrier_all for init_server
                self._role_maker._barrier_all()
                self.all_ips_ = self._role_maker._all_gather(self._local_ip)

                self._fleet_ptr.gather_servers(self.all_ips_,
                                               self._role_maker._get_size())
                # barrier_all for init_worker, wait all workers start
                self._role_maker._barrier_all()
            else:
                raise Exception(
                    "You should run DistributedOptimizer.minimize() first")

    def stop_worker(self):
        """
        Close this executor.

        For the distributed training, this method would free the resource on PServers related to
        the current Trainer.

        Returns:
            None
        """

        if self._inner_mode == PSMode.TRANSPILER:
            self._communicator.stop()
            if isinstance(self._role_maker, MPISymetricRoleMaker):
                self._role_maker._finalize()
            self._executor.close()
        else:
            self._role_maker._barrier_worker()
            # all worker should be finalize first
            if self._role_maker.is_worker():
                self._fleet_ptr.finalize_worker()
            self._role_maker._barrier_worker()
            if self._role_maker.is_first_worker():
                self._fleet_ptr.stop_server()
            self._role_maker._barrier_worker()
            self._role_maker._barrier_all()
            self._role_maker._finalize()

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributeTranspilerConfig): Extra properties for distributed optimizer.

        Returns:
            TranspilerOptimizer: subclass of DistributedOptimizer.
        """

        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of Optimizer")
        if not self._is_initialized:
            raise ValueError(
                "fleet.init(role) to initialize before optimizer.minimize(loss)")

        _strategy = strategy

        if not strategy:
            _strategy = StrategyFactory.create_async_strategy()

        if isinstance(strategy, DistributedStrategy):
            _strategy = strategy
        elif isinstance(strategy, DistributeTranspilerConfig):
            if strategy.sync_mode:
                _strategy = SyncStrategy()
            else:
                if strategy.runtime_split_send_recv:
                    if strategy.geo_sgd_mode:
                        _strategy = GeoStrategy(strategy.geo_sgd_need_push_nums)
                    elif strategy.half_async:
                        _strategy = HalfAsyncStrategy()
                    else:
                        _strategy = AsyncStrategy()
                else:
                    _strategy = HalfAsyncStrategy()
                    # for half_async compatibility
                    strategy.half_async = True
                    strategy.runtime_split_send_recv = True
            self._strategy.set_program_config(strategy)
        elif isinstance(strategy, dict):
            if self._inner_mode != PSMode.PSLIB:
                raise TypeError("Dict strategy can only be used at PSLIB Mode")

            _strategy = StrategyFactory.create_async_strategy()
            _strategy.set_pslib_runtime_config(strategy)
        else:
            raise TypeError(
                "strategy must be an instance of DistributeTranspilerConfig, DistributedStrategy"
            )

        self._strategy = _strategy
        self._optimizer = ParameterServerOptimizer(optimizer, _strategy)
        return self._optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        """
        Prune the given `main_program` to build a new program especially for inference,
        and then save it and all related parameters to given `dirname` by the `executor`.
        """

        if self._inner_mode == PSMode.PSLIB:
            self._fleet_ptr.save_model(dirname, 0)
            return

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save_inference_model() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save_inference_model() function, executor must be as Executor type"
            )

        if main_program is not None:
            if isinstance(main_program, CompiledProgram):
                raise TypeError(
                    "in fleet.save_inference_model() function, main_program must be as Program type, CompiledProgram is not allowed"
                )
            fluid.io.save_inference_model(dirname, feeded_var_names,
                                          target_vars, executor, main_program,
                                          None, None, export_for_deployment)
        else:
            fluid.io.save_inference_model(dirname, feeded_var_names,
                                          target_vars, executor,
                                          self._origin_main_program, None, None,
                                          export_for_deployment, True)

            model_basename = "__model__"
            model_filename = os.path.join(dirname, model_basename)

            with open(model_filename, "rb") as f:
                program_desc_str = f.read()

            program = Program.parse_from_string(program_desc_str)
            program._copy_dist_param_info_from(self.main_program)
            self.save_persistables(executor, dirname, program)

    def save_persistables(self, executor, dirname, main_program=None, **kwargs):
        """
        This function filters out all variables with `persistable==True` from the
        give `main_program` and then saves these variables to the folder `dirname`
        or file `filename`.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None; if you would like to save all variables in a
        single file, use `filename` to specify the file name.
        """

        if self._inner_mode == PSMode.PSLIB:
            mode = kwargs.get("mode", 0)
            self._fleet_ptr.client_flush()
            self._role_maker._barrier_worker()
            if self._role_maker.is_first_worker():
                self._fleet_ptr.save_model(dirname, mode)
            self._role_maker._barrier_worker()
            return

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save_persistables() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save_persistables() function, executor must be as Executor type"
            )

        if main_program is None:
            main_program = self.main_program

        if isinstance(main_program, CompiledProgram):
            raise TypeError(
                "in fleet.save_persistables() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        if not main_program._is_distributed:
            raise ValueError(
                "main_program is for local, may not use fleet.save_persistables")

        fluid.io.save_persistables(executor, dirname, main_program, None)

    def _set_opt_info(self, opt_info):
        """
        this function saves the result from DistributedOptimizer.minimize()
        """
        self._opt_info = opt_info

    def save_cache_model(self, executor, dirname, main_program=None, **kwargs):
        """
        save sparse cache table,
        when using fleet, it will save sparse cache table

        Args:
            executor(Executor): fluid executor
            dirname(str): save path. It can be hdfs/afs path or local path
            main_program(Program): fluid program, default None
            kwargs: use define property, current support following
                mode(int): define for feature extension in the future,
                           currently no use, will pass a default value 0
                table_id(int): which table to save cache, default is 0

        Returns:
            feasign_num(int): cache feasign num

        Example:
            .. code-block:: python

              fleet.save_cache_model(None, dirname="/you/path/to/model", mode = 0)

        """
        mode = kwargs.get("mode", 0)
        table_id = kwargs.get("table_id", 0)
        self._fleet_ptr.client_flush()
        self._role_maker._barrier_worker()
        cache_threshold = 0.0

        if self._role_maker.is_first_worker():
            cache_threshold = self._fleet_ptr.get_cache_threshold(table_id)
        # check cache threshold right or not
        self._role_maker._barrier_worker()

        if self._role_maker.is_first_worker():
            self._fleet_ptr.cache_shuffle(table_id, dirname, mode,
                                          cache_threshold)

        self._role_maker._barrier_worker()

        feasign_num = -1
        if self._role_maker.is_first_worker():
            feasign_num = self._fleet_ptr.save_cache(table_id, dirname, mode)

        self._role_maker._barrier_worker()
        return feasign_num

    def shrink_sparse_table(self):
        """
        shrink cvm of all sparse embedding in pserver, the decay rate
        is defined as "show_click_decay_rate" in fleet_desc.prototxt

        Example:
            >>> fleet.shrink_sparse_table()

        """
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            tables = []
            for tp in self._opt_info["fleet_desc"].trainer_param:
                for i in tp.sparse_table:
                    tables.append(i.table_id)
            for i in list(set(tables)):
                self._fleet_ptr.shrink_sparse_table(i)
        self._role_maker._barrier_worker()

    def shrink_dense_table(self, decay, emb_dim=11, scope=None, table_id=None):
        """
        shrink batch_sum in pserver by multiplying by decay

        Args:
            decay(float): the decay rate, usually range in (0, 1)
            emb_dim(int): one element's length in datanorm layer
            scope(Scope): Scope object, default is fluid.global_scope()
            table_id(int): table id of shrinking dense table. None means shrink all,
                           you should specify it when using multiple scopes,
                           default is None.

        Example:
            >>> fleet.shrink_dense_table(0.98, 11, myscope1, 1)
            >>> fleet.shrink_dense_table(0.98, 11, myscope1, 2)
            >>> fleet.shrink_dense_table(0.98, 11, myscope2, 3)

        """
        if scope is None:
            scope = fluid.global_scope()
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            for tp in self._opt_info["fleet_desc"].trainer_param:
                for i in tp.dense_table:
                    if table_id is not None and table_id != i.table_id:
                        continue
                    var_list = [var for var in i.dense_variable_name]
                    skip = False
                    for var in var_list:
                        if scope.find_var(var) is None:
                            skip = True
                            break
                    if skip:
                        continue
                    self._fleet_ptr.shrink_dense_table(i.table_id, scope,
                                                       var_list, decay, emb_dim)
        self._role_maker._barrier_worker()

    def clear_model(self):
        """
        clear_model() will be called by user. It will clear sparse model.

        Examples:
            .. code-block:: python

              fleet.clear_model()

        """
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            self._fleet_ptr.clear_model()
        self._role_maker._barrier_worker()

    def load_one_table(self, table_id, model_path, **kwargs):
        """
        load pslib model for one table or load params from paddle model

        Args:
            table_id(int): load table id
            model_path(str): load model path, can be local or hdfs/afs path
            kwargs(dict): user defined params, currently support following:
                only for load pslib model for one table:
                    mode(int): load model mode. 0 is for load whole model, 1 is
                               for load delta model (load diff), default is 0.
                only for load params from paddle model:
                    scope(Scope): Scope object
                    model_proto_file(str): path of program desc proto binary
                                           file, can be local or hdfs/afs file
                    var_names(list): var name list
                    load_combine(bool): load from a file or split param files
                                        default False.

        Examples:
            .. code-block:: python

              # load pslib model for one table
              fleet.load_one_table(0, "hdfs:/my_fleet_model/20190714/0/")
              fleet.load_one_table(1, "hdfs:/xx/xxx", mode = 0)

              # load params from paddle model
              fleet.load_one_table(2, "hdfs:/my_paddle_model/",
                                   scope = my_scope,
                                   model_proto_file = "./my_program.bin",
                                   load_combine = False)

              # below is how to save proto binary file
              with open("my_program.bin", "wb") as fout:
                  my_program = fluid.default_main_program()
                  fout.write(my_program.desc.serialize_to_string())

        """
        self._role_maker._barrier_worker()
        mode = kwargs.get("mode", 0)
        scope = kwargs.get("scope", None)
        model_proto_file = kwargs.get("model_proto_file", None)
        var_names = kwargs.get("var_names", None)
        load_combine = kwargs.get("load_combine", False)
        self._role_maker._barrier_worker()
        if scope is not None and model_proto_file is not None:
            self._load_one_table_from_paddle_model(scope, table_id, model_path,
                                                   model_proto_file, var_names,
                                                   load_combine)
        elif self._role_maker.is_first_worker():
            self._fleet_ptr.load_model_one_table(table_id, model_path, mode)
        self._role_maker._barrier_worker()

    def _load_one_table_from_paddle_model(self,
                                          scope,
                                          table_id,
                                          model_path,
                                          model_proto_file,
                                          var_names=None,
                                          load_combine=False):
        """
        load params from paddle model, and push params to pserver

        Args:
            scope(Scope): Scope object
            table_id(int): the id of table to load
            model_path(str): path of paddle model, can be local or hdfs/afs file
            model_proto_file(str): path of program desc proto binary file,
                                   can be local or hdfs/afs file
            var_names(list): load var names
            load_combine(bool): load from a file or split param files

        """
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            # get fs config from fleet_desc
            fs_name = self._opt_info["fleet_desc"].fs_client_param.uri
            fs_ugi = self._opt_info["fleet_desc"].fs_client_param.user + "," + \
                     self._opt_info["fleet_desc"].fs_client_param.passwd
            hadoop_bin = self._opt_info["fleet_desc"].fs_client_param.hadoop_bin
            # download model_path if it's hdfs/afs
            if model_path.startswith("hdfs:") or model_path.startswith("afs:"):
                dest = "./model_for_load_table_%s" % table_id
                cmd = hadoop_bin + " fs -D fs.default.name=" + fs_name + \
                      " -D hadoop.job.ugi=" + fs_ugi + " -get " + model_path + \
                      " " + dest
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError("download model failed")
                model_path = dest
            # download model_proto_file if it's hdfs/afs
            if model_proto_file.startswith("hdfs:") or \
                    model_proto_file.startswith("afs:"):
                dest = "./model_proto_file_for_load_table_%s" % table_id
                cmd = hadoop_bin + " fs -D fs.default.name=" + fs_name + \
                      " -D hadoop.job.ugi=" + fs_ugi + " -get " + \
                      model_proto_file + " " + dest
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError("download model proto file failed")
                model_proto_file = dest
            for tp in self._opt_info["fleet_desc"].trainer_param:
                for i in tp.dense_table:
                    if table_id is not None and table_id != i.table_id:
                        continue
                    table_var_names = [var for var in i.dense_variable_name]
                    skip = False
                    for var in table_var_names:
                        if scope.find_var(var) is None:
                            skip = True
                            break
                    if skip:
                        continue
                    self._fleet_ptr.load_from_paddle_model(
                        scope, table_id, var_names, model_path,
                        model_proto_file, table_var_names, load_combine)
        self._role_maker._barrier_worker()


# fleet is a global instance for parameter server.
fleet = FleetTranspiler()


class ParameterServerOptimizer(DistributedOptimizer):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    Fleet() instance who holds the global information about current distributed
    training.

    Args:
        optimizer(Optimizer): subclass of Optimizer.
        strategy(DistributeTranspilerConfig): instance of DistributeTranspilerConfig.

    Returns:
        None
    """

    def __init__(self, optimizer, strategy, mode=PSMode.TRANSPILER):
        super(ParameterServerOptimizer, self).__init__(optimizer, strategy)
        self._mode = mode
        if self._mode == PSMode.PSLIB:
            self._optimizer_name = "Distributed%s" % optimizer.type.capitalize()
            if optimizer.type != "adam":
                print("Currently, distributed optimizer only support Adam"
                      "Will config built-in adam for you."
                      "We will support more functions in DistributedOptimizer",
                      sys.stderr)
                self._optimizer_name = "DistributedAdam"

            self._optimizer = globals()[self._optimizer_name](optimizer)
        else:
            self._optimizer = optimizer

        self._window = 1
        self.type = "downpour"
        self.data_norm_name = [
            ".batch_size", ".batch_square_sum", ".batch_sum",
            ".batch_size@GRAD", ".batch_square_sum@GRAD", ".batch_sum@GRAD"
        ]

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        raise NotImplementedError()

    def apply_gradients(self, params_grads):
        raise NotImplementedError()

    def _build_trainer_programs(self, compiled_config):
        _main = fleet._origin_main_program.clone()
        _startup = fleet._origin_startup_program.clone()

        # for main program
        _main = worker.delete_optimizer_pass(_main, compiled_config)
        _main = worker.distributed_ops_pass(_main, compiled_config)
        _main = worker.append_send_ops_pass(_main, compiled_config)

        # for startup program
        _startup = worker.fake_init_ops_pass(_startup, compiled_config)
        _startup = worker.init_from_server_pass(_startup, compiled_config)
        _startup = worker.delet_extra_optimizes_pass(_startup, compiled_config)

        return _main, _startup

    def _build_pserver_programs(self, compiled_config):
        _main = fluid.Program()
        _startup = fluid.Program()

        _main = server.add_listen_and_serv_pass(_main, compiled_config)
        _main = server.add_rpc_global_flags_pass(_main, compiled_config)
        _main = server.add_optimizer_pass(_main, compiled_config)
        _main = server.add_recv_inputs_pass(_main, compiled_config)

        _startup = server.build_pserver_startup_program_pass(_startup, _main,
                                                             compiled_config)

        with open("_main.po", "w") as wb:
            wb.write(str(_main))

        with open("_startup.po", "w") as wb:
            wb.write(str(_startup))

        return _main, _startup

    def minimize(self,
                 losses,
                 scopes=None,
                 startup_programs=None,
                 parameter_list=None,
                 no_grad_set=None):

        if isinstance(losses, list):
            raise ValueError("need implement later")

        self._optimizer.minimize(losses, startup_programs, parameter_list,
                                 no_grad_set)

        fleet._origin_main_program = default_main_program().clone(
            for_test=False)
        fleet._origin_startup_program = default_startup_program().clone(
            for_test=False)

        compiled_config = public.CompileTimeStrategy(
            fleet._origin_main_program, fleet._origin_startup_program,
            self._strategy, fleet._role_maker)

        fleet.compiled_config = compiled_config
        fleet.main_program, fleet.startup_program = \
            self._build_trainer_programs(compiled_config) if fleet.is_worker() \
                else self._build_pserver_programs(compiled_config)
