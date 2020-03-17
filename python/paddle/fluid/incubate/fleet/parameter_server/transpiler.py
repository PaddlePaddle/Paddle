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

import paddle.fluid.io as io
import paddle.fluid as fluid
import paddle.fluid.core as core
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

from paddle.fluid.incubate.fleet.parameter_server import version
from paddle.fluid.incubate.fleet.parameter_server.distributed_strategy import TrainerRuntimeConfig, DistributedStrategy, \
    SyncStrategy, AsyncStrategy, HalfAsyncStrategy, GeoStrategy, StrategyFactory

from paddle.fluid.incubate.fleet.parameter_server.mode import PSMode
from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer


class FleetTranspiler(Fleet):
    """
    A subclass for compatibility with fluid.transpiler.DistributeTranspiler.
    """

    def __init__(self):
        super(FleetTranspiler, self).__init__(Mode.PS)

        self._inner_mode = PSMode.TRAINSPILER if version.is_transpiler(
        ) else PSMode.PSLIB

        self._strategy = None
        self._transpiler = None
        self._origin_program = None
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
            kwargs["trainer_id"] = self._role_maker.worker_index()
            return kwargs

        # if MPISymetricRoleMaker is defined
        # we suppose a user wants to submit job on mpi cluster
        if isinstance(self._role_maker, MPISymetricRoleMaker):
            # check whether server has been initialized
            from paddle.fluid.transpiler.details.checkport import wait_server_ready
            wait_server_ready(self.server_endpoints(to_string=False))

        trainer_config = self._strategy.get_trainer_runtime_config()

        print(trainer_config)

        kwargs = None

        if isinstance(self._strategy, GeoStrategy):
            kwargs = geo_strategy_envs()
        if isinstance(self._strategy, GeoStrategy):
            kwargs = sync_strategy_envs()

        self._communicator = Communicator(
            self.main_program, trainer_config.mode, kwargs,
            trainer_config.get_communicator_flags())

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
        # worker_index * 2 is for compatible with older versions of pslib
        self._fleet_ptr.init_worker(self._dist_desc_str, self.all_ips_,
                                    self._role_maker._get_size(),
                                    self._role_maker.worker_index() * 2)
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
        if self._inner_mode == PSMode.TRAINSPILER:
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

            io.load_persistables(self._executor, model_dir, self.main_program)

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

        if self._inner_mode == PSMode.TRAINSPILER:
            self._init_transpiler_server(model_dir)
        else:
            self._init_pslib_server(model_dir, **kwargs)

    def run_server(self):
        """
        `run_server` execute executor to start pserver main program.

        Returns:
            None
        """

        if self._inner_mode == PSMode.TRAINSPILER:
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
                # server_index * 2 is for compatible with older versions of pslib
                self._fleet_ptr.init_server(self._dist_desc_str,
                                            self._role_maker.server_index() * 2)
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

        if self._inner_mode == PSMode.TRAINSPILER:
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
                "strategy must be an instance of DistributeTranspilerConfig, SyncStrategy, HalfAsyncStrategy, AsyncStrategy or GeoStratey."
            )

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
            io.save_inference_model(dirname, feeded_var_names, target_vars,
                                    executor, main_program, None, None,
                                    export_for_deployment)
        else:
            io.save_inference_model(dirname, feeded_var_names, target_vars,
                                    executor, self._origin_program, None, None,
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

        io.save_persistables(executor, dirname, main_program, None)

    def _transpile(self, config):
        program_config = self._strategy.get_program_config()

        # _origin_program is a deep copy for default_main_program, for inference
        self._origin_program = default_main_program().clone(for_test=False)

        if program_config.geo_sgd_mode:
            from paddle.fluid.transpiler.geo_sgd_transpiler import GeoSgdTranspiler
            self._transpiler = GeoSgdTranspiler(program_config)
        else:
            self._transpiler = OriginTranspiler(program_config)
        self._transpiler._set_server_config(
            self._strategy.get_server_runtime_config())

        if self.is_worker():
            self._transpiler.transpile(
                trainer_id=fleet.worker_index(),
                pservers=fleet.server_endpoints(to_string=True),
                trainers=fleet.worker_num(),
                sync_mode=program_config.sync_mode)

            if isinstance(self._role_maker, MPISymetricRoleMaker):
                program_config.wait_port = False
                self._strategy.set_program_config(program_config)

            self.main_program = self._transpiler.get_trainer_program(
                wait_port=program_config.wait_port)
            self.startup_program = default_startup_program()
            if program_config.geo_sgd_mode:
                self.vars_info = self._transpiler._get_vars_info()
                self.startup_program = self._transpiler.trainer_startup_program
        else:
            self._transpiler.transpile(
                trainer_id=fleet.worker_index(),
                pservers=fleet.server_endpoints(to_string=True),
                trainers=fleet.worker_num(),
                sync_mode=program_config.sync_mode,
                current_endpoint=self.server_endpoints()[self.server_index()])
            self.main_program, self.startup_program = \
                self._transpiler.get_pserver_programs(
                    self.server_endpoints()[self.server_index()])

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

    def __init__(self, optimizer, strategy, mode=PSMode.TRAINSPILER):
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

        self._learning_rate = optimizer._learning_rate

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

    def minimize(self,
                 losses,
                 scopes=None,
                 startup_programs=None,
                 parameter_list=None,
                 no_grad_set=None):

        if not isinstance(losses, list):
            losses = [losses]

        optimize_ops, param_grads, opt_info = \
            self._optimizer._minimize(
                losses,
                startup_programs,
                parameter_list,
                no_grad_set,
                self._strategy)
        opt_info["mpi_rank"] = fleet.worker_index()
        opt_info["mpi_size"] = fleet.worker_num()

        fleet._set_opt_info(opt_info)

        programs = [loss.block.program for loss in losses]

        if scopes is None:
            scopes = [fluid.global_scope()] * len(programs)

        if len(scopes) != len(programs):
            raise ValueError(
                "You should make sure len(scopes) == len(programs) or set scopes None"
            )

        fleet._main_programs = programs
        fleet._scopes = scopes

        return [optimize_ops, param_grads]

    def _find_distributed_lookup_table_inputs(self, program, table_names):
        """
        Find input variable of distribute lookup table in program.
        We could support multi-distribute table now.
        Args:
            program(Program): given program, locate distributed lookup table
            table_name(str): given table names that is found beforehand
        Returns:
            inputs
        """
        local_vars = program.current_block().vars
        inputs_dict = dict()
        for table_name in table_names:
            inputs_dict[table_name] = []

        for op in program.global_block().ops:
            if op.type == "lookup_table":
                if op.input("W")[0] in table_names:
                    inputs_dict[op.input("W")[0]].extend(
                        [local_vars[name] for name in op.input("Ids")])
        return inputs_dict

    def _find_distributed_lookup_table_outputs(self, program, table_names):
        """
        Find output variable of distribute lookup table in program.
        We could support multi-distribute table now.
        Args:
            programs(Program): given program, locate distributed lookup table
            table_name(str): given table name that is found beforehand
        Returns:
            outputs
        """
        local_vars = program.current_block().vars
        outputs_dict = dict()
        for table_name in table_names:
            outputs_dict[table_name] = []

        for op in program.global_block().ops:
            if op.type == "lookup_table":
                if op.input("W")[0] in table_names:
                    outputs_dict[op.input("W")[0]].extend(
                        [local_vars[name] for name in op.output("Out")])
        return outputs_dict

    def _find_distributed_lookup_table_grads(self, program, table_names):
        local_vars = program.current_block().vars
        grads_dict = dict()
        for table_name in table_names:
            grads_dict[table_name] = []

        for op in program.global_block().ops:
            if op.type == "lookup_table_grad" and op.input("W")[
                    0] in table_names:
                grads_dict[op.input("W")[0]].extend(
                    [local_vars[name] for name in op.input("Out@GRAD")])
        return grads_dict

    def _find_multi_distributed_lookup_table(self, losses):
        """
        find multi-sparse-table
        """
        table_names = set()
        cnt = 0
        tmp_list = []
        ret_list = []
        for loss in losses:
            for op in loss.block.program.global_block().ops:
                if op.type == "lookup_table":
                    if op.attr('is_distributed') is True:
                        table_name = op.input("W")[0]
                        if table_name not in table_names:
                            table_names.add(table_name)
                            tmp_list.append([table_name, cnt])
                            cnt += 1
        tmp_list.sort(key=lambda k: k[1])
        for x in tmp_list:
            ret_list.append(x[0])
        return ret_list

    def _minimize(self,
                  losses,
                  startup_program=None,
                  parameter_list=None,
                  no_grad_set=None,
                  strategy={}):
        """
        DownpounSGD is a distributed optimizer so
        that user can call minimize to generate backward
        operators and optimization operators within minimize function
        Args:
            loss(Variable): loss variable defined by user
            startup_program(Program): startup program that defined by user
            parameter_list(str list): parameter names defined by users
            no_grad_set(set): a set of variables that is defined by users
            so that these variables do not need gradient computation
            strategy(dict): user-defined properties
        Returns:
            [optimize_ops, grads_and_weights]
        """
        # sparse table names of each program
        prog_id_to_sparse_table = OrderedDict()
        # inputs_dict and outputs_dict of sparse tables of each program
        prog_id_to_inputs_dict = OrderedDict()
        prog_id_to_outputs_dict = OrderedDict()
        # related to PSParameter
        ps_param = pslib.PSParameter()
        # related to ServerParameter
        server = DownpourServer()
        # program to worker (related to DownpourTrainerParameter)
        prog_id_to_worker = OrderedDict()
        # param_grads of each program
        prog_id_to_param_grads = OrderedDict()
        # sparse_grads of each program
        prog_id_to_sparse_grads = OrderedDict()
        # unique program set
        program_id_set = set()

        sparse_table_to_index = OrderedDict()
        sparse_table_index = 0
        for loss in losses:
            prog_id = str(id(loss.block.program))
            # param_grads of program
            params_grads = sorted(
                fluid.backward.append_backward(loss, parameter_list,
                                               no_grad_set),
                key=lambda x: x[0].name)

            if prog_id not in program_id_set:
                program_id_set.add(prog_id)
                sparse_table = self._find_multi_distributed_lookup_table([loss])
                prog_id_to_sparse_table[prog_id] = sparse_table

                # get sparse_table_to_index
                for tn in sparse_table:
                    if sparse_table_to_index.get(tn) is None:
                        sparse_table_to_index[tn] = sparse_table_index
                        sparse_table_index += 1

                # get inputs_dict
                inputs_dict = self._find_distributed_lookup_table_inputs(
                    loss.block.program, sparse_table)
                prog_id_to_inputs_dict[prog_id] = inputs_dict
                # get outputs_dict
                outputs_dict = self._find_distributed_lookup_table_outputs(
                    loss.block.program, sparse_table)
                prog_id_to_outputs_dict[prog_id] = outputs_dict

                prog_id_to_worker[prog_id] = DownpourWorker(self._window)

                grads_dict = self._find_distributed_lookup_table_grads(
                    loss.block.program, sparse_table)
                prog_id_to_sparse_grads[prog_id] = grads_dict

            if prog_id not in prog_id_to_param_grads:
                prog_id_to_param_grads[prog_id] = []
            prog_id_to_param_grads[prog_id].append(params_grads)

        # if strategy.get("parallel_compute")

        # if user specify a fleet_desc.prototxt file, then load the file
        # instead of creating default fleet_desc.prototxt.
        # user can specify server_param or trainer_param or fs_client_param.
        if strategy.get("fleet_desc_file") is not None:
            fleet_desc_file = strategy["fleet_desc_file"]
            with open(fleet_desc_file) as f:
                text_format.Merge(f.read(), ps_param)
            server.get_desc().CopyFrom(ps_param.server_param)
            if len(ps_param.trainer_param) == 1:
                for k in prog_id_to_worker:
                    prog_id_to_worker[k].get_desc().CopyFrom(
                        ps_param.trainer_param[0])
            else:
                if len(ps_param.trainer_param) != len(prog_id_to_worker):
                    raise ValueError(
                        "trainer param size != program size, %s vs %s" %
                        (len(ps_param.trainer_param), len(prog_id_to_worker)))
                idx = 0
                # prog_id_to_worker is OrderedDict
                for k in prog_id_to_worker:
                    prog_id_to_worker[k].get_desc().CopyFrom(
                        ps_param.trainer_param[idx])
                    idx += 1

        # ServerParameter add all sparse tables
        for tn in sparse_table_to_index:
            sparse_table_index = sparse_table_to_index[tn]
            if strategy.get(tn) is not None:
                server.add_sparse_table(sparse_table_index, strategy[tn])
            else:
                server.add_sparse_table(sparse_table_index, None)

        # each DownpourTrainerParameter add its own sparse tables
        program_id_set.clear()
        for loss in losses:
            prog_id = str(id(loss.block.program))
            if prog_id not in program_id_set:
                program_id_set.add(prog_id)
                worker = prog_id_to_worker[prog_id]
                inputs_dict = prog_id_to_inputs_dict[prog_id]
                outputs_dict = prog_id_to_outputs_dict[prog_id]
                for tn in prog_id_to_sparse_table[prog_id]:
                    sparse_table_index = sparse_table_to_index[tn]
                    grads_dict = prog_id_to_sparse_grads[prog_id]
                    worker.add_sparse_table(sparse_table_index, inputs_dict[tn],
                                            outputs_dict[tn], grads_dict[tn])

        dense_start_table_id = len(sparse_table_to_index)
        dense_table_index = len(sparse_table_to_index)
        program_configs = {}
        # ServerParameter add all dense tables
        # each DownpourTrainerParameter add its own dense tables
        program_id_set.clear()
        for loss_index in range(len(losses)):
            program_id = str(id(losses[loss_index].block.program))
            if program_id not in program_id_set:
                program_id_set.add(program_id)
                worker = prog_id_to_worker[program_id]
                sparse_table_names = prog_id_to_sparse_table[program_id]
                sparse_table_index = \
                    [sparse_table_to_index[i] for i in sparse_table_names]

                program_configs[program_id] = {
                    "pull_sparse": [t_index for t_index in sparse_table_index],
                    "push_sparse": [t_index for t_index in sparse_table_index]
                }

                params_grads = prog_id_to_param_grads[program_id]
                for pg in params_grads:
                    params = []
                    grads = []
                    data_norm_params = []
                    data_norm_grads = []
                    for i in pg:
                        is_data_norm_data = False
                        for data_norm_name in self.data_norm_name:
                            if i[0].name.endswith(data_norm_name):
                                is_data_norm_data = True
                                data_norm_params.append(i[0])
                        if not is_data_norm_data:
                            params.append(i[0])

                    for i in pg:
                        is_data_norm_data = False
                        for data_norm_grad in self.data_norm_name:
                            if i[0].name.endswith(data_norm_grad):
                                is_data_norm_data = True
                                data_norm_grads.append(i[1])
                        if not is_data_norm_data:
                            grads.append(i[1])

                    if strategy.get('dense_table') is not None:
                        server.add_dense_table(dense_table_index, params, grads,
                                               strategy['dense_table'],
                                               sparse_table_names)
                    else:
                        server.add_dense_table(dense_table_index, params, grads,
                                               None, sparse_table_names)
                    worker.add_dense_table(
                        dense_table_index, self._learning_rate, params, grads,
                        dense_start_table_id, sparse_table_names)
                    if "pull_dense" in program_configs[
                            program_id] and "push_dense" in program_configs[
                                program_id] and len(program_configs[program_id][
                                    "pull_dense"]) > 0:
                        program_configs[program_id]["pull_dense"].extend(
                            [dense_table_index])
                        program_configs[program_id]["push_dense"].extend(
                            [dense_table_index])
                    else:
                        program_configs[program_id][
                            "pull_dense"] = [dense_table_index]
                        program_configs[program_id][
                            "push_dense"] = [dense_table_index]
                    if len(data_norm_params) != 0 and len(data_norm_grads) != 0:
                        dense_table_index += 1
                        if strategy.get('datanorm_table') is not None:
                            server.add_data_norm_table(
                                dense_table_index, self._learning_rate,
                                data_norm_params, data_norm_grads,
                                strategy['datanorm_table'], sparse_table_names)
                        else:
                            server.add_data_norm_table(
                                dense_table_index, self._learning_rate,
                                data_norm_params, data_norm_grads, None,
                                sparse_table_names)

                        worker.add_dense_table(
                            dense_table_index, self._learning_rate,
                            data_norm_params, data_norm_grads,
                            dense_start_table_id, sparse_table_names)
                    program_configs[program_id]["pull_dense"].extend(
                        [dense_table_index])
                    program_configs[program_id]["push_dense"].extend(
                        [dense_table_index])
                    dense_table_index += 1

            # Todo(guru4elephant): figure out how to support more sparse parameters
            # currently only support lookup_table
            worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
            if len(worker.get_desc().skip_op) == 0:
                worker.get_desc().skip_op.extend(worker_skipped_ops)

        ps_param.server_param.CopyFrom(server.get_desc())
        # prog_id_to_worker is OrderedDict
        if len(ps_param.trainer_param) == 0:
            for k in prog_id_to_worker:
                tp = ps_param.trainer_param.add()
                tp.CopyFrom(prog_id_to_worker[k].get_desc())

        if strategy.get("fs_uri") is not None:
            ps_param.fs_client_param.uri = strategy["fs_uri"]
        elif ps_param.fs_client_param.uri == "":
            ps_param.fs_client_param.uri = "hdfs://your_hdfs_uri"
        if strategy.get("fs_user") is not None:
            ps_param.fs_client_param.user = strategy["fs_user"]
        elif ps_param.fs_client_param.user == "":
            ps_param.fs_client_param.user = "your_hdfs_user"
        if strategy.get("fs_passwd") is not None:
            ps_param.fs_client_param.passwd = strategy["fs_passwd"]
        elif ps_param.fs_client_param.passwd == "":
            ps_param.fs_client_param.passwd = "your_hdfs_passwd"
        if strategy.get("fs_hadoop_bin") is not None:
            ps_param.fs_client_param.hadoop_bin = strategy["fs_hadoop_bin"]
        elif ps_param.fs_client_param.hadoop_bin == "":
            ps_param.fs_client_param.hadoop_bin = "$HADOOP_HOME/bin/hadoop"

        opt_info = {}
        opt_info["program_id_to_worker"] = prog_id_to_worker
        opt_info["program_configs"] = program_configs
        opt_info["trainer"] = "DistMultiTrainer"
        opt_info["device_worker"] = strategy.get("device_worker", "DownpourSGD")
        opt_info["optimizer"] = "DownpourSGD"
        opt_info["fleet_desc"] = ps_param
        opt_info["worker_skipped_ops"] = worker_skipped_ops
        opt_info["use_cvm"] = strategy.get("use_cvm", False)
        opt_info["no_cvm"] = strategy.get("no_cvm", False)
        opt_info["stat_var_names"] = strategy.get("stat_var_names", [])
        opt_info["local_tables"] = strategy.get("local_tables", [])
        opt_info["async_tables"] = strategy.get("async_tables", [])
        opt_info["async_tables"] = strategy.get("async_tables", [])
        opt_info["scale_datanorm"] = strategy.get("scale_datanorm", -1)
        opt_info["check_nan_var_names"] = strategy.get("check_nan_var_names",
                                                       [])
        opt_info["dump_slot"] = False
        opt_info["dump_converter"] = ""
        opt_info["dump_fields"] = strategy.get("dump_fields", [])
        opt_info["dump_file_num"] = strategy.get("dump_file_num", 16)
        opt_info["dump_fields_path"] = strategy.get("dump_fields_path", "")
        opt_info["dump_param"] = strategy.get("dump_param", [])
        if server._server.downpour_server_param.downpour_table_param[
                0].accessor.accessor_class == "DownpourCtrAccessor":
            opt_info["dump_slot"] = True
        opt_info["adjust_ins_weight"] = strategy.get("adjust_ins_weight", {})
        opt_info["copy_table"] = strategy.get("copy_table", {})
        opt_info["loss_names"] = strategy.get("loss_names", [])

        for loss in losses:
            loss.block.program._fleet_opt = opt_info

        param_grads_list = []
        for loss in losses:
            prog_id = str(id(loss.block.program))
            param_grads_list.append(prog_id_to_param_grads[prog_id])
        return None, param_grads_list, opt_info
