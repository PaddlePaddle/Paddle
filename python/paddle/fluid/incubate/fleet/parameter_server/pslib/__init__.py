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

import os
import sys
from .optimizer_factory import *
from google.protobuf import text_format
import paddle.fluid as fluid
from paddle.fluid.framework import Program

from paddle.fluid.incubate.fleet.base.fleet_base import Fleet
from paddle.fluid.incubate.fleet.base.fleet_base import Mode
from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer
from paddle.fluid.incubate.fleet.base.role_maker import MPISymetricRoleMaker


class PSLib(Fleet):
    def __init__(self):
        super(PSLib, self).__init__(Mode.PSLIB)
        self._opt_info = None
        self._local_ip = 0
        self._fleet_ptr = None
        self._main_programs = []
        self._scopes = []
        self._client2client_request_timeout_ms = 500000
        self._client2client_connect_timeout_ms = 10000
        self._client2client_max_retry = 3

    def init(self, role_maker=None):
        super(PSLib, self).init(MPISymetricRoleMaker())
        self._fleet_ptr = fluid.core.Fleet()

    def _set_client_communication_config(self, request_timeout_ms,
                                         connect_timeout_ms, max_retry):
        self._client2client_request_timeout_ms = request_timeout_ms
        self._client2client_connect_timeout_ms = connect_timeout_ms
        self._client2client_max_retry = max_retry

    def init_worker(self):
        """
        init_worker(): will be called by user. When a user knows current process is_server(), he/she
                    should call init_worker() to initialize global information about worker and connect
                    worker with pserver. You should run startup program before init_worker.

        Args:
            executor(Executor): The executor to run for init server.
            programs(Program|None): The program that need to run.
        """

        if len(self._main_programs) == 0:
            raise ValueError(
                "You should run DistributedOptimizer.minimize() first")

        if self._opt_info:
            if "fleet_desc" in self._opt_info:
                self._dist_desc_str = text_format.MessageToString(
                    self._opt_info["fleet_desc"])
                self._dist_desc = self._opt_info["fleet_desc"]
            else:
                raise Exception(
                    "You should run DistributedOptimizer.minimize() first")
            # barrier_all for init_server, wait for server starts
            self._role_maker._barrier_all()
            self.all_ips_ = self._role_maker._all_gather(self._local_ip)
            self._fleet_ptr.init_worker(self._dist_desc_str, self.all_ips_,
                                        self._role_maker._get_size(),
                                        self._role_maker._get_rank())
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
                tables = self._dist_desc.trainer_param.dense_table
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
                                    "var " + var_name + " not found in scope, "
                                    + "you should run startup program first")
                            var_name_list.append(var_name)
                        self._fleet_ptr.init_model(scope,
                                                   int(table.table_id),
                                                   var_name_list)
            # barrier for init model done
            self._role_maker._barrier_worker()
        else:
            raise NameError(
                "You should run DistributedOptimizer.minimize() first")

    def init_server(self, model_dir=None, **kwargs):
        """
        init_server() will be called by user. It will load model from model_dir.

        Args:
            model_dir(str): load model path, can be local or hdfs/afs path.
            kwargs: user-defined attributes, currently support following:
                model(int): load model mode.
                            0 is for load whole model,
                            1 is for load delta model (load diff),
                            default is 0.

        Example:
            >>> fleet.init_server("/you/path/to/model", mode = 0)

        """
        mode = kwargs.get("mode", 0)
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            self._fleet_ptr.load_model(model_dir, mode)
        self._role_maker._barrier_worker()

    def run_server(self):
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
            self._fleet_ptr.init_server(self._dist_desc_str,
                                        self._role_maker._get_rank())
            self._local_ip = self._fleet_ptr.run_server()

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
        stop(): will be called after a user finishes his/her training task. Fleet instance will be
            destroyed when stop() is called.
        """
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            self._fleet_ptr.stop_server()
        self._role_maker._barrier_worker()
        self._role_maker._barrier_all()
        self._role_maker._finalize()

    def distributed_optimizer(self, optimizer, strategy={}):
        """
        distributed_optimizer

        Args:
            optimizer(Optimizer): optimizer
            strategy(dict): strategy

        Examples:
            .. code-block:: python

              fleet.distributed_optimizer(optimizer)

        Returns:
            optimizer(DownpourOptimizer): downpour optimizer

        """
        self._optimizer = DownpourOptimizer(optimizer, strategy)
        return self._optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names=None,
                             target_vars=None,
                             main_program=None,
                             export_for_deployment=True):
        """
        save pserver model called from a worker

        Args:
            executor(Executor): fluid executor
            dirname(str): save model path
            feeded_var_names(list): default None
            target_vars(list): default None
            main_program(Program): default None
            export_for_deployment(bool): default None

        Examples:
            .. code-block:: python

              fleet.save_inference_model(dirname="hdfs:/my/path")

        """
        self._fleet_ptr.save_model(dirname)

    def save_persistables(self, executor, dirname, main_program=None, **kwargs):
        """
        save presistable parameters,
        when using fleet, it will save sparse and dense feature

        Args:
            executor(Executor): fluid executor
            dirname(str): save path. It can be hdfs/afs path or local path
            main_program(Program): fluid program, default None
            kwargs: use define property, current support following
                mode(int): 0 means save all pserver model,
                           1 means save delta pserver model (save diff),
                           2 means save xbox base,
                           3 means save batch model.

        Example:
            >>> fleet.save_persistables(dirname="/you/path/to/model", mode = 0)

        """
        mode = kwargs.get("mode", 0)
        self._fleet_ptr.client_flush()
        self._role_maker._barrier_worker()
        if self._role_maker.is_first_worker():
            self._fleet_ptr.save_model(dirname, mode)
        self._role_maker._barrier_worker()

    def save_cache_model(self, executor, dirname, main_program=None, **kwargs):
        """
        save sparse cache table,
        when using fleet, it will save sparse cache table

        Args:
            dirname(str): save path. It can be hdfs/afs path or local path
            main_program(Program): fluid program, default None
            kwargs: use define property, current support following
                mode(int): define for feature extension in the future,
                           currently no use, will pass a default value 0 

        Example:
            .. code-block:: python
            >>> fleet.save_cache_model(None, dirname="/you/path/to/model", mode = 0)

        """
        mode = kwargs.get("mode", 0)
        self._fleet_ptr.client_flush()
        self._role_maker._barrier_worker()
        cache_threshold = 0.0

        if self._role_maker.is_first_worker():
            cache_threshold = self._fleet_ptr.get_cache_threshold()
        #check cache threshold right or not
        self._role_maker._barrier_worker()

        if self._role_maker.is_first_worker():
            self._fleet_ptr.cache_shuffle(0, dirname, mode, cache_threshold)

        self._role_maker._barrier_worker()

        feasign_num = -1
        if self._role_maker.is_first_worker():
            feasign_num = self._fleet_ptr.save_cache(0, dirname, mode)

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
            for i in self._opt_info["fleet_desc"].trainer_param.sparse_table:
                self._fleet_ptr.shrink_sparse_table(i.table_id)
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
            for i in self._opt_info["fleet_desc"].trainer_param.dense_table:
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
                self._fleet_ptr.shrink_dense_table(i.table_id, scope, var_list,
                                                   decay, emb_dim)
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
                    load_combine(bool): load from a file or splited param files
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
            load_combine(bool): load from a file or splited param files

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
            for i in self._opt_info["fleet_desc"].trainer_param.dense_table:
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
                    scope, table_id, var_names, model_path, model_proto_file,
                    table_var_names, load_combine)
        self._role_maker._barrier_worker()

    def _set_opt_info(self, opt_info):
        """
        this function saves the result from DistributedOptimizer.minimize()
        """
        self._opt_info = opt_info


fleet = PSLib()


class DownpourOptimizer(DistributedOptimizer):
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
        strategy(any): config for DownpourOptimizer.

    Returns:
        None
    """

    def __init__(self, optimizer, strategy=None):
        super(DownpourOptimizer, self).__init__(optimizer, strategy)

        self._optimizer = optimizer
        self._optimizer_name = "Distributed%s" % optimizer.type.capitalize()
        if optimizer.type != "adam":
            print("Currently, distributed optimizer only support Adam"
                  "Will config built-in adam for you."
                  "We will support more functions in DistributedOptimizer",
                  sys.stderr)
            self._optimizer_name = "DistributedAdam"

        self._distributed_optimizer = globals()[self._optimizer_name](optimizer)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Currently, backward function can not be called through DistributedOptimizer
        """
        raise NotImplementedError()

    def apply_gradients(self, params_grads):
        """
        Currently, apply_gradients function can not be called through DistributedOptimizer
        """
        raise NotImplementedError()

    def minimize(self,
                 losses,
                 scopes=None,
                 startup_programs=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        minimize a program through loss, loss can be a list in DistributedOptimizer.
        Note that in parameter server mode, a worker will not get anything about optimize_os
        Because optmizer algorithms run on pserver side. We will make this usable in pserver
        process, but currently the optimization part is written into Fleet(). A user does not
        need to care about how to startup a pserver node.

        Args:
            losses (Variable|Variable List): loss variable or loss variable list to run optimization.
            scopes (Scope| Scope List): scope instance.
            startup_programs (Program|Program List): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.

        Returns:
            tuple: (optimize_ops, params_grads) which are, list of operators appended;
            and list of (param, grad) Variables pair for optimization.
        """

        if not isinstance(losses, list):
            losses = [losses]

        optimize_ops, param_grads, opt_info = \
                      self._distributed_optimizer._minimize(
                          losses,
                          startup_programs,
                          parameter_list,
                          no_grad_set,
                          self._strategy)
        opt_info["mpi_rank"] = fleet._role_maker._get_rank()
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
