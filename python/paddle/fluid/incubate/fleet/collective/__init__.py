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

import sys
import os
from ..base.role_maker import MPIDeviceRoleMaker
from google.protobuf import text_format
import paddle.fluid.optimizer as local_optimizer
import paddle.fluid as fluid


class Fleet(object):
    """
    Fleet in Python. Fleet is used in distributed training. It is designed as a singlton instance
    in c++. A Fleet() object will be initialized automatically when a user import this package as
    fleet. The General interface Fleet supports are:
    init(): which should be called only once in user's python scripts. init() will initialize
            FleetWrapper in CPP, it will also initialize a RoleMaker which is used for identifying
            current node's role, e.g. worker, server, etc.
    stop(): will be called after a user finishes his/her training task. Fleet instance will be
            destroyed when stop() is called.
    init_pserver(): will be called by user. When a user knows current process is_worker(), he/she
                    should call init_pserver() to initialize global information about parameter server
    init_worker(): will be called by user. When a user knows current process is_server(), he/she
                    should call init_worker() to initialize global information about worker and connect
                    worker with pserver.
    get_worker_num(): return the number of current task's worker node
    get_server_num(): return the number of current task's pserver node
    is_worker(): return whether current process is a worker
    is_server(): return thether current process is a server
    init_pserver_model(): initialize model parameters in pserver, called from a worker node
    save_pserver_model(): save model parameters in pserver, called from a server node

    Example:

        .. code-block:: python
           import paddle.fluid.incubate.fleet.parameter_server as fleet
           from my_model import bow_net
           model = bow_net()
           fleet.init()
           sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.0001)
           sgd_optimizer = fleet.DistributedOptimizer(sgd_optimizer)
           sgd_optimizer.minimize(model.loss)
           exe = paddle.fluid.Executor(paddle.fluid.CPUPlace())
           if fleet.is_worker():
              exe.run(paddle.fluid.default_startup_program())
              fleet.init_worker() # init worker should be called before training
              # do other things like training
           elif fleet.is_server():
              fleet.init_pserver()
           fleet.stop()
    """

    def __init__(self):
        self._opt_info = None  # for fleet only
        self._role_maker = None
        self._local_ip = 0
        self._is_initialized = False

    def init(self):
        # TODO(guru4elephant)
        # this is a temporary solution
        # we will support more configurable RoleMaker for users in the future
        """
        init(): which should be called only once in user's python scripts. init() will initialize
            FleetWrapper in CPP, it will also initialize a RoleMaker which is used for identifying
            current node's role, e.g. worker, server, etc.
        """
        if not self._is_initialized:
            self._role_maker = MPIDeviceRoleMaker()
            self._role_maker._generate_role()
            self._nccl_ptr = fluid.core.Nccl()
            self._is_initialized = True

    def stop(self):
        """
        stop(): will be called after a user finishes his/her training task. Fleet instance will be
            destroyed when stop() is called.
        """
        self._role_maker._barrier_worker()
        self._role_maker._finalize()

    def init_pserver(self):
        """
        init_pserver(): will be called by user. When a user knows current process is_worker(), he/she
            should call init_pserver() to initialize global information about parameter server
        """
        raise NotImplementedError()

    def init_worker(self, programs, scopes=None):
        """
        init_worker(): will be called by user. When a user knows current process is_server(), he/she
                    should call init_worker() to initialize global information about worker and connect
                    worker with pserver. You should run startup program before init_worker.

        Args:
            programs(Program|list): a Program or a list of Programs
            scopes(Scope|list): a Scope or  a list of Scopes, default None.
        """
        if not isinstance(programs, list):
            programs = [programs]
        if scopes is None:
            scopes = [fluid.global_scope()] * len(programs)
        if len(scopes) != len(programs):
            print(
                "You should make sure len(scopes) == len(programs) or set scopes None"
            )
            sys.exit(-1)

        self._role_maker._barrier_all()
        self._local_ip = self._role_maker._get_local_ip()
        self._all_ips = self._role_maker._all_gather(self._local_ip)
        print(self._all_ips)
        local_rank = 0
        global_rank = self._role_maker._get_rank()
        total_rank = self._role_maker._get_size()
        for i, ip in enumerate(self._all_ips):
            if ip == self._local_ip and i != global_rank:
                local_rank += 1
            if i == global_rank:
                break
        self._nccl_ptr.set_rank_info(local_rank, global_rank, total_rank)
        if global_rank == 0:
            nccl_id = self._nccl_ptr.get_nccl_id()
        else:
            nccl_id = None
        nccl_id = self._role_maker._broadcast(nccl_id, root=0)
        if global_rank != 0:
            self._nccl_ptr.set_nccl_id(nccl_id)
        self._role_maker._barrier_all()
        self._nccl_ptr.init_nccl()
        print("node %d done on nccl init" % global_rank)

    def get_worker_num(self):
        """
        return the number of current job's worker num
        """
        return self._role_maker._worker_num()

    def get_server_num(self):
        """
        return the number of current job's server num
        """
        return self._role_maker._server_num()

    def get_worker_index(self):
        """
        return the mpi rank of current worker
        """
        return self._role_maker._worker_index()

    def is_worker(self):
        """
        return whether current node is a worker
        """
        return self._role_maker._is_worker()

    def is_server(self):
        """
        return whether current node is pserver
        """
        return self._role_maker._is_server()

    def init_pserver_model(self):
        """
        init pserver model called from pserver
        """
        if self._role_maker._is_first_worker():
            self._fleet_ptr.init_model()
        self._role_maker._barrier_worker()

    def save_pserver_model(self, save_path):
        """
        save pserver model called from a worker
        """
        raise NotImplementedError()

    def _set_opt_info(self, opt_info):
        """
        this function saves the result from DistributedOptimizer.minimize()
        """
        self._opt_info = opt_info


class DistributedOptimizer(object):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    Fleet() instance who holds the global information about current distributed
    training.
    """

    def __init__(self, optimizer, dist_config={}):
        super(DistributedOptimizer, self).__init__()
        self._optimizer = optimizer
        self._distributed_optimizer = optimizer

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
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        minimize a program through loss, loss can be a list in DistributedOptimizer
        Args:
            loss (Variable|Variable List): loss variable or loss variable list to run optimization.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.
        Returns:
            tuple: (optimize_ops, params_grads) which are, list of operators appended;
            and list of (param, grad) Variables pair for optimization.
        Note that in parameter server mode, a worker will not get anything about optimize_os
        Because optmizer algorithms run on pserver side. We will make this usable in pserver
        process, but currently the optimization part is written into Fleet(). A user does not
        need to care about how to startup a pserver node.
        """
        '''
        pg = self._optimizer.backward(loss)
        to_apply = []
        for p, g in pg:
            r_g = fluid.layers.collective._allreduce(g)
            to_apply.append([p, r_g])
        optimizer_ops = optimizer.apply_gradients(to_apply)
        '''
        optimize_ops, pg = self._optimizer.minimize(loss, startup_program,
                                                    parameter_list, no_grad_set)
        opt_info = {}
        opt_info["trainer"] = "MultiTrainer"
        opt_info["device_worker"] = "Mirrored"
        opt_info["worker_skipped_ops"] = [""]
        opt_info["param_grads"] = pg

        loss.block.program._fleet_opt = opt_info
        return optimize_ops, pg


# this is a temporary solution
# TODO(guru4elephant)
# will make this more flexible for more Parameter Server Archs
fleet_instance = Fleet()

init = fleet_instance.init
stop = fleet_instance.stop
init_pserver = fleet_instance.init_pserver
init_worker = fleet_instance.init_worker
is_worker = fleet_instance.is_worker
is_server = fleet_instance.is_server
init_pserver_model = fleet_instance.init_pserver_model
save_pserver_model = fleet_instance.save_pserver_model
worker_num = fleet_instance.get_worker_num
server_num = fleet_instance.get_server_num
worker_index = fleet_instance.get_worker_index
