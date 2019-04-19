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
# limitations under the License.

from __future__ import print_function

import abc
import sys

from enum import Enum

from paddle.fluid.optimizer import Optimizer

from role_maker import RoleMakerBase, Role
from role_maker import MPISymetricRoleMaker
from role_maker import UserDefinedRoleMaker


class Mode(Enum):
    TRANSPILER = 1,
    PSLIB = 2,
    COLLECTIVE = 3


class Fleet(object):
    """
    Fleet is the base class, transpiler and pslib are implementation of Fleet.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode):
        """
        Init Fleet with mode, mode must be an instance of Mode.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None; if you would like to save all variables in a
        single file, use `filename` to specify the file name.

        Args:
            mode(Mode): the implementation of Fleet's mode.

        Returns:
            None
        """

        assert isinstance(mode, Mode)
        self.is_initialized = False
        self.mode = mode
        self.workers = 0
        self.servers = 0
        self.worker_endpoints = []
        self.server_endpoints = []
        self.role = Role.WORKER
        self.current_endpoint = None
        self.current_id = 0
        self.optimizer = None
        self.role_maker_ = None

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        """
        return self.is_worker() and self.current_id == 0

    def worker_id(self):
        """
        Get current worker id.

        Returns:
            int: node id
        """
        return self.current_id

    def get_workers(self):
        """
        Get current total worker number.

        Returns:
            int: worker number
        """
        return self.workers

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.
        """
        return self.role == Role.WORKER

    def is_server(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.
        """
        return self.role == Role.SERVER

    def split_files(self, files):
        """
        split files before distributed training,
        for example, files is [a, b, c ,d, e]  and trainer_num = 2,
        then trainer 0 gets [a, b, c] and trainer 1 gets [d, e]

        Args:
            files(list): file list need to be read.

        Returns:
            list: files belongs to this worker.
        """
        file_num = len(files)
        trainer_id = self.worker_id()
        trainer_num = self.get_workers()
        if trainer_num > file_num:
            raise ValueError("trainer_num should be <= file_num : "
                             "%s > %s" % (trainer_num, file_num))
        start = 0
        end = 0
        for i in range(0, trainer_id + 1):
            length = file_num / trainer_num + (i < (file_num % trainer_num))
            start = end
            end += length
        return files[start:end]

    def init(self, role_maker=None):
        """
        should be called only once in user's python scripts,
        init() will initialize RoleMaker which is used for identifying
            current node's role, e.g. worker, server, etc.

        Args:
            role_maker(RoleMakerBase): subclass of RoleMakerBase.

        Returns:
            None
        """

        if role_maker and not isinstance(role_maker, RoleMakerBase):
            raise ValueError("role_maker must be an instance of RoleMakerBase")

        self.role_maker_ = role_maker

        if isinstance(role_maker, MPISymetricRoleMaker):
            self.role_maker_._generate_role()
            self.role = Role.WORKER if role_maker._is_worker() else Role.SERVER
            self.workers = role_maker._worker_num()
            self.servers = role_maker._server_num()
            self.server_endpoints = role_maker._get_pserver_endpoints()
            self.worker_endpoints = role_maker._get_trainer_endpoints()
            self.current_id = role_maker._worker_index(
            ) if role_maker._is_worker() else role_maker._server_index()
            self.current_endpoint = self.worker_endpoints[self.current_id] \
                if role_maker._is_worker() else self.server_endpoints[self.current_id]

        elif isinstance(role_maker, UserDefinedRoleMaker):
            self.current_id = role_maker.current_id
            self.current_endpoint = role_maker.current_endpoint
            self.workers = role_maker.workers
            self.worker_endpoints = role_maker.worker_endpoints
            self.servers = role_maker.servers
            self.server_endpoints = role_maker.server_endpoints
            self.role = role_maker.role

        else:
            raise ValueError(
                "role_maker must be an instance of UserDefinedRoleMaker/MPISymetricRoleMaker"
            )

        self.is_initialized = True

    @abc.abstractmethod
    def init_worker(self, executor):
        pass

    @abc.abstractmethod
    def run_worker(self, executor, main_program=None):
        pass

    @abc.abstractmethod
    def init_server(self, executor, model_dir=None):
        pass

    @abc.abstractmethod
    def run_server(self, executor):
        pass

    @abc.abstractmethod
    def stop_worker(self):
        pass

    @abc.abstractmethod
    def stop(self, executor):
        pass

    @abc.abstractmethod
    def distributed_optimizer(self, optimizer, strategy=None):
        pass

    @abc.abstractmethod
    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        pass

    @abc.abstractmethod
    def save_persistables(self, executor, dirname, main_program=None):
        pass

    def to_string(self):
        infos = """
        mode             = {}
        workers          = {}
        server_endpoints = {}
        role             = {}
        current_endpoint = {}
        current_id       = {}
        """.format(self.mode, self.workers, self.server_endpoints, self.role,
                   self.current_endpoint, self.current_id)
        return infos


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
    __metaclass__ = abc.ABCMeta

    def __init__(self, optimizer, strategy=None):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of Optimizer")

        if strategy and not isinstance(strategy, dict):
            raise ValueError("strategy must be an instance of Dict")

        self._optimizer = optimizer
        self._strategy = strategy

    @abc.abstractmethod
    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Currently, backward function can not be called through DistributedOptimizer
        """
        pass

    @abc.abstractmethod
    def apply_gradients(self, params_grads):
        """
        Currently, apply_gradients function can not be called through DistributedOptimizer
        """
        pass

    @abc.abstractmethod
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
        pass
