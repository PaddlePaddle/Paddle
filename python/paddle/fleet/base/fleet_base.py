#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from .strategy_compiler import StrategyCompiler
from .meta_optimizer_factory import MetaOptimizerFactory

__all__ = ['Fleet']


class Fleet(object):
    """
    Unified API for distributed training of PaddlePaddle
    Fleet is initialized through init function
    """

    def __init__(self):
        self._runtime_handle = None
        self._util = None

    def init(self, role_maker):
        self.role_maker = role_maker
        self.strategy_compiler = StrategyCompiler()

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        
        """
        #return self._role_maker.is_first_worker()
        return True

    def worker_index(self):
        """
        Get current worker index.

        Returns:
            int: node id
        """
        #return self._role_maker.worker_index()
        return 0

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker numbers
        """
        #return self._role_maker.worker_num()
        return 1

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.
        """
        #return self._role_maker.is_worker()
        return True

    def worker_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """
        '''
        if to_string:
            return ",".join(self._role_maker.get_trainer_endpoints())
        else:
            return self._role_maker.get_trainer_endpoints()
        '''
        return ["127.0.0.1:1001", "127.0.0.1:1002"]

    def server_num(self):
        """
        Get current total worker number.

        Returns:
            int: server number
        """
        #return len(self._role_maker.get_pserver_endpoints())
        return 1

    def server_index(self):
        """
        Get current server index.

        Returns:
            int: node id
        """
        #return self._role_maker.server_index()
        return 0

    def server_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """
        '''
        if to_string:
            return ",".join(self._role_maker.get_pserver_endpoints())
        else:
            return self._role_maker.get_pserver_endpoints()
        '''
        return ["127.0.0.1:1001", "127.0.0.1:1002"]

    def is_server(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.
        """
        #return self._role_maker.is_server()
        return True

    @property
    def util(self):
        """
        return util
        """
        return self._util

    @util.setter
    def util(self, util):
        """
        set util
        """
        self._util = util

    def barrier_worker(self):
        """
        barrier between workers
        """
        self._role_maker.barrier_worker()

    def init_worker(self):
        assert self._runtime_handle is not None
        self._runtime_handle.init_worker()

    def init_server(self, model_dir=None):
        assert self._runtime_handle is not None
        self._runtime_handle.init_server()

    def run_server(self):
        assert self._runtime_handle is not None
        self._runtime_handle.run_server()

    def stop_worker(self):
        assert self._runtime_handle is not None
        self._runtime_handle.stop_worker()

    def distributed_optimizer(self, optimizer, strategy):
        self.user_defined_optimizer = optimizer
        self.user_defined_strategy = strategy
        return self

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        # cache original feed forward program
        self.origin_main_program = loss.block.program
        if startup_program == None:
            self.origin_startup_program = \
                paddle.default_startup_program().clone(for_test=False)
            startup_program = paddle.default_startup_program()
        else:
            self.origin_startup_program = \
                startup_program.clone(for_test=False)

        # compile time
        distributed_optimizer_list = \
            MetaOptimizerFactory()._get_valid_meta_optimizers(
                self.user_defined_optimizer)
        valid_optimizer_list = []
        # recall meta optimizers for ranking
        for opt in distributed_optimizer_list:
            opt._set_basic_info(loss, self.role_maker,
                                self.user_defined_optimizer,
                                self.user_defined_strategy)
            if opt._can_apply():
                valid_optimizer_list.append(opt)
        # combine recalled meta optimizers to be a valid meta optimizer
        meta_optimizer, compiled_strategy = \
                self.strategy_compiler.generate_optimizer(
                    loss, self.role_maker, self.user_defined_optimizer,
                    self.user_defined_strategy, valid_optimizer_list)
        optimize_ops, params_grads = meta_optimizer.minimize(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        if self._runtime_handle is not None:
            self._runtime_handle = RuntimeFactory()._create_runtime(
                final_dist_strategy, self.role_maker, optimize_ops,
                params_grads)

        if self._util is not None:
            self._util = UtilFactory()._create_util(final_dist_strategy,
                                                    self.role_maker,
                                                    optimize_ops, params_grads)

        return optimize_ops, params_grads
