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
from .runtime_factory import RuntimeFactory
from .util_factory import UtilFactory

__all__ = ['Fleet']


class Fleet(object):
    """
    Unified API for distributed training of PaddlePaddle
    Please reference the https://github.com/PaddlePaddle/Fleet for details


    Returns:
        Fleet: A Fleet instance

    Examples:
        .. code-block:: python

            import paddle.fleet as fleet
            import paddle.fluid.incubate.fleet.base.role_maker as role_maker
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)
            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
            if fleet.is_first_worker():
                print("this is first worker")
            print("current node index: {}".format(fleet.worker_index()))
            print("total number of worker num: {}".format(fleet.worker_num()))
            if fleet.is_worker():
                print("this is worker")
            print("worker endpoints: {}".format(fleet.worker_endpoints(to_string=True)))
            print("server num: {}".format(fleet.server_num()))
            print("server endpoints: {}".format(fleet.server_endpoints(to_string=True)))
            if fleet.is_server():
                print("this is server")
            fleet.stop_worker()
    """

    def __init__(self):
        self._runtime_handle = None
        self._util = None

    def init(self, role_maker):
        self._role_maker = role_maker
        self.strategy_compiler = StrategyCompiler()

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        
        """
        return self._role_maker.is_first_worker()

    def worker_index(self):
        """
        Get current worker index.

        Returns:
            int: node id
        """
        return self._role_maker.worker_index()

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker numbers
        """
        return self._role_maker.worker_num()

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.
        """
        return self._role_maker.is_worker()

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
        return len(self._role_maker.get_pserver_endpoints())

    def server_index(self):
        """
        Get current server index.

        Returns:
            int: node id
        """
        return self._role_maker.server_index()

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
        return self._role_maker.is_server()

    @property
    def util(self):
        """
        Utility functions that can be used under certain runtime
        return util
        """
        return self._util

    @util.setter
    def util(self, util):
        """
        Set Utility functions for userd-defined runtime
        set util
        """
        self._util = util

    def barrier_worker(self):
        """
        barrier between workers
        """
        self._role_maker.barrier_worker()

    def init_worker(self):
        """
        init worker
        """
        assert self._runtime_handle is not None
        self._runtime_handle._init_worker()

    def init_server(self, model_dir=None):
        """
        init server
        """
        assert self._runtime_handle is not None
        self._runtime_handle._init_server()

    def run_server(self):
        """
        run server
        """
        assert self._runtime_handle is not None
        self._runtime_handle._run_server()

    def stop_worker(self):
        """
        stop worker
        """
        assert self._runtime_handle is not None
        self._runtime_handle._stop_worker()

    def distributed_optimizer(self, optimizer, strategy):
        """
        distirbuted_optimizer
        Returns:
            Fleet instance with minimize interface like optimizers

        Examples:
            .. code-block:: python
            import paddle.fleet as fleet
            import paddle.fluid.incubate.fleet.base.role_maker as role_maker
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)
            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        """
        self.user_defined_optimizer = optimizer
        self.user_defined_strategy = strategy
        self.valid_strategy = None
        return self

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Add distributed operations to minimize ``loss`` by updating ``parameter_list``.

        Args:
            loss (Variable): A ``Variable`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (Iterable, optional): Iterable of ``Variable`` or ``Variable.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable``  or ``Variable.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) variable pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to 
            indicate program pruning. If so, the program will be pruned by ``feed`` and 
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:
            import paddle
            import paddle.fleet as fleet
            import paddle.fluid.incubate.fleet.base.role_maker as role_maker

            fc_1 = paddle.layers.fc(input=input_x, size=hid_dim, act='tanh')
            fc_2 = paddlen.layers.fc(input=fc_1, size=hid_dim, act='tanh')
            prediction = paddle.layers.fc(input=[fc_2], size=label_dim, act='softmax')
            cost = paddle.layers.cross_entropy(input=prediction, label=input_y)
            avg_cost = paddle.layers.mean(x=cost)

            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)
            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)

            # for more examples, please reference https://github.com/PaddlePaddle/Fleet

        """
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
        valid_graph_optimizer_list = []
        can_not_apply_optimizer_list = []
        # recall meta optimizers for ranking
        for opt in distributed_optimizer_list:
            opt._set_basic_info(loss, self._role_maker,
                                self.user_defined_optimizer,
                                self.user_defined_strategy)
            if opt._can_apply() and not opt._is_graph_out():
                valid_optimizer_list.append(opt)
            elif opt._can_apply() and opt._is_graph_out():
                valid_graph_optimizer_list.append(opt)
            else:
                can_not_apply_optimizer_list.append(opt)
        # combine recalled meta optimizers to be a valid meta optimizer
        meta_optimizer, graph_optimizer = \
                self.strategy_compiler.generate_optimizer(
                    loss, self._role_maker, self.user_defined_optimizer,
                    self.user_defined_strategy, valid_optimizer_list,
                    valid_graph_optimizer_list)

        valid_strategy = self.strategy_compiler._get_valid_strategy(
            self.user_defined_strategy, can_not_apply_optimizer_list)
        self.valid_strategy = valid_strategy

        optimize_ops = []
        params_grads = []
        if meta_optimizer:
            optimize_ops, params_grads = meta_optimizer.minimize(
                loss,
                startup_program=startup_program,
                parameter_list=parameter_list,
                no_grad_set=no_grad_set)

        if graph_optimizer:
            optimizer_ops, params_grads = graph_optimizer.minimize(
                loss,
                startup_program=startup_program,
                parameter_list=parameter_list,
                no_grad_set=no_grad_set)
            # since we do not encourage users to use graph operations
            # if a graph optimizer takes effect, mostly
            # optimizers_ops and params_grads are None
            # i.e. users can not modify current computation graph anymore

        if self._runtime_handle is None:
            self._runtime_handle = RuntimeFactory()._create_runtime(
                valid_strategy, self._role_maker, optimize_ops, params_grads)

        if self._util is None:
            self._util = UtilFactory()._create_util(
                valid_strategy, self._role_maker, optimize_ops, params_grads)

        return optimize_ops, params_grads
