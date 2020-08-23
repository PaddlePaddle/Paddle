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
from .role_maker import UserDefinedRoleMaker, PaddleCloudRoleMaker, RoleMakerBase
from .strategy_compiler import StrategyCompiler
from .distributed_strategy import DistributedStrategy
from .meta_optimizer_factory import MetaOptimizerFactory
from .runtime_factory import RuntimeFactory
from .util_factory import UtilFactory
from paddle.fluid.wrapped_decorator import wrap_decorator

__all__ = ['Fleet']


def _inited_runtime_handler_(func):
    def __impl__(*args, **kwargs):
        cls = args[0]

        if cls._runtime_handle is None:
            raise ValueError("Fleet can not find suitable runtime handler")

        return func(*args, **kwargs)

    return __impl__


inited_runtime_handler = wrap_decorator(_inited_runtime_handler_)


class Fleet(object):
    """
    Unified API for distributed training of PaddlePaddle
    Please reference the https://github.com/PaddlePaddle/Fleet for details


    Returns:
        Fleet: A Fleet instance

    Examples:
        .. code-block:: python

            import paddle.distributed.fleet as fleet
            role = fleet.role_maker.PaddleCloudRoleMaker(is_collective=True)
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
        self._role_maker = None
        self._is_collective = False

    def init(self, role_maker=None, is_collective=False):
        """
        Initialize role_maker in Fleet.

        This function is responsible for the distributed architecture 
        what you want to run your code behind,such as Transpiler,
        Collective in PaddleCloudRoleMaker or UserDefinedRoleMaker 
        
        """
        if isinstance(role_maker, RoleMakerBase):
            self._role_maker = role_maker
        elif role_maker == None:
            if isinstance(is_collective, bool):
                self._is_collective = is_collective
                self._role_maker = PaddleCloudRoleMaker(
                    is_collective=self._is_collective)
            else:
                raise ValueError(
                    "Something wrong occurred, please check whether is_collective is bool value"
                )
        else:
            raise ValueError(
                "Something wrong occurred, please check whether rolemaker is instance of RoleMakerBase"
            )
        self.strategy_compiler = StrategyCompiler()
        return None

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

    @inited_runtime_handler
    def init_worker(self):
        """
        init worker
        """
        self._runtime_handle._init_worker()

    @inited_runtime_handler
    def init_server(self, *args, **kwargs):
        """
        init server
        """
        self._runtime_handle._init_server(*args, **kwargs)

    @inited_runtime_handler
    def run_server(self):
        """
        run server
        """
        self._runtime_handle._run_server()

    @inited_runtime_handler
    def stop_worker(self):
        """
        stop worker
        """
        self._runtime_handle._stop_worker()

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        self._runtime_handle._save_inference_model(
            executor, dirname, feeded_var_names, target_vars, main_program,
            export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None):
        self._runtime_handle._save_persistables(executor, dirname, main_program)

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        distirbuted_optimizer
        Returns:
            Fleet instance with minimize interface like optimizers

        Examples:
            .. code-block:: python
            import paddle.distributed.fleet as fleet
            role = fleet.role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)
            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        """
        self.user_defined_optimizer = optimizer
        if strategy == None:
            strategy = DistributedStrategy()
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
            import paddle.distributed.fleet as fleet

            fc_1 = paddle.layers.fc(input=input_x, size=hid_dim, act='tanh')
            fc_2 = paddlen.layers.fc(input=fc_1, size=hid_dim, act='tanh')
            prediction = paddle.layers.fc(input=[fc_2], size=label_dim, act='softmax')
            cost = paddle.layers.cross_entropy(input=prediction, label=input_y)
            avg_cost = paddle.layers.mean(x=cost)

            role = fleet.role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)
            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)

            # for more examples, please reference https://github.com/PaddlePaddle/Fleet

        """
        context = {}
        # cache original feed forward program
        self.origin_main_program = loss.block.program
        context["origin_main_program"] = self.origin_main_program
        context["loss"] = loss
        if startup_program == None:
            self.origin_startup_program = \
                paddle.static.default_startup_program().clone(for_test=False)
            startup_program = paddle.static.default_startup_program()
        else:
            self.origin_startup_program = \
                startup_program.clone(for_test=False)

        context["origin_startup_program"] = startup_program
        context["role_maker"] = self._role_maker

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

        context["valid_strategy"] = valid_strategy

        self.valid_strategy = valid_strategy

        optimize_ops = []
        params_grads = []

        if meta_optimizer:
            optimize_ops, params_grads = meta_optimizer.minimize(
                loss,
                startup_program=startup_program,
                parameter_list=parameter_list,
                no_grad_set=no_grad_set)

            default_program = paddle.static.default_main_program()

            if id(default_program) != id(loss.block.program):
                paddle.fluid.framework.switch_main_program(loss.block.program)

        else:
            optimize_ops, params_grads = self.user_defined_optimizer.minimize(
                loss,
                startup_program=startup_program,
                parameter_list=parameter_list,
                no_grad_set=no_grad_set)

        context["program_optimize_ops"] = optimize_ops
        context["program_params_grads"] = params_grads

        if graph_optimizer:
            optimize_ops, params_grads = graph_optimizer.minimize(
                loss,
                startup_program=startup_program,
                parameter_list=parameter_list,
                no_grad_set=no_grad_set)
            # since we do not encourage users to use graph operations
            # if a graph optimizer takes effect, mostly
            # optimizers_ops and params_grads are None
            # i.e. users can not modify current computation graph anymore
            context["graph_optimize_ops"] = optimize_ops
            context["graph_optimize_grads"] = params_grads

        if self._runtime_handle is None:
            self._runtime_handle = RuntimeFactory()._create_runtime(context)

        if self._util is None:
            self._util = UtilFactory()._create_util(context)

        return optimize_ops, params_grads
