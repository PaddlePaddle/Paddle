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
    Please reference the https://github.com/PaddlePaddle/FleetX for details


    Returns:
        Fleet: A Fleet instance

    Example for collective training:
        .. code-block:: python

            import paddle.distributed.fleet as fleet

            fleet.init(is_collective=True)

            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

            # do distributed training


    Example for parameter server training:

        .. code-block:: python

            import paddle.distributed.fleet as fleet

            fleet.init()

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
        self._role_maker = None
        self.strategy_compiler = None
        self._is_collective = False
        self._runtime_handle = None
        self._util = None

    def init(self, role_maker=None, is_collective=False):
        """
        Initialize role_maker in Fleet.

        This function is responsible for the distributed architecture
        what you want to run your code behind.

        Args:
            role_maker (RoleMakerBase, optional): A ``RoleMakerBase`` containing the configuration
                of environment variables related to distributed training.If you did not initialize 
                the rolemaker by yourself, it will be automatically initialized to PaddleRoleMaker.
                The default value is None.
            is_collective (Boolean, optional): A ``Boolean`` variable determines whether the program 
                runs on the CPU or GPU. False means set distributed training using CPU, and True means
                GPU.The default value is False.The default value is False.
        Returns:
            None

        Examples1:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

        Examples2:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init(is_collective=True)

        Examples3:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                role = fleet.PaddleCloudRoleMaker
                fleet.init(role)

        """

        if role_maker is None:
            if isinstance(is_collective, bool):
                self._is_collective = is_collective
                self._role_maker = PaddleCloudRoleMaker(
                    is_collective=self._is_collective)
            else:
                raise ValueError(
                    "`is_collective` should be instance of `bool`, but got {}".
                    format(type(is_collective)))
        else:
            if isinstance(role_maker, RoleMakerBase):
                self._role_maker = role_maker
            else:
                raise ValueError(
                    "`role_maker` should be subclass of `RoleMakerBase`, but got {}".
                    format(type(role_maker)))
        self.strategy_compiler = StrategyCompiler()
        return None

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.is_first_worker()

        """
        return self._role_maker.is_first_worker()

    def worker_index(self):
        """
        Get current worker index.

        Returns:
            int: node id

        Examples:

            .. code-block:: python
                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.worker_index()

        """
        return self._role_maker.worker_index()

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker numbers
        
        Examples:
            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.worker_num()

        """
        return self._role_maker.worker_num()

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.

        Examples:
            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.is_worker()

        """
        return self._role_maker.is_worker()

    def worker_endpoints(self, to_string=False):
        """
        Get current worker endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints

        Examples:
            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.worker_endpoints()

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

        Examples:
            .. code-block:: python
            import paddle.distributed.fleet as fleet
            fleet.init()
            fleet.server_num()
        """
        return len(self._role_maker.get_pserver_endpoints())

    def server_index(self):
        """
        Get current server index.

        Returns:
            int: node id

        Examples:
            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.server_index()

        """
        return self._role_maker.server_index()

    def server_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints

        Examples:
            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.server_endpoints()

        """

        if to_string:
            return ",".join(self._role_maker.get_pserver_endpoints())
        else:
            return self._role_maker.get_pserver_endpoints()

    def is_server(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.

        Examples:

            .. code-block:: python
                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.is_server()

        """
        return self._role_maker.is_server(
        ) or self._role_maker._is_heter_worker()

    @property
    def util(self):
        """
        Utility functions that can be used under certain runtime
        return util

        Returns:
            UtilBase: instance of UtilBase, can use distributed ops/tools easily.

        Examples:

            .. code-block:: python
                import paddle.distributed.fleet as fleet
                fleet.init()
                util = fleet.util
                files = ["1.log", "2.log", "3.log", "4.log"]
                files = util.get_file_shard()

        """
        return self._util

    @util.setter
    def util(self, util):
        """
        Set Utility functions for userd-defined runtime

        Returns:
            None
        """
        self._util = util

    def barrier_worker(self):
        """
        barrier all workers

        Returns:
            None
        """
        self._role_maker.barrier_worker()

    @inited_runtime_handler
    def init_worker(self):
        """
        initialize `Communicator` for parameter server training.


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_worker()

        """
        self._runtime_handle._init_worker()

    @inited_runtime_handler
    def init_server(self, *args, **kwargs):
        """
        init_server executor to initialize startup program,
        if the `args` is not empty, it will run load_persistables for increment training.


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_server()

        """
        self._runtime_handle._init_server(*args, **kwargs)

    @inited_runtime_handler
    def run_server(self):
        """
        run server will run pserver main program with executor.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                if fleet.is_server():
                    fleet.init_server()

        """
        self._runtime_handle._run_server()

    @inited_runtime_handler
    def stop_worker(self):
        """
        stop `Communicator` and give training complete notice to parameter server.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_server()

        """
        self._runtime_handle._stop_worker()

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        """
        save inference model for inference.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_server()

        """

        self._runtime_handle._save_inference_model(
            executor, dirname, feeded_var_names, target_vars, main_program,
            export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None):
        """

        saves all persistable variables from :code:`main_program` to
        the folder :code:`dirname`. You can refer to

        The :code:`dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set :code:`filename` None.

        Args:
            executor(Executor): The executor to run for saving persistable variables.
                                You can refer to :ref:`api_guide_executor_en` for
                                more details.

            dirname(str, optional): The saving directory path.
                                When you need to save the parameter to the memory, set it to None.
            main_program(Program, optional): The program whose persistbale variables will
                                             be saved. Default: None.


        Returns:
            None

        Examples:

            .. code-block:: text

                import paddle.distributed.fleet as fleet
                import paddle.fluid as fluid

                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                exe = fluid.Executor(fluid.CPUPlace())
                fleet.save_persistables(exe, "dirname", fluid.default_main_program())

        """

        self._runtime_handle._save_persistables(executor, dirname, main_program)

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributedStrategy): Extra properties for distributed optimizer.

        Returns:
            Fleet: instance of fleet.

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
            .. code-block:: python

                import paddle
                import paddle.distributed.fleet as fleet

                fc_1 = paddle.fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
                fc_2 = paddle.fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
                prediction = paddle.fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
                cost = paddle.fluid.layers.cross_entropy(input=prediction, label=input_y)
                avg_cost = paddle.fluid.layers.mean(x=cost)

                role = fleet.role_maker.PaddleCloudRoleMaker(is_collective=True)
                fleet.init(role)
                strategy = fleet.DistributedStrategy()
                optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
                optimizer.minimize(avg_cost)

                # for more examples, please reference https://github.com/PaddlePaddle/FleetX

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
        self.valid_strategy._enable_env()

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
