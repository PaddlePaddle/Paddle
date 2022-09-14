#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import warnings
import paddle
import os
from types import MethodType
import numpy as np
from paddle.fluid.framework import _global_flags
from paddle.fluid import compiler
from .base.role_maker import UserDefinedRoleMaker, PaddleCloudRoleMaker, RoleMakerBase
from .base.strategy_compiler import StrategyCompiler
from .base.distributed_strategy import DistributedStrategy
from .base.meta_optimizer_factory import MetaOptimizerFactory
from .base.runtime_factory import RuntimeFactory
from paddle.fluid.wrapped_decorator import wrap_decorator
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.ir import apply_build_strategy
from .base import topology as tp
from .meta_parallel import model_parallel_random_seed
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid import core

__all__ = []


def apply_ir_passes(main_program, startup_program, config):
    build_strategy = config._user_defined_strategy.build_strategy._copy()
    if not _global_flags()['FLAGS_apply_pass_to_program']:
        return build_strategy

    pipeline_opt = getattr(main_program, "_pipeline_opt", {})
    if pipeline_opt:
        main_program = pipeline_opt["section_program"]
        startup_program = startup_program._pipeline_opt["startup_program"]

    pass_attrs = {"use_cuda": config._is_collective}
    fuse_all_reduce = config._user_defined_strategy.fuse_all_reduce_ops
    if fuse_all_reduce and build_strategy.fuse_all_optimizer_ops:
        # FIXME(zjl): currently, fuse_all_optimizer_ops
        # have conflict with fuse_all_reduce_ops because
        # RawProgramOptimizer also inserts coalesce_tensor
        # into program. These two procedures may conflict
        # in which vars are to be fused.
        warnings.warn(
            'Currently, the fuse_all_optimizer_ops pass has conflict with fuse_all_reduce_ops pass. Disable the fuse_all_optimizer_ops pass temporarily.'
        )
        build_strategy.fuse_all_optimizer_ops = False

    return apply_build_strategy(main_program, startup_program, build_strategy,
                                pass_attrs)


def _inited_runtime_handler_(func):

    def __impl__(*args, **kwargs):
        cls = args[0]

        if cls._runtime_handle is None:
            raise ValueError("Fleet can not find suitable runtime handler")

        return func(*args, **kwargs)

    return __impl__


def _is_non_distributed_check_(func):

    def __impl__(*args, **kwargs):
        cls = args[0]

        if cls._role_maker is not None and cls._role_maker._is_non_distributed(
        ) is True:
            warnings.warn(
                "%s() function doesn't work when use non_distributed fleet." %
                (func.__name__))
            return

        return func(*args, **kwargs)

    return __impl__


inited_runtime_handler = wrap_decorator(_inited_runtime_handler_)
is_non_distributed_check = wrap_decorator(_is_non_distributed_check_)


class Fleet(object):
    """
    Unified API for distributed training of PaddlePaddle
    Please reference the https://github.com/PaddlePaddle/FleetX for details


    Returns:
        Fleet: A Fleet instance

    Example for collective training:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            import paddle.distributed.fleet as fleet

            fleet.init(is_collective=True)

            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

            # do distributed training


    Example for parameter server training:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            fleet.init(strategy=strategy)

            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer)

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
        self._context = {}
        self.user_defined_optimizer = paddle.optimizer.Optimizer(0.0)

    def init(self, role_maker=None, is_collective=False, strategy=None):
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
                runs on Collective mode or ParameterServer mode. True means the program runs on
                Collective mode, and False means running on ParameterServer mode. The default value 
                is False.
            strategy (DistributedStrategy): Extra properties for distributed training. 
                For details, please refer to paddle.distributed.fleet.DistributedStrategy. Default: None.


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
                role = fleet.PaddleCloudRoleMaker()
                fleet.init(role)

        Examples4:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                strategy = fleet.DistributedStrategy()
                fleet.init(strategy=strategy)

        """
        if strategy is None:
            strategy = DistributedStrategy()
        self._user_defined_strategy = copy.deepcopy(strategy)

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
                self._is_collective = role_maker._is_collective
            else:
                raise ValueError(
                    "`role_maker` should be subclass of `RoleMakerBase`, but got {}"
                    .format(type(role_maker)))
        self._role_maker._generate_role()

        import paddle.distributed.fleet as fleet
        fleet.util._set_role_maker(self._role_maker)

        self.strategy_compiler = StrategyCompiler()

        if self._role_maker._is_non_distributed() and self._is_collective:
            if paddle.fluid.core.is_compiled_with_cuda():
                gpus_num = paddle.fluid.core.get_cuda_device_count()
                if gpus_num != 1:
                    raise ValueError(
                        "CUDA_VISIBLE_DEVICES shoule be set only 1 card if you use `python` to launch fleet program."
                    )

        if paddle.fluid.framework._non_static_mode():
            if self.worker_num() == 1:
                # if worker_num is 1, should construct default topology & hcg
                self._topology = tp.CommunicateTopology()
                self._hcg = tp.HybridCommunicateGroup(self._topology)
                return
            if parallel_helper._is_parallel_ctx_initialized():
                warnings.warn(
                    "The dygraph parallel environment has been initialized.")
            else:
                # FLAGS_nccl_nrings is used for dynamic graph multi-stream communication
                if "FLAGS_nccl_nrings" in os.environ:
                    warnings.warn(
                        "You have set the environment variable FLAGS_nccl_nrings "
                        "outside the program, so the nccl_comm_num in "
                        "DistributedStrategy will not take effect here.")
                else:
                    os.environ["FLAGS_nccl_nrings"] = str(
                        self._user_defined_strategy.nccl_comm_num)
                paddle.distributed.init_parallel_env()

            # hybrid parallel not support for npu/xpu
            if self._user_defined_strategy.heter_ccl_mode == False:
                # init hybrid parallel environment in dygraph
                if tp._HYBRID_PARALLEL_GROUP is None:
                    self._init_hybrid_parallel_env()
                else:
                    warnings.warn(
                        "The dygraph hybrid parallel environment has been initialized."
                    )
        elif self._is_collective:
            use_sharding = self._user_defined_strategy.sharding

            # global group
            global_rank = self.worker_index()
            global_world_size = self.worker_num()
            # NOTE(wangxi): see sharding_optimizer
            global_ring_id = 3 if use_sharding else 0
            global_ranks = list(range(global_world_size))

            if tp._HYBRID_PARALLEL_GROUP is None: tp._CommunicateGroup()
            cg = tp._HYBRID_PARALLEL_GROUP
            self._hcg = cg
            cg.set_comm_group('global', global_rank, global_world_size,
                              global_ring_id, global_ranks)

            use_tensor_parallel = self._user_defined_strategy.tensor_parallel
            use_mp = use_sharding or use_tensor_parallel

            # hybrid group
            if use_mp is False: return

            mp_degree_sharding = 1
            mp_degree_tensor_parallel = 1
            if use_sharding:
                sharding_configs = self._user_defined_strategy.sharding_configs
                mp_degree_sharding = int(sharding_configs['mp_degree'])

            if use_tensor_parallel:
                tensor_parallel_configs = self._user_defined_strategy.tensor_parallel_configs
                mp_degree_tensor_parallel = int(
                    tensor_parallel_configs['tensor_parallel_degree'])

            if use_sharding and use_tensor_parallel:
                assert mp_degree_sharding == mp_degree_tensor_parallel

            mp_degree = mp_degree_sharding if use_sharding else mp_degree_tensor_parallel

            if mp_degree > 1:
                assert global_world_size % mp_degree == 0
                # NOTE(wangxi): mp_ring_id sync with sharding_optimizer.py _build_groups
                mp_ring_id = 0
                mp_rank = global_rank % mp_degree
                mp_group_id = global_rank // mp_degree
                mp_group_ranks = [
                    idx for idx in global_ranks
                    if idx // mp_degree == mp_group_id
                ]
                cg.set_comm_group('model', mp_rank, mp_degree, mp_ring_id,
                                  mp_group_ranks)
        return self

    def _init_hybrid_parallel_env(self):
        """initialize the hybrid environment
        """
        self.hybrid_configs = self._user_defined_strategy.hybrid_configs
        self.dp_degree = self.hybrid_configs["dp_degree"]
        self.mp_degree = self.hybrid_configs["mp_degree"]
        self.pp_degree = self.hybrid_configs["pp_degree"]
        self.sharding_degree = self.hybrid_configs["sharding_degree"]

        assert self.mp_degree >= 0, "mp_degree should be greater or equal to 0"
        assert self.pp_degree >= 0, "pp_degree should be greater or equal to 0"
        assert self.sharding_degree >= 0, "sharding_degree should be greater or equal to 0"

        self.mp_degree = max(self.mp_degree, 1)
        self.pp_degree = max(self.pp_degree, 1)

        if self.dp_degree < 0:
            nranks = paddle.distributed.get_world_size()
            self.dp_degree = nranks // (self.mp_degree * self.pp_degree)

        self.dp_degree = max(self.dp_degree, 1)

        self._topology = tp.CommunicateTopology(
            hybrid_group_names=["data", "pipe", "sharding", "model"],
            dims=[
                self.dp_degree, self.pp_degree, self.sharding_degree,
                self.mp_degree
            ])

        self._hcg = tp.HybridCommunicateGroup(self._topology)

        if self.mp_degree > 1:
            tensor_parallel_configs = self._user_defined_strategy.tensor_parallel_configs
            tensor_init_seed = tensor_parallel_configs["tensor_init_seed"]
            if tensor_init_seed == -1:
                model_parallel_random_seed()
            else:
                model_parallel_random_seed(tensor_init_seed)

    def get_hybrid_communicate_group(self):
        assert self._hcg is not None
        return self._hcg

    def get_hybrid_parallel_topology(self):
        assert self._topology is not None
        return self._topology

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
        return self._role_maker._is_first_worker()

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
        return self._role_maker._worker_index()

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
        return self._role_maker._worker_num()

    def node_num(self):
        return self._role_maker._get_node_num()

    def local_rank(self):
        return self._role_maker._get_local_rank()

    def local_device_ids(self):
        return self._role_maker._get_local_device_ids()

    def world_device_ids(self):
        return self._role_maker._get_world_device_ids()

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
        return self._role_maker._is_worker()

    def is_coordinator(self):
        return self._role_maker._is_coordinator()

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
        if to_string:
            return ",".join(self._role_maker._get_trainer_endpoints())
        else:
            return self._role_maker._get_trainer_endpoints()

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
        return len(self._role_maker._get_pserver_endpoints())

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
        return self._role_maker._server_index()

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
            return ",".join(self._role_maker._get_pserver_endpoints())
        else:
            return self._role_maker._get_pserver_endpoints()

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
        return self._role_maker._is_server()

    def barrier_worker(self):
        """
        barrier all workers

        Returns:
            None
        """
        self._role_maker._barrier("worker")

    @is_non_distributed_check
    @inited_runtime_handler
    def init_worker(self, scopes=None):
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
        self._runtime_handle._init_worker(scopes)

    @is_non_distributed_check
    @inited_runtime_handler
    def init_coordinator(self, scopes=None):
        """
        initialize coordinator node
        """
        self._runtime_handle._init_coordinator(scopes)

    def make_fl_strategy(self):
        self._runtime_handle._make_fl_strategy()

    @is_non_distributed_check
    @inited_runtime_handler
    def get_fl_client(self):
        """
        get worker(training node) ptr
        """
        return self._runtime_handle._worker

    @is_non_distributed_check
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

    @is_non_distributed_check
    @inited_runtime_handler
    def load_model(self, path, mode):
        """
        load fleet model from path


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.load_model("path", mode=0)

        """
        self._runtime_handle._load_persistables(path, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def load_one_table(self, table_id, path, mode):
        """
        load fleet one table from path


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.load_one_table(0, "path", mode=0)

        """
        self._runtime_handle._load_one_table(table_id, path, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def load_inference_model(self, path, mode):
        """
        load fleet inference model from path


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.load_inference_model("path", mode=1)

        """
        self._runtime_handle._load_inference_model(path, mode)

    @is_non_distributed_check
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

    @is_non_distributed_check
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

    @is_non_distributed_check
    @inited_runtime_handler
    def save(self, dirname, feed=[], fetch=[], **configs):
        inference = True

        if not feed and not fetch:
            inference = False

        place = paddle.CPUPlace()
        executor = paddle.static.Executor(place)

        if inference:
            feeded_var_names = []
            fetch_var_names = []

            for var in feed:
                if isinstance(var, str):
                    feeded_var_names.append(var)
                elif isinstance(var, paddle.static.Variable):
                    feeded_var_names.append(var.name)
                else:
                    raise ValueError("feed must be [str|Variable]")

            for var in fetch:
                if isinstance(var, str):
                    fetch_var_names.append(var)
                elif isinstance(var, paddle.static.Variable):
                    fetch_var_names.append(var.name)
                else:
                    raise ValueError("feed must be [str|Variable]")

            fetch_vars = [
                paddle.static.default_main_program().global_block().var(name)
                for name in fetch_var_names
            ]

            self._runtime_handle._save_inference_model(executor, dirname,
                                                       feeded_var_names,
                                                       fetch_vars, None, True,
                                                       0)
        else:
            increment_mode = 0
            if "mode" in configs:
                increment_mode = int(configs["mode"])
            self._runtime_handle._save_persistables(executor,
                                                    dirname,
                                                    main_program=None,
                                                    mode=increment_mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True,
                             mode=0):
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
        # warnings.warn(
        #     "'save_inference_model' is a deprecated, will be deleted after v2.2.0, Please use fleet.save instead."
        # )

        self._runtime_handle._save_inference_model(executor, dirname,
                                                   feeded_var_names,
                                                   target_vars, main_program,
                                                   export_for_deployment, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def save_persistables(self, executor, dirname, main_program=None, mode=0):
        """

        saves all persistable tensors from :code:`main_program` to
        the folder :code:`dirname`. You can refer to

        The :code:`dirname` is used to specify the folder where persistable tensors
        are going to be saved. If you would like to save tensors in separate
        files, set :code:`filename` None.

        Args:
            executor(Executor): The executor to run for saving persistable tensors.
                                You can refer to :ref:`api_guide_executor_en` for
                                more details.

            dirname(str, optional): The saving directory path.
                                When you need to save the parameter to the memory, set it to None.
            main_program(Program, optional): The program whose persistbale tensors will
                                             be saved. Default: None.


        Returns:
            None

        Examples:

            .. code-block:: text

                import paddle
                paddle.enable_static()
                import paddle.distributed.fleet as fleet

                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                exe = paddle.static.Executor(paddle.CPUPlace())
                fleet.save_persistables(exe, "dirname", paddle.static.default_main_program())

        """
        # warnings.warn(
        #     "'save_persistables' is a deprecated, will be deleted after v2.2.0, Please use fleet.save instead."
        # )

        self._runtime_handle._save_persistables(executor, dirname, main_program,
                                                mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def save_cache_model(self, dirname, **configs):
        return self._runtime_handle._save_cache_model(dirname, **configs)

    @is_non_distributed_check
    @inited_runtime_handler
    def check_save_pre_patch_done(self):
        return self._runtime_handle._check_save_pre_patch_done()

    @is_non_distributed_check
    @inited_runtime_handler
    def save_one_table(self, table_id, path, mode):
        """
        save fleet one table from path


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.save_one_table(0, "path", mode=0)

        """
        self._runtime_handle._save_one_table(table_id, path, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def save_dense_params(self,
                          executor,
                          dirname,
                          scope,
                          program,
                          var_names=None):
        """
        save fleet one table from path


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                import paddle
                place = paddle.fluid.CPUPlace()
                exe = paddle.fluid.Executor(place)

                # build net
                # fleet.distributed_optimizer(...)

                fleet.save_dense_params(exe, "path", scope=paddle.static.global_scope(), program=paddle.static.default_main_program())

        """
        self._runtime_handle._save_dense_params(executor, dirname, scope,
                                                program, var_names)

    def shrink(self, threshold=None):
        self._runtime_handle._shrink(threshold)

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributedStrategy): Extra properties for distributed optimizer. 
                It is recommended to use DistributedStrategy in fleet.init(). The strategy
                here is for compatibility. If the strategy in fleet.distributed_optimizer() 
                is not None, then it will overwrite the DistributedStrategy in fleet.init(), 
                which will take effect in distributed training.

        Returns:
            Fleet: instance of fleet.

        Examples:

            .. code-block:: python

                import paddle
                import paddle.distributed.fleet as fleet
                fleet.init(is_collective=True)
                strategy = fleet.DistributedStrategy()
                optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

        """
        self.user_defined_optimizer = optimizer

        if strategy is not None:
            if self._is_collective:
                warnings.warn(
                    "It is recommended to use DistributedStrategy "
                    "in fleet.init(). The strategy here is only for compatibility. "
                    "If the strategy in fleet.distributed_optimizer() is "
                    "not None, then it will overwrite the DistributedStrategy in fleet.init(), "
                    "which will take effect in distributed training.")
            self._user_defined_strategy = copy.deepcopy(strategy)

        self._context = {}

        return self

    def _get_amp_optimizer(self):
        # imitate target optimizer retrieval
        amp_optimizer = None
        for optimizer in self.strategy_compiler._get_applied_meta_optimizer():
            if hasattr(optimizer, 'amp_init'):
                amp_optimizer = optimizer
                break

        if amp_optimizer is None:
            if hasattr(self.user_defined_optimizer, 'amp_init'):
                amp_optimizer = self.user_defined_optimizer

        assert amp_optimizer is not None, \
            "amp_init can only be used when the amp(auto mixed precision) strategy is turned on."
        return amp_optimizer

    def get_loss_scaling(self):
        """Return the real-time loss scaling factor.
        """
        amp_optimizer = self._get_amp_optimizer()
        return amp_optimizer.get_loss_scaling()

    def amp_init(self,
                 place,
                 scope=None,
                 test_program=None,
                 use_fp16_test=False):
        """
        Init the amp training, such as cast fp32 parameters to fp16 type.
  
        Args:
            place(CUDAPlace): place is used to initialize 
                fp16 parameters with fp32 values.
            scope(Scope): The scope is used to find fp32 parameters.
            test_program(Program): The program is used for testing.
            use_fp16_test(bool): Whether to use fp16 testing.
            
        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                import paddle.nn.functional as F
                paddle.enable_static()

                def run_example_code():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
                    conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)
                    # 1) Use fp16_guard to control the range of fp16 kernels used.
                    with paddle.static.amp.fp16_guard():
                        bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                        pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                        hidden = paddle.static.nn.fc(pool, size=10)
                        loss = paddle.mean(hidden)
                    # 2) Create the optimizer and set `multi_precision` to True.
                    # Setting `multi_precision` to True can avoid the poor accuracy
                    # or the slow convergence in a way. 
                    optimizer = paddle.optimizer.Momentum(learning_rate=0.01, multi_precision=True)
                    # 3) These ops in `custom_black_list` will keep in the float32 computation type.
                    amp_list = paddle.static.amp.CustomOpLists(
                        custom_black_list=['pool2d'])
                    # 4) The entry of Paddle AMP.
                    # Enable pure fp16 training by setting `use_pure_fp16` to True.
                    optimizer = paddle.static.amp.decorate(
                        optimizer,
                        amp_list,
                        init_loss_scaling=128.0,
                        use_dynamic_loss_scaling=True,
                        use_pure_fp16=True)
                    # If you don't use the default_startup_program(), you sholud pass
                    # your defined `startup_program` into `minimize`.
                    optimizer.minimize(loss)
                    exe.run(paddle.static.default_startup_program())
                    # 5) Use `amp_init` after FP32 parameters initialization(such as `exe.run(startup_program)`).
                    # If you want to perform the testing process, you should pass `test_program` into `amp_init`.
                    optimizer.amp_init(place, scope=paddle.static.global_scope())
                    
                if paddle.is_compiled_with_cuda() and len(paddle.static.cuda_places()) > 0:
                    run_example_code()       
        """
        amp_optimizer = self._get_amp_optimizer()
        return amp_optimizer.amp_init(place, scope, test_program, use_fp16_test)

    def _final_strategy(self):
        if "valid_strategy" not in self._context:
            print(
                "WARNING: You may need to call minimize function before this function is called"
            )
            return {}
        else:
            return self._context["valid_strategy"]

    def _get_applied_meta_list(self):
        if "applied_meta_list" not in self._context:
            print(
                "WARNING: You may need to call minimize function before _get_applied_meta_list called"
            )
            return []
        else:
            return self._context["applied_meta_list"]

    def _get_applied_graph_list(self):
        if "applied_graph_list" not in self._context:
            print(
                "WARNING: You may need to call minimize function before _get_applied_graph_list called"
            )
            return []
        else:
            return self._context["applied_graph_list"]

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Add distributed operations to minimize ``loss`` by updating ``parameter_list``.

        Args:
            loss (Tensor): A ``Tensor`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (Iterable, optional): Iterable of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) tensor pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to
            indicate program pruning. If so, the program will be pruned by ``feed`` and
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:

            .. code-block:: python

                import paddle
                paddle.enable_static()
                import paddle.distributed.fleet as fleet
                import paddle.nn.functional as F

                hid_dim = 10
                label_dim = 2
                input_x = paddle.static.data(name='x', shape=[None, 13], dtype='float32')
                input_y = paddle.static.data(name='y', shape=[None, 1], dtype='int64')
                fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim, activation='tanh')
                fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim, activation='tanh')
                prediction = paddle.static.nn.fc(x=[fc_2], size=label_dim, activation='softmax')
                cost = F.cross_entropy(input=prediction, label=input_y)
                avg_cost = paddle.mean(x=cost)

                fleet.init(is_collective=True)
                strategy = fleet.DistributedStrategy()
                optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
                optimizer.minimize(avg_cost)

                # for more examples, please reference https://github.com/PaddlePaddle/FleetX

        """
        if not isinstance(loss, list):
            return self._minimize_impl(loss, startup_program, parameter_list,
                                       no_grad_set)
        else:
            if paddle.fluid.framework._non_static_mode(
            ) or self._role_maker._is_non_distributed() or self._is_collective:
                raise ValueError("loss can be list only in PS mode")
            return self._minimize_losses_impl(loss, startup_program,
                                              parameter_list, no_grad_set)

    def _minimize_impl(self,
                       loss,
                       startup_program=None,
                       parameter_list=None,
                       no_grad_set=None):
        context = {}
        context["user_defined_strategy"] = copy.deepcopy(
            self._user_defined_strategy)
        if paddle.fluid.framework._non_static_mode():
            # imitate target optimizer retrieval
            target_opt = self.user_defined_optimizer
            self._context = context
            return target_opt.minimize(loss)

        # cache original feed forward program
        self.origin_main_program = loss.block.program
        # add distributed attr
        if not hasattr(self.origin_main_program, "distributed_info_"):
            setattr(self.origin_main_program, "distributed_info_", dict())
            self.origin_main_program.distributed_info_[
                "dp_degree"] = self._user_defined_strategy.sharding_configs[
                    "dp_degree"]
            self.origin_main_program.distributed_info_[
                "mp_degree"] = self._user_defined_strategy.sharding_configs[
                    "mp_degree"]
            self.origin_main_program.distributed_info_[
                "pp_degree"] = self._user_defined_strategy.sharding_configs[
                    "pp_degree"]
            self.origin_main_program.distributed_info_[
                "sharding_degree"] = self._user_defined_strategy.sharding_configs[
                    "sharding_degree"]

        context["origin_main_program"] = self.origin_main_program
        context["origin_main_programs"] = [self.origin_main_program]
        context["loss"] = loss
        if startup_program == None:
            self.origin_startup_program = \
                paddle.static.default_startup_program().clone(for_test=False)
            startup_program = paddle.static.default_startup_program()
        else:
            self.origin_startup_program = \
                startup_program.clone(for_test=False)

        context["origin_startup_program"] = startup_program
        context["origin_startup_programs"] = [startup_program]
        context["role_maker"] = self._role_maker

        # Use the auto-parallel's routines instead
        if self._user_defined_strategy.semi_auto or self._user_defined_strategy.auto_search:
            from ..auto_parallel.parallelizer import AutoParallelizer
            auto_parallelizer = AutoParallelizer(self)
            optimize_ops, params_grads, dist_startup_prog, dist_main_prog = auto_parallelizer.parallelize(
                loss, startup_program, parameter_list, no_grad_set)

            return optimize_ops, params_grads, dist_startup_prog, dist_main_prog

        # compile time
        distributed_optimizer_list = \
            MetaOptimizerFactory()._get_valid_meta_optimizers(
                self.user_defined_optimizer)

        context["user_defined_strategy"] = copy.deepcopy(
            self._user_defined_strategy)
        copy_user_defined_strategy = copy.deepcopy(self._user_defined_strategy)

        # trigger the auto-parallel in very strict condition
        # strategy = DistributedStrategy()
        # strategy.auto = True
        # optimizer = paddle.optimizer.SGD(learning_rate=0.1)
        # optimizer = fleet.distributed_optimizer(optimizer, strategy)
        if copy_user_defined_strategy._is_strict_auto():
            # turn on all the strategy for each optimizer
            for opt in distributed_optimizer_list:
                opt._enable_strategy(copy_user_defined_strategy, context)

        valid_optimizer_list = []
        valid_graph_optimizer_list = []
        can_not_apply_optimizer_list = []
        # recall meta optimizers for ranking
        for opt in distributed_optimizer_list:
            opt._set_basic_info(loss, self._role_maker,
                                self.user_defined_optimizer,
                                copy_user_defined_strategy)
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
                copy_user_defined_strategy, valid_optimizer_list,
                valid_graph_optimizer_list)

        valid_strategy = self.strategy_compiler._get_valid_strategy(
            copy_user_defined_strategy, can_not_apply_optimizer_list)

        context["valid_strategy"] = copy.deepcopy(valid_strategy)
        # print("valid_strategy:", context["valid_strategy"])
        # print("user_defined_strategy:", context["user_defined_strategy"])

        applied_meta_list = self.strategy_compiler._get_applied_meta_list()
        applied_graph_list = self.strategy_compiler._get_applied_graph_list()

        context['applied_meta_list'] = applied_meta_list
        context['applied_graph_list'] = applied_graph_list

        self._context = context

        self.valid_strategy = valid_strategy
        self.valid_strategy._enable_env()

        optimize_ops = []
        params_grads = []

        if self._role_maker._is_non_distributed() and not self._is_collective:
            if self._runtime_handle is None:
                self._runtime_handle = RuntimeFactory()._create_runtime(context)

            compiled_program = compiler.CompiledProgram(
                self.origin_main_program).with_data_parallel(
                    loss_name=loss.name, share_vars_from=None)
            loss.block.program._graph = compiled_program
            return self.user_defined_optimizer.minimize(loss,
                                                        startup_program,
                                                        parameter_list,
                                                        no_grad_set=no_grad_set)

        if meta_optimizer:
            # print("before minimize program id:", id(loss.block.program))
            optimize_ops, params_grads = meta_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set=no_grad_set)
            # print("after minimize program id:", id(loss.block.program))

            default_program = paddle.static.default_main_program()
            # print("default program id:", id(default_program))

            if id(default_program) != id(loss.block.program):
                paddle.fluid.framework.switch_main_program(loss.block.program)
            # print("default program id after switch:", id(default_program))

        else:
            optimize_ops, params_grads = self.user_defined_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set=no_grad_set)

        context["program_optimize_ops"] = optimize_ops
        context["program_params_grads"] = params_grads

        if graph_optimizer:
            # print("before graph minimize program id:", id(loss.block.program))
            optimize_ops, params_grads = graph_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set=no_grad_set)
            # since we do not encourage users to use graph operations
            # if a graph optimizer takes effect, mostly
            # optimizers_ops and params_grads are None
            # i.e. users can not modify current computation graph anymore
            context["graph_optimize_ops"] = optimize_ops
            context["graph_optimize_grads"] = params_grads
        else:
            apply_ir_passes(loss.block.program, startup_program, self)

        if not self._role_maker._is_heter_parameter_server_mode:
            program = paddle.static.default_main_program()
            opt_info = {} if program._fleet_opt is None else program._fleet_opt
            opt_info["mpi_size"] = self.worker_num()
            opt_info["mpi_rank"] = self.worker_index()
            for k, v in self._user_defined_strategy.trainer_desc_configs.items(
            ):
                if v or k not in opt_info:
                    opt_info[k] = v
            program._fleet_opt = opt_info

        if self._runtime_handle is None:
            self._runtime_handle = RuntimeFactory()._create_runtime(context)

        import paddle.distributed.fleet as fleet
        fleet.util._set_strategy(context["valid_strategy"])

        return optimize_ops, params_grads

    def _minimize_losses_impl(self,
                              losses,
                              startup_programs=None,
                              parameter_list=None,
                              no_grad_set=None):
        context = {}

        # cache original feed forward program
        self.origin_main_program = losses[0].block.program
        context["origin_main_program"] = self.origin_main_program
        context["origin_main_programs"] = []
        for loss in losses:
            context["origin_main_programs"].append(loss.block.program)
        context["loss"] = losses

        if startup_programs is None:
            if len(losses) == 1:
                startup_programs = [paddle.static.default_startup_program()]
            else:
                raise ValueError(
                    "startup_program can't be None when loss is list.")
        self.origin_startup_program = startup_programs[0].clone(for_test=False)
        context["origin_startup_program"] = startup_programs[0]
        context["origin_startup_programs"] = []
        for program in startup_programs:
            context["origin_startup_programs"].append(program)

        context["role_maker"] = self._role_maker

        context["user_defined_strategy"] = copy.deepcopy(
            self._user_defined_strategy)

        context["valid_strategy"] = copy.deepcopy(self._user_defined_strategy)

        self._context = context

        self.valid_strategy = context["valid_strategy"]
        self.valid_strategy._enable_env()

        optimize_ops = []
        params_grads = []

        from .meta_optimizers import ParameterServerOptimizer
        ps_optimizer = ParameterServerOptimizer(self.user_defined_optimizer)
        ps_optimizer._set_basic_info(losses, self._role_maker,
                                     self.user_defined_optimizer,
                                     self._user_defined_strategy)
        optimize_ops, params_grads = ps_optimizer.minimize_losses_impl(
            losses, startup_programs, parameter_list, no_grad_set=no_grad_set)

        # default_program = paddle.static.default_main_program()

        # if id(default_program) != id(losses[0].block.program):
        #     paddle.fluid.framework.switch_main_program(losses[0].block.program)

        context["program_optimize_ops"] = optimize_ops
        context["program_params_grads"] = params_grads

        for loss in losses:
            program = loss.block.program
            opt_info = {} if program._fleet_opt is None else program._fleet_opt
            opt_info["mpi_size"] = self.worker_num()
            opt_info["mpi_rank"] = self.worker_index()
            for k, v in self._user_defined_strategy.trainer_desc_configs.items(
            ):
                if v or k not in opt_info:
                    opt_info[k] = v
            program._fleet_opt = opt_info
            # print("fleet base opt info:", id(program), program._fleet_opt)

        if self._runtime_handle is None:
            self._runtime_handle = RuntimeFactory()._create_runtime(context)

        import paddle.distributed.fleet as fleet
        fleet.util._set_strategy(context["valid_strategy"])

        return optimize_ops, params_grads
