#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import multiprocessing
import os
import sys
import warnings
from . import framework
from .framework import _get_paddle_place, _get_paddle_place_list
from .framework import cuda_places, cpu_places, xpu_places
from . import core
import logging
from .data_feeder import convert_dtype
import hashlib

__all__ = [
    'CompiledProgram',
    'ExecutionStrategy',
    'BuildStrategy',
    'IpuCompiledProgram',
    'IpuStrategy',
    'GcuOptions',
]

ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy
BuildStrategy = core.ParallelExecutor.BuildStrategy
InferNativeConfig = core.NativeConfig
InferAnalysisConfig = core.AnalysisConfig
DeviceType = core.DeviceType


def _place_obj(place):
    p = core.Place()
    p.set_place(place)
    return p


def _is_pserver_mode(main_program):
    main = main_program if main_program else framework.default_main_program()
    for op in main.global_block().ops:
        if op.type in ["send", "recv"]:
            return True
    return False


def _has_backward_op(graph):
    for node in graph.nodes():
        if (
            node.is_op()
            and node.op() is not None
            and node.op().type().endswith("_grad")
        ):
            return True
    return False


def _prune_feed_ops(program):
    # prune the feed ops in the program.
    pop_idx = []
    for i, op in enumerate(program.global_block().ops):
        if op.type == "feed":
            pop_idx.append(i)
    for index in pop_idx[::-1]:
        program.global_block()._remove_op(index)


def _has_optimize_op(block):
    for op in block.ops:
        op_maker = core.op_proto_and_checker_maker
        optimize = core.op_proto_and_checker_maker.OpRole.Optimize
        if op_maker.kOpRoleVarAttrName() in op.attr_names and int(
            op.all_attrs()[op_maker.kOpRoleAttrName()]
        ) == int(optimize):
            return True
    return False


def _should_broadcast_or_not_exists(program, var_name):
    block = program.global_block()
    var = block.vars.get(var_name, None)
    if var is None:
        return True
    is_distributed = getattr(var, '_is_distributed', False) or getattr(
        var, 'is_distributed', False
    )
    return not is_distributed


class CompiledProgram:
    """
    :api_attr: Static Graph

    The CompiledProgram is used to transform a program or graph for
    various optimizations according to the configuration of build_strategy,
    for example, the operators' fusion in the computation graph, memory
    optimization during the execution of the computation graph, etc.
    For more information about build_strategy, please refer to
    :code:`paddle.static.BuildStrategy`.

    Args:
        program_or_graph (Graph|Program): This argument is the Program or Graph
            being executed.
        build_strategy(BuildStrategy): This argument is used to compile the
            program or graph with the specified options, such as operators' fusion
            in the computational graph and memory optimization during the execution
            of the computational graph. For more information about build_strategy,
            please refer to :code:`paddle.static.BuildStrategy`. The default is None.

    Returns:
        CompiledProgram

    Example:
        .. code-block:: python

            import numpy
            import paddle
            import paddle.static as static

            paddle.enable_static()

            place = paddle.CUDAPlace(0) # paddle.CPUPlace()
            exe = static.Executor(place)

            data = static.data(name='X', shape=[None, 1], dtype='float32')
            hidden = static.nn.fc(x=data, size=10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

            exe.run(static.default_startup_program())
            compiled_prog = static.CompiledProgram(
                static.default_main_program())

            x = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_prog,
                                feed={"X": x},
                                fetch_list=[loss.name])
    """

    def __init__(self, program_or_graph, build_strategy=None):
        if isinstance(program_or_graph, core.Graph):
            self._graph = program_or_graph
            # don't not create a new program here.
            self._program = None
        elif isinstance(program_or_graph, framework.Program):
            _prune_feed_ops(program_or_graph)
            self._graph = core.Graph(program_or_graph.desc)
            self._program = program_or_graph
        else:
            raise TypeError(
                "The type of program_to_graph parameter is wrong, expected Graph or Program, but received %s"
                % type(program_or_graph)
            )

        self._scope = None
        self._place = None
        self._executor = None
        self._compiled = False
        self._is_inference = False
        self._share_vars_from = None
        self._places = None
        self._build_strategy = build_strategy
        self._exec_strategy = None

    def _with_inference_optimize(self, config):
        """Add inference optimize

        Args:
            config: instance of `NativeConfig` or `AnalysisConfig` to create predictor
        Returns:
            self
        """
        assert (
            not self._is_inference
        ), "Already compiled with inference, cannot be recompiled."

        assert any(
            [
                isinstance(config, InferNativeConfig),
                isinstance(config, InferAnalysisConfig),
            ]
        )
        self._is_inference = True
        self._infer_config = config
        return self

    def _with_distributed(self):
        raise NotImplementedError(
            "Subclass of CompiledProgram should implement _with_distributed method."
        )

    def _compile_data_parallel(self, places, use_device, scope=None):
        if self._share_vars_from:
            if scope:
                sys.stderr.write("share_vars_from is set, scope is ignored.\n")
            if self._share_vars_from._executor is None:
                raise ValueError(
                    "The shared Program is not compiled and executed, so there is no "
                    "variables to share."
                )
            self._local_scopes = self._share_vars_from._executor.local_scopes()
        else:
            assert scope is not None, ""
            self._local_scopes = []

        assert isinstance(places, tuple) or isinstance(
            places, list
        ), "Currently , The places type can only be list or tuple, but the input type is {}.".format(
            type(places)
        )

        if self._build_strategy is None:
            self._build_strategy = BuildStrategy()
        self._build_strategy.is_distribution = _is_pserver_mode(self._program)

        if self._exec_strategy is None:
            self._exec_strategy = ExecutionStrategy()
        self._exec_strategy._use_device = use_device

        if self._exec_strategy.num_threads == 0:
            if self._exec_strategy._use_device == DeviceType.CUDA:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                self._exec_strategy.num_threads = len(places) * 4
            elif self._exec_strategy._use_device == DeviceType.XPU:
                # Currently only single thread is supported in Kunlun XPU.
                self._exec_strategy.num_threads = 1
            else:
                self._exec_strategy.num_threads = len(places) * 2

        if (
            "FLAGS_use_cinn" in core.globals()
            and core.globals()["FLAGS_use_cinn"]
            and self._exec_strategy.num_threads != 1
        ):
            warnings.warn(
                "At present, when CINN is turned on, each process can "
                "only contain one thread, so reset the number of threads to 1 here."
            )
            self._exec_strategy.num_threads = 1

        # TODO(wuyi): trainer endpoints should be passed in through
        # build_strategy, not program.xxx.
        # TODO(gongwb): let user to set them once.
        if (
            self._program
            and self._build_strategy.num_trainers > 1
            and self._program._trainers_endpoints
        ):
            tps = self._program._trainers_endpoints

            assert self._build_strategy.num_trainers == len(
                tps
            ), "The trainer numbers is not equal to endpoint numbers."
            self._build_strategy.trainers_endpoints = tps

        if self._program:
            self._build_strategy.nccl_comm_num = self._program._nccl_comm_num
            self._build_strategy.use_hierarchical_allreduce = (
                self._program._use_hierarchical_allreduce
            )
            self._build_strategy.hierarchical_allreduce_inter_nranks = (
                self._program._hierarchical_allreduce_inter_nranks
            )

        if self._build_strategy.sync_batch_norm:
            self._build_strategy.enable_sequential_execution = True

        if self._program is not None and self._program._enable_dgc:
            assert (
                self._exec_strategy._use_device == DeviceType.CUDA
            ), "DGC only used under CUDA environment."
            assert (
                self._build_strategy.num_trainers * len(places) > 1
            ), "DGC is not avaliable for single card training."
            assert (
                self._build_strategy.reduce_strategy
                == BuildStrategy.ReduceStrategy.AllReduce
            ), "DGC \
                only can be used for AllReduce BuildStrategy."

            # DGC doesn't support fuse for now, close fuse.
            self._build_strategy.fuse_all_reduce_ops = False

        self._persistable_vars = []
        for node in self._graph.nodes():
            if (
                node.is_var()
                and node.var() is not None
                and node.var().persistable()
                and node.var().type() != core.VarDesc.VarType.RAW
            ):
                name = node.name()
                if (
                    self._program is not None
                    and _should_broadcast_or_not_exists(self._program, name)
                ):
                    self._persistable_vars.append(node.name())

        places = list(map(_place_obj, places))

        # ParallelExecutor would broadcast all the parameters during initializing.
        # The parameters of each process should be in the same ordered for the data-parallelism
        # distributed training to keep the broadcast correct.
        self._persistable_vars = list(set(self._persistable_vars))
        self._persistable_vars.sort()

        if core.is_cuda_graph_capturing():
            raise RuntimeError(
                "CUDA Graph is not allowed to capture when running the first batch."
            )
        return core.ParallelExecutor(
            places,
            self._persistable_vars,
            '',
            self._scope,
            self._local_scopes,
            self._exec_strategy,
            self._build_strategy,
            self._graph,
        )

    def _compile_inference(self):
        return core.create_paddle_predictor(self._infer_config)

    def _compile(self, scope, place):
        """Compile the program based on the configs.

        Args:
            scope: The variables (resources) that are associated with
               this compiled program.
            place: The location that the compiled program will be run on.

        Returns:
            self
        """
        if self._compiled:
            if scope and self._scope != scope:
                raise ValueError("Cannot compile program with different scope.")
            if place and not self._place._equals(place):
                raise ValueError("Cannot compile program with different place.")
            return self
        self._compiled = True

        self._scope = scope
        self._place = place

        if self._is_inference:
            self._executor = self._compile_inference()
        else:
            self._places = [self._place]

            if isinstance(self._place, core.CUDAPlace):
                use_device = DeviceType.CUDA
            elif isinstance(self._place, core.XPUPlace):
                use_device = DeviceType.XPU
            else:
                use_device = DeviceType.CPU
            self._executor = self._compile_data_parallel(
                use_device=use_device, scope=self._scope, places=self._places
            )
        return self

    def _get_places(self, place, place_list):
        has_set_place = place_list is not None
        if has_set_place:
            for p in place_list:
                assert (
                    p._type() == place._type()
                ), "Place type not match. You may set wrong type of places."
        else:
            if isinstance(place, core.CUDAPlace):
                place_list = cuda_places()
            elif isinstance(place, core.XPUPlace):
                place_list = xpu_places()
            else:
                place_list = cpu_places()
        assert place_list, "No places for execution."
        return place_list


class IpuDynamicPatcher:
    """
    Patcher for IPU dynamic2static support.
    """

    patcher_cache = []

    def __init__(self):
        pass

    @staticmethod
    def convert_concrete_program(
        ipu_strategy, concrete_program, class_instance=None
    ):
        """
        Convert the ConcreteProgram to IPUConcreteProgram.
        """
        from ..fluid.dygraph.base import switch_to_static_graph
        from ..fluid import backward
        from ..fluid.framework import device_guard
        import paddle

        inputs = concrete_program.inputs
        outputs = concrete_program.outputs
        startup_program = concrete_program.startup_program

        scope = paddle.static.global_scope()

        @switch_to_static_graph
        def append_backward_desc():
            program = concrete_program.main_program

            # backward with optimizer to add backward graph to program
            backward.gradients_with_optimizer(program, ipu_strategy._optimizer)

            # initialize backward parameters
            exe = paddle.static.Executor(paddle.CPUPlace())
            startup_program = paddle.static.default_startup_program()
            exe.run(startup_program)

            return program

        if ipu_strategy.enable_fp16:
            class_instance.to(dtype="float16")

        # copy the bias and filters
        for param_or_buffer in concrete_program.parameters:
            param_or_buffer_tensor = scope.var(
                param_or_buffer.name
            ).get_tensor()
            src_tensor = param_or_buffer.value().get_tensor()
            param_or_buffer_tensor._share_data_with(src_tensor)

        # TODO(czr): feed and fetch list needs to consider more type
        if class_instance:
            feed_list = [elem.name for elem in inputs[1:] if elem is not None]
        else:
            feed_list = [elem.name for elem in inputs if elem is not None]
        fetch_list = [elem.name for elem in outputs]

        if ipu_strategy.is_training:
            concrete_program.main_program = append_backward_desc()
            # copy optimizer parameters
            optimizer = ipu_strategy._optimizer
            for k, v in optimizer._accumulators.items():
                for param_name, var_tmp in v.items():
                    var = optimizer.helper.create_global_variable(
                        name=var_tmp.name,
                        persistable=True,
                        dtype=var_tmp.dtype,
                        type=var_tmp.type,
                        shape=var_tmp.shape,
                        belong_to_optimizer=True,
                    )
                    device = optimizer._get_device_for_param(param_name)
                    with device_guard(device):
                        optimizer.helper.set_variable_initializer(
                            var,
                            initializer=paddle.nn.initializer.Constant(
                                value=0.0
                            ),
                        )
                    param_or_lr_tensor = scope.find_var(
                        var_tmp.name
                    ).get_tensor()
                    optim_tensor = var.value().get_tensor()
                    param_or_lr_tensor._share_data_with(optim_tensor)
                    optimizer._accumulators[k][param_name] = var

        @switch_to_static_graph
        def func_compile():
            if ipu_strategy.enable_fp16:
                amp_list = paddle.static.amp.CustomOpLists()
                amp_list.unsupported_list = {"cumsum"}
                to_fp16_var_names = paddle.static.amp.cast_model_to_fp16(
                    concrete_program.main_program,
                    amp_list,
                    use_fp16_guard=False,
                )
                paddle.static.amp.cast_parameters_to_fp16(
                    paddle.CPUPlace(),
                    concrete_program.main_program,
                    to_fp16_var_names=to_fp16_var_names,
                )

            program = IpuCompiledProgram(
                concrete_program.main_program,
                ipu_strategy=ipu_strategy,
                scope=scope,
            ).compile(feed_list, fetch_list)
            return program

        main_program = func_compile()
        concrete_program.main_program = main_program
        return concrete_program

    @staticmethod
    def patch_program_cache(ipu_strategy):
        """Monkey patch ProgramCache discriptor to support dynamic2static in IPU.

        Args:
            ipu_strategy: The ipu_strategy used in dynamic graph.

        Returns:
            None
        """
        from paddle.jit.dy2static.program_translator import (
            CacheKey,
            ProgramCache,
            MAX_TRACED_PROGRAM_COUNT,
        )
        from paddle.jit.dy2static import logging_utils
        from paddle.jit.dy2static.partial_program import (
            partial_program_from,
        )

        old_getter = ProgramCache.__getitem__

        def patch_getter(self, item):
            if not isinstance(item, CacheKey):
                raise ValueError(
                    'type(item) should be CacheKey, but received %s'
                    % type(item).__name__
                )
            item_id = hash(item)
            self._recent_key = item_id
            if item_id not in self._caches or ipu_strategy.need_compile:
                if item_id in self._caches:
                    logging_utils.warn(
                        "ipu_strategy chances detected. Please sync weights."
                    )
                if self._caches and not ipu_strategy.need_compile:
                    logging_utils.warn(
                        "dynamic2static on IPU doesn't support mutiple caches. Please make sure"
                        "dynamic inputs is not used."
                    )
                concrete_program, _ = self._build_once(item)
                concrete_program = IpuDynamicPatcher.convert_concrete_program(
                    ipu_strategy, concrete_program, item.class_instance
                )

                self._caches[item_id] = (
                    concrete_program,
                    partial_program_from(
                        concrete_program, item.class_instance is not None
                    ),
                )
                # Note: raise warnings if number of traced program is more than `max_tracing_count`
                current_tracing_count = len(self._caches)
                if current_tracing_count > MAX_TRACED_PROGRAM_COUNT:
                    logging_utils.warn(
                        "Current traced program number: {} > `max_tracing_count`:{}. Too much cached programs will bring expensive overhead. "
                        "The reason may be: (1) passing tensors with different shapes, (2) passing python objects instead of tensors.".format(
                            current_tracing_count, MAX_TRACED_PROGRAM_COUNT
                        )
                    )

            return self._caches[item_id]

        setattr(ProgramCache, '__getitem__', patch_getter)
        IpuDynamicPatcher.patcher_cache.append(
            [ProgramCache, '__getitem__', old_getter]
        )

    @staticmethod
    def patch_lr_scheduler(ipu_strategy):
        from paddle.optimizer.lr import LRScheduler

        # For IPU dynamic graph usage, lr_var is not synced in executor as static graph mode do.
        # Manually set lr to ipu_strategy to update the lr.
        old_step = LRScheduler.step

        def patch_step(self, epoch=None):
            old_step(self, epoch)
            ipu_strategy.set_options({"lr": self.last_lr})

        setattr(LRScheduler, 'step', patch_step)
        IpuDynamicPatcher.patcher_cache.append([LRScheduler, 'step', old_step])

    @staticmethod
    def register_patch(ipu_strategy):
        IpuDynamicPatcher.patch_program_cache(ipu_strategy)
        IpuDynamicPatcher.patch_lr_scheduler(ipu_strategy)

    @staticmethod
    def release_patch():
        for module, key, attr in IpuDynamicPatcher.patcher_cache:
            setattr(module, key, attr)


class IpuStrategy:
    """
    Help users precisely control the graph building in :code:`paddle.static.IpuCompiledProgram` .

    Returns:
        The IpuStrategy instance.

    Examples:
        .. code-block:: python

            # required: ipu

            import paddle
            import paddle.static as static

            paddle.enable_static()

            ipu_strategy = static.IpuStrategy()
    """

    def __init__(self):
        if core.is_compiled_with_ipu():
            self._ipu_strategy = core.IpuStrategy()
            default_options = {
                'location_optimizer': {
                    'on_chip': 0,
                    'use_replicated_tensor_sharding': 1,
                },  # set optimizer location
                'accumulation_and_replication_reduction_type': 1,  # popart::ReductionType::Mean
                'mean_accumulation_and_replication_reduction_strategy': 1,  # popart::MeanReductionStrategy::Post
            }
            self._ipu_strategy.set_options(default_options)
            self.has_custom_ops = False
            self.custom_op_names = []
            self.need_compile = True
        else:
            raise RuntimeError(
                "Can not use IpuStrategy in non IPU compiled environment, please re-compile with WITH_IPU=ON."
            )
        from paddle import in_dynamic_mode

        if in_dynamic_mode():
            self.register_patch()

    def register_patch(self):
        """
        Register patchs function to support dynamic to static on IPU. This operation would break the dy2static functionality on CPU.
        Use `release_patch` to release the patch.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                ipu_strategy = static.IpuStrategy()

                ipu_strategy.register_patch()
        """
        IpuDynamicPatcher.register_patch(self)

    def release_patch(self):
        """
        Release the registered IPU functions.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                ipu_strategy = static.IpuStrategy()

                ipu_strategy.release_patch()
        """
        IpuDynamicPatcher.release_patch()

    def set_optimizer(self, optimizer):
        """
        Set optimizer to ipu_strategy in dynamic mode.

          Args:
              optimizer (Optimizer): Optimizer to be used in training.

          Returns:
              None.

          Examples:
              .. code-block:: python

                  # required: ipu

                  import paddle
                  import paddle.static as static

                  linear = paddle.nn.Linear(10, 10)
                  optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                                   parameters=linear.parameters())
                  ipu_strategy = static.IpuStrategy()
                  ipu_strategy.set_optimizer(optimizer)
        """
        from paddle import in_dynamic_mode

        if in_dynamic_mode():
            self._optimizer = optimizer
            optimizer_attrs = self.parse_optimizer(optimizer)
            self._ipu_strategy.set_options(optimizer_attrs)
        else:
            raise RuntimeError("Only needs to set optimizer in dynamic mode.")

    def parse_optimizer(self, optimizer):
        """
        Parse optimizer attributes for IPU dynamic to static support. Currently only support parse lr.

          Args:
              optimizer (Optimizer): Optimizer to be parsed.

          Returns:
              Dict.

          Examples:
              .. code-block:: python

                  # required: ipu

                  import paddle
                  import paddle.static as static

                  linear = paddle.nn.Linear(10, 10)
                  optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                                   parameters=linear.parameters())
                  ipu_strategy = static.IpuStrategy()
                  attrs = ipu_strategy.parse_optimizer(optimizer)
        """

        def get_lr():
            from paddle.optimizer.lr import LRScheduler

            if isinstance(optimizer._learning_rate, float):
                return {"lr": optimizer._learning_rate}
            elif isinstance(optimizer._learning_rate, LRScheduler):
                return {"lr": optimizer._learning_rate()}

        attr_fn = [get_lr]
        optimizer_attrs = {"is_dynamic": True}
        for fn in attr_fn:
            optimizer_attrs.update(fn())
        return optimizer_attrs

    def set_graph_config(
        self,
        num_ipus=1,
        is_training=True,
        micro_batch_size=1,
        enable_manual_shard=False,
    ):
        """
        Set graph configuration to the IpuStrategy instance.

        Args:
            num_ipus (int, optional): Number of IPU devices. Default 1, which means only use 1 IPU.
            is_training (bool, optional): True is training graph, False is inference graph. Default True, which means is training mode.
            batch_size (int, optional): The batch-size in the graph. Used to make the graph batch-size fixed,
                if the batch-size in the graph is dynamic. Default 1, which means the batch-size would be set 1, if the batch-size is dynamice.
            enable_manual_shard (bool, optional): Enable graph sharding or not. Only if num_ipus > 1, enable_manual_shard is able to be set True.
                Default False, which means disabled.

        Returns:
            None.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                ipu_strategy.set_graph_config(num_ipus=1,
                                            is_training=True,
                                            micro_batch_size=1,
                                            enable_manual_shard=False)
        """
        if num_ipus == 1 and enable_manual_shard:
            raise RuntimeError(
                "Only if num_ipus > 1, enable_manual_shard is able to be set True."
            )
        options = {
            'num_ipus': num_ipus,
            'is_training': is_training,
            'micro_batch_size': micro_batch_size,
            'enable_manual_shard': enable_manual_shard,
        }
        self.set_options(options)

    def set_pipelining_config(
        self,
        enable_pipelining=False,
        batches_per_step=1,
        enable_gradient_accumulation=False,
        accumulation_factor=1,
    ):
        """
        Set pipelining configuration to the IpuStrategy instance. Used to optimize the throughput performance.

        Args:
            enable_pipelining (bool, optional): Enable data pipelining between subgraphs. Only if enable_manual_shard=True, enable_pipelining is able to be set True.
                Default False, which means disabled.
            batches_per_step (int, optional): Set the batches per run in data pipelining mode. Only if enable_pipelining=True, batches_per_step is able to be set > 1.
                Default 1, which means no data pipelining.
            enable_gradient_accumulation (bool, optional): Enable to accumulate gradients before updating the weights in training mode. Only if enable_pipelining=True,
                enable_gradient_accumulation is able to be set True. Default False, which means no gradient accumulation.
            accumulation_factor (int, optional): Specify the number of micro-batches to accumulate
                before applying the varUpdate. Default 1, which means disable the accumulation.

        Returns:
            None.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                ipu_strategy.set_pipelining_config(enable_pipelining=False,
                                                    batches_per_step=1,
                                                    enable_gradient_accumulation=False,
                                                    accumulation_factor=1)
        """
        enable_manual_shard = self.get_option('enable_manual_shard')
        if not enable_manual_shard and enable_pipelining:
            raise RuntimeError(
                "Only if enable_manual_shard=True, enable_pipelining is able to be set True."
            )
        options = {
            'enable_pipelining': enable_pipelining,
            'batches_per_step': batches_per_step,
            'enable_gradient_accumulation': enable_gradient_accumulation,
            'accumulation_factor': accumulation_factor,
        }
        self.set_options(options)

    def set_precision_config(self, enable_fp16=False):
        """
        Set half computation configuration to the IpuStrategy instance. Used to optimize the performance.

        Args:
            enable_fp16 (bool, optional): Enable FLOAT16 mode and transform FLOAT32 to FLOAT16. Default False, which means disable FLOAT16 mode.

        Returns:
            None.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                ipu_strategy.set_precision_config(enable_fp16=False)
        """
        options = {
            'enable_fp16': enable_fp16,
        }
        self.set_options(options)

    def add_custom_op(
        self, paddle_op, popart_op=None, domain='custom.ops', version=1
    ):
        """
        Add a mapping to use popart custom ops running on the IPU.

        Args:
            paddle_op(str): the name of custom op in paddle.

            popart_op(str): the name of custom op in popart.

            domain(str): domain name of custom op in popart.

            version(int): version of custom op in popart.

        Returns:
            None.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                ipu_strategy.add_custom_op('paddle_relu', 'popart_relu')
        """
        if popart_op is None:
            popart_op = paddle_op
        custom_op = {
            'paddle_op': paddle_op,
            'popart_op': popart_op,
            'domain': domain,
            'version': version,
        }
        self.set_options({'custom_op': custom_op})
        self.custom_op_names.append(paddle_op)
        if not self.has_custom_ops:
            self.has_custom_ops = True

    def set_options(self, options):
        """
        Set options from dict.

        Args:
            options(dict): dict of options.

        Returns:
            None.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                options = {'num_ipus':1, 'enable_fp16': True}
                ipu_strategy.set_options(options)
        """
        self._ipu_strategy.set_options(options)
        # check whether to recompile program with updated ipu options.
        recompile_white_list = {'lr'}
        if options.keys() - recompile_white_list:
            self.need_compile = True

    def get_option(self, option):
        """
        Get option.

        Args:
            option(str): name of option.

        Returns:
            option value.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                num_ipus = ipu_strategy.get_option('num_ipus')
        """
        return self._ipu_strategy.get_option(option)['value']

    def enable_pattern(self, pattern):
        """
        Enable PopART pattern to optimize the graph.

        Args:
            pattern(string): the name of the pattern.

        Returns:
            None.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                ipu_strategy.enable_pattern("ViewSimplifyPattern")
        """
        self._ipu_strategy.enable_pattern(pattern)

    def disable_pattern(self, pattern):
        """
        Disable PopART pattern.

        Args:
            pattern(string): the name of the pattern.

        Returns:
            None.

        Examples:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                ipu_strategy = static.IpuStrategy()
                ipu_strategy.disable_pattern("ViewSimplifyPattern")
        """
        self._ipu_strategy.disable_pattern(pattern)

    @property
    def num_ipus(self):
        """
        Get the number of IPU devices from IpuStrategy instance.
        """
        return self.get_option('num_ipus')

    @property
    def is_training(self):
        """
        Get the boolean of training or inference from IpuStrategy instance.
        """
        return self.get_option('is_training')

    @property
    def enable_pipelining(self):
        """
        Get the boolean of enable pipelining or not from IpuStrategy instance.
        """
        return self.get_option('enable_pipelining')

    @property
    def enable_fp16(self):
        """
        Get the boolean of float16 mode or not from IpuStrategy instance.
        """
        return self.get_option('enable_fp16')


class IpuCompiledProgram:
    """
    The IpuCompiledProgram is used to transform a program to a ipu-target program,
    such as forward graph extraction, computing graph transformation, useless scale Ops clean, etc.

    Args:
        program(Program, optional): This parameter represents the :code:`Program`
            to be executed. Default is None, which means the program will be set to
            the default program :code:`paddle.static.default_main_program()` .
        scope(Scope, optional): The scope used to run this program, you can switch
            it to different scope. Default is None, which means use the global
            scope :code:`paddle.static.global_scope()` .
        ipu_strategy(IpuStrategy, optional): This argument is used to build the program with the
            specified options, such as half computation, training or inference session, the number of IPUs, etc.
            Default is None, which means build the program based on the default `ipu_strategy`.

    Returns:
        IpuCompiledProgram

    Example:
        .. code-block:: python

            # required: ipu

            import paddle
            import paddle.static as static

            paddle.enable_static()

            a = static.data(name='data', shape=[None, 1], dtype='int32')
            b = a + 1
            main_prog = static.default_main_program()

            ipu_strategy = static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=True, micro_batch_size=1)
            ipu_strategy.set_pipelining_config(enable_pipelining=False, batches_per_step=1, enable_gradient_accumulation=False, accumulation_factor=1)
            ipu_strategy.set_precision_config(enable_fp16=False)

            ipu_compiled_program = static.IpuCompiledProgram(
                main_prog,
                ipu_strategy=ipu_strategy)
    """

    def __init__(self, program=None, scope=None, ipu_strategy=None):
        if not core.is_compiled_with_ipu():
            raise ValueError(
                "Can not use this function since PaddlePaddle is not compiled with IPU"
            )

        if program is None:
            program = framework.default_main_program()

        if not isinstance(program, framework.Program):
            raise TypeError(
                "The type of program is wrong, expected Program, but got %s"
                % type(program)
            )

        self._program = program
        self._compiled = False

        if scope is not None:
            self._scope = scope
        else:
            # import here to avoiding confused
            import paddle

            self._scope = paddle.static.global_scope()

        if ipu_strategy is not None:
            self._ipu_strategy = ipu_strategy
        else:
            self._ipu_strategy = IpuStrategy()

        if ipu_strategy.has_custom_ops:
            self._custom_op_names = set(ipu_strategy.custom_op_names)
        else:
            self._custom_op_names = ()

        self._backend = core.IpuBackend.get_instance()

    def compile(self, feed_list, fetch_list):
        """
        This interface is used to compile the input Program to a program
        to run the model on the ipu.

        Args:
            feed_list(list): This parameter represents the input Tensors of the model.

            fetch_list(list): This parameter represents the Tensors that need to be returned
                after the model.

        Returns:
            Program

        Example:
            .. code-block:: python

                # required: ipu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                a = static.data(name='data', shape=[None, 1], dtype='int32')
                b = a + 1
                main_prog = static.default_main_program()

                ipu_strategy = static.IpuStrategy()
                ipu_strategy.set_graph_config(num_ipus=1, is_training=True, micro_batch_size=1)
                ipu_strategy.set_pipelining_config(enable_pipelining=False, batches_per_step=1, enable_gradient_accumulation=False, accumulation_factor=1)
                ipu_strategy.set_precision_config(enable_fp16=False)

                program = static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile([a.name], [b.name])
        """
        self._backend.set_scope(self._scope)
        self._backend.set_ipu_strategy(self._ipu_strategy._ipu_strategy)

        # feed and fetch doesn't have corresponding popart op, so we rm both here
        global_block = self._program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(global_block.ops):
            op.desc.set_is_target(False)
            if op.type == 'feed' or op.type == 'fetch':
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            global_block._remove_op(index)

        for var in ['feed', 'fetch']:
            if global_block.has_var(var):
                global_block._remove_var(var)

        self._program.desc.flush()
        self._graph = core.Graph(self._program.desc)

        if self._ipu_strategy.is_training:
            passes = [
                'optimizer_extract_pass',
                'optimizer_state_align_pass',
            ]
            for pass_name in passes:
                a_pass = core.get_pass(pass_name)
                a_pass.apply(self._graph)

        passes = [
            'forward_graph_extract_pass',
            'infer_shape_pass',
            'avg_shard_pass',
            'delete_scale_op_pass',
        ]
        for pass_name in passes:
            a_pass = core.get_pass(pass_name)
            if pass_name == 'infer_shape_pass':
                a_pass.set('feed_list', feed_list)
            a_pass.apply(self._graph)

        a_pass = core.get_pass('popart_canonicalization_pass')
        if self._custom_op_names:
            a_pass.set('custom_ops', self._custom_op_names)
        a_pass.apply(self._graph)

        passes = [
            'ipu_inplace_pass',
            'ipu_graph_builder_pass',
            'ipu_runtime_replacer_pass',
        ]
        for pass_name in passes:
            a_pass = core.get_pass(pass_name)
            a_pass.set('feed_list', feed_list)
            a_pass.set('fetch_list', fetch_list)
            a_pass.apply(self._graph)

        convert_pass = core.get_pass('graph_to_program_pass')
        desc = core.ProgramDesc()
        convert_pass.set_not_owned('program', desc)
        convert_pass.apply(self._graph)
        program = framework.Program._construct_from_desc(desc)

        if hasattr(self._program, 'lr_scheduler'):
            # how to share var between two different block ?
            lr_var_name = self._program.lr_scheduler._var_name

            program.lr_scheduler = self._program.lr_scheduler
            # Program.clone will clone lr_scheduler, so i set lr_var as
            # lr_scheduler attribute
            global_block = self._program.global_block()
            program.lr_scheduler.lr_var = global_block.vars[lr_var_name]

        # with popart, we need to support batches_per_step, what means
        # the shape of feed_var and feed_tensor(maybe numpy array) will
        # mismatch, so we set need_check_feed to False. Thus we can avoid
        # modify logic of run.
        program_global_block = program.global_block()
        for feed_name in feed_list:
            feed_var = program_global_block.var(feed_name)
            feed_var.desc.set_need_check_feed(False)

        if not hasattr(program, 'org_program'):
            program.org_program = self._program

        self._ipu_strategy.need_compile = False

        return program


class GcuCompiledProgram(object):
    """
    The GcuCompiledProgram is used to transform a program to a gcu-target program,
    such as forward graph extraction, computing graph transformation, useless scale Ops clean, etc.

    Args:
        program(Program, optional): This parameter represents the :code:`Program`
            to be executed. Default is None, which means the program will be set to
            the default program :code:`paddle.static.default_main_program()` .
        scope(Scope, optional): The scope used to run this program, you can switch
            it to different scope. Default is None, which means use the global
            scope :code:`paddle.static.global_scope()` .
        gcu_strategy(GcuStrategy, optional): This argument is used to build the program with the
            specified options. Default is None, which means build the program
            based on the default `gcu_strategy`.

    Returns:
        Program

    Example:
        .. code-block:: python

            # required: gcu

            import paddle
            import paddle.static as static

            paddle.enable_static()

            a = static.data(name='data', shape=[None, 1], dtype='int32')
            b = a + 1
            main_prog = static.default_main_program()

            gcu_strategy = static.GcuStrategy()
            gcu_strategy.set_batch_size(batch_num = 16)
            program = static.GcuCompiledProgram(
                main_prog,
                gcu_strategy=gcu_strategy).compile([a.name], [b.name])
    """

    def __init__(self, program=None, scope=None, gcu_strategy=None):
        if not core.is_compiled_with_custom_device('gcu'):
            raise ValueError(
                "Can not use this function since PaddlePaddle is not compiled with GCU"
            )

        if program is None:
            program = framework.default_main_program()
        else:
            program = (
                program._program
                if isinstance(program, CompiledProgram)
                else program
            )

        if not isinstance(program, framework.Program):
            raise TypeError(
                "The type of program is wrong, expected Program, but got %s"
                % type(program)
            )

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

        self._program = program.clone()
        self._compiled = False

        program_key_str = str(id(self._program)) + str(':') + str(self._program)
        self._program_key = self._hash_md5(program_key_str)

        self._logger.info(
            'gcu compile, program key is:{}, program is:{}'.format(
                self._program_key, self._program
            )
        )

        if scope is not None:
            self._scope = scope
        else:
            # import here to avoiding confused
            import paddle

            self._scope = paddle.static.global_scope()

        self._backend = core.GcuBackend.get_instance()
        self.dist = (
            True if os.getenv('_gcu_enable_distributed') == 'true' else False
        )
        local_rank = int(os.getenv('PADDLE_RANK_IN_NODE', 0))
        world_rank = int(os.getenv('PADDLE_TRAINER_ID', 0))
        world_size = int(os.getenv('PADDLE_TRAINERS_NUM', 1))
        device_id = int(os.getenv('FLAGS_selected_gcus', 0))
        node_id = world_rank // 8
        self._backend.init_backend(
            self.dist, local_rank, device_id, world_size, node_id
        )
        if program.random_seed != 0:
            core.set_gcu_random_seed(program.random_seed)
            self._logger.info(
                'GcuCompiledProgram, init_backend, set gcu random seed:{}'.format(
                    program.random_seed
                )
            )

        self._logger.info(
            'GcuCompiledProgram init_backend, dist:{}, rank_id:{}, device_id:{}, world_size:{}, node_id:{}'.format(
                self.dist, local_rank, device_id, world_size, node_id
            )
        )

        # self._backend = core.GcuBackend()

    def compile_in_run(self, feed, fetch_list):
        """
        This interface is used to compile the input Program to a program
        to run the model on the gcu. Used in executor.run.

        Args:
            feed(list|dict): This parameter represents the input Tensors of the model.
                If it is single card training, the feed is dict type, and if it is multi-card
                training, the parameter feed can be dict or list of Tensors. If the
                parameter type is dict, the data in the feed will be split and sent to
                multiple devices, that is to say, the input data will be evenly
                sent to different devices, so you should make sure the number of samples of
                the current mini-batch must be greater than the number of places;
                if the parameter type is list, those data are copied directly to each device,
                so the length of this list should be equal to the number of places.
            fetch_list(list): This parameter represents the names of Tensors that need to be
                returned after the model runs.
        Returns:
            Program

        Example:
            .. code-block:: python

                # required: gcu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                a = static.data(name='data', shape=[None, 1], dtype='int32')
                b = a + 1
                main_prog = static.default_main_program()

                gcu_strategy = static.GcuStrategy()
                gcu_strategy.set_batch_size(batch_num = 16)
                program = static.GcuCompiledProgram(
                    main_prog,
                    gcu_strategy=gcu_strategy).compile([a], [b.name])
        """
        # get feed var name list and fetch var name list
        feed_var_names = []
        feed_vars = {}
        if isinstance(feed, dict):
            feed_var_names = list(feed.keys())
            feed_vars = feed
        elif isinstance(feed, list) or isinstance(feed, tuple):
            for i, each in enumerate(feed):
                feed_var_names += list(each.keys())
                feed_vars.update(each)

        # set feed shape and dtype
        global_block = self._program.global_block()
        for name in feed_var_names:
            if global_block.has_var(name):
                input_data = global_block.var(name)
                feed_var = feed_vars[name]
                old_shape = input_data.shape
                feed_shape = (
                    feed_var.shape()
                    if isinstance(feed_var, (core.Tensor, core.LoDTensor))
                    else feed_var.shape
                )
                feed_dtype = (
                    feed_var._dtype()
                    if isinstance(feed_var, (core.Tensor, core.LoDTensor))
                    else feed_var.dtype
                )
                input_data.desc.set_shape(feed_shape)
                input_dtype = input_data.dtype
                if convert_dtype(feed_dtype) != convert_dtype(input_dtype):
                    raise ValueError(
                        "The previous data type of Variable {} is {}, the feed "
                        "data type is {}. They are not "
                        "matched.".format(name, input_dtype, feed_dtype)
                    )
                self._logger.info(
                    'gcu compile, input:{}, old_shape:{}, set shape:{}, dtype:{}'.format(
                        name, old_shape, input_data.shape, input_data.dtype
                    )
                )
            else:
                warnings.warn(
                    "The variable %s is not found in program. It is not declared or is pruned."
                    % name
                )

        # compile
        return self.compile(feed_var_names, fetch_list, allreduce_mean=True)

    def compile(self, feed_list, fetch_list, allreduce_mean=True):
        """
        This interface is used to compile the input Program to a program
        to run the model on the gcu.

        Args:
            feed_list(list): This parameter represents the names of input Tensors of the model.
            fetch_list(list): This parameter represents the names of Tensors that need to be
                returned after the model runs.
        Returns:
            Program

        Example:
            .. code-block:: python

                # required: gcu

                import paddle
                import paddle.static as static

                paddle.enable_static()

                a = static.data(name='data', shape=[None, 1], dtype='int32')
                b = a + 1
                main_prog = static.default_main_program()

                gcu_strategy = static.GcuStrategy()
                gcu_strategy.set_batch_size(batch_num = 16)
                program = static.GcuCompiledProgram(
                    main_prog,
                    gcu_strategy=gcu_strategy).compile([a.name], [b.name])
        """
        if len(self._program.global_block().ops) == 0:
            return self._program

        self._backend.set_scope(self._scope)
        # self._backend.set_gcu_strategy(self._gcu_strategy)

        # self._graph = framework.IrGraph(core.Graph(self._program.desc))
        self._graph = core.Graph(self._program.desc)
        self.origin_ir_graph = framework.IrGraph(self._graph)
        if self.dist:
            ar, ar_ins, ar_outs = self.allreduce_fusion(self.origin_ir_graph)

        passes = [
            'infer_shape_pass',
        ]
        for pass_name in passes:
            a_pass = core.get_pass(pass_name)
            if pass_name == 'infer_shape_pass':
                a_pass.set('feed_list', feed_list)
                # batch_size = self._gcu_strategy.get_batch_size()
                a_pass.set('batch_size', str(16))
            a_pass.apply(self._graph)

        if self.dist and ar is not None:
            # ar, ar_ins, ar_outs = self.allreduce_fusion(self.origin_ir_graph)
            fp_bp_graph, fp_bp_feed, fp_bp_fetch = self.generate_fp_bp_graph(
                self.origin_ir_graph, feed_list, fetch_list, ar, ar_ins, ar_outs
            )
            (
                update_graph,
                update_feed,
                update_fetch,
            ) = self.generate_update_graph(
                self.origin_ir_graph, feed_list, fetch_list, allreduce_mean
            )

            graph_list = [self._graph, fp_bp_graph.graph, update_graph.graph]
            all_feeds = [feed_list, fp_bp_feed, update_feed]
            all_fetches = [fetch_list, fp_bp_fetch, update_fetch]

            self._logger.info('gcu compile, all_feeds:{}'.format(all_feeds))
            self._logger.info('gcu compile, all_fetches:{}'.format(all_fetches))
            self._logger.info(fp_bp_graph.to_program())
            # paddle.fluid.save(fp_bp_graph.to_program(), "./dist_gcu/fp_bp_graph")
            self._logger.info(update_graph.to_program())
            # paddle.fluid.save(update_graph.to_program(), "./dist_gcu/update_graph")
        else:
            graph_list = [self._graph]
            all_feeds = [feed_list]
            all_fetches = [fetch_list]

        self._backend.compile(
            graph_list, all_feeds, all_fetches, self._program_key
        )

        passes = [
            'gcu_runtime_replacer_pass',
        ]
        for pass_name in passes:
            a_pass = core.get_pass(pass_name)
            a_pass.set('feed_list', feed_list)
            a_pass.set('fetch_list', fetch_list)
            a_pass.set('program_key', self._program_key)
            a_pass.apply(self._graph)

        convert_pass = core.get_pass('graph_to_program_pass')
        desc = core.ProgramDesc()
        convert_pass.set_not_owned('program', desc)
        convert_pass.apply(self._graph)
        program = framework.Program._construct_from_desc(desc)
        if hasattr(self._program, 'lr_scheduler'):
            program.lr_scheduler = self._program.lr_scheduler
            self._logger.info('gcu compile, program has lr_scheduler')
        self._compiled = True
        return program

    def allreduce_fusion(self, ir_graph):
        all_var_nodes = ir_graph.all_var_nodes()
        allreduce_ops = []
        allreduce_input_vars = []
        allreduce_output_vars = []
        for op_node in ir_graph.all_op_nodes():
            if op_node.op().type() == 'eccl_allreduce':
                allreduce_ops.append(op_node)
                input_name = op_node.input('InputList')[0]
                output_name = op_node.output('OutputList')[0]
                self._logger.info(
                    'allreduce_fusion, eccl_allreduce input name:{}, output name:{}'.format(
                        input_name, output_name
                    )
                )
                allreduce_input_vars.append(
                    ir_graph._find_node_by_name(all_var_nodes, input_name)
                )
                allreduce_output_vars.append(
                    ir_graph._find_node_by_name(all_var_nodes, output_name)
                )
        if not allreduce_ops:
            self._logger.info(
                'allreduce_fusion, don not have allreduce ops, not need fusion'
            )
            return None, None, None
        # Create fusion allreduce
        # The GCU does not support multi stream parallelism at this time,
        # so it is only merged into one allreduce op here
        fusion_allreduce = ir_graph.create_op_node(
            op_type='eccl_allreduce',
            inputs={'InputList': allreduce_input_vars},
            outputs={'OutputList': allreduce_output_vars},
            attrs={
                'op_role': 8,  # 0: Forward   1: Backward   2: Optimize   4: RPC   8: Dist   16: LRSched   256: Loss
                'reduce_type': 0,  # 0: sum   1: prod   2: max   3: min
                'sync_mode': True,
            },
        )
        for input_var in allreduce_input_vars:
            ir_graph.link_to(input_var, fusion_allreduce)
        for output_var in allreduce_output_vars:
            ir_graph.link_to(fusion_allreduce, output_var)
        ir_graph.safe_remove_nodes(allreduce_ops)
        ir_graph.topology_sort()
        return fusion_allreduce, allreduce_input_vars, allreduce_output_vars

    def generate_fp_bp_graph(
        self,
        ir_graph,
        global_feed_list,
        global_fetch_list,
        fusion_allreduce,
        allreduce_input_vars,
        allreduce_output_vars,
    ):
        fp_bp_graph = ir_graph.clone()
        var_node_map = {}
        for var in fp_bp_graph.all_var_nodes():
            var_node_map[var.name()] = var
        new_ar_out_vars = [
            var_node_map[ar_in_var.name()]
            for ar_in_var in allreduce_output_vars
        ]

        all_op_nodes = fp_bp_graph.all_op_nodes()
        remove_nodes = []
        remove_ar_out_name = []
        optimize_op_inputs = []
        for ar_out_var in new_ar_out_vars:
            count = len(ar_out_var.outputs)
            for out_op_node in ar_out_var.outputs:
                if self._is_optimize_op(out_op_node):
                    optimize_op_inputs += out_op_node.inputs
                    # Assume that outputs of the optimizer are all in-place operation
                    remove_nodes.append(out_op_node)
                    count -= 1
                else:
                    warnings.warn(
                        'NOT Support, allreduce output to non-opt op:'.format(
                            out_op_node.name()
                        )
                    )
            assert (
                count == 0
            ), "Allreduce can only be output to the optimizer operator"
            remove_nodes.append(ar_out_var)
            remove_ar_out_name.append(ar_out_var.name())

        remove_nodes.append(
            fp_bp_graph._find_node_by_name(
                all_op_nodes, fusion_allreduce.name()
            )
        )

        # get fetch list (before node is removed)
        fetch_1 = [
            ar_in_var.name() if not ar_in_var.is_ctrl_var() else None
            for ar_in_var in allreduce_input_vars
        ]
        fetch_2 = [
            opt_in.name()
            if not opt_in.is_ctrl_var()
            and not opt_in.name() in remove_ar_out_name
            else None
            for opt_in in optimize_op_inputs
        ]
        fetch_list = fetch_1 + fetch_2
        fetch_list = list(filter(lambda x: x is not None, fetch_list))
        fetch_list.sort()  # Sorted by name
        fetch_list = global_fetch_list + fetch_list
        fetch_list = sorted(
            set(fetch_list), key=fetch_list.index
        )  # delete duplication

        # remove nodes, may cause isolated node
        fp_bp_graph.safe_remove_nodes(remove_nodes)

        # remove isolated node
        isolated_nodes = [
            node if not node.inputs and not node.outputs else None
            for node in fp_bp_graph.all_nodes()
        ]
        isolated_nodes = list(filter(lambda x: x is not None, isolated_nodes))
        isolated_node_names = [node.name() for node in isolated_nodes]
        fp_bp_graph.safe_remove_nodes(isolated_nodes)
        self._logger.info(
            'fp_bp_graph remove isolated_nodes:{}'.format(isolated_node_names)
        )
        fp_bp_graph.topology_sort()

        # get fetch list
        new_all_var_name = [var.name() for var in fp_bp_graph.all_var_nodes()]
        self._logger.info(
            'fp_bp_graph new_all_var_name:{}'.format(new_all_var_name)
        )
        fetch_list = list(
            filter(
                lambda x: x in new_all_var_name
                and x not in isolated_node_names,
                fetch_list,
            )
        )

        # get feed list
        feed_list = global_feed_list
        feed_list = list(filter(lambda x: x in new_all_var_name, feed_list))

        return fp_bp_graph, feed_list, fetch_list

    def _is_optimize_op(self, op_node):
        if op_node.op().type() in [
            'merged_momentum',
            'merged_adam',
            'merged_adamw',
        ]:
            return True
        if not op_node.op().has_attr('_gcu_graph_op_category'):
            return False
        op_category = op_node.op().attr('_gcu_graph_op_category')
        if op_category == 'gcu_optimizer':
            return True
        return False

    def generate_update_graph(
        self, ir_graph, global_feed_list, global_fetch_list, allreduce_mean
    ):
        update_graph = ir_graph.clone()
        remove_nodes = []
        optimize_op_input_vars = []
        optimize_op_output_vars = []
        for op_node in update_graph.all_op_nodes():
            if self._is_optimize_op(op_node):
                optimize_op_input_vars += op_node.inputs
                optimize_op_output_vars += op_node.outputs
            else:
                remove_nodes.append(op_node)

        # Vars to be retained
        input_vars = [
            opt_in if not opt_in.is_ctrl_var() else None
            for opt_in in optimize_op_input_vars
        ]
        input_vars = list(filter(lambda x: x is not None, input_vars))
        output_vars = [
            opt_out if not opt_out.is_ctrl_var() else None
            for opt_out in optimize_op_output_vars
        ]
        output_vars = list(filter(lambda x: x is not None, output_vars))
        persist_vars = input_vars + output_vars

        # feed_list, not contain feeds in global_feed_list
        feed_list = [
            input_var.name()
            if input_var.inputs
            and input_var.outputs
            and input_var.name() not in global_feed_list
            else None  # input is not empty and output is not empty
            for input_var in input_vars
        ]
        feed_list = list(filter(lambda x: x is not None, feed_list))
        feed_list = sorted(set(feed_list))  # delete duplication and sort

        # will remove other vars
        persist_var_names = [var.name() for var in persist_vars]
        for var in update_graph.all_var_nodes():
            if var.name() not in persist_var_names:
                remove_nodes.append(var)
                self._logger.info(
                    'update_graph will remove var:{}'.format(var.name())
                )

        # remove nodes
        update_graph.safe_remove_nodes(remove_nodes)
        update_graph.topology_sort()

        if allreduce_mean:
            self._logger.info(
                'Will enable allreduce_mean, NOTE: Option for allreduce must be SUM'
            )
            self._grads_mean(update_graph)

        # Adjust the global input or output to the front
        new_all_var_name = [var.name() for var in update_graph.all_var_nodes()]
        self._logger.info(
            'update_graph new_all_var_name:{}'.format(new_all_var_name)
        )
        feed_list = global_feed_list + feed_list
        feed_list = list(filter(lambda x: x in new_all_var_name, feed_list))

        fetch_list = global_fetch_list
        fetch_list = list(filter(lambda x: x in new_all_var_name, fetch_list))

        return update_graph, feed_list, fetch_list

    def _grads_mean(self, ir_graph):
        world_size = int(os.getenv('PADDLE_TRAINERS_NUM', 1))
        grads_vars = []
        for var in ir_graph.all_var_nodes():
            if '_gcu_all_reduce' in var.name():
                grads_vars.append(var)
        if not grads_vars:
            return
        var_type = grads_vars[0].type()
        world_size_out = ir_graph.create_var_node(
            name='gcu_grads_mean_world_size',
            var_type=var_type,
            shape=[1],
            var_dtype=core.VarDesc.VarType.FP32,
        )
        world_size_op = ir_graph.create_op_node(
            op_type='fill_constant',
            inputs={},
            outputs={'Out': [world_size_out]},
            attrs={
                'shape': [1],
                'dtype': core.VarDesc.VarType.FP32,
                'value': float(world_size),
            },
        )
        ir_graph.link_to(world_size_op, world_size_out)
        for grad in grads_vars:
            grads_outputs = grad.outputs
            mean_grad = ir_graph.create_var_node(
                name=grad.name() + '_mean',
                var_type=grad.type(),
                shape=grad.shape(),
                var_dtype=grad.dtype(),
            )
            div_op = ir_graph.create_op_node(
                op_type='elementwise_div',
                inputs={'X': grad, 'Y': world_size_out},
                outputs={'Out': mean_grad},
                attrs={'axis': -1},
            )
            ir_graph.link_to(grad, div_op)
            ir_graph.link_to(world_size_out, div_op)
            ir_graph.link_to(div_op, mean_grad)
            for grads_out_node in grads_outputs:
                ir_graph.update_input_link(grad, mean_grad, grads_out_node)

    def _hash_md5(self, string_key):
        try:
            hashinst = hashlib.md5()
            hashinst.update(str(string_key).encode('utf-8'))
            md5sum = hashinst.hexdigest()
        except UnicodeDecodeError as e:
            md5sum = None
            err = e
        assert (
            md5sum is not None
        ), 'Error({}) occurred when do hash_md5.'.format(str(err))
        return str(md5sum)


class GcuStrategy(object):
    """
    Help users precisely control the graph building in :code:`paddle.static.GcuCompiledProgram` .

    Returns:
        The GcuStrategy instance.

    Examples:
        .. code-block:: python

            # required: gcu

            import paddle
            import paddle.static as static

            paddle.enable_static()

            gcu_strategy = static.GcuStrategy()
    """

    def __init__(self):
        if core.is_compiled_with_custom_device('gcu'):
            self._gcu_strategy = core.GcuStrategy()
        else:
            raise RuntimeError(
                "Can not use GcuStrategy in non GCU compiled environment, please re-compile with WITH_GCU=ON."
            )

    def set_batch_size(self, batch_num):
        self._gcu_strategy.set_batch_size(batch_num)

    def get_batch_size(self):
        batch_size = self._gcu_strategy.get_batch_size()
        return batch_size


class GcuOptions(object):
    """
    Help users precisely control the compilation and execution of program.

    Returns:
        The GcuOptions instance.

    Examples:
        .. code-block:: python

            # required: gcu

            import paddle

            gcu_options = paddle.framework.GcuOptions()
    """

    def __init__(self):
        self._invalid_option_keys = [
            'gcu_weights_sync_interval',
            'gcu_weights_sync_mode',
        ]

    def _check_key(self, key):
        if not key in self._invalid_option_keys:
            raise RuntimeError('Key[{}] is not supported yet.'.format(key))

    def _check_keys(self, options):
        for k, v in options:
            if not k in self._invalid_option_keys:
                raise RuntimeError('Key[{}] is not supported yet.'.format(k))

    def get_option(self, key):
        return core.get_option(key)

    def set_graph_option(self, key, value):
        self._check_key(key)
        core.set_graph_option(key, value)

    def set_global_option(self, key, value):
        self._check_key(key)
        core.set_global_option(key, value)

    def reset_graph_options(self, options):
        self._check_keys(options)
        core.reset_graph_options(options)

    def reset_global_options(self, options):
        self._check_keys(options)
        core.reset_global_options(options)

    def get_all_graph_options(self):
        return core.get_all_graph_options()

    def get_all_options(self):
        return core.get_all_options()

    def clear_graph_option(self, key):
        core.clear_graph_option(key)

    def clear_global_option(self, key):
        core.clear_global_option(key)

    def clear_all_options(self):
        core.clear_all_options()
