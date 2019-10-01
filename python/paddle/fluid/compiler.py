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
import six
import sys
from .. import compat as cpt
from . import framework
from .framework import cuda_places, cpu_places

from . import core

__all__ = ['CompiledProgram', 'ExecutionStrategy', 'BuildStrategy']

ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy
BuildStrategy = core.ParallelExecutor.BuildStrategy
InferNativeConfig = core.NativeConfig
InferAnalysisConfig = core.AnalysisConfig


def _place_obj(place):
    p = core.Place()
    p.set_place(place)
    return p


def _is_pserver_mode(main_program):
    main = main_program if main_program \
        else framework.default_main_program()
    for op in main.global_block().ops:
        if op.type in ["send", "recv"]:
            return True
    return False


def _has_backward_op(graph):
    for node in graph.nodes():
        if node.is_op() and node.op() is not None and \
                node.op().type().endswith("_grad"):
            return True
    return False


def _prune_feed_ops(program):
    # prune the feed ops in the program.
    pop_idx = []
    for i, op in enumerate(program.global_block().ops):
        if op.type == "feed": pop_idx.append(i)
    for index in pop_idx[::-1]:
        program.global_block()._remove_op(index)


class CompiledProgram(object):
    """
    Compiles to Graph for execution.

    1. Users first create the program with layers.
    2. Optionally, users use CompiledProgram to optimize the program before run.
    3. The original program or CompiledProgram is run by executor.

    The CompiledProgram is used to transform a program for various
    optimizations, for example.
      * Pre-compute some logic once so that each run is faster.
      * Transform the program so that it can run in multiple devices.
      * Transform the program for optimized inference or distributed
        training. **Note that: this part is not finished.**

    Example:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.fluid.compiler as compiler
          import numpy
          import os

          place = fluid.CUDAPlace(0) # fluid.CPUPlace()
          exe = fluid.Executor(place)

          data = fluid.layers.data(name='X', shape=[1], dtype='float32')
          hidden = fluid.layers.fc(input=data, size=10)
          loss = fluid.layers.mean(hidden)
          fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

          fluid.default_startup_program().random_seed=1
          exe.run(fluid.default_startup_program())
          compiled_prog = compiler.CompiledProgram(
                   fluid.default_main_program())

          x = numpy.random.random(size=(10, 1)).astype('float32')
          loss_data, = exe.run(compiled_prog,
                               feed={"X": x},
                               fetch_list=[loss.name])

    Args:
        program_or_graph (Graph|Program): If it's Program, it will be first
            lowered to a graph for further optimizations. If it's a graph
            (potentially optimized before), it will be directly used for
            further optimizations. Note: graph is only supported when compiled
            with with_data_parallel option.
        build_strategy(BuildStrategy): build_strategy is used to
            build the graph with the specified options.
            For more information, please refer to fluid.BuildStrategy.
            Default None.
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
            raise ValueError("Wrong program_to_graph type: %s" %
                             type(program_or_graph))

        self._scope = None
        self._place = None
        self._executor = None
        self._compiled = False
        self._is_data_parallel = False
        self._is_inference = False
        self._loss_name = None
        self._share_vars_from = None
        self._places = None
        self._build_strategy = build_strategy
        self._exec_strategy = None

    def with_data_parallel(self,
                           loss_name=None,
                           build_strategy=None,
                           exec_strategy=None,
                           share_vars_from=None,
                           places=None):
        """Configs the program to run in data parallel way.

        Example:
            .. code-block:: python

              import paddle.fluid as fluid
              import paddle.fluid.compiler as compiler
              import numpy
              import os

              use_cuda = True
              place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

              # NOTE: If you use CPU to run the program, you need
              # to specify the CPU_NUM, otherwise, fluid will use
              # all the number of the logic core as the CPU_NUM,
              # in that case, the batch size of the input should be
              # greater than CPU_NUM, if not, the process will be
              # failed by an exception.
              if not use_cuda:
                  os.environ['CPU_NUM'] = str(2)

              exe = fluid.Executor(place)

              data = fluid.layers.data(name='X', shape=[1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

              fluid.default_startup_program().random_seed=1
              exe.run(fluid.default_startup_program())
              compiled_prog = compiler.CompiledProgram(
                       fluid.default_main_program()).with_data_parallel(
                                loss_name=loss.name)

              x = numpy.random.random(size=(10, 1)).astype('float32')
              loss_data, = exe.run(compiled_prog,
                                   feed={"X": x},
                                   fetch_list=[loss.name])

        Args:
            loss_name (str): The loss name must set in training. Default None.
            build_strategy(BuildStrategy): build_strategy is used to
                build the graph with the specified options.
                For more information, please refer to fluid.BuildStrategy.
                Note that, if you set build_strategy in the argument list when
                creating CompiledProgram and calling with_data_parallel,
                the build_strategy in CompiledProgram will be overwritten by the latter.
                Default None.
            exec_strategy(ExecutionStrategy): exec_strategy is used to
                to select the a way to execute the graph, for example how many
                threads are used, how many iterations to clean up the temp
                variables. For more information, please refer
                to fluid.ExecutionStrategy. Default None.
            share_vars_from(CompiledProgram): If provided, this CompiledProgram
                will share variables from `share_vars_from`. `share_vars_from`
                must be run by the executor before this CompiledProgram so that
                vars are ready.
            places(list(CUDAPlace)|list(CPUPlace)|None): If provided, only compile
                program in the given places. Otherwise, the places used when compiled 
                is determined by the Executor, and the places used are controlled 
                by environment variables: FLAGS_selected_gpus or CUDA_VISIBLE_DEVICES
                if using GPU; or CPU_NUM if using CPU. For example, if you want to 
                run on GPU 0 and 1, set places=[fluid.CUDAPlace(0), fluid.CUDAPlace(1)].
                If you want to run on 2 CPU cores, set places=[fluid.CPUPlace()]*2.  

        Returns:
            self
        """
        assert not self._is_data_parallel, "Already compiled with parallel."
        assert not self._is_inference, "Cannot compile both data parallel and inference"
        self._is_data_parallel = True
        # FIXME(zcd): Currently, the build_strategy can be set during creating
        # CompiledProgram or calling with_data_parallel, and it may be confusing,
        # but in the long run, we should set up build_strategy only when creating
        # CompiledProgram, and exec_strategy should be deprecated.
        if build_strategy is not None: self._build_strategy = build_strategy
        self._exec_strategy = exec_strategy
        self._loss_name = loss_name
        self._share_vars_from = share_vars_from
        self._places = places

        if _has_backward_op(self._graph):
            assert self._loss_name is not None, "The loss_name should be set here."

        if self._places is not None:
            if not isinstance(self._places, (list, tuple)):
                self._places = [self._places]

        return self

    def _with_inference_optimize(self, config):
        """ Add inference optimize

        Args:
            config: instance of `NativeConfig` or `AnalysisConfig` to create predictor
        Returns:
            self
        """
        assert not self._is_data_parallel, "Cannot compile both data parallel and inference"
        assert not self._is_inference, "Already compiled with inference"

        assert any([
            isinstance(config, InferNativeConfig),
            isinstance(config, InferAnalysisConfig)
        ])
        self._is_inference = True
        self._infer_config = config
        return self

    def _with_distributed(self):
        raise NotImplementedError()

    def _compile_data_parallel(self, places, use_cuda=False, scope=None):
        if self._share_vars_from:
            if scope:
                sys.stderr.write("share_vars_from is set, scope is ignored.\n")
            if not self._is_data_parallel:
                raise ValueError(
                    "Currently, only data parallel mode need share_vars_from.")
            if not self._share_vars_from._is_data_parallel:
                raise ValueError("share_vars_from is not data parallel. Cannot "
                                 "share vars from it.")
            if self._share_vars_from._executor is None:
                raise ValueError(
                    "share_vars_from is not compiled and run, so there is no "
                    "var to share.")
            self._local_scopes = self._share_vars_from._executor.local_scopes()
        else:
            assert scope is not None, ""
            self._local_scopes = []

        assert isinstance(places, tuple) or isinstance(places, list), \
            "Currently , The places type only should be list or tuple, \n" \
            "but the input type is {}.".format(type(places))

        if self._build_strategy is None:
            self._build_strategy = BuildStrategy()
        self._build_strategy.is_distribution = _is_pserver_mode(self._program)

        if self._exec_strategy is None:
            self._exec_strategy = ExecutionStrategy()
        self._exec_strategy.use_cuda = use_cuda

        if self._exec_strategy.num_threads == 0:
            if self._exec_strategy.use_cuda:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                self._exec_strategy.num_threads = len(places) * 4
            else:
                self._exec_strategy.num_threads = len(places) * 2

        if self._build_strategy.num_trainers > 1:
            assert self._is_data_parallel, \
                "If you use multi-trainer to train the model, you should use "\
                "the data parallel model, i.e. calling with_data_parallel function."

        # TODO(wuyi): trainer endpoings should be passed in through
        # build_strategy, not program.xxx.
        # TODO(gongwb): let user to set them once.
        if self._program and self._build_strategy.num_trainers > 1 and \
                self._program._trainers_endpoints:
            tps = self._program._trainers_endpoints

            assert self._build_strategy.num_trainers == len(
                tps), "num_trainers == len(end_points)"
            self._build_strategy.trainers_endpoints = tps

        if self._program:
            self._build_strategy.nccl_comm_num = self._program._nccl_comm_num
            self._build_strategy.use_hierarchical_allreduce = self._program._use_hierarchical_allreduce
            self._build_strategy.hierarchical_allreduce_inter_nranks = self._program._hierarchical_allreduce_inter_nranks

        if self._build_strategy.sync_batch_norm:
            self._build_strategy.enable_sequential_execution = True

        self._persistable_vars = []
        for node in self._graph.nodes():
            if node.is_var() and node.var() is not None and node.var().persistable() and \
                    node.var().type() != core.VarDesc.VarType.RAW:
                self._persistable_vars.append(cpt.to_text(node.name()))

        places = list(map(_place_obj, places))

        # ParallelExecutor would broadcast all the parameters during initializing.
        # The parameters of each process should be in the same ordered for the data-parallelism
        # distributed training to keep the broadcast correct.
        self._persistable_vars = list(set(self._persistable_vars))
        self._persistable_vars.sort()

        return core.ParallelExecutor(
            places, self._persistable_vars,
            cpt.to_text(self._loss_name)
            if self._loss_name else six.u(''), self._scope, self._local_scopes,
            self._exec_strategy, self._build_strategy, self._graph)

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
                raise ValueError("Cannot compile with different scope")
            if place and not self._place._equals(place):
                raise ValueError("Cannot compile with different place")
            return self
        self._compiled = True

        self._scope = scope
        self._place = place

        if self._is_inference:
            self._executor = self._compile_inference()
        else:
            if self._is_data_parallel:
                self._places = self._get_places(self._place, self._places)
            else:
                self._places = [self._place]
            self._executor = self._compile_data_parallel(
                use_cuda=isinstance(self._place, core.CUDAPlace),
                scope=self._scope,
                places=self._places)
        return self

    def _get_places(self, place, place_list):
        has_set_place = (place_list is not None)
        if has_set_place:
            for p in place_list:
                assert p._type() == place._type(), \
                    "Place type not match. You may set the wrong type of places"
        else:
            place_list = cuda_places() if isinstance(
                place, core.CUDAPlace) else cpu_places()
        assert place_list, "no place for execution"
        return place_list
