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

from . import core
from . import framework

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


def get_available_places(use_cuda):
    if use_cuda:
        gpus_env = os.getenv("FLAGS_selected_gpus")
        if gpus_env:
            gpus = [int(s) for s in gpus_env.split(",")]
        else:
            gpus = [i for i in six.moves.range(core.get_cuda_device_count())]
        places = [core.CUDAPlace(i) for i in gpus]
    else:
        cpu_num = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        places = [core.CPUPlace() for _ in six.moves.range(cpu_num)]
    assert places, "no place for execution"
    return places


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
      * TODO: transform the program for optimized inference or distributed
              training.

    Example:
        .. code-block:: python
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup)
            compiled_prog = compiler.CompiledProgram(main).with_data_parallel(
                loss_name=loss.name)
            for i in range(5):
                test_loss, = exe.run(compiled_prog,
                                     feed=feed_dict,
                                     fetch_list=[loss.name])

    Args:
        program_or_graph (Graph|Program): If it's Program, it will be first
            lowered to a graph for further optimizations. If it's a graph
            (potentially optimized before), it will be directly used for
            further optimizations. Note: graph is only supported when compiled
            with with_data_parallel option.
    """

    def __init__(self, program_or_graph):
        if isinstance(program_or_graph, core.Graph):
            self._graph = program_or_graph
            self._program = None
        elif isinstance(program_or_graph, framework.Program):
            self._graph = core.Graph(program_or_graph.desc)
            self._program = program_or_graph
        else:
            raise ValueError("Wrong program_to_graph type: %s" %
                             type(program_or_graph))

        self._program_desc = self._graph.origin_program_desc()
        self._scope = None
        self._place = None
        self._executor = None
        self._compiled = False
        self._is_data_parallel = False
        self._is_inference = False

    def with_data_parallel(self,
                           loss_name=None,
                           build_strategy=None,
                           exec_strategy=None,
                           share_vars_from=None):
        """Configs the program to run in data parallel way.

        Args:
            loss_name (str): The loss name must set in training. Default None.
            build_strategy(BuildStrategy): build_strategy is used to
                build the graph so it can run on multiple devices/cores with
                optimized topology.
                For more information, please refer to fluid.BuildStrategy.
                Default None.
            exec_strategy(ExecutionStrategy): exec_strategy is used to
                to select the a way to execute the graph, for example how many
                threads are used, how many iterations to clean up the temp
                variables. For more information, please refer
                to fluid.ExecutionStrategy. Default None.
            share_vars_from(CompiledProgram): If provide, this CompiledProgram
                will share variables from `share_vars_from`. `share_vars_from`
                must be run by the executor before this CompiledProgram so that
                vars are ready.
        Returns:
            self
        """
        assert not self._is_data_parallel, "Already compiled with parallel."
        assert not self._is_inference, "Cannot compile both data parallel and inference"
        self._is_data_parallel = True
        self._build_strategy = build_strategy
        self._exec_strategy = exec_strategy
        self._loss_name = loss_name
        self._share_vars_from = share_vars_from
        if self._exec_strategy is None:
            self._exec_strategy = ExecutionStrategy()
        if self._build_strategy is None:
            self._build_strategy = BuildStrategy()
        self._build_strategy.is_distribution = _is_pserver_mode(self._program)
        return self

    def with_inference_optimize(self, config):
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

    def _compile_data_parallel(self, use_cuda=False, scope=None):
        if self._share_vars_from:
            if scope:
                sys.stderr.write("share_vars_from is set, scope is ignored.\n")
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

        self._exec_strategy.use_cuda = use_cuda
        self._places = get_available_places(self._exec_strategy.use_cuda)

        if self._exec_strategy.num_threads == 0:
            if self._exec_strategy.use_cuda:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                self._exec_strategy.num_threads = len(self._places) * 4
            else:
                cpu_num = int(
                    os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
                self._exec_strategy.num_threads = cpu_num * 2

        # FIXME(dzhwinter): enable_inplace should be after memory_optimize
        # if turn on python memory optimize, turn off the inplace_pass.
        # memory_optimize and enable_inplace default are True, but we can disable them on purpose
        if self._program and self._program._is_mem_optimized:
            self._build_strategy.memory_optimize = False

        if self._program and self._program._is_mem_optimized:
            self._build_strategy.enable_inplace = False

        # TODO(wuyi): trainer endpoings should be passed in through
        # build_strategy, not program.xxx.
        if self._program and self._build_strategy.num_trainers > 1 and \
                self._program._trainers_endpoints:
            tps = self._program._trainers_endpoints

            assert self._build_strategy.num_trainers == len(
                tps), "num_trainers == len(end_points)"
            self._build_strategy.trainers_endpoints = tps

        if self._build_strategy.sync_batch_norm:
            self._build_strategy.enable_sequential_execution = True

        self._persistable_vars = []
        for node in self._graph.nodes():
            if node.is_var() and node.var() is not None and node.var().persistable() and \
                    node.var().type() != core.VarDesc.VarType.RAW:
                self._persistable_vars.append(cpt.to_text(node.name()))

        places = list(map(_place_obj, self._places))
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
        if self._is_data_parallel:
            self._executor = self._compile_data_parallel(
                use_cuda=isinstance(self._place, core.CUDAPlace),
                scope=self._scope)
        elif self._is_inference:
            self._executor = self._compile_inference()
        else:
            p = _place_obj(self._place)
            self._executor = core.Executor(p)
        return self
