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

from __future__ import print_function
from . import core
from . import framework
from . import executor
from . import compiler
import sys

__all__ = ['ParallelExecutor']

ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy
BuildStrategy = core.ParallelExecutor.BuildStrategy


class ParallelExecutor(object):
    """
    ParallelExecutor is designed for data parallelism, which focuses on distributing
    the data across different nodes and every node operates on the data in parallel.
    If you use ParallelExecutor to run the current program on GPU, the node means GPU
    device, and ParallelExecutor will get the available GPU device automatically on
    the current machine. If you use ParallelExecutor to run the current program on CPU,
    the node means the CPU device, and you can specify the CPU device number by adding
    'CPU_NUM' environment variable, for example 'CPU_NUM=4', if the environment variable
    is not found, ParallelExecutor will call `multiprocessing.cpu_count` to get the number
    of CPUs in the system.

    Args:
        use_cuda (bool): Whether to use CUDA or not.
        loss_name (str): The loss name must set in training. Default None.
        main_program (Program): The program that need to run, if not provided,
            then default_main_program will be used. Default None.
        share_vars_from(ParallelExecutor): If provide, it will share variables
            from the specified ParallelExecutor. Default None.
        exec_strategy(ExecutionStrategy): exec_strategy is used to control how to run
            the program in ParallelExecutor, for example how many threads are used to
            execute the program, how many iterations to clean up the temp variables
            which is generated during execution. For more information, please refer
            to fluid.ExecutionStrategy. Default None.
        build_strategy(BuildStrategy): build_strategy is used to control how to
            build the SSA Graph in ParallelExecutor by setting the property,
            for example reduce_strategy, gradient_scale_strategy. For more information,
            please refer to fluid.BuildStrategy. Default None.
        num_trainers(int): If greater than 1, NCCL will be initialized with
            multiple rank of nodes, each node should have same number of GPUs.
            Distributed training will be enabled then. Default 1.
        trainer_id(int): Must use together with num_trainers. trainer_id is the
            "rank" of current node starts from 0. Default 0.
        scope(Scope): scope to run with, default use fluid.global_scope().

    Returns:
        ParallelExecutor: The initialized ParallelExecutor object.

    Raises:
        TypeError: If share_vars_from is provided, but not ParallelExecutor object.

    Examples:
        .. code-block:: python

          train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
          test_exe = fluid.ParallelExecutor(use_cuda=True,
                                            main_program=test_program,
                                            share_vars_from=train_exe)

          train_loss, = train_exe.run([loss.name], feed=feed_dict)
          test_loss, = test_exe.run([loss.name], feed=feed_dict)
    """

    def __init__(self,
                 use_cuda,
                 loss_name=None,
                 main_program=None,
                 share_vars_from=None,
                 exec_strategy=None,
                 build_strategy=None,
                 num_trainers=1,
                 trainer_id=0,
                 scope=None):
        sys.stderr.write(
            'ParallelExecutor is deprecated. '
            'Please use CompiledProgram and Executor. CompiledProgram '
            'is a central place for optimization and Executor is the '
            'unified executor. Example can be found in compiler.py.\n')

        if build_strategy is None:
            build_strategy = BuildStrategy()
        build_strategy.num_trainers = num_trainers
        build_strategy.trainer_id = trainer_id

        self._places = framework.cuda_places(
        ) if use_cuda else framework.cpu_places()
        self._scope = scope if scope is not None else executor.global_scope()

        if main_program is not None and main_program._enable_dgc:
            assert num_trainers > 1
            assert build_strategy.reduce_strategy == BuildStrategy.ReduceStrategy.AllReduce
            assert num_trainers * len(
                self._places) > 1, "dgc is not useful for single card training"
            assert use_cuda

        main_program = main_program if main_program is not None \
            else framework.default_main_program()

        self._compiled_program = compiler.CompiledProgram(main_program)
        if share_vars_from:
            assert isinstance(
                share_vars_from, ParallelExecutor
            ), "The share_vars_from should be ParallelExecutor."
        self._compiled_program.with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            share_vars_from=share_vars_from._compiled_program
            if share_vars_from else None)

        # FIXME(gongwb): I will move dgc from dist mode to allreduce mode in next pr.
        if main_program._enable_dgc:
            self._compiled_program._build_strategy.is_distribution = True

        self._place = core.CUDAPlace(0) if use_cuda else core.CPUPlace()
        self._exe = executor.Executor(self._place)
        self._compiled_program._compile(place=self._place, scope=self._scope)

    def run(self, fetch_list, feed=None, feed_dict=None, return_numpy=True):
        """
        Run a parallel executor with fetch_list.

        The feed parameter can be a dict or a list. If feed is a dict, the
        feed data will be split into multiple devices. If feed is a list, we
        assume the data has been splitted into multiple devices, the each
        element in the list will be copied to each device directly.

        For example, if the feed is a dict:

        >>> exe = ParallelExecutor()
        >>> # the image will be splitted into devices. If there is two devices
        >>> # each device will process an image with shape (24, 1, 28, 28)
        >>> exe.run(feed={'image': numpy.random.random(size=(48, 1, 28, 28))})

        For example, if the feed is a list:

        >>> exe = ParallelExecutor()
        >>> # each device will process each element in the list.
        >>> # the 1st device will process an image with shape (48, 1, 28, 28)
        >>> # the 2nd device will process an image with shape (32, 1, 28, 28)
        >>> #
        >>> # you can use exe.device_count to get the device number.
        >>> exe.run(feed=[{"image": numpy.random.random(size=(48, 1, 28, 28))},
        >>>               {"image": numpy.random.random(size=(32, 1, 28, 28))},
        >>>              ])

        Args:
            fetch_list(list): The fetched variable names
            feed(list|dict|None): The feed variables. If the feed is a dict,
                tensors in that dict will be splitted into each devices. If
                the feed is a list, each element of the list will be copied
                to each device. Default None.
            feed_dict: Alias for feed parameter, for backward compatibility.
                This parameter has been deprecated. Default None.
            return_numpy(bool): Whether converts the fetched tensor to numpy.
                Default: True.

        Returns:
            List: The fetched result list.

        Raises:
            ValueError: If the feed is a list, but its length is not equal the
                length of active places, or its element's is not dict.

        NOTES:
            1. If the feed's type is dict, the number of data that feeds to
               ParallelExecutor must be bigger than active places. Otherwise,
               it will throw exception from C++ side. Special attention should be
               paid to check whether the last batch of the dataset is bigger
               than active places.
            2. If active places are more than one, the fetch results for each
               variable is a list, and each element of this list is the variable of
               respective active place.

        Examples:
            .. code-block:: python

                pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                            loss_name=avg_cost.name,
                                            main_program=fluid.default_main_program())
                loss = pe.run(feed=feeder.feed(cur_batch),
                              fetch_list=[avg_cost.name]))
        """
        return self._exe.run(program=self._compiled_program,
                             scope=self._scope,
                             feed=feed,
                             fetch_list=fetch_list,
                             return_numpy=return_numpy)

    @property
    def device_count(self):
        return len(self._places)
