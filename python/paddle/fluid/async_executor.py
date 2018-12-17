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

import numpy as np
import contextlib
import six
from .framework import Program, default_main_program, Variable
from . import core
from .executor import global_scope, Executor
from paddle.fluid.proto import data_feed_pb2
from google.protobuf import text_format
from . import io
from .data_feed_desc import DataFeedDesc

__all__ = ['AsyncExecutor']


class AsyncExecutor(object):
    """
    An asynchronous Executor in Python. Through exploiting the power of
    multi-core processor and data queueing, AsyncExecutor makes data reading
    and cosuming decoupled, each run in multiple threads in parallel.

    Instead of reading data in python side, AsyncExecutor accepts a training
    file list, which will be retrieved in C++, then training inputs will be
    read, parsed and fed to training network within C++ code.

    AsyncExecutor is in active development and the API might change in the near
    future.

    Example:
        >>> data_feed = fluid.DataFeedDesc('data.proto')
        >>> startup_program = fluid.default_startup_program()
        >>> main_program = fluid.default_main_program()
        >>> filelist = ["train_data/part-%d" % i for i in range(100)]
        >>> thread_num = len(filelist) / 4
        >>>
        >>> place = fluid.CPUPlace()
        >>> async_executor = fluid.AsyncExecutor(place)
        >>>
        >>> async_executor.run_startup_program(startup_program)
        >>>
        >>> epoch = 10
        >>> for i in range(epoch):
        >>>     async_executor.run(main_program,
        >>>                        data_feed,
        >>>                        filelist,
        >>>                        thread_num,
        >>>                        [acc],
        >>>                        debug=False)

    Args:
        place(fluid.CPUPlace|None): indicate the executor run on which device.
                                   Only CPUPlace supported

    Note:
        For debugging complicated network in parallel-GPUs, you can test it
        on the executor. They has the exactly same arguments, and expected
        the same results.

    Note: Only running on CPUPlace supported.
    """

    def __init__(self, place=None):
        if place is None:
            place = core.CPUPlace()
        if not isinstance(place, core.CPUPlace):
            raise ValueError("AsyncExecutor only supports CPU device")

        p = core.Place()
        p.set_place(place)

        scope = global_scope()
        self.executor = core.AsyncExecutor(scope, p)

    def run(self, program, data_feed, filelist, thread_num, fetch, debug=False):
        """
        Run program by this AsyncExecutor. Training dataset will be in filelist.
        Users can also inspect certain variables by naming them in parameter
        :code:`fetch`, like in fluid.Executor. Unlike fluid.Executor, however,
        AsyncExecutor doesn't return fetched variables, instead, it will dump
        the values of each fetched variable to stdandard output.

        Running the dataset will be on multiple threads, within each a thread
        local scope will be created, then all OPs also created in that scope.
        Parameters are updated by all the OPs simultaneously.

        Args:
            program(Program): the program that need to run, if not provied,
                              then default_main_program will be used.
            data_feed(DataFeedDesc): A DataFeedDesc object
            filelist(str): a file containing the training dataset file list
            thread_num(int): number of concurrent training threads. See
                             :code:`Note` for how to set this properly
            fetch(str|list): the var name or a list of var names to inspect
            debug(bool): When set to True, fetch vars will be printed to
                         standard output after each minibatch

        Note:
            the executor will run all operators in the program but not only
            the operators dependent by the fetch_list.

        Note:
            Running AsyncExecutor will be on multiple threads, each bound to a
            CPU core. To achieve best performance, it's suggested to set thread
            num to be equal or slightly less than that of CPU cores.
        """
        if program is None:
            program = default_main_program()
        program_desc = program.desc

        if data_feed is None:
            raise ValueError('ValueError: data_feed should be provided')

        if filelist is None:
            raise ValueError('ValueError: filelist should be provided')

        if isinstance(filelist, str):
            filelist = [filelist]

        if not isinstance(thread_num, int):
            raise TypeError('TypeError: thread_num should be a positive number')

        if fetch is not None:
            if isinstance(fetch, Variable):
                fetch = [fetch]
            fetch_var_names = [var.name for var in fetch]
            for fetch_var in fetch:
                shape = fetch_var.shape
                if shape[len(shape) - 1] != 1:
                    raise AssertionError(
                        "%s: Fetch variable has wrong shape. Only varibles "
                        "with the last dimension size 1 supported." %
                        (fetch_var.name))

        self.executor.run_from_files(program_desc,
                                     data_feed.desc(), filelist, thread_num,
                                     fetch_var_names, debug)
