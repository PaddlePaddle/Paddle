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

__all__ = ['DataFeedDesc', 'AsyncExecutor']


class DataFeedDesc(object):
    """
    Datafeed descriptor, describing input training data format. This class is
    currently only used for AsyncExecutor (See comments for class AsyncExecutor
    for a brief introduction)

    DataFeedDesc shall be initialized from a valid protobuf message from disk:
    >>> data_feed = fluid.DataFeedDesc('data.proto')

    See :code:`paddle/fluid/framework/data_feed.proto` for message definition.
    A typical message might look like:

    >>> name: "MultiSlotDataFeed"
    >>> batch_size: 2
    >>> multi_slot_desc {
    >>>     slots {
    >>>         name: "words"
    >>>         type: "uint64"
    >>>         is_dense: false
    >>>         is_used: true
    >>>     }
    >>>     slots {
    >>>         name: "label"
    >>>         type: "uint64"
    >>>         is_dense: false
    >>>         is_used: true
    >>>     }
    >>> }

    However, users usually shouldn't care about the message format; instead,
    they are encouragd to use :code:`Data Generator` as a tool to generate a
    valid data description, in the process of converting their raw log files to
    training files acceptable to AsyncExecutor.

    DataFeedDesc can also be changed during runtime. Once you got familiar with
    what each field mean, you can modify it to better suit your need. E.g.:
    >>> data_feed.set_batch_size(128)
    >>> data_feed.set_dense_slots('wd')  # The slot named 'wd' will be dense
    >>> data_feed.set_use_slots('wd')    # The slot named 'wd' will be used

    Finally, the content can be dumped out for debugging purpose:
    >>> print(data_feed.desc())

    Args:
        proto_file(string): Disk file containing a data feed description.
    
    """

    def __init__(self, proto_file):
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        if self.proto_desc.name == "MultiSlotDataFeed":
            self.__name_to_index = {
                slot.name: i
                for i, slot in enumerate(self.proto_desc.multi_slot_desc.slots)
            }

    def set_batch_size(self, batch_size):
        """
        Set batch size. Will be effective during training

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> data_feed.set_batch_size(128)

        Args:
            batch_size: batch size

        """
        self.proto_desc.batch_size = batch_size

    def set_dense_slots(self, dense_slots_name):
        """
        Set if a specific slot will be dense. Will be effective during training.
        features for a dense slot will be fed into a Tensor, while those for a
        sparse slot will be fed into a LoDTensor

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> data_feed.set_dense_slots(['words'])

        Args:
            dense_slots_name: a list of slot names which will be set dense

        Note:
            Default is sparse for all slots
        """
        if self.proto_desc.name != "MultiSlotDataFeed":
            raise ValueError(
                "Only MultiSlotDataFeed need set_dense_slots, pls check your datafeed.proto"
            )
        for name in dense_slots_name:
            self.proto_desc.multi_slot_desc.slots[self.__name_to_index[
                name]].is_dense = True

    def set_use_slots(self, use_slots_name):
        """
        Set if a specific slot will be used for training. A dataset shall
        contain a lot of features, through this function one can select which
        ones will be used for a specific model.

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> data_feed.set_use_slots(['words'])

        Args:
            use_slots_name: a list of slot names which will be used in training

        Note:
            Default is not used for all slots
        """
        if self.proto_desc.name != "MultiSlotDataFeed":
            raise ValueError(
                "Only MultiSlotDataFeed need set_use_slots, pls check your datafeed.proto"
            )
        for name in use_slots_name:
            self.proto_desc.multi_slot_desc.slots[self.__name_to_index[
                name]].is_used = True

    def desc(self):
        """
        Returns a protobuf message for this DataFeedDesc

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> print(data_feed.desc())

        Returns:
            A string message
        """
        return text_format.MessageToString(self.proto_desc)


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
