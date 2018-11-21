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
from .executor import global_scope
from paddle.fluid.proto import data_feed_pb2
from google.protobuf import text_format

__all__ = ['DataFeedDesc', 'AsyncExecutor']

g_scope = core.Scope()

class DataFeedDesc(object):
    def __init__(self, proto_file):
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        self.__name_to_index = {}
        for i, slot in enumerate(self.proto_desc.multi_slot_desc.slots):
            self.__name_to_index[slot.name] = i

    def set_data_feed_type(self, data_feed):
        self.proto_desc.name = datafeed

    def set_batch_size(self, batch_size):
        self.proto_desc.batch = batch_size

    def get_slot(self, name):
        return self.proto_desc.multi_slot_desc.slots[self.__name_to_index[name]]

    def set_use_slots(self, use_slots_name):
        for name in use_slots_name:
            self.proto_desc.multi_slot_desc.slots[self.__name_to_index[name]].use = True

    def desc(self):
        return text_format.MessageToString(self.proto_desc)

class AsyncExecutor(object):
    """
    An asynchronous Executor in Python

    Args:
        place(core.CPUPlace|core.CUDAPlace(n)): indicate the executor run on which device

    Note: For debugging complicated network in parallel-GPUs, you can test it on the executor.
    They has the exactly same arguments, and expected the same results.
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

    def run(self, program, data_feed, filelist, thread_num, fetch):
        """
        Run program by this Executor. Feed data by feed map, fetch result by fetch_list.
        Python executor takes a program, add feed operators and fetch operators to this program according
        to feed map and fetch_list. Feed map provides input data for the program. fetch_list provides
        the variables(or names) that user want to get after program run.

        Note: the executor will run all
        operators in the program but not only the operators dependent by the fetch_list

        Args:
            program(Program): the program that need to run, if not provied, then default_main_program will be used.
            feed(dict): feed variable map, e.g. {"image": ImageData, "label": LableData}
            fetch_list(list): a list of variable or variable names that user want to get, run will return them according to this list.
            feed_var_name(str): the name for the input variable of feed Operator.
            fetch_var_name(str): the name for the output variable of fetch Operator.
            scope(Scope): the scope used to run this program, you can switch it to different scope. default is global_scope
            return_numpy(bool): if convert the fetched tensor to numpy
            use_program_cache(bool): set use_program_cache to true if program not changed compare to the last step.

        Returns:

            list(numpy.array): fetch result according to fetch_list.


        Examples:

            >>> data = layers.data(name='X', shape=[1], dtype='float32')
            >>> hidden = layers.fc(input=data, size=10)
            >>> layers.assign(hidden, out)
            >>> loss = layers.mean(out)
            >>> adam = fluid.optimizer.Adam()
            >>> adam.minimize(loss)

            >>> cpu = core.CPUPlace()
            >>> exe = Executor(cpu)
            >>> exe.run(default_startup_program())

            >>> x = numpy.random.random(size=(10, 1)).astype('float32')
            >>> outs = exe.run(
            >>>     feed={'X': x},
            >>>     fetch_list=[loss.name])
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

        evaluation = self.executor.run_from_files(program_desc, data_feed, filelist, thread_num, fetch_var_names)
        return evaluation

