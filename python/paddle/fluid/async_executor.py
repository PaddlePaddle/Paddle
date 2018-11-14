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
from . import Executor

__all__ = ['TextDataFeed', 'AsyncExecutor']

g_scope = core.Scope()

class TextDataFeed():
    def __init__(self):
        self.feed = core.TextDataFeed()

    def set_filelist(self, filelist):
        self.feed.set_filelist(filelist)

    def set_batch_size(self, batch_size):
        self.feed.set_batch_size(batch_size)

    def set_field_names(self, field_names):
        if isinstance(field_names, Variable):
            field_names = [field_names]

        self.feed.set_field_names(field_names)

    def start_an_epoch(self):
        self.feed.start_one_epoch()

class AsyncExecutor(object):
    """
    An asynchronous Executor in Python

    Args:
        place(core.CPUPlace|core.CUDAPlace(n)): indicate the executor run on which device

    Note: For debugging complicated network in parallel-GPUs, you can test it on the executor.
    They has the exactly same arguments, and expected the same results.
    """

    def __init__(self,
                 program,
                 param_names,
                 data_feed,
                 thread_num,
                 place=None,
                 scope=None):
        if program is None:
            program = default_main_program()
        program_desc = program.desc

        if not isinstance(data_feed, TextDataFeed):
            raise ValueError("data_feed for AsyncExecutor.run() type error")

        if place is None:
            place = core.CPUPlace()
        if not isinstance(place, core.CPUPlace):
            raise ValueError("AsyncExecutor only supports CPU device")

        if isinstance(param_names, Variable):
            param_names = [param_names]

        p = core.Place()
        p.set_place(place)
        self.executor = core.AsyncExecutor(program_desc, param_names, data_feed.feed, thread_num, p)

    def run_startup_program(self,
                            program=None,
                            scope=None):
        if program is None:
            program = default_startup_program()
        program_desc = program._get_desc()

        if scope is None:
            scope = g_scope

        self.executor.run_startup_program(program_desc, scope)

    def run(self, inspect_vars, scope=None):
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
        if inspect_vars is not None:
            if isinstance(inspect_vars, Variable):
                inspect_vars = [inspect_vars]
            inspect_var_names = [var.name for var in inspect_vars]

        if scope is None:
            scope = g_scope

        self.executor.init_root_scope(scope)

        evaluation = self.executor.run(inspect_var_names)
        return evaluation

