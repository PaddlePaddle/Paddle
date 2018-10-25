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

__all__ = ['AsyncExecutorParameter', 'AsyncExecutor']

g_scope = core.Scope()

class AsyncExecutorParameter(object):
    """
    AsyncExecutor configure parameter

    Args:
        None    	
    """
    def __init__(self):
        self.parameter = core.AsyncExecutorParameter()

    def parse(self, conf_file):
        self.parameter.parse(conf_file)

class AsyncExecutor(object):
    """
    An asynchronous Executor in Python

    Args:
        place(core.CPUPlace|core.CUDAPlace(n)): indicate the executor run on which device

    Note: For debugging complicated network in parallel-GPUs, you can test it on the executor.
    They has the exactly same arguments, and expected the same results.
    """

    def __init__(self,
                 async_executor_parameter,
                 place,
                 scope):
        if not isinstance(async_executor_parameter, AsyncExecutorParameter):
            raise TypeError(
                "AsyncExecutor requires AsyncExecutorParameter as its parameter. "
                "But you passed in %s" %s (type(async_executor_parameter))
            )

        self.place = place
        p = core.Place()
        p.set_place(place)
        self.executor = core.AsyncExecutor(p)
        self.executor.init(async_executor_parameter.parameter, scope)
        self._closed = False
        self.parameter = async_executor_parameter.parameter

    def close(self):
        """
        Close this executor.

        You can no long use this executor after calling this method.
        For the distributed training, this method would free the resource on PServers related to
        the current Trainer.

        Example:
            >>> cpu = core.CPUPlace()
            >>> exe = Executor(cpu)
            >>> ...
            >>> exe.close()
        """
        if not self._closed:
            self._closed = True
    def run_startup_program(self,
                            program=None,
                            scope=None):
        if program is None:
            program = default_startup_program()
        program_desc = program._get_desc()

        if scope is None:
            scope = g_scope

        self.executor.run_startup_program(program_desc, scope)
        
    def run(self, program=None, scope=None):
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

        if self._closed:
            raise RuntimeError("Attempted to use a closed Executor")

        if program is None:
            program = default_main_program()
        program_desc = program.desc

        if not isinstance(program, Program):
            raise TypeError(
                "Executor requires Program as its Parameter. But you passed in %s"
                % (type(program)))

        if scope is None:
            scope = g_scope
 
        self.executor.run(program.desc)

    def load_init_model(self):
        return self.executor.load_init_model()
