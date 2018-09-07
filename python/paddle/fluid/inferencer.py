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

import contextlib

import core

import executor
import framework
import io
import parallel_executor
import unique_name
from trainer import check_and_get_place

__all__ = ['Inferencer', ]


class Inferencer(object):
    def __init__(self, infer_func, param_path, place=None, parallel=False):
        """
        :param infer_func: a function that will return predict Variable
        :param param_path: the path where the inference model is saved by fluid.io.save_params
        :param place: place to do the inference
        :param parallel: use parallel_executor to run the inference, it will use multi CPU/GPU.
        """
        self.param_path = param_path
        self.scope = core.Scope()
        self.parallel = parallel
        self.place = check_and_get_place(place)

        self.inference_program = framework.Program()
        with framework.program_guard(self.inference_program):
            with unique_name.guard():
                self.predict_var = infer_func()

        with self._prog_and_scope_guard():
            # load params from param_path into scope
            io.load_params(executor.Executor(self.place), param_path)

        if parallel:
            with self._prog_and_scope_guard():
                self.exe = parallel_executor.ParallelExecutor(
                    use_cuda=isinstance(self.place, core.CUDAPlace),
                    loss_name=self.predict_var.name)
        else:
            self.exe = executor.Executor(self.place)

    def infer(self, inputs, return_numpy=True):
        """
        :param inputs: a map of {"input_name": input_var} that will be feed into the inference program
        to get the predict value
        :return: the predict value of the inference model
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "inputs should be a map of {'input_name': input_var}")

        with executor.scope_guard(self.scope):
            results = self.exe.run(self.inference_program,
                                   feed=inputs,
                                   fetch_list=[self.predict_var],
                                   return_numpy=return_numpy)

        return results

    @contextlib.contextmanager
    def _prog_and_scope_guard(self):
        with framework.program_guard(main_program=self.inference_program):
            with executor.scope_guard(self.scope):
                yield
