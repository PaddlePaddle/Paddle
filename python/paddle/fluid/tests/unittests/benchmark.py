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

import numpy as np
import unittest
import time

import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest


class BenchmarkSuite(OpTest):
    def timeit_function(self, callback, iters, *args):
        assert iters != 0, "Iters should >= 1"
        start = time.time()
        for i in range(iters):
            callback(*args)
            elapse = time.time() - start
        return elapse / iters

    def _get_places(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(self.op_type):
            places.append(core.CUDAPlace(0))
        return places

    def timeit_output_with_place(self, place, iters):
        return self.timeit_function(self.calc_output, iters, place)

    def timeit_output(self, iters=100):
        places = self._get_places()
        elapses = []
        for place in places:
            elapses.append(self.timeit_output_with_place(place, iters))
        for place, elapse in zip(places, elapses):
            print("One pass of {3} at {0} cost {1}".format(
                str(place), elapse, self.op_type))

    def timeit_grad_with_place(self, place, iters=100):
        return self.timeit_function(self.calc_gra, iters, place)

    def timeit_grad(self, iters=100):
        analytic_grads = self._get_gradient(
            inputs_to_check, place, output_names, no_grad_set=None)

    def _get_input_names(self):
        inputs = []
        for name, value in self.inputs.iteritems():
            if isinstance(value, list):
                inputs.append([sub_name for sub_name, _ in value])
            inputs.append(name)
        return inputs
