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

import unittest
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
import paddle.fluid.layers as layers
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import switch_main_program
from paddle.fluid.framework import Program
import numpy as np


class TestPrintOpCPU(unittest.TestCase):
    def setUp(self):
        self.place = core.CPUPlace()
        self.x_tensor = core.LoDTensor()
        tensor_np = np.random.random(size=(2, 3)).astype('float32')
        self.x_tensor.set(tensor_np, self.place)
        self.x_tensor.set_recursive_sequence_lengths([[1, 1]])

    def build_network(self, only_forward, **kargs):
        x = layers.data('x', shape=[3], dtype='float32', lod_level=1)
        x.stop_gradient = False
        printed = layers.Print(input=x, **kargs)
        if only_forward: return printed
        loss = layers.mean(printed)
        append_backward(loss=loss)
        return loss

    def test_forward(self):
        switch_main_program(Program())
        printed = self.build_network(True, print_phase='forward')
        exe = Executor(self.place)
        outs = exe.run(feed={'x': self.x_tensor},
                       fetch_list=[printed],
                       return_numpy=False)

    def test_backward(self):
        switch_main_program(Program())
        loss = self.build_network(False, print_phase='backward')
        exe = Executor(self.place)
        outs = exe.run(feed={'x': self.x_tensor},
                       fetch_list=[loss],
                       return_numpy=False)


class TestPrintOpGPU(TestPrintOpCPU):
    def setUp(self):
        self.place = core.CUDAPlace(0)
        self.x_tensor = core.LoDTensor()
        tensor_np = np.random.random(size=(2, 3)).astype('float32')
        self.x_tensor.set(tensor_np, self.place)
        self.x_tensor.set_recursive_sequence_lengths([[1, 1]])


if __name__ == '__main__':
    unittest.main()
