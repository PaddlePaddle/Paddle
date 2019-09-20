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

import unittest
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import switch_main_program
from paddle.fluid.framework import Program
import numpy as np
from simple_nets import simple_fc_net, init_data


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
        layers.Print(input=x, **kargs)
        loss = layers.mean(x)
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

    def test_all_parameters(self):
        x = layers.data('x', shape=[3], dtype='float32', lod_level=1)
        x.stop_gradient = False

        for print_tensor_name in [True, False]:
            for print_tensor_type in [True, False]:
                for print_tensor_shape in [True, False]:
                    for print_tensor_lod in [True, False]:
                        layers.Print(
                            input=x,
                            print_tensor_name=print_tensor_name,
                            print_tensor_type=print_tensor_type,
                            print_tensor_shape=print_tensor_shape,
                            print_tensor_lod=print_tensor_lod, )
        loss = layers.mean(x)
        append_backward(loss=loss)
        exe = Executor(self.place)
        outs = exe.run(feed={'x': self.x_tensor},
                       fetch_list=[loss],
                       return_numpy=False)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestPrintOpGPU(TestPrintOpCPU):
    def setUp(self):
        self.place = core.CUDAPlace(0)
        self.x_tensor = core.LoDTensor()
        tensor_np = np.random.random(size=(2, 3)).astype('float32')
        self.x_tensor.set(tensor_np, self.place)
        self.x_tensor.set_recursive_sequence_lengths([[1, 1]])


class TestPrintOpBackward(unittest.TestCase):
    def check_backward(self, use_cuda):
        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = simple_fc_net()
            loss = fluid.layers.Print(loss)
            fluid.optimizer.Adam().minimize(loss)

        print_ops = [op for op in main.blocks[0].ops if op.type == u'print']
        assert len(print_ops) == 2, "The number of print op should be 2"

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)

        binary = fluid.compiler.CompiledProgram(main).with_data_parallel(
            loss_name=loss.name)

        img, label = init_data()
        feed_dict = {"image": img, "label": label}
        exe.run(binary, feed_dict)

    def test_fw_bw(self):
        if core.is_compiled_with_cuda():
            self.check_backward(use_cuda=True)
        self.check_backward(use_cuda=False)


if __name__ == '__main__':
    unittest.main()
