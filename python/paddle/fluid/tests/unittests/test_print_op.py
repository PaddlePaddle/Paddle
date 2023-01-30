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

<<<<<<< HEAD
import unittest

import numpy as np
from simple_nets import init_data, simple_fc_net

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import switch_main_program
=======
from __future__ import print_function
import unittest

import numpy as np

from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid import core
from paddle.fluid.framework import switch_main_program
from simple_nets import simple_fc_net, init_data
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.static import Program, program_guard

paddle.enable_static()


class TestPrintOpCPU(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.place = paddle.CPUPlace()
        self.x_tensor = fluid.core.LoDTensor()
        tensor_np = np.random.random(size=(2, 3)).astype('float32')
        self.x_tensor.set(tensor_np, self.place)
        self.x_tensor.set_recursive_sequence_lengths([[1, 1]])

    def build_network(self, only_forward, **kargs):
<<<<<<< HEAD
        x = paddle.static.data('x', shape=[-1, 3], dtype='float32', lod_level=1)
=======
        x = layers.data('x', shape=[3], dtype='float32', lod_level=1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x.stop_gradient = False
        paddle.static.Print(input=x, **kargs)
        loss = paddle.mean(x)
        paddle.static.append_backward(loss=loss)
        return loss

    def test_forward(self):
        switch_main_program(Program())
        printed = self.build_network(True, print_phase='forward')
        exe = paddle.static.Executor(self.place)
<<<<<<< HEAD
        outs = exe.run(
            feed={'x': self.x_tensor}, fetch_list=[printed], return_numpy=False
        )
=======
        outs = exe.run(feed={'x': self.x_tensor},
                       fetch_list=[printed],
                       return_numpy=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_backward(self):
        switch_main_program(Program())
        loss = self.build_network(False, print_phase='backward')
        exe = paddle.static.Executor(self.place)
<<<<<<< HEAD
        outs = exe.run(
            feed={'x': self.x_tensor}, fetch_list=[loss], return_numpy=False
        )

    def test_all_parameters(self):
        x = paddle.static.data('x', shape=[-1, 3], dtype='float32', lod_level=1)
=======
        outs = exe.run(feed={'x': self.x_tensor},
                       fetch_list=[loss],
                       return_numpy=False)

    def test_all_parameters(self):
        x = layers.data('x', shape=[3], dtype='float32', lod_level=1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x.stop_gradient = False

        for print_tensor_name in [True, False]:
            for print_tensor_type in [True, False]:
                for print_tensor_shape in [True, False]:
                    for print_tensor_lod in [True, False]:
                        paddle.static.Print(
                            input=x,
                            print_tensor_name=print_tensor_name,
                            print_tensor_type=print_tensor_type,
                            print_tensor_shape=print_tensor_shape,
                            print_tensor_lod=print_tensor_lod,
                        )
        loss = paddle.mean(x)
        paddle.static.append_backward(loss=loss)
        exe = paddle.static.Executor(self.place)
<<<<<<< HEAD
        outs = exe.run(
            feed={'x': self.x_tensor}, fetch_list=[loss], return_numpy=False
        )
=======
        outs = exe.run(feed={'x': self.x_tensor},
                       fetch_list=[loss],
                       return_numpy=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_no_summarize(self):
        switch_main_program(Program())
        printed = self.build_network(True, summarize=-1, print_phase='forward')
        exe = paddle.static.Executor(self.place)
<<<<<<< HEAD
        outs = exe.run(
            feed={'x': self.x_tensor}, fetch_list=[printed], return_numpy=False
        )


class TestPrintOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of Print_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], paddle.CPUPlace()
            )
=======
        outs = exe.run(feed={'x': self.x_tensor},
                       fetch_list=[printed],
                       return_numpy=False)


class TestPrintOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of Print_op must be Variable.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         paddle.CPUPlace())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertRaises(TypeError, paddle.static.Print, x1)
            # The input dtype of Print_op must be float32, float64, int32_t, int64_t or bool.
            x2 = paddle.static.data(name='x2', shape=[4], dtype="float16")
            self.assertRaises(TypeError, paddle.static.Print, x2)


<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestPrintOpGPU(TestPrintOpCPU):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestPrintOpGPU(TestPrintOpCPU):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.x_tensor = fluid.core.LoDTensor()
        tensor_np = np.random.random(size=(2, 3)).astype('float32')
        self.x_tensor.set(tensor_np, self.place)
        self.x_tensor.set_recursive_sequence_lengths([[1, 1]])


class TestPrintOpBackward(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def check_backward(self, use_cuda):
        main = paddle.static.Program()
        startup = paddle.static.Program()

        with program_guard(main, startup):
            loss = simple_fc_net()
            loss = paddle.static.Print(loss)
            paddle.optimizer.Adam().minimize(loss)

<<<<<<< HEAD
        print_ops = [op for op in main.blocks[0].ops if op.type == 'print']
=======
        print_ops = [op for op in main.blocks[0].ops if op.type == u'print']
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        assert len(print_ops) == 2, "The number of print op should be 2"

        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup)

        binary = paddle.static.CompiledProgram(main).with_data_parallel(
<<<<<<< HEAD
            loss_name=loss.name
        )
=======
            loss_name=loss.name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        img, label = init_data()
        feed_dict = {"image": img, "label": label}
        exe.run(binary, feed_dict)

    def test_fw_bw(self):
        if paddle.is_compiled_with_cuda():
            self.check_backward(use_cuda=True)
        self.check_backward(use_cuda=False)


if __name__ == '__main__':
    unittest.main()
