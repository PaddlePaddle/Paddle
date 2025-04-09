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

import numpy as np
from op_test import convert_float_to_uint16
from simple_nets import init_data, simple_fc_net

import paddle
from paddle import base
from paddle.autograd.ir_backward import grad
from paddle.base import core
from paddle.framework import in_dynamic_or_pir_mode
from paddle.static import Program, program_guard

paddle.enable_static()


class TestPrintOpCPU(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float32'
        self.place = paddle.CPUPlace()
        self.x_tensor = base.core.DenseTensor()
        tensor_np = np.random.random(size=(2, 3)).astype(self.dtype)
        self.x_tensor.set(tensor_np, self.place)

    def build_network(self, only_forward, **kargs):
        x = paddle.static.data('x', shape=[-1, 3], dtype=self.dtype)
        x.stop_gradient = False
        paddle.static.Print(input=x, **kargs)
        loss = paddle.mean(x)

        if in_dynamic_or_pir_mode():
            dx = grad(loss, [x])
        else:
            paddle.static.append_backward(loss=loss)
        return loss

    def test_forward(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            printed = self.build_network(True, print_phase='forward')
            exe = paddle.static.Executor(self.place)
            outs = exe.run(
                feed={'x': self.x_tensor},
                fetch_list=[printed],
                return_numpy=False,
            )

    def test_backward(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            loss = self.build_network(False, print_phase='backward')
            exe = paddle.static.Executor(self.place)
            outs = exe.run(
                feed={'x': self.x_tensor}, fetch_list=[loss], return_numpy=False
            )

    def test_all_parameters(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog, paddle.static.Program()):
            x = paddle.static.data('x', shape=[-1, 3], dtype=self.dtype)
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
            if in_dynamic_or_pir_mode():
                dx = grad(loss, [x])
            else:
                paddle.static.append_backward(loss=loss)
            exe = paddle.static.Executor(self.place)
            outs = exe.run(
                feed={'x': self.x_tensor}, fetch_list=[loss], return_numpy=False
            )

    def test_no_summarize(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            printed = self.build_network(
                True, summarize=-1, print_phase='forward'
            )
            exe = paddle.static.Executor(self.place)
            outs = exe.run(
                feed={'x': self.x_tensor},
                fetch_list=[printed],
                return_numpy=False,
            )


class TestPrintOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of Print_op must be Variable.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], paddle.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.static.Print, x1)
            # The input dtype of Print_op must be uint16, float16, float32, float64, int32_t, int64_t or bool.
            x2 = paddle.static.data(name='x2', shape=[4], dtype="int8")
            self.assertRaises(TypeError, paddle.static.Print, x2)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestPrintOpGPU(TestPrintOpCPU):
    def setUp(self):
        self.dtype = 'float32'
        self.place = paddle.CUDAPlace(0)
        self.x_tensor = base.core.DenseTensor()
        tensor_np = np.random.random(size=(2, 3)).astype(self.dtype)
        self.x_tensor.set(tensor_np, self.place)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestPrintOpGPUFP16(TestPrintOpCPU):
    def setUp(self):
        self.dtype = 'float16'
        self.place = paddle.CUDAPlace(0)
        self.x_tensor = base.core.DenseTensor()
        tensor_np = np.random.random(size=(2, 3)).astype(self.dtype)
        self.x_tensor.set(tensor_np, self.place)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestPrintOpGPUBFP16(TestPrintOpCPU):
    def setUp(self):
        self.dtype = 'bfloat16'
        self.place = paddle.CUDAPlace(0)
        self.x_tensor = base.core.DenseTensor()
        tensor_np = convert_float_to_uint16(np.random.random(size=(2, 3)))
        self.x_tensor.set(tensor_np, self.place)


class TestPrintOpBackward(unittest.TestCase):
    def check_backward(self, use_cuda):
        with paddle.pir_utils.OldIrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()

            with paddle.static.program_guard(main, startup):
                loss = simple_fc_net()
                loss = paddle.static.Print(loss)
                paddle.optimizer.Adam().minimize(loss)

            print_ops = [op for op in main.blocks[0].ops if op.type == 'print']
            assert len(print_ops) == 2, "The number of print op should be 2"

            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup)

            binary = paddle.static.CompiledProgram(main)

            img, label = init_data()
            feed_dict = {"image": img, "label": label}
            exe.run(binary, feed_dict)

    # fc is not supported in pir
    #

    def test_fw_bw(self):
        if paddle.is_compiled_with_cuda():
            self.check_backward(use_cuda=True)
        self.check_backward(use_cuda=False)


if __name__ == '__main__':
    unittest.main()
