# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest


class TestL1Loss(unittest.TestCase):
    def test_L1Loss_mean(self):
        input_np = np.random.random(size=(10, 1)).astype(np.float32)
        label_np = np.random.random(size=(10, 1)).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.layers.data(
                name='input', shape=[10, 1], dtype='float32')
            label = fluid.layers.data(
                name='label', shape=[10, 1], dtype='float32')
            l1_loss = paddle.nn.loss.L1Loss()
            ret = l1_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[ret])

        with fluid.dygraph.guard():
            l1_loss = paddle.nn.loss.L1Loss()
            dy_ret = l1_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_ret.numpy()

        expected = np.mean(np.abs(input_np - label_np))
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))
        self.assertTrue(dy_result.shape, [1])

    def test_L1Loss_sum(self):
        input_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        label_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.layers.data(
                name='input', shape=[10, 10, 5], dtype='float32')
            label = fluid.layers.data(
                name='label', shape=[10, 10, 5], dtype='float32')
            l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
            ret = l1_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[ret])

        with fluid.dygraph.guard():
            l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
            dy_ret = l1_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_ret.numpy()

        expected = np.sum(np.abs(input_np - label_np))
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))
        self.assertTrue(dy_result.shape, [1])

    def test_L1Loss_none(self):
        input_np = np.random.random(size=(10, 5)).astype(np.float32)
        label_np = np.random.random(size=(10, 5)).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.layers.data(
                name='input', shape=[10, 5], dtype='float32')
            label = fluid.layers.data(
                name='label', shape=[10, 5], dtype='float32')
            l1_loss = paddle.nn.loss.L1Loss(reduction='none')
            ret = l1_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[ret])

        with fluid.dygraph.guard():
            l1_loss = paddle.nn.loss.L1Loss(reduction='none')
            dy_ret = l1_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_ret.numpy()

        expected = np.abs(input_np - label_np)
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))
        self.assertTrue(dy_result.shape, input.shape)


if __name__ == "__main__":
    unittest.main()
