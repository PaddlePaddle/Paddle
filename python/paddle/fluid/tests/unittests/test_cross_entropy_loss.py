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


class CrossEntropyLoss(unittest.TestCase):
    def test_cross_entropy_loss_mean(self):
        input_np = np.random.random([5, 100]).astype(np.float32)
        label_np = np.random.random([5, 1]).astype(np.int64)
        weight_np = np.random.random([100]).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.layers.data(
                name='input', shape=[5, 100], dtype='float32')
            label = fluid.layers.data(name='label', shape=[5, 1], dtype='int64')
            weight = fluid.layers.data(
                name='weight', shape=[100], dtype='float32')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight)
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        self.assertTrue(np.allclose(static_ret, dy_ret_value))

    def test_cross_entropy_loss_sum(self):
        input_np = np.random.random([5, 100]).astype(np.float32)
        label_np = np.random.random([5, 1]).astype(np.int64)
        weight_np = np.random.random([100]).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.layers.data(
                name='input', shape=[5, 100], dtype='float32')
            label = fluid.layers.data(name='label', shape=[5, 1], dtype='int64')
            weight = fluid.layers.data(
                name='weight', shape=[100], dtype='float32')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='sum')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='sum')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        self.assertTrue(np.allclose(static_ret, dy_ret_value))

    def test_cross_entropy_loss_none(self):
        input_np = np.random.random([5, 100]).astype(np.float32)
        label_np = np.random.random([5, 1]).astype(np.int64)
        weight_np = np.random.random([100]).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.layers.data(
                name='input', shape=[5, 100], dtype='float32')
            label = fluid.layers.data(name='label', shape=[5, 1], dtype='int64')
            weight = fluid.layers.data(
                name='weight', shape=[100], dtype='float32')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='none')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='none')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        self.assertTrue(np.allclose(static_ret, dy_ret_value))


if __name__ == "__main__":
    unittest.main()
