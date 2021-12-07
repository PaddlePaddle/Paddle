# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import paddle
import paddle.optimizer as optimizer


class TestOptimizerForVarBase(unittest.TestCase):
    def setUp(self):
        self.lr = 0.01

    def run_optimizer_step_with_varbase_list_input(self, optimizer):
        x = paddle.zeros([2, 3])
        y = paddle.ones([2, 3])
        x.stop_gradient = False

        z = x + y

        opt = optimizer(
            learning_rate=self.lr, parameters=[x], weight_decay=0.01)

        z.backward()
        opt.step()

        self.assertTrue(np.allclose(x.numpy(), np.full([2, 3], -self.lr)))

    def run_optimizer_minimize_with_varbase_list_input(self, optimizer):
        x = paddle.zeros([2, 3])
        y = paddle.ones([2, 3])
        x.stop_gradient = False

        z = x + y

        opt = optimizer(learning_rate=self.lr, parameters=[x])

        z.backward()
        opt.minimize(z)

        self.assertTrue(np.allclose(x.numpy(), np.full([2, 3], -self.lr)))

    def test_adam_with_varbase_list_input(self):
        self.run_optimizer_step_with_varbase_list_input(optimizer.Adam)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Adam)

    def test_sgd_with_varbase_list_input(self):
        self.run_optimizer_step_with_varbase_list_input(optimizer.SGD)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.SGD)

    def test_adagrad_with_varbase_list_input(self):
        self.run_optimizer_step_with_varbase_list_input(optimizer.Adagrad)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Adagrad)

    def test_adamw_with_varbase_list_input(self):
        self.run_optimizer_step_with_varbase_list_input(optimizer.AdamW)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.AdamW)

    def test_adamax_with_varbase_list_input(self):
        self.run_optimizer_step_with_varbase_list_input(optimizer.Adamax)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Adamax)

    def test_momentum_with_varbase_list_input(self):
        self.run_optimizer_step_with_varbase_list_input(optimizer.Momentum)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Momentum)

    def test_optimizer_with_varbase_input(self):
        x = paddle.zeros([2, 3])
        with self.assertRaises(TypeError):
            optimizer.Adam(learning_rate=self.lr, parameters=x)

    def test_create_param_lr_with_1_for_coverage(self):
        x = paddle.fluid.framework.ParamBase(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="x",
            optimize_attr={'learning_rate': 1.0})
        x.value().get_tensor().set(
            np.random.random((5, 10)).astype('float32'),
            paddle.fluid.framework._current_expected_place())

        y = paddle.ones([5, 10])
        z = x + y
        opt = optimizer.Adam(learning_rate=self.lr, parameters=[x])
        z.backward()
        opt.step()

    def test_create_param_lr_with_no_1_value_for_coverage(self):
        x = paddle.fluid.framework.ParamBase(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="x",
            optimize_attr={'learning_rate': 0.12})
        x.value().get_tensor().set(
            np.random.random((5, 10)).astype('float32'),
            paddle.fluid.framework._current_expected_place())

        y = paddle.ones([5, 10])
        z = x + y
        opt = optimizer.Adam(learning_rate=self.lr, parameters=[x])
        z.backward()
        opt.step()


if __name__ == "__main__":
    unittest.main()
