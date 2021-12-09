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
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


class TestLogLossOp(OpTest):
    def setUp(self):
        self.op_type = 'log_loss'
        samples_num = 100

        x = np.random.random((samples_num, 1)).astype("float32")
        predicted = sigmoid_array(x)
        labels = np.random.randint(0, 2, (samples_num, 1)).astype("float32")
        epsilon = 1e-7
        self.inputs = {
            'Predicted': predicted,
            'Labels': labels,
        }

        self.attrs = {'epsilon': epsilon}
        loss = -labels * np.log(predicted + epsilon) - (
            1 - labels) * np.log(1 - predicted + epsilon)
        self.outputs = {'Loss': loss}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Predicted'], 'Loss', max_relative_error=0.03)


class TestLogLossOpError(unittest.TestCase):
    def test_errors(self):
        with fluid.program_guard(fluid.Program()):

            def test_x_type():
                input_data = np.random.random(100, 1).astype("float32")
                fluid.layers.log_loss(input_data)

            self.assertRaises(TypeError, test_x_type)

            def test_x_dtype():
                x2 = fluid.layers.data(name='x2', shape=[100, 1], dtype='int32')
                fluid.layers.log_loss(x2)

            self.assertRaises(TypeError, test_x_dtype)

            def test_label_type():
                input_data = np.random.random(100, 1).astype("float32")
                fluid.layers.log_loss(input_data)

            self.assertRaises(TypeError, test_label_type)

            def test_label_dtype():
                x2 = fluid.layers.data(name='x2', shape=[100, 1], dtype='int32')
                fluid.layers.log_loss(x2)

            self.assertRaises(TypeError, test_label_dtype)


if __name__ == '__main__':
    unittest.main()
