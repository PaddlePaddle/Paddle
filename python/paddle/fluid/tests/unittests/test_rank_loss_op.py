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
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestRankLossOp(OpTest):

    def setUp(self):
        self.op_type = "rank_loss"
        shape = (100, 1)
        # labels_{i} = {0, 1.0} or {0, 0.5, 1.0}
        label_shape, left_shape, right_shape = self.set_shape()
        label = np.random.randint(0, 2, size=shape).astype("float32")
        left = np.random.random(shape).astype("float32")
        right = np.random.random(shape).astype("float32")
        loss = np.log(1.0 + np.exp(left - right)) - label * (left - right)
        loss = np.reshape(loss, label_shape)
        self.inputs = {
            'Label': label.reshape(label_shape),
            'Left': left.reshape(left_shape),
            'Right': right.reshape(right_shape)
        }
        self.outputs = {'Out': loss.reshape(label_shape)}

    def set_shape(self):
        batch_size = 100
        return (batch_size, 1), (batch_size, 1), (batch_size, 1)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Left", "Right"], "Out")

    def test_check_grad_ignore_left(self):
        self.check_grad(["Right"], "Out", no_grad_set=set('Left'))

    def test_check_grad_ignore_right(self):
        self.check_grad(["Left"], "Out", no_grad_set=set('Right'))


class TestRankLossOp1(TestRankLossOp):

    def set_shape(self):
        batch_size = 100
        return (batch_size), (batch_size, 1), (batch_size, 1)


class TestRankLossOp2(TestRankLossOp):

    def set_shape(self):
        batch_size = 100
        return (batch_size, 1), (batch_size), (batch_size, 1)


class TestRankLossOp3(TestRankLossOp):

    def set_shape(self):
        batch_size = 100
        return (batch_size, 1), (batch_size, 1), (batch_size)


class TestRankLossOp4(TestRankLossOp):

    def set_shape(self):
        batch_size = 100
        return (batch_size), (batch_size), (batch_size, 1)


class TestRankLossOp5(TestRankLossOp):

    def set_shape(self):
        batch_size = 100
        return (batch_size), (batch_size), (batch_size)


class TestRankLossOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            label = fluid.data(name="label", shape=[16, 1], dtype="float32")
            left = fluid.data(name="left", shape=[16, 1], dtype="float32")
            right = fluid.data(name="right", shape=[16, 1], dtype="float32")

            def test_label_Variable():
                label_data = np.random.rand(16, 1).astype("float32")
                out = fluid.layers.rank_loss(label_data, left, right)

            self.assertRaises(TypeError, test_label_Variable)

            def test_left_Variable():
                left_data = np.random.rand(16, 1).astype("float32")
                out = fluid.layers.rank_loss(label, left_data, right)

            self.assertRaises(TypeError, test_left_Variable)

            def test_right_Variable():
                right_data = np.random.rand(16, 1).astype("float32")
                out = fluid.layers.rank_loss(label, left, right_data)

            self.assertRaises(TypeError, test_right_Variable)


if __name__ == '__main__':
    unittest.main()
