# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import sys

sys.path.append("..")
from op_test import OpTest
import numpy as np
import os

paddle.enable_static()


def sample_output_one_dimension(out, dim):
    # count numbers of different categories
    sample_prob = np.zeros(dim).astype("float32")
    sample_index_prob = np.unique(out, return_counts=True)
    sample_prob[sample_index_prob[0]] = sample_index_prob[1]
    sample_prob /= sample_prob.sum()
    return sample_prob


def sample_output_two_dimension(out, shape):
    num_dist = shape[0]
    out_list = np.split(out, num_dist, axis=0)
    sample_prob = np.zeros(shape).astype("float32")
    for i in range(num_dist):
        sample_index_prob = np.unique(out_list[i], return_counts=True)
        sample_prob[i][sample_index_prob[0]] = sample_index_prob[1]
    sample_prob /= sample_prob.sum(axis=-1, keepdims=True)
    return sample_prob


class TestMultinomialOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = "multinomial"
        self.init_data()
        self.inputs = {"X": self.input_np}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_data(self):
        # input probability is a vector, and replacement is True
        self.input_np = np.random.rand(4)
        self.outputs = {"Out": np.zeros(100000).astype("int64")}
        self.attrs = {"num_samples": 100000, "replacement": True}

    def test_check_output(self):
        self.check_output_customized(self.verify_output,
                                     custom_place=self.place)

    def sample_output(self, out):
        return sample_output_one_dimension(out, 4)

    def verify_output(self, outs):
        # normalize the input to get the probability
        prob = self.input_np / self.input_np.sum(axis=-1, keepdims=True)
        sample_prob = self.sample_output(np.array(outs[0]))
        np.testing.assert_allclose(sample_prob, prob, rtol=0, atol=0.01)


class TestMultinomialOp2(TestMultinomialOp):

    def init_data(self):
        # input probability is a matrix
        self.input_np = np.random.rand(3, 4)
        self.outputs = {"Out": np.zeros((3, 100000)).astype("int64")}
        self.attrs = {"num_samples": 100000, "replacement": True}

    def sample_output(self, out):
        return sample_output_two_dimension(out, [3, 4])


class TestMultinomialOp3(TestMultinomialOp):

    def init_data(self):
        # replacement is False. number of samples must be less than number of categories.
        self.input_np = np.random.rand(1000)
        self.outputs = {"Out": np.zeros(100).astype("int64")}
        self.attrs = {"num_samples": 100, "replacement": False}

    def verify_output(self, outs):
        out = np.array(outs[0])
        unique_out = np.unique(out)
        self.assertEqual(
            len(unique_out), 100,
            "replacement is False. categories can't be sampled repeatedly")


class TestMultinomialApi(unittest.TestCase):

    def test_dygraph(self):
        # input probability is a vector, and replacement is True
        paddle.set_device('npu:0')
        paddle.disable_static()
        x_numpy = np.random.rand(4)
        x = paddle.to_tensor(x_numpy)
        out = paddle.multinomial(x, num_samples=100000, replacement=True)

        sample_prob = sample_output_one_dimension(out.numpy(), 4)
        prob = x_numpy / x_numpy.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(sample_prob, prob, rtol=0, atol=0.01)
        paddle.enable_static()

    def test_dygraph2(self):
        # input probability is a matrix, and replacement is True
        paddle.set_device('npu:0')
        paddle.disable_static()
        x_numpy = np.random.rand(3, 4)
        x = paddle.to_tensor(x_numpy)
        out = paddle.multinomial(x, num_samples=100000, replacement=True)

        sample_prob = sample_output_two_dimension(out.numpy(), [3, 4])
        prob = x_numpy / x_numpy.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(sample_prob, prob, rtol=0, atol=0.01)
        paddle.enable_static()

    def test_dygraph3(self):
        # replacement is False. number of samples must be less than number of categories.
        paddle.set_device('npu:0')
        paddle.disable_static()
        x_numpy = np.random.rand(1000)
        x = paddle.to_tensor(x_numpy)
        out = paddle.multinomial(x, num_samples=100, replacement=False)

        unique_out = np.unique(out.numpy())
        self.assertEqual(
            len(unique_out), 100,
            "replacement is False. categories can't be sampled repeatedly")
        paddle.enable_static()

    def test_dygraph4(self):
        paddle.set_device('npu:0')
        paddle.disable_static()
        logits = -1 * paddle.ones([2800])
        # Categorical.sample API will call multinomial op with replacement=True
        cat = paddle.distribution.Categorical(logits.exp())
        cat.sample([1])
        paddle.enable_static()

    def test_static(self):
        paddle.set_device('npu:0')
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            x = fluid.data('x', shape=[4], dtype='float32')
            out = paddle.multinomial(x, num_samples=100000, replacement=True)

            place = fluid.NPUPlace(0)
            exe = fluid.Executor(place)

        exe.run(startup_program)
        x_np = np.random.rand(4).astype('float32')
        out = exe.run(train_program, feed={'x': x_np}, fetch_list=[out])

        sample_prob = sample_output_one_dimension(out, 4)
        prob = x_np / x_np.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(sample_prob, prob, rtol=0, atol=0.01)


class TestMultinomialAlias(unittest.TestCase):

    def test_alias(self):
        paddle.set_device('npu:0')
        x = paddle.rand([4])
        out1 = paddle.multinomial(x, num_samples=10, replacement=True)
        out2 = paddle.tensor.multinomial(x, num_samples=10, replacement=True)
        out3 = paddle.tensor.random.multinomial(x,
                                                num_samples=10,
                                                replacement=True)


class TestMultinomialError(unittest.TestCase):

    def setUp(self):
        paddle.set_device('npu:0')
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_num_sample(self):

        def test_num_sample_less_than_0():
            x = paddle.rand([4])
            out = paddle.multinomial(x, num_samples=-2)

        self.assertRaises(ValueError, test_num_sample_less_than_0)

    def test_input_probs_dim(self):

        def test_dim_larger_than_2():
            x = paddle.rand([2, 3, 3])
            out = paddle.multinomial(x)

        self.assertRaises(ValueError, test_dim_larger_than_2)

        def test_dim_less_than_1():
            x_np = np.random.random([])
            x = paddle.to_tensor(x_np)
            out = paddle.multinomial(x)

        self.assertRaises(ValueError, test_dim_less_than_1)

        with self.assertRaises(ValueError):
            prob = paddle.rand([20, 1000])
            prob[1:0] = 0
            out = paddle.multinomial(prob)


if __name__ == "__main__":
    unittest.main()
