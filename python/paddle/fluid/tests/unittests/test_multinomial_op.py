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
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from op_test import OpTest
import numpy as np
import os


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
        paddle.enable_static()
        self.op_type = "multinomial"
        self.init_data()
        self.inputs = {"X": self.input_np}

    def init_data(self):
        # input probability is a vector, and replacement is True
        self.input_np = np.random.rand(4)
        self.outputs = {"Out": np.zeros(100000).astype("int64")}
        self.attrs = {"num_samples": 100000, "replacement": True}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def sample_output(self, out):
        return sample_output_one_dimension(out, 4)

    def verify_output(self, outs):
        # normalize the input to get the probability
        prob = self.input_np / self.input_np.sum(axis=-1, keepdims=True)
        sample_prob = self.sample_output(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))


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
        paddle.disable_static()
        x_numpy = np.random.rand(4)
        x = paddle.to_tensor(x_numpy)
        out = paddle.multinomial(x, num_samples=100000, replacement=True)
        paddle.enable_static()

        sample_prob = sample_output_one_dimension(out.numpy(), 4)
        prob = x_numpy / x_numpy.sum(axis=-1, keepdims=True)
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))

    def test_dygraph2(self):
        # input probability is a matrix, and replacement is True
        paddle.disable_static()
        x_numpy = np.random.rand(3, 4)
        x = paddle.to_tensor(x_numpy)
        out = paddle.multinomial(x, num_samples=100000, replacement=True)

        sample_prob = sample_output_two_dimension(out.numpy(), [3, 4])
        prob = x_numpy / x_numpy.sum(axis=-1, keepdims=True)
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))
        paddle.enable_static()

    def test_dygraph3(self):
        # replacement is False. number of samples must be less than number of categories.
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
        paddle.disable_static()
        logits = -1 * paddle.ones([2800])
        # Categorical.sample API will call multinomial op with replacement=True
        cat = paddle.distribution.Categorical(logits.exp())
        cat.sample([1])
        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            x = fluid.data('x', shape=[4], dtype='float32')
            out = paddle.multinomial(x, num_samples=100000, replacement=True)

            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

        exe.run(startup_program)
        x_np = np.random.rand(4).astype('float32')
        out = exe.run(train_program, feed={'x': x_np}, fetch_list=[out])

        sample_prob = sample_output_one_dimension(out, 4)
        prob = x_np / x_np.sum(axis=-1, keepdims=True)
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))


class TestMultinomialAlias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.rand([4])
        paddle.multinomial(x, num_samples=10, replacement=True)
        paddle.tensor.multinomial(x, num_samples=10, replacement=True)
        paddle.tensor.random.multinomial(x, num_samples=10, replacement=True)


class TestMultinomialError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_num_sample(self):
        def test_num_sample_less_than_0():
            x = paddle.rand([4])
            paddle.multinomial(x, num_samples=-2)

        self.assertRaises(ValueError, test_num_sample_less_than_0)

    def test_replacement_False(self):
        def test_samples_larger_than_categories():
            x = paddle.rand([4])
            paddle.multinomial(x, num_samples=5, replacement=False)

        self.assertRaises(ValueError, test_samples_larger_than_categories)

    def test_input_probs_dim(self):
        def test_dim_larger_than_2():
            x = paddle.rand([2, 3, 3])
            paddle.multinomial(x)

        self.assertRaises(ValueError, test_dim_larger_than_2)

        def test_dim_less_than_1():
            x_np = np.random.random([])
            x = paddle.to_tensor(x_np)
            paddle.multinomial(x)

        self.assertRaises(ValueError, test_dim_less_than_1)


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        # Different GPU generatte different random value. Only test V100 here.
        if not "V100" in paddle.device.cuda.get_device_name():
            return

        if os.getenv("FLAGS_use_curand", None) in ('0', 'False', None):
            return

        print("Test Fixed Random number on V100 GPU------>")
        paddle.disable_static()
        paddle.set_device('gpu')
        paddle.seed(100)

        x = paddle.randint(0, 100, [1024, 10000]).astype('float32')
        y = paddle.multinomial(x, 1, replacement=False).numpy()
        self.assertEqual(np.sum(y), 5187793)
        self.assertEqual(np.mean(y), 5066.2041015625)
        expect = [9982, 1655, 4741, 1323, 9319, 3298, 6473, 7477, 2507, 2628]
        self.assertTrue(np.array_equal(y[100:110, :].flatten(), expect))

        y = paddle.multinomial(x, 5000, replacement=False).numpy()
        self.assertEqual(np.sum(y), 25603962316)
        self.assertEqual(np.mean(y), 5000.77388984375)
        expect = [7300, 6055, 8714, 5401, 7360, 161, 5035, 7002, 6788, 2916]
        self.assertTrue(np.array_equal(y[100, 1000:1010], expect))

        y = paddle.multinomial(x, 5000, replacement=False).numpy()
        self.assertEqual(np.sum(y), 25592855710)
        self.assertEqual(np.mean(y), 4998.604630859375)
        expect = [5700, 6567, 4399, 5688, 7472, 545, 6894, 526, 2124, 385]
        self.assertTrue(np.array_equal(y[300, 3000:3010], expect))

        y = paddle.multinomial(x, 20000, replacement=True).numpy()
        self.assertEqual(np.sum(y), 102371362581)
        self.assertEqual(np.mean(y), 4998.60168852539)
        self.assertEqual(np.std(y), 2886.316308500771)
        expect = [7630, 8235, 8445, 3275, 5580, 4591, 1331, 342, 1662, 7156]
        self.assertTrue(np.array_equal(y[100, 0:10], expect))

        y = paddle.multinomial(x, 20000, replacement=True).numpy()
        self.assertEqual(np.sum(y), 102400672117)
        self.assertEqual(np.mean(y), 5000.032818212891)
        self.assertEqual(np.std(y), 2886.913426124017)
        expect = [4159, 7849, 9305, 5759, 4422, 122, 345, 2897, 5200, 5911]
        self.assertTrue(np.array_equal(y[100, 0:10], expect))

        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
