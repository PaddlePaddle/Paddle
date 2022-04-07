#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import _test_eager_guard
paddle.enable_static()


class TestGumbelSoftmaxOp(OpTest):
    def init_attrs(self):
        self.shape = [20, 10]
        self.attrs = {"hard": True, "axis": -1}
        self.count_expected = 20
        self.dtype = "float64"

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        out_np.shape = self.shape
        self.assertTrue(list(out_np.shape) == self.shape)
        self.assertEqual(out_np.sum(), self.count_expected)

    def setUp(self):
        self.op_type = "gumbel_softmax"
        self.init_attrs()
        np.random.seed(0)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.zeros(self.shape).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestGumbelSoftmaxOp2(TestGumbelSoftmaxOp):
    def init_attrs(self):
        self.shape = [20, 10]
        self.attrs = {"hard": True, "axis": 0}
        self.count_expected = 10
        self.dtype = "float64"


class TestGumbelSoftmaxOp3(TestGumbelSoftmaxOp):
    def init_attrs(self):
        self.shape = [100]
        self.attrs = {"hard": True, "axis": -1}
        self.count_expected = 1
        self.dtype = "float64"


class TestGumbelSoftmaxOp4(TestGumbelSoftmaxOp):
    def init_attrs(self):
        self.shape = [20, 10, 5]
        self.attrs = {"hard": True, "axis": -1}
        self.count_expected = 200
        self.dtype = "float64"


class TestGumbelSoftmaxOp5(TestGumbelSoftmaxOp):
    def init_attrs(self):
        self.shape = [20, 10, 5]
        self.attrs = {"hard": True, "axis": 1}
        self.count_expected = 100
        self.dtype = "float64"


class TestGumbelSoftmaxOpSampleDistribution(OpTest):
    def softmax(self, x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
        softmax = x_exp / x_exp_row_sum
        return softmax

    def init_attrs(self):
        self.shape = [100, 3]
        self.attrs = {"hard": True, "axis": -1}
        self.counts = np.zeros(self.shape).astype(self.dtype)
        self._cpu_only = True

    def accumulate_output(self, outs):
        out_np = np.array(outs)
        out_np = out_np.reshape(self.shape)
        self.counts = np.sum(out_np, axis=0)

    def setUp(self):
        self.op_type = "gumbel_softmax"
        self.init_attrs()
        single_x = np.array([0.2, 0.3, 0.5])
        batch_x = np.ones(self.shape) * single_x
        out = np.zeros(self.shape).astype(self.dtype)
        self.probs = self.softmax(single_x)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(batch_x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_customized(self.accumulate_output)
        # Experiment should result in batch num .
        self.assertEqual(self.counts.sum(), self.shape[0])

        # Treat the probability from softmax as 
        # the probability of binomial distribution.
        # Samples from gumbel softmax meet this binomial distribution.
        # Construct statistics z for samples and 
        # z is approximately N(0,1) for unbiased count
        expected = self.probs * self.shape[0]
        z = (self.counts - expected) / np.sqrt((expected * (1 - self.probs)))
        # A (lazy) approximate 99% two-sided test:
        # occurs with prob alpha~>=0.01 if unbiased
        self.assertLess(np.max(np.abs(z)).item(), 2.58)

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestGumbelSoftmaxOpGrad(unittest.TestCase):
    def init_attrs(self):
        self.shape = [20, 10]
        self.dtype = "float64"

    def setUp(self):
        self.init_attrs()
        np.random.seed(0)
        self.x_np = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)

    def test_dygraph_check(self):
        paddle.disable_static()
        x_hard = paddle.to_tensor(self.x_np, stop_gradient=False)
        x_soft = paddle.to_tensor(self.x_np, stop_gradient=False)
        out_hard = paddle.nn.functional.gumbel_softmax(x_hard, hard=True)
        out_soft = paddle.nn.functional.gumbel_softmax(x_soft, hard=False)

        out_hard.sum().backward()
        out_soft.sum().backward()

        self.assertEqual(
            np.allclose(x_hard.grad.numpy(), x_soft.grad.numpy()), True)
        paddle.enable_static()


class TestGumbelSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1., 1., self.x_shape).astype(np.float32)
        self.count_expected = 24
        self.place = paddle.CUDAPlace(0) \
            if paddle.fluid.core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_check_api(self):
        # test static api
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data(name='x', shape=self.x_shape)
            y = paddle.nn.functional.gumbel_softmax(x, hard=True)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
            out_np = np.array(out[0])
        self.assertEqual(out_np.sum(), self.count_expected)

        # test dygrapg api
        with paddle.fluid.dygraph.base.guard():
            x = paddle.to_tensor(self.x)
            y = paddle.nn.functional.gumbel_softmax(x, hard=True)
            out_np = np.array(y)
            self.assertEqual(out_np.sum(), self.count_expected)

            with _test_eager_guard():
                x = paddle.to_tensor(self.x)
                y = paddle.nn.functional.gumbel_softmax(x, hard=True)
                out_np = np.array(y)
                self.assertEqual(out_np.sum(), self.count_expected)


class TestGumbelSoftmaxOpError(unittest.TestCase):
    def test_errors(self):
        paddle.disable_static()

        def test_Variable():
            x1 = fluid.create_lod_tensor(
                np.zeros((100, 784)), [[10, 10, 10, 70]], fluid.CPUPlace())
            paddle.nn.functional.gumbel_softmax(x1)

        self.assertRaises(ValueError, test_Variable)

        def test_Variable2():
            x1 = np.zeros((100, 784))
            paddle.nn.functional.gumbel_softmax(x1)

        self.assertRaises(ValueError, test_Variable2)

        def test_argument1():
            x = paddle.to_tensor([0.2, 0.3, 0.4])
            paddle.nn.functional.gumbel_softmax(x, temperature=-1)

        self.assertRaises(ValueError, test_argument1)

        def test_argument2():
            x = paddle.to_tensor([0.2, 0.3, 0.4])
            paddle.nn.functional.gumbel_softmax(x, axis=1.1)

        self.assertRaises(ValueError, test_argument2)

        paddle.enable_static()

        def test_dtype():
            with paddle.static.program_guard(paddle.static.Program()):
                x_int32 = paddle.fluid.data(
                    name='x_int32', shape=[2, 3], dtype='int32')
                paddle.nn.functional.gumbel_softmax(x_int32)

        self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    unittest.main()
