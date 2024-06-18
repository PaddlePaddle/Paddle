#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0D Tensor indicates that the tensor's dimension is 0
# 0D Tensor's shape is always [], numel is 1
# which can be created by paddle.rand([])

import unittest

import numpy as np
from decorator_helper import prog_scope

import paddle
import paddle.nn.functional as F


class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.x = paddle.full([], 2.0)

    def test_Bernoulli(self):
        d = paddle.distribution.Bernoulli(probs=0.3)
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.entropy().shape, [])
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.cdf(self.x).shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])

        d_other = paddle.distribution.Bernoulli(probs=0.7)
        self.assertEqual(d.kl_divergence(d_other).shape, [])

    def test_Geometric(self):
        d = paddle.distribution.Geometric(0.5)
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.entropy().shape, [])
        self.assertEqual(d.stddev.shape, [])
        self.assertEqual(d.pmf(self.x).shape, [])
        self.assertEqual(d.log_pmf(self.x).shape, [])
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.cdf(self.x).shape, [])

        d_other = paddle.distribution.Geometric(probs=0.7)
        self.assertEqual(d.kl_divergence(d_other).shape, [])

    def test_Cauchy(self):
        d = paddle.distribution.Cauchy(loc=0.1, scale=1.2)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])
        self.assertEqual(d.cdf(self.x).shape, [])
        self.assertEqual(d.entropy().shape, [])

        d_other = paddle.distribution.Cauchy(
            loc=paddle.to_tensor(1.2), scale=paddle.to_tensor(2.3)
        )
        self.assertEqual(d.kl_divergence(d_other).shape, [])

    def test_Categorical(self):
        logits = paddle.rand([6])
        d = paddle.distribution.Categorical(logits)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.probs(paddle.full([], 2, dtype='int64')).shape, [])
        self.assertEqual(
            d.log_prob(paddle.full([], 2, dtype='int64')).shape, []
        )
        self.assertEqual(d.entropy().shape, [])

    def test_Normal(self):
        normal = paddle.distribution.Normal(0.0, 3.0)
        self.assertEqual(normal.sample([]).shape, [])
        self.assertEqual(normal.rsample([]).shape, [])
        self.assertEqual(normal.mean.shape, [])
        self.assertEqual(normal.variance.shape, [])
        self.assertEqual(normal.probs(self.x).shape, [])
        self.assertEqual(normal.log_prob(self.x).shape, [])
        self.assertEqual(normal.entropy().shape, [])

        normal = paddle.distribution.Normal(
            paddle.full([], 0.0), paddle.full([], 3.0)
        )
        self.assertEqual(normal.sample([]).shape, [])
        self.assertEqual(normal.rsample([]).shape, [])
        self.assertEqual(normal.mean.shape, [])
        self.assertEqual(normal.variance.shape, [])
        self.assertEqual(normal.probs(self.x).shape, [])
        self.assertEqual(normal.log_prob(self.x).shape, [])
        self.assertEqual(normal.entropy().shape, [])

    def test_Uniform(self):
        uniform = paddle.distribution.Uniform(0.0, 1.0)
        self.assertEqual(uniform.sample([]).shape, [])
        self.assertEqual(uniform.probs(self.x).shape, [])
        self.assertEqual(uniform.log_prob(self.x).shape, [])
        self.assertEqual(uniform.entropy().shape, [])

        uniform = paddle.distribution.Uniform(
            paddle.full([], 0.0), paddle.full([], 1.0)
        )
        self.assertEqual(uniform.sample([]).shape, [])
        self.assertEqual(uniform.probs(self.x).shape, [])
        self.assertEqual(uniform.log_prob(self.x).shape, [])
        self.assertEqual(uniform.entropy().shape, [])

    def test_Beta(self):
        beta = paddle.distribution.Beta(alpha=0.5, beta=0.5)
        self.assertEqual(beta.sample([]).shape, [])
        self.assertEqual(beta.mean.shape, [])
        self.assertEqual(beta.variance.shape, [])
        self.assertEqual(beta.prob(self.x).shape, [])
        self.assertEqual(beta.log_prob(self.x).shape, [])
        self.assertEqual(beta.entropy().shape, [])

    def test_kl_divergence(self):
        p = paddle.distribution.Beta(alpha=0.5, beta=0.5)
        q = paddle.distribution.Beta(alpha=0.2, beta=1.0)
        kl = paddle.distribution.kl_divergence(p, q)
        self.assertEqual(kl.shape, [])

    def test_TransformedDistribution(self):
        d = paddle.distribution.TransformedDistribution(
            paddle.distribution.Normal(0.0, 1.0),
            [
                paddle.distribution.AffineTransform(
                    paddle.full([], 1.0), paddle.full([], 2.0)
                )
            ],
        )
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])

    def test_Laplace(self):
        d = paddle.distribution.Laplace(0.0, 1.0)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.stddev.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])
        self.assertEqual(d.cdf(self.x).shape, [])
        self.assertEqual(d.icdf(self.x).shape, [])
        self.assertEqual(d.entropy().shape, [])

    def test_LogNormal(self):
        d = paddle.distribution.LogNormal(0.0, 1.0)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.entropy().shape, [])
        self.assertEqual(d.probs(self.x).shape, [])

    def test_Gumbel(self):
        d = paddle.distribution.Gumbel(0.0, 1.0)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.stddev.shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])
        self.assertEqual(d.cdf(self.x).shape, [])
        self.assertEqual(d.entropy().shape, [])

    def test_Multinomial(self):
        d = paddle.distribution.Multinomial(
            10, paddle.to_tensor([0.2, 0.3, 0.5])
        )
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])
        self.assertEqual(d.entropy().shape, [])


class TestLossAPI(unittest.TestCase):
    def test_sigmoid_focal_loss(self):
        logit = paddle.to_tensor(
            [[0.97, 0.91, 0.03], [0.55, 0.43, 0.71]],
            dtype='float32',
            stop_gradient=False,
        )
        logit.retain_grads()
        label = paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32'
        )
        fg_num_0 = paddle.full([], 2.0)
        fg_num_1 = paddle.full([1], 2.0)

        out0 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_0, reduction='sum'
        )
        out1 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_1, reduction='sum'
        )
        out0.retain_grads()

        np.testing.assert_array_equal(
            out0.numpy(),
            out1.numpy(),
        )

        out0.backward()
        self.assertEqual(out0.shape, [])
        self.assertEqual(out1.shape, [])
        self.assertEqual(out0.grad.shape, [])
        self.assertEqual(logit.grad.shape, [2, 3])

    def test_cross_entropy(self):
        input = paddle.rand([3, 5])
        input.stop_gradient = False
        label = paddle.randint(0, 5, shape=[3])

        loss = paddle.nn.functional.cross_entropy(input, label, reduction='sum')
        loss.backward()

        self.assertEqual(loss.shape, [])
        self.assertEqual(input.grad.shape, [3, 5])

    def test_l1_loss(self):
        input = paddle.rand([3, 5])
        input.stop_gradient = False
        label = paddle.rand([3, 5])

        loss = paddle.nn.functional.l1_loss(input, label, reduction='mean')
        loss.backward()

        self.assertEqual(loss.shape, [])
        self.assertEqual(input.grad.shape, [3, 5])

    def test_nll_loss(self):
        input = paddle.rand([5, 3])
        input.stop_gradient = False
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        log_out = log_softmax(input)
        label = paddle.randint(0, 3, [5], "int64")

        loss = paddle.nn.functional.nll_loss(log_out, label)
        loss.backward()

        self.assertEqual(loss.shape, [])
        self.assertEqual(input.grad.shape, [5, 3])

        input = paddle.rand([5, 3, 2, 4])
        input.stop_gradient = False
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        log_out = log_softmax(input)
        label = paddle.randint(0, 3, [5, 2, 4], "int64")

        loss = paddle.nn.functional.nll_loss(log_out, label)
        loss.backward()

        self.assertEqual(loss.shape, [])
        self.assertEqual(input.grad.shape, [5, 3, 2, 4])


class TestLossAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    @prog_scope()
    def test_sigmoid_focal_loss(self):
        logit = paddle.rand([2, 3])
        logit.stop_gradient = False

        label = paddle.randint(0, 1, [2, 3]).astype('float32')
        label.stop_gradient = False

        fg_num_0 = paddle.full([], 2.0)
        fg_num_1 = paddle.full([1], 2.0)

        out0 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_0, reduction='mean'
        )
        out1 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_1, reduction='mean'
        )
        [(_, out0_grad), (_, logit_grad)] = paddle.static.append_backward(
            out0.sum(), parameter_list=[out0, logit]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out0, out1, out0_grad, logit_grad])
        np.testing.assert_allclose(res[0], res[1])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, (2, 3))

    @prog_scope()
    def test_cross_entropy(self):
        input = paddle.rand([3, 5])
        input.stop_gradient = False
        label = paddle.randint(0, 5, shape=[3])
        label.stop_gradient = False

        loss = paddle.nn.functional.cross_entropy(
            input, label, reduction='mean'
        )
        [(_, input_grad)] = paddle.static.append_backward(
            loss, parameter_list=[input]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[loss, input_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 5))

    @prog_scope()
    def test_l1_loss(self):
        input = paddle.rand([3, 5])
        input.stop_gradient = False
        label = paddle.rand([3, 5])

        loss = paddle.nn.functional.l1_loss(input, label, reduction='sum')
        [(_, input_grad)] = paddle.static.append_backward(
            loss, parameter_list=[input]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[loss, input_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 5))

    @prog_scope()
    def test_nll_loss(self):
        input = paddle.rand([5, 3])
        input.stop_gradient = False
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        log_out = log_softmax(input)

        label = paddle.randint(0, 3, shape=[5])
        label.stop_gradient = False

        loss = paddle.nn.functional.nll_loss(log_out, label)
        [(_, input_grad)] = paddle.static.append_backward(
            loss, parameter_list=[input]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[loss, input_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (5, 3))

        input = paddle.rand([5, 3, 2, 4])
        input.stop_gradient = False
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        log_out = log_softmax(input)

        label = paddle.randint(0, 3, shape=[5, 2, 4])
        label.stop_gradient = False

        loss = paddle.nn.functional.nll_loss(log_out, label)
        [(_, input_grad)] = paddle.static.append_backward(
            loss, parameter_list=[input]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[loss, input_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (5, 3, 2, 4))


if __name__ == "__main__":
    unittest.main()
