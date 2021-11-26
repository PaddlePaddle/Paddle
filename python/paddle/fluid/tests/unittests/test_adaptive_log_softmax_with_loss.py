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
# limitations under the License.

import unittest
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F


class TestNNAdaptiveLogSoftmaxWithLossAPI(unittest.TestCase):
    def test_error(self):
        # args validation
        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 15, 15], div_value=2.)

        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 15, 10], div_value=2.)

        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 25], div_value=2.)

        with self.assertRaisesRegex(ValueError,
                                    "cutoffs should be a sequence of unique,"):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 20], div_value=2.)
        # not raise
        _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 19], div_value=2.)

    def test_shape(self):
        # input shapes
        with self.assertRaisesRegex(
                RuntimeError, r"Input and target should have the same size"):
            asfm = nn.AdaptiveLogSoftmaxWithLoss(
                16, 20, [5, 10, 15], div_value=2.)
            x = paddle.randn((2, 16))
            y = paddle.to_tensor([0, 5, 10])
            asfm(x, y)

        # out-of-bound targets
        with self.assertRaisesRegex(RuntimeError,
                                    r"Target values should be in"):
            asfm = nn.AdaptiveLogSoftmaxWithLoss(
                16, 20, [5, 10, 15], div_value=2.)
            x = paddle.randn((128, 16))
            y = paddle.randint(low=21, high=200, shape=[128])
            asfm(x, y)

    def test_cluster(self):
        # cluster sizes
        asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
        x = paddle.randn((128, 16))
        y = paddle.randint(low=0, high=20, shape=[128])

        self.assertEqual(
            asfm.head.weight.shape,
            [16, 5 + 3])  # 5 targets in head, 3 clusters, dimensionality 16
        self.assertEqual(asfm.tail[0][1].weight.shape,
                         [8, 5])  # 5 targets in this cluster, dimensionality 8
        self.assertEqual(asfm.tail[1][1].weight.shape, [4, 5])
        self.assertEqual(asfm.tail[2][1].weight.shape, [2, 5])

        self.assertEqual(asfm(x, y).output.shape, [128])

    def test_log_probs(self):
        # log_probs actually returns log_proba
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 4, [2], div_value=2.)
        x = paddle.randn((4, 8))
        logprob_out = asfm.log_prob(x)
        np.testing.assert_array_almost_equal(
            paddle.exp(logprob_out).sum(1), paddle.ones([4]))

        # forward returns the same thing as log_probs
        for v in [0, 1, 2, 3]:
            y = paddle.full((4, ), v, dtype='int64')
            out, loss = asfm(x, y)
            np.testing.assert_array_almost_equal(
                out,
                logprob_out.gather(y.unsqueeze(1), 1).slice([1], [0],
                                                            [1]).squeeze())
            np.testing.assert_array_almost_equal(loss,
                                                 F.nll_loss(logprob_out, y))

    def test_correct(self):
        # predict
        x = paddle.abs(paddle.randn((64, 8)))

        # argmax in shortlist
        asfm = nn.AdaptiveLogSoftmaxWithLoss(
            8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.detach().abs()
        asfm.head.bias.detach().abs()
        asfm.head.weight.detach()[asfm.shortlist_size:, :] *= 0.

        out = asfm.predict(x)
        np.testing.assert_array_almost_equal(
            out, asfm.log_prob(x).argmax(axis=1))

        # argmax outside of shortlist
        asfm = nn.AdaptiveLogSoftmaxWithLoss(
            8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.detach().abs()
        asfm.head.bias.detach().abs()
        asfm.head.weight.detach()[:asfm.shortlist_size, :] *= 0.

        out = asfm.predict(x)
        np.testing.assert_array_almost_equal(
            out, asfm.log_prob(x).argmax(axis=1))

        # half of the argmax in shortlist, half in clusters
        asfm = nn.AdaptiveLogSoftmaxWithLoss(
            8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.detach().abs()
        asfm.head.bias.detach().abs()

        x[:32, :asfm.shortlist_size] *= 0.
        x[32:, asfm.shortlist_size:] *= 0.

        asfm.head.weight.detach()[:asfm.shortlist_size,
                                  asfm.shortlist_size:] *= 0.
        asfm.head.weight.detach()[asfm.shortlist_size:, :
                                  asfm.shortlist_size] *= 0.

        out = asfm.predict(x)
        np.testing.assert_array_almost_equal(
            out, asfm.log_prob(x).argmax(axis=1))

    def linear_ref(self, x, weight, bias):
        x = x.numpy() if isinstance(x, paddle.Tensor) else x
        weight = weight.numpy() if isinstance(weight, paddle.Tensor) else weight
        bias = bias.numpy() if isinstance(bias, paddle.Tensor) else bias
        return np.matmul(x, weight) + bias

    def test_approx_to_lsfm(self):
        """
        Test error between AdaptiveLogSoftmaxWithLoss and log_softmax less then 3%, according to http://arxiv.org/abs/1609.04309
        """
        x = paddle.abs(paddle.randn((64, 8)))
        asfm = nn.AdaptiveLogSoftmaxWithLoss(
            8, 10, [6, 8], div_value=2., head_bias=True)

        head_weight = asfm.head.weight.detach()
        head_bias = asfm.head.bias.detach()
        ref_head_output = self.linear_ref(x, head_weight, head_bias)

        out = asfm.log_prob(x).argmax(axis=1)
        ref = F.log_softmax(
            paddle.to_tensor(
                ref_head_output, dtype='float32')).argmax(axis=1)
        self.assertTrue(out[out != ref].shape[0] < out.shape[0] / 30)


if __name__ == "__main__":
    unittest.main()
