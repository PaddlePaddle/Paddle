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

from __future__ import division
from __future__ import print_function

import unittest
import os
import six
import numpy as np
import shutil
import copy

import paddle
from paddle import fluid

from paddle.incubate.hapi.model import Model, Input
from paddle.incubate.hapi.loss import CrossEntropy, SoftmaxWithCrossEntropy


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def randomize_probability(batch_size, class_num, dtype='float32'):
    prob = np.random.uniform(
        0.1, 1.0, size=(batch_size, class_num)).astype(dtype)
    prob_sum = prob.sum(axis=1)
    for i in six.moves.xrange(len(prob)):
        prob[i] /= prob_sum[i]
    return prob


def numpy_ce(x, label):
    return np.asmatrix(
        [[-np.log(x[i][label[i][0]])] for i in range(x.shape[0])],
        dtype="float32").mean()


class TestLoss(unittest.TestCase):
    def test_cross_entropy(self):
        class_num = 100
        batch_size = 128
        inputs = [randomize_probability(128, class_num) for _ in range(2)]

        labels = [
            np.random.randint(
                0, class_num, (batch_size, 1), dtype="int64") for _ in range(2)
        ]

        gt_out = [numpy_ce(inputs[i], labels[i]) for i in range(2)]

        fluid.enable_dygraph()
        cross_entropy = CrossEntropy()
        out = cross_entropy(
            [fluid.dygraph.to_variable(x) for x in inputs],
            [fluid.dygraph.to_variable(label) for label in labels])
        out = [o.numpy() for o in out]

        for o, g in zip(out, gt_out):
            np.testing.assert_allclose(o, g, atol=1e-5)

    def test_soft_cross_entronpy(self):
        class_num = 100
        batch_size = 128

        inputs = [randomize_probability(128, class_num) for _ in range(2)]

        labels = [
            np.random.randint(
                0, class_num, (batch_size, 1), dtype="int64") for _ in range(2)
        ]

        fluid.enable_dygraph()
        softmax_cross_entropy = SoftmaxWithCrossEntropy()

        softmax_cross_entropy(
            [fluid.dygraph.to_variable(x) for x in inputs],
            [fluid.dygraph.to_variable(label) for label in labels])

        softmax_cross_entropy = SoftmaxWithCrossEntropy(average=False)

        inputs = [randomize_probability(128, class_num)]

        labels = [
            np.random.randint(
                0, class_num, (batch_size, 1), dtype="int64")
        ]

        softmax_cross_entropy([fluid.dygraph.to_variable(x) for x in inputs],
                              fluid.dygraph.to_variable(labels[0]))


if __name__ == '__main__':
    unittest.main()
