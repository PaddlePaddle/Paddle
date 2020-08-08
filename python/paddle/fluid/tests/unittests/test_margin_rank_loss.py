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

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.imperative as imperative
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


def calc_margin_rank_loss(x, y, label, margin=0.0, reduction='none'):
    result = (-1 * label) * (x - y) + margin
    result = np.maximum(result, 0)
    if reduction == 'none':
        return result
    elif reduction == 'sum':
        return np.sum(result)
    elif reduction == 'mean':
        return np.mean(result)


def create_test_case(margin, reduction):
    class MarginRankingLossCls(unittest.TestCase):
        def setUp(self):
            self.x_data = np.random.rand(10, 10).astype("float64")
            self.y_data = np.random.rand(10, 10).astype("float64")
            self.label_data = np.random.choice(
                [-1, 1], size=[10, 10]).astype("float64")
            self.places = []
            self.places.append(fluid.CPUPlace())
            if core.is_compiled_with_cuda():
                self.places.append(paddle.CUDAPlace(0))

        def run_staic_api(self, place):
            expected = calc_margin_rank_loss(
                self.x_data,
                self.y_data,
                self.label_data,
                margin=margin,
                reduction=reduction)
            with program_guard(Program(), Program()):
                x = paddle.nn.data(name="x", shape=[10, 10], dtype="float64")
                y = paddle.nn.data(name="y", shape=[10, 10], dtype="float64")
                label = paddle.nn.data(
                    name="label", shape=[10, 10], dtype="float64")
                margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                    margin=margin, reduction=reduction)
                result = margin_rank_loss(x, y, label)
                exe = fluid.Executor(place)
                result_numpy, = exe.run(feed={
                    "x": self.x_data,
                    "y": self.y_data,
                    "label": self.label_data
                },
                                        fetch_list=[result])
                self.assertTrue(np.allclose(result_numpy, expected))
                self.assertTrue('loss' in result.name)

        def run_imperative_api(self):
            x = imperative.to_variable(self.x_data)
            y = imperative.to_variable(self.y_data)
            label = imperative.to_variable(self.label_data)
            margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                margin=margin, reduction=reduction)
            result = margin_rank_loss(x, y, label)
            expected = calc_margin_rank_loss(
                self.x_data,
                self.y_data,
                self.label_data,
                margin=margin,
                reduction=reduction)
            self.assertTrue(np.allclose(result.numpy(), expected))

        def run_imperative_broadcast_api(self):
            label_data = np.random.choice([-1, 1], size=[10]).astype("float64")
            x = imperative.to_variable(self.x_data)
            y = imperative.to_variable(self.y_data)
            label = imperative.to_variable(label_data)
            margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                margin=margin, reduction=reduction)
            result = margin_rank_loss(x, y, label)
            expected = calc_margin_rank_loss(
                self.x_data,
                self.y_data,
                label_data,
                margin=margin,
                reduction=reduction)
            self.assertTrue(np.allclose(result.numpy(), expected))

        def test_case(self):
            for place in self.places:
                with paddle.imperative.guard(place):
                    self.run_imperative_api()
                    self.run_imperative_broadcast_api()
                self.run_staic_api(place)

    cls_name = "TestMarginRankLossCase_{}_{}".format(margin, reduction)
    MarginRankingLossCls.__name__ = cls_name
    globals()[cls_name] = MarginRankingLossCls


for margin in [0.0, 0.2]:
    for reduction in ['none', 'mean', 'sum']:
        create_test_case(margin, reduction)


# test case the raise message
class MarginRakingLossError(unittest.TestCase):
    def test_errors(self):
        def test_margin_value_error():
            margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                margin=0.1, reduction="reduce_mean")

        self.assertRaises(ValueError, test_margin_value_error)


if __name__ == "__main__":
    unittest.main()
