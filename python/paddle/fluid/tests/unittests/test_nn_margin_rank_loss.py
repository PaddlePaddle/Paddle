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

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.static import Program, program_guard


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
            self.label_data = np.random.choice([-1, 1],
                                               size=[10, 10]).astype("float64")
            self.places = []
            self.places.append(fluid.CPUPlace())
            if core.is_compiled_with_cuda():
                self.places.append(paddle.CUDAPlace(0))

        def run_static_functional_api(self, place):
            paddle.enable_static()
            expected = calc_margin_rank_loss(self.x_data,
                                             self.y_data,
                                             self.label_data,
                                             margin=margin,
                                             reduction=reduction)
            with program_guard(Program(), Program()):
                x = paddle.static.data(name="x",
                                       shape=[10, 10],
                                       dtype="float64")
                y = paddle.static.data(name="y",
                                       shape=[10, 10],
                                       dtype="float64")
                label = paddle.static.data(name="label",
                                           shape=[10, 10],
                                           dtype="float64")
                result = paddle.nn.functional.margin_ranking_loss(
                    x, y, label, margin, reduction)
                exe = paddle.static.Executor(place)
                result_numpy, = exe.run(feed={
                    "x": self.x_data,
                    "y": self.y_data,
                    "label": self.label_data
                },
                                        fetch_list=[result])
                np.testing.assert_allclose(result_numpy, expected, rtol=1e-05)

        def run_static_api(self, place):
            paddle.enable_static()
            expected = calc_margin_rank_loss(self.x_data,
                                             self.y_data,
                                             self.label_data,
                                             margin=margin,
                                             reduction=reduction)
            with program_guard(Program(), Program()):
                x = paddle.static.data(name="x",
                                       shape=[10, 10],
                                       dtype="float64")
                y = paddle.static.data(name="y",
                                       shape=[10, 10],
                                       dtype="float64")
                label = paddle.static.data(name="label",
                                           shape=[10, 10],
                                           dtype="float64")
                margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                    margin=margin, reduction=reduction)
                result = margin_rank_loss(x, y, label)
                exe = paddle.static.Executor(place)
                result_numpy, = exe.run(feed={
                    "x": self.x_data,
                    "y": self.y_data,
                    "label": self.label_data
                },
                                        fetch_list=[result])
                np.testing.assert_allclose(result_numpy, expected, rtol=1e-05)
                self.assertTrue('loss' in result.name)

        def run_dynamic_functional_api(self, place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x_data)
            y = paddle.to_tensor(self.y_data)
            label = paddle.to_tensor(self.label_data)

            result = paddle.nn.functional.margin_ranking_loss(
                x, y, label, margin, reduction)
            expected = calc_margin_rank_loss(self.x_data,
                                             self.y_data,
                                             self.label_data,
                                             margin=margin,
                                             reduction=reduction)
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)

        def run_dynamic_api(self, place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x_data)
            y = paddle.to_tensor(self.y_data)
            label = paddle.to_tensor(self.label_data)
            margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                margin=margin, reduction=reduction)
            result = margin_rank_loss(x, y, label)
            expected = calc_margin_rank_loss(self.x_data,
                                             self.y_data,
                                             self.label_data,
                                             margin=margin,
                                             reduction=reduction)
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)

        def run_dynamic_broadcast_api(self, place):
            paddle.disable_static(place)
            label_data = np.random.choice([-1, 1], size=[10]).astype("float64")
            x = paddle.to_tensor(self.x_data)
            y = paddle.to_tensor(self.y_data)
            label = paddle.to_tensor(label_data)
            margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                margin=margin, reduction=reduction)
            result = margin_rank_loss(x, y, label)
            expected = calc_margin_rank_loss(self.x_data,
                                             self.y_data,
                                             label_data,
                                             margin=margin,
                                             reduction=reduction)
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)

        def test_case(self):
            for place in self.places:
                self.run_static_api(place)
                self.run_static_functional_api(place)
                self.run_dynamic_api(place)
                self.run_dynamic_functional_api(place)
                self.run_dynamic_broadcast_api(place)

    cls_name = "TestMarginRankLossCase_{}_{}".format(margin, reduction)
    MarginRankingLossCls.__name__ = cls_name
    globals()[cls_name] = MarginRankingLossCls


for margin in [0.0, 0.2]:
    for reduction in ['none', 'mean', 'sum']:
        create_test_case(margin, reduction)


# test case the raise message
class MarginRakingLossError(unittest.TestCase):
    paddle.enable_static()

    def test_errors(self):

        def test_margin_value_error():
            margin_rank_loss = paddle.nn.loss.MarginRankingLoss(
                margin=0.1, reduction="reduce_mean")

        self.assertRaises(ValueError, test_margin_value_error)

        def test_functional_margin_value_error():
            x = paddle.static.data(name="x", shape=[10, 10], dtype="float64")
            y = paddle.static.data(name="y", shape=[10, 10], dtype="float64")
            label = paddle.static.data(name="label",
                                       shape=[10, 10],
                                       dtype="float64")
            result = paddle.nn.functional.margin_ranking_loss(
                x, y, label, margin=0.1, reduction="reduction_mean")

        self.assertRaises(ValueError, test_functional_margin_value_error)


if __name__ == "__main__":
    unittest.main()
