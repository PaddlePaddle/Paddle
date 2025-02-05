#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
import paddle.optimizer as optim
from paddle import nn
from paddle.base import Program
from paddle.nn import functional as F


class SimpleModel(nn.Layer):
    def __init__(
        self,
        in_features,
        n_classes,
        cutoffs,
        div_value=4.0,
        head_bias=False,
    ):
        super().__init__()
        self.fc = paddle.nn.Linear(in_features, in_features)
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features,
            n_classes,
            cutoffs,
            div_value=div_value,
            head_bias=head_bias,
        )

    def forward(self, input, label=None):
        x = self.fc(input)
        if label is not None:
            return self.adaptive_softmax(x, label)
        else:
            return self.adaptive_softmax.log_prob(x)

    def predict(self, input):
        logprob = self.adaptive_softmax.predict(self.fc(input))
        return logprob


class TestNNAdaptiveLogSoftmaxWithLossAPI(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.place.append('cpu')
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')
        self.log_np = np.random.randn(4, 8).astype('float32')
        self.predict_np = np.abs(np.random.randn(64, 8).astype('float32'))

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x = paddle.randn((4, 8))
            self._test_log_probs_dygraph(x)
            x = paddle.abs(paddle.randn((4, 8)))
            self._test_correct_dygraph(x)

    def _test_log_probs_dygraph(self, x):
        model = SimpleModel(8, 4, [2], div_value=2.0)
        logprob_out = model(x)
        np.testing.assert_array_almost_equal(
            paddle.exp(logprob_out).sum(1), paddle.ones([4])
        )

        for v in [0, 1, 2, 3]:
            y = paddle.full((4,), v, dtype='int64')
            out, loss = model(x, y)
            np.testing.assert_array_almost_equal(
                out,
                logprob_out.gather(y.unsqueeze(1), 1)
                .slice([1], [0], [1])
                .squeeze(),
            )
            np.testing.assert_array_almost_equal(
                loss, F.nll_loss(logprob_out, y)
            )

    def _test_correct_dygraph(self, x):
        model = SimpleModel(8, 10, [4, 8], div_value=2.0, head_bias=True)
        model.adaptive_softmax.head_weight.detach().abs()
        model.adaptive_softmax.head_bias.detach().abs()
        model.adaptive_softmax.head_weight.detach()[
            model.adaptive_softmax.shortlist_size :, :
        ] *= 0.0

        out = model.predict(x)
        np.testing.assert_array_almost_equal(out, model(x).argmax(axis=1))

        model = SimpleModel(8, 10, [4, 8], div_value=2.0, head_bias=True)
        model.adaptive_softmax.head_weight.detach().abs()
        model.adaptive_softmax.head_bias.detach().abs()
        model.adaptive_softmax.head_weight.detach()[
            : model.adaptive_softmax.shortlist_size, :
        ] *= 0.0

        out = model.predict(x)
        np.testing.assert_array_almost_equal(out, model(x).argmax(axis=1))

        model = SimpleModel(8, 10, [4, 8], div_value=2.0, head_bias=True)
        model.adaptive_softmax.head_weight.detach().abs()
        model.adaptive_softmax.head_bias.detach().abs()

        x[:32, : model.adaptive_softmax.shortlist_size] *= 0.0
        x[32:, model.adaptive_softmax.shortlist_size :] *= 0.0
        model.adaptive_softmax.head_weight.detach()[
            : model.adaptive_softmax.shortlist_size,
            model.adaptive_softmax.shortlist_size :,
        ] *= 0.0
        model.adaptive_softmax.head_weight.detach()[
            model.adaptive_softmax.shortlist_size :,
            : model.adaptive_softmax.shortlist_size,
        ] *= 0.0

        out = model.predict(x)
        np.testing.assert_array_almost_equal(out, model(x).argmax(axis=1))

    def _test_log_probs_static(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(Program()):
            model = SimpleModel(8, 4, [2], div_value=2.0)
            x = paddle.static.data(
                name="log_input", shape=[4, 8], dtype='float32'
            )
            out = model(x)
            exe = paddle.static.Executor(place=place)
            feed_list = {"log_input": self.log_np}
            logprob_out = exe.run(
                paddle.static.default_main_program(),
                feed=feed_list,
                fetch_list=[out],
            )

            np.testing.assert_array_almost_equal(
                paddle.exp(logprob_out).sum(1), paddle.ones([4])
            )

            for v in [0, 1, 2, 3]:
                y = paddle.full((4,), v, dtype='int64')
                out, loss = model(x, y)
                f_out, f_loss = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_list,
                    fetch_list=[out, loss],
                )
                np.testing.assert_array_almost_equal(
                    f_out,
                    logprob_out.gather(y.unsqueeze(1), 1)
                    .slice([1], [0], [1])
                    .squeeze(),
                )
                np.testing.assert_array_almost_equal(
                    f_loss, F.nll_loss(logprob_out, y)
                )

    def _test_correct_static(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(Program()):
            model = SimpleModel(8, 10, [4, 8], div_value=2.0, head_bias=True)
            exe = paddle.static.Executor(place=place)
            feed_list = {"predict_input": self.predict_np}
            x = paddle.static.data(
                name="predict_input", shape=[64, 8], dtype='float32'
            )
            model.adaptive_softmax.head_weight.detach().abs()
            model.adaptive_softmax.head_bias.detach().abs()
            paddle.static.setitem(
                model.adaptive_softmax.head_weight.detach(),
                (
                    slice(model.adaptive_softmax.shortlist_size, None, None),
                    slice(None, None, None),
                ),
                0.0,
            )
            out = model.predict(x)
            predict_out1 = exe.run(
                paddle.static.default_main_program(),
                feed=feed_list,
                fetch_list=[out],
            )
            np.testing.assert_array_almost_equal(
                predict_out1, model(x).argmax(axis=1)
            )

            model = SimpleModel(8, 10, [4, 8], div_value=2.0, head_bias=True)
            model.adaptive_softmax.head_weight.detach().abs()
            model.adaptive_softmax.head_bias.detach().abs()
            paddle.static.setitem(
                model.adaptive_softmax.head_weight.detach(),
                (
                    slice(None, model.adaptive_softmax.shortlist_size, None),
                    slice(None, None, None),
                ),
                0.0,
            )
            out = model.predict(x)
            predict_out2 = exe.run(
                paddle.static.default_main_program(),
                feed=feed_list,
                fetch_list=[out],
            )
            np.testing.assert_array_almost_equal(
                predict_out2, model(x).argmax(axis=1)
            )

            model = SimpleModel(8, 10, [4, 8], div_value=2.0, head_bias=True)
            model.adaptive_softmax.head_weight.detach().abs()
            model.adaptive_softmax.head_bias.detach().abs()
            paddle.static.setitem(
                x,
                (
                    slice(None, 32, None),
                    slice(None, model.adaptive_softmax.shortlist_size, None),
                ),
                0.0,
            )
            paddle.static.setitem(
                x,
                (
                    slice(32, None, None),
                    slice(model.adaptive_softmax.shortlist_size, None, None),
                ),
                0.0,
            )
            paddle.static.setitem(
                model.adaptive_softmax.head_weight.detach(),
                (
                    slice(
                        None, model.adaptive_softmaxasfm.shortlist_size, None
                    ),
                    slice(model.adaptive_softmax.shortlist_size, None, None),
                ),
                0.0,
            )
            paddle.static.setitem(
                model.adaptive_softmax.head_weight.detach(),
                (
                    slice(model.adaptive_softmax.shortlist_size, None, None),
                    slice(None, model.adaptive_softmax.shortlist_size, None),
                ),
                0.0,
            )
            out = model.predict(x)
            predict_out3 = exe.run(
                paddle.static.default_main_program(),
                feed=feed_list,
                fetch_list=[out],
            )
            np.testing.assert_array_almost_equal(
                predict_out3, model(x).argmax(axis=1)
            )

    def test_shape(self):
        with self.assertRaises(ValueError):
            model = SimpleModel(16, 20, [5, 10, 15], div_value=2.0)
            x = paddle.randn((2, 16))
            y = paddle.to_tensor([0, 5, 10])
            model(x, y)

    def test_forward(self):
        n_classes = 4
        in_features = 8
        cutoffs = [2]

        x = paddle.to_tensor(
            [
                [
                    0.99785769,
                    -1.14492130,
                    0.62956816,
                    0.77550924,
                    -1.97198308,
                    0.50906199,
                    0.76702958,
                    1.31143034,
                ],
                [
                    0.17371807,
                    2.68322444,
                    1.90870595,
                    0.58601201,
                    -0.78898108,
                    0.42098731,
                    -0.74253917,
                    -0.37492049,
                ],
                [
                    -0.77694625,
                    -0.11529812,
                    0.38232428,
                    0.70575434,
                    0.73429769,
                    0.81399834,
                    0.14212975,
                    0.12567955,
                ],
                [
                    0.44165909,
                    0.23613696,
                    0.81143701,
                    0.60473150,
                    0.77017546,
                    0.27865678,
                    -0.03236491,
                    0.31634274,
                ],
                [
                    0.15336825,
                    -0.66177142,
                    -0.01784009,
                    0.08901446,
                    0.85228783,
                    1.49427640,
                    -1.66938102,
                    0.86154014,
                ],
                [
                    -0.60814697,
                    1.26191938,
                    -0.21735200,
                    -0.88890392,
                    0.49093658,
                    -1.28960681,
                    1.06943762,
                    0.15803306,
                ],
                [
                    -0.12136814,
                    -0.16133699,
                    0.15643604,
                    0.79464215,
                    -1.02201688,
                    0.26957786,
                    -0.31038952,
                    0.93334937,
                ],
                [
                    0.66997373,
                    0.95807010,
                    -0.66944563,
                    -0.89887059,
                    1.00404060,
                    0.69594669,
                    -0.82105070,
                    1.15200853,
                ],
            ],
            dtype='float32',
        )
        labels = paddle.to_tensor([3, 3, 3, 2, 3, 0, 0, 0], dtype='int64')
        model = SimpleModel(in_features, n_classes, cutoffs, div_value=2.0)

        optimizer = optim.Adam(
            parameters=model.parameters(), learning_rate=0.001
        )
        for _ in range(2):
            _, loss = model(x, labels)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        tail_weights_before_training = [
            proj[0].numpy().copy()
            for proj in model.adaptive_softmax.tail_weights
        ]

        with paddle.no_grad():
            output, loss = model(x, labels)

        tail_weights_after_training = [
            proj[0].numpy() for proj in model.adaptive_softmax.tail_weights
        ]

        for before, after in zip(
            tail_weights_before_training, tail_weights_after_training
        ):
            assert not np.any(before != after)

    def test_cluster(self):
        model = SimpleModel(16, 20, [5, 10, 15], div_value=2.0)
        x = paddle.randn((128, 16))
        y = paddle.randint(low=0, high=20, shape=[128])
        output, _ = model(x, y)
        self.assertEqual(model.adaptive_softmax.head_weight.shape, [16, 5 + 3])
        self.assertEqual(
            model.adaptive_softmax.tail_weights[0][1].shape, [8, 5]
        )
        self.assertEqual(
            model.adaptive_softmax.tail_weights[1][1].shape, [4, 5]
        )
        self.assertEqual(
            model.adaptive_softmax.tail_weights[2][1].shape, [2, 5]
        )

        self.assertEqual(output.shape, [128])

    def test_error(self):
        with self.assertRaises(ValueError):
            _ = SimpleModel(16, 20, [5, 15, 15], div_value=2.0)

        with self.assertRaises(ValueError):
            _ = SimpleModel(16, 20, [5, 15, 10], div_value=2.0)

        with self.assertRaises(ValueError):
            _ = SimpleModel(16, 20, [5, 10, 25], div_value=2.0)

        with self.assertRaisesRegex(
            ValueError, "cutoffs should be a sequence of unique,"
        ):
            _ = SimpleModel(16, 20, [5, 10, 20], div_value=2.0)

    def test_dim_error(self):
        with self.assertRaises(ValueError):
            model = SimpleModel(16, 20, [5, 10, 15], div_value=2.0)
            x = paddle.randn((129, 16))
            y = paddle.randint(low=0, high=20, shape=[128])
            _ = model(x, y)

        with self.assertRaises(ValueError):
            model = SimpleModel(16, 20, [5, 10, 15], div_value=2.0)
            x = paddle.randn((128, 16))
            y = paddle.randint(low=0, high=20, shape=[])
            _ = model(x, y)

        with self.assertRaises(ValueError):
            model = SimpleModel(16, 20, [5, 10, 15], div_value=2.0)
            x = paddle.randn((128, 16))
            y = paddle.randint(low=0, high=20, shape=[128, 1])
            _ = model(x, y)

    def test_grad(self):
        n_classes = 4
        in_features = 8
        cutoffs = [2]

        x = paddle.to_tensor(
            [
                [
                    0.99785769,
                    -1.14492130,
                    0.62956816,
                    0.77550924,
                    -1.97198308,
                    0.50906199,
                    0.76702958,
                    1.31143034,
                ],
                [
                    0.17371807,
                    2.68322444,
                    1.90870595,
                    0.58601201,
                    -0.78898108,
                    0.42098731,
                    -0.74253917,
                    -0.37492049,
                ],
                [
                    -0.77694625,
                    -0.11529812,
                    0.38232428,
                    0.70575434,
                    0.73429769,
                    0.81399834,
                    0.14212975,
                    0.12567955,
                ],
                [
                    0.44165909,
                    0.23613696,
                    0.81143701,
                    0.60473150,
                    0.77017546,
                    0.27865678,
                    -0.03236491,
                    0.31634274,
                ],
                [
                    0.15336825,
                    -0.66177142,
                    -0.01784009,
                    0.08901446,
                    0.85228783,
                    1.49427640,
                    -1.66938102,
                    0.86154014,
                ],
                [
                    -0.60814697,
                    1.26191938,
                    -0.21735200,
                    -0.88890392,
                    0.49093658,
                    -1.28960681,
                    1.06943762,
                    0.15803306,
                ],
                [
                    -0.12136814,
                    -0.16133699,
                    0.15643604,
                    0.79464215,
                    -1.02201688,
                    0.26957786,
                    -0.31038952,
                    0.93334937,
                ],
                [
                    0.66997373,
                    0.95807010,
                    -0.66944563,
                    -0.89887059,
                    1.00404060,
                    0.69594669,
                    -0.82105070,
                    1.15200853,
                ],
            ],
            dtype='float32',
        )
        labels = paddle.to_tensor([3, 3, 3, 2, 3, 0, 0, 0], dtype='int64')
        model = SimpleModel(in_features, n_classes, cutoffs, div_value=2.0)

        _, loss = model(x, labels)

        weights = model.adaptive_softmax.head_weight
        loss.backward()
        analytic_grads = weights.grad.numpy()

        h = 1e-5
        weights_np = weights.numpy().copy()
        grad_numerical = np.zeros_like(weights_np)

        it = np.nditer(
            weights_np, flags=['multi_index'], op_flags=['readwrite']
        )
        while not it.finished:
            ix = it.multi_index
            oldval = weights_np[ix]
            weights_np[ix] = oldval + h
            model.adaptive_softmax.head_weight.set_value(
                paddle.to_tensor(weights_np)
            )
            _, y_pos = model(x, labels)
            loss_pos = y_pos.mean()

            weights_np[ix] = oldval - h
            model.adaptive_softmax.head_weight.set_value(
                paddle.to_tensor(weights_np)
            )
            _, y_neg = model(x, labels)
            loss_neg = y_neg.mean()

            grad_numerical[ix] = (loss_pos - loss_neg) / (2 * h)
            weights_np[ix] = oldval
            it.iternext()

        np.allclose(analytic_grads, grad_numerical, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
