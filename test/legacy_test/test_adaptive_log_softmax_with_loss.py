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
        logprob = self.adaptive_softmax.log_prob(self.fc(input))
        return logprob.argmax(axis=1)


class TestNNAdaptiveLogSoftmaxWithLossAPI(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.place = ['cpu']
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

    def test_forwadr(self):
        n_classes = 4
        in_features = 8
        cutoffs = [2]

        x = paddle.randn([8, in_features])
        print(x)
        labels = paddle.randint(0, n_classes, [8])
        print(labels)
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


if __name__ == "__main__":
    unittest.main()
