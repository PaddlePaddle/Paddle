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

"""Integration tests for the new paddle.metric API (Accuracy, Precision, Recall, AUROC)."""

import unittest

import numpy as np

import paddle
from paddle.metric import Accuracy, Precision, Recall


class TestAccuracy(unittest.TestCase):
    def test_acc(self, squeeze_y=False):
        x = paddle.to_tensor(
            np.array(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.1, 0.4, 0.3, 0.2],
                    [0.1, 0.2, 0.4, 0.3],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            )
        )

        y = np.array([3, 1, 2, 3])
        if squeeze_y:
            y = y.squeeze()
        y = paddle.to_tensor(y)

        m = Accuracy(task="multiclass", num_classes=4)

        m.update(x, y)
        res = m.compute()
        self.assertAlmostEqual(float(res), 1.0)

        x = paddle.to_tensor(
            np.array(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.1, 0.3, 0.4, 0.2],
                    [0.1, 0.2, 0.4, 0.3],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            )
        )
        y = paddle.to_tensor(np.array([3, 2, 2, 3]))
        m.update(x, y)
        res = m.compute()
        self.assertAlmostEqual(float(res), 1.0)

        m.reset()
        self.assertEqual(m._update_count, 0)

    def test_1d_label(self):
        self.test_acc(True)

    def test_topk(self):
        x = paddle.to_tensor(np.random.rand(10, 4).astype('float32'))
        y = paddle.to_tensor(np.random.randint(0, 4, (10,)).astype('int64'))

        m = Accuracy(task="multiclass", num_classes=4, top_k=2)
        m.update(x, y)
        res = m.compute()
        self.assertGreaterEqual(float(res), 0.0)
        self.assertLessEqual(float(res), 1.0)


class TestPrecision(unittest.TestCase):
    def test_binary(self):
        x = paddle.to_tensor(np.array([0.1, 0.5, 0.6, 0.7]))
        y = paddle.to_tensor(np.array([1, 0, 1, 1]))

        m = Precision(task="binary")
        m.update(x, y)
        r = float(m.compute())
        self.assertAlmostEqual(r, 1.0)

        x = paddle.to_tensor(np.array([0.1, 0.5, 0.6, 0.7, 0.2]))
        y = paddle.to_tensor(np.array([1, 0, 1, 1, 1]))
        m.update(x, y)
        r = float(m.compute())
        self.assertAlmostEqual(r, 1.0)

        m.reset()
        self.assertEqual(m._update_count, 0)


class TestRecall(unittest.TestCase):
    def test_binary(self):
        x = paddle.to_tensor(np.array([0.1, 0.5, 0.6, 0.7]))
        y = paddle.to_tensor(np.array([1, 0, 1, 1]))

        m = Recall(task="binary")
        m.update(x, y)
        r = float(m.compute())
        self.assertAlmostEqual(r, 2.0 / 3.0)

        x = paddle.to_tensor(np.array([0.1, 0.5, 0.6, 0.7]))
        y = paddle.to_tensor(np.array([1, 0, 0, 1]))
        m.update(x, y)
        r = float(m.compute())
        self.assertAlmostEqual(r, 3.0 / 5.0)

        m.reset()
        self.assertEqual(m._update_count, 0)


class TestAuc(unittest.TestCase):
    def test_auc(self):
        x = paddle.to_tensor(
            np.array(
                [
                    [0.78, 0.22],
                    [0.62, 0.38],
                    [0.55, 0.45],
                    [0.30, 0.70],
                    [0.14, 0.86],
                    [0.59, 0.41],
                    [0.91, 0.08],
                    [0.16, 0.84],
                ]
            )
        )
        y = paddle.to_tensor(np.array([0, 1, 1, 0, 1, 0, 0, 1]))

        from paddle.metric import AUROC

        m = AUROC(task="binary")
        m.update(x[:, 1], y)
        r = float(m.compute())
        self.assertAlmostEqual(r, 0.8125, places=3)

        m.reset()
        self.assertEqual(m._update_count, 0)


if __name__ == '__main__':
    unittest.main()
