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
from paddle import base
from paddle.hapi.model import to_list


def one_hot(x, n_class):
    res = np.eye(n_class)[np.array(x).reshape(-1)]
    res = res.reshape([*x.shape, n_class])
    return res


def accuracy(pred, label, topk=(1,)):
    maxk = max(topk)
    pred = np.argsort(pred)[..., ::-1][..., :maxk]
    if len(label.shape) == 1:
        label = label.reshape(-1, 1)
    elif label.shape[-1] != 1:
        label = np.argmax(label, axis=-1)
        label = label[..., np.newaxis]
    correct = pred == np.repeat(label, maxk, -1)

    total = np.prod(np.array(label.shape[:-1]))

    res = []
    for k in topk:
        correct_k = correct[..., :k].sum()
        res.append(float(correct_k) / total)
    return res


def convert_to_one_hot(y, C):
    oh = np.random.choice(np.arange(C), C, replace=False).astype('float32') / C
    oh = np.tile(oh[np.newaxis, :], (y.shape[0], 1))
    for i in range(y.shape[0]):
        oh[i, int(y[i])] = 1.0
    return oh


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

        y = np.array([[0], [1], [2], [3]])
        if squeeze_y:
            y = y.squeeze()

        y = paddle.to_tensor(y)

        m = paddle.metric.Accuracy(name='my_acc')

        # check name
        self.assertEqual(m.name(), ['my_acc'])

        correct = m.compute(x, y)
        # check shape and results
        self.assertEqual(correct.shape, [4, 1])
        self.assertEqual(m.update(correct), 0.75)
        self.assertEqual(m.accumulate(), 0.75)

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
        y = paddle.to_tensor(np.array([[0], [1], [2], [3]]))
        correct = m.compute(x, y)
        # check results
        self.assertEqual(m.update(correct), 0.5)
        self.assertEqual(m.accumulate(), 0.625)

        # check reset
        m.reset()
        self.assertEqual(m.total[0], 0.0)
        self.assertEqual(m.count[0], 0.0)

    def test_1d_label(self):
        self.test_acc(True)

    def compare(self, x_np, y_np, k=(1,)):
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)

        m = paddle.metric.Accuracy(name='my_acc', topk=k)
        correct = m.compute(x, y)

        acc_np = accuracy(x_np, y_np, k)
        acc_np = acc_np[0] if len(acc_np) == 1 else acc_np

        # check shape and results
        self.assertEqual(correct.shape, list(x_np.shape)[:-1] + [max(k)])
        self.assertEqual(m.update(correct), acc_np)
        self.assertEqual(m.accumulate(), acc_np)

    def test_3d(self):
        x_np = np.random.rand(2, 3, 4)
        y_np = np.random.randint(4, size=(2, 3, 1))
        self.compare(x_np, y_np)

    def test_one_hot(self):
        x_np = np.random.rand(2, 3, 4)
        y_np = np.random.randint(4, size=(2, 3))
        y_one_hot_np = one_hot(y_np, 4)
        self.compare(x_np, y_one_hot_np, (1, 2))


class TestAccuracyDynamic(unittest.TestCase):
    def setUp(self):
        self.topk = (1,)
        self.class_num = 5
        self.sample_num = 1000
        self.name = None
        self.squeeze_label = False

    def random_pred_label(self):
        label = np.random.randint(
            0, self.class_num, (self.sample_num, 1)
        ).astype('int64')
        pred = np.random.randint(
            0, self.class_num, (self.sample_num, 1)
        ).astype('int32')
        if self.squeeze_label:
            label = label.squeeze()
        pred_one_hot = convert_to_one_hot(pred, self.class_num)
        pred_one_hot = pred_one_hot.astype('float32')

        return label, pred_one_hot

    def test_main(self):
        with base.dygraph.guard(base.CPUPlace()):
            acc = paddle.metric.Accuracy(topk=self.topk, name=self.name)
            for _ in range(10):
                label, pred = self.random_pred_label()
                label_var = paddle.to_tensor(label)
                pred_var = paddle.to_tensor(pred)
                state = to_list(acc.compute(pred_var, label_var))
                acc.update(*[s.numpy() for s in state])
                res_m = acc.accumulate()
                res_f = accuracy(pred, label, self.topk)
                assert np.all(
                    np.isclose(
                        np.array(res_m, dtype='float64'),
                        np.array(res_f, dtype='float64'),
                        rtol=1e-3,
                    )
                ), f"Accuracy precision error: {res_m} != {res_f}"
                acc.reset()
                assert np.sum(acc.total) == 0
                assert np.sum(acc.count) == 0


class TestAccuracyDynamicMultiTopk(TestAccuracyDynamic):
    def setUp(self):
        self.topk = (1, 5)
        self.class_num = 10
        self.sample_num = 1000
        self.name = "accuracy"
        self.squeeze_label = True


class TestPrecision(unittest.TestCase):
    def test_1d(self):
        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([1, 0, 1, 1])

        m = paddle.metric.Precision()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 2.0 / 3.0)

        x = np.array([0.1, 0.5, 0.6, 0.7, 0.2])
        y = np.array([1, 0, 1, 1, 1])
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 4.0 / 6.0)

    def test_2d(self):
        x = np.array([0.1, 0.5, 0.6, 0.7]).reshape(-1, 1)
        y = np.array([1, 0, 1, 1]).reshape(-1, 1)

        m = paddle.metric.Precision()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 2.0 / 3.0)

        x = np.array([0.1, 0.5, 0.6, 0.7, 0.2]).reshape(-1, 1)
        y = np.array([1, 0, 1, 1, 1]).reshape(-1, 1)
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 4.0 / 6.0)

        # check reset
        m.reset()
        self.assertEqual(m.tp, 0.0)
        self.assertEqual(m.fp, 0.0)
        self.assertEqual(m.accumulate(), 0.0)


class TestRecall(unittest.TestCase):
    def test_1d(self):
        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([1, 0, 1, 1])

        m = paddle.metric.Recall()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 2.0 / 3.0)

        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([1, 0, 0, 1])
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 3.0 / 5.0)

        # check reset
        m.reset()
        self.assertEqual(m.tp, 0.0)
        self.assertEqual(m.fn, 0.0)
        self.assertEqual(m.accumulate(), 0.0)


class TestAuc(unittest.TestCase):
    def test_auc_numpy(self):
        x = np.array(
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
        y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
        m = paddle.metric.Auc()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 0.8125)

        m.reset()
        self.assertEqual(m.accumulate(), 0.0)

    def test_auc_tensor(self):
        x = np.array(
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
        y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
        m = paddle.metric.Auc()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 0.8125)

        m.reset()
        self.assertEqual(m.accumulate(), 0.0)


if __name__ == '__main__':
    unittest.main()
